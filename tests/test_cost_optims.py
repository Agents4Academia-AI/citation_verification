"""
Offline tests for the token-cost / latency optimizations:

  - STEP 1 (agentic): the correctness pass dedups by reference, so a work cited
    in N places resolves ONCE, copies its verdict to the siblings, and preserves
    the original stub order.
  - STEP 2 transport (relevance judge): mode selection prefers the direct
    Messages API when a key is configured, falls back to the query() subscription
    path, and yields no judge when neither SDK is available — and the Messages API
    usage mapper prices a response correctly.

No SDK and no network: stage functions are faked and module availability is
monkeypatched.
"""

from __future__ import annotations

from types import SimpleNamespace

from citation_verifier.backends import relevance_judge as rj
from citation_verifier.backends.agentic import AgenticBackend
from citation_verifier.backends.usage import usage_from_message
from citation_verifier.grounding.resolver import MultiSourceResolver, _title_tokens_contradict
from citation_verifier.interfaces import Candidate, RunUsage, VerificationResult
from citation_verifier.schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Exists,
    MatchMethod,
    Resolved,
    SupportsClaim,
)
from citation_verifier.stages.relevance import RelevanceVerdict


def _stub(claim_id: str, cite_key: str, raw: str) -> CitationRecord:
    return CitationRecord(
        paper_id="p",
        claim_id=claim_id,
        cite_key=cite_key,
        claim=Claim(claim_id=claim_id, text="some claim"),
        cited_as=CitedAs(raw=raw),
    )


# ── STEP 2 dedup (agentic correctness pass) ──────────────────────────
def test_correctness_pass_dedups_by_reference_and_keeps_order():
    """A reference cited twice resolves once; siblings get a deep copy; order kept."""
    calls = {"n": 0}

    def fake_fill_correctness(rec, *, resolver):  # the real stage's signature
        calls["n"] += 1
        rec.exists = Exists.YES
        rec.resolved = Resolved(match_method=MatchMethod.DOI, title="T", abstract="A")
        rec.metadata_issues = []
        return rec

    # c1 and c2 cite the SAME work (cite_key k1); c3 cites a different one.
    stubs = [
        _stub("c1", "k1", "ref one"),
        _stub("c2", "k1", "ref one"),
        _stub("c3", "k2", "ref two"),
    ]
    backend = AgenticBackend(settings=None)
    result = VerificationResult(paper_id="p", backend="agentic", usage=RunUsage(backend="agentic"))

    backend._correctness_pass(stubs, fake_fill_correctness, resolver=None, usage=result.usage, result=result)

    assert calls["n"] == 2  # resolved once per UNIQUE reference, not 3 times
    assert [r.claim_id for r in result.records] == ["c1", "c2", "c3"]  # original order
    assert all(Exists(r.exists) is Exists.YES for r in result.records)

    rep, sib = result.records[0], result.records[1]
    assert sib.resolved is not None and sib.resolved.title == "T"
    assert sib.resolved is not rep.resolved  # deep copy, not a shared object
    assert sib.evidence is not rep.evidence  # each record owns its evidence list

    # Bulk-tier usage is still recorded per record (one slice each).
    assert result.usage.by_tier["bulk"].tool_calls == 3


# ── STEP 1 transport selection ───────────────────────────────────────
def test_has_api_key_reads_settings_and_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rj._has_api_key(None) is False
    assert rj._has_api_key({"anthropic_api_key": "sk-x"}) is True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert rj._has_api_key(None) is True


def test_select_mode_prefers_messages_with_key(monkeypatch):
    # Only `anthropic` importable + a key => the cheaper Messages API path.
    monkeypatch.setattr(rj, "_module_available", lambda name: name == "anthropic")
    assert rj._select_mode({"anthropic_api_key": "k"}) == "messages"


def test_select_mode_falls_back_to_query_without_key(monkeypatch):
    # No key => the raw Messages API has no auth; fall back to the subscription.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(rj, "_module_available", lambda name: True)
    assert rj._select_mode(None) == "query"


def test_select_mode_none_without_any_sdk(monkeypatch):
    monkeypatch.setattr(rj, "_module_available", lambda name: False)
    assert rj._select_mode({"anthropic_api_key": "k"}) is None
    assert rj.build_relevance_judge({"anthropic_api_key": "k"}) is None  # keyless floor


# ── STEP 1 usage mapping (Messages API response) ─────────────────────
def test_judge_batch_groups_evidence_by_citation(monkeypatch):
    """A citation cited twice is sent to the model ONCE (evidence shared); each
    claim still gets its own verdict, mapped back to every input item in order."""
    judge = rj.LLMRelevanceJudge(settings=None, mode="query")
    seen_chunks = []

    def fake_chunk(chunk):  # chunk: [(cite_key, group)]
        seen_chunks.append(chunk)
        return {
            ck: {cid: RelevanceVerdict(supports_claim="supports") for (_i, cid, _c) in g["members"]}
            for ck, g in chunk
        }

    monkeypatch.setattr(judge, "_judge_citation_chunk", fake_chunk)

    items = [  # k1 cited twice (same evidence), k2 once
        {"cite_key": "k1", "claim_id": "c1", "claim": "A", "abstract": "EV1", "resolved": None},
        {"cite_key": "k1", "claim_id": "c2", "claim": "B", "abstract": "EV1", "resolved": None},
        {"cite_key": "k2", "claim_id": "c3", "claim": "C", "abstract": "EV2", "resolved": None},
    ]
    out = judge.judge_batch(items)

    assert len(out) == 3  # one verdict per claim-site, aligned to input
    assert all(SupportsClaim(v.supports_claim) is SupportsClaim.SUPPORTS for v in out)
    # 2 unique citations sent (not 3 pairs); k1's evidence appears once for 2 claims
    grouped = dict(seen_chunks[0])
    assert set(grouped) == {"k1", "k2"}
    assert grouped["k1"]["evidence"] == "EV1" and len(grouped["k1"]["members"]) == 2


def test_gate_corroborates_messy_reference_authors():
    """The author gate must corroborate the CANDIDATE's surnames against the raw
    reference (robust to 'Last, First' order + initials), not parse the reference's
    author list — which used to mangle e.g. 'Diederik P. Kingma' to {'p'} and veto a
    perfect title match. Hard rejection of subset false positives belongs to the
    title-token gate, not noisy author/year metadata."""
    g = MultiSourceResolver._gate

    # exact-title candidate; reference uses 'First M. Last' with a middle initial
    adam = Candidate(
        source="arxiv",
        title="Adam: A Method for Stochastic Optimization",
        authors=["Diederik P. Kingma", "Jimmy Ba"],
        year=2014,
    )
    ref_adam = 'Diederik P. Kingma, Jimmy Ba. "Adam: A Method for Stochastic Optimization". ICLR 2015'
    assert g(adam, ref_adam, 2015) is True  # was wrongly False before the fix

    # 'Last, First, Last, First' format must still corroborate
    di = Candidate(
        source="arxiv",
        title="Diff-Instruct",
        authors=["Weijian Luo", "Tianyang Hu", "Zhenguo Li"],
        year=2023,
    )
    ref_di = "Luo, Weijian, Hu, Tianyang, Li, Zhenguo. Diff-Instruct: A Universal Approach. 2024"
    assert g(di, ref_di, 2024) is True

    # Token-subset false positives are blocked before author/year corroboration.
    spurious = Candidate(source="crossref", title="Mathieu, Emile", authors=["Emile Mathieu"], year=2001)
    ref_spurious = "Fjelde, Mathieu, Dutordoir. Introduction to Flow Matching. 2024"
    assert _title_tokens_contradict(ref_spurious, spurious.title)
    assert MultiSourceResolver(validate_urls=False)._match(ref_spurious, [spurious]) is None

    # A different work with no meaningful title similarity is still rejected.
    other = Candidate(source="crossref", title="X", authors=["Alice Smith", "Bob Jones"], year=2024)
    ref_other = "Carol White, Dan Black. High-Resolution Image Synthesis. 2024"
    assert MultiSourceResolver(validate_urls=False)._match(ref_other, [other]) is None


def test_usage_from_message_maps_and_prices():
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=0,
        )
    )
    u = usage_from_message(resp, "claude-opus-4-6", "agentic")
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (100, 20, 80)
    assert u.num_turns == 1
    assert u.cost_usd > 0  # priced via the pricing table
