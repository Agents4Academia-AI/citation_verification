"""
Offline tests for the relevance judge seam (STEP 2).

No network and no SDK: we exercise the pure parsing/slicing helpers, the
honest-abstain path (empty evidence => unverified, never calls the model), and
that `fill_relevance` actually applies an injected judge's verdict.
"""

from __future__ import annotations

from citation_verifier.backends.relevance_judge import (
    LLMRelevanceJudge,
    _parse_batch_verdicts,
    _parse_verdict,
    _slice_intro,
)
from citation_verifier.schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Priority,
    SupportsClaim,
)
from citation_verifier.stages.relevance import (
    RelevanceVerdict,
    fill_relevance,
    fill_relevance_batch,
)


class _NullResolver:
    """Resolver stub — never matches (so no network is touched)."""

    def resolve(self, cite_key, reference, /):  # noqa: D401
        return None

    def candidates(self, reference, /, max_results=4):
        return []


def _stub(text: str = "Background on transformers.") -> CitationRecord:
    return CitationRecord(
        paper_id="p", claim_id="c1", cite_key="k1",
        claim=Claim(claim_id="c1", text=text), cited_as=CitedAs(),
    )


# ── pure parsing ─────────────────────────────────────────────────────
def test_parse_verdict_handles_loose_does_not():
    v = _parse_verdict('{"supports_claim":"does not","priority":"obligatory","confidence":0.8,"justification":"x"}')
    assert v.supports_claim is SupportsClaim.DOES_NOT
    assert v.priority is Priority.OBLIGATORY
    assert v.confidence == 0.8


def test_parse_verdict_fenced_and_pross():
    text = "Here is my verdict:\n```json\n{\"supports_claim\": \"supports\"}\n```\nDone."
    v = _parse_verdict(text)
    assert v.supports_claim is SupportsClaim.SUPPORTS


def test_parse_verdict_garbage_abstains():
    v = _parse_verdict("no json at all")
    assert v.supports_claim is SupportsClaim.INCONCLUSIVE
    assert v.priority is None


# ── intro slicing ────────────────────────────────────────────────────
def test_slice_intro_cuts_at_next_section():
    text = "Title Abstract blah. 1 Introduction this is the intro body. Related Work other stuff."
    intro = _slice_intro(text)
    assert "this is the intro body" in intro
    assert "other stuff" not in intro


def test_slice_intro_absent_returns_empty():
    assert _slice_intro("no such heading here") == ""


# ── honest abstain without evidence (no SDK/network) ─────────────────
def test_judge_abstains_without_evidence():
    judge = LLMRelevanceJudge(settings=None)
    assert judge.model == "claude-opus-4-8"
    verdict = judge(claim="X improves Y.", abstract="", resolved=None)
    assert verdict.supports_claim is SupportsClaim.INCONCLUSIVE


# ── the seam: fill_relevance applies an injected judge ────────────────
def test_fill_relevance_applies_injected_judge():
    rec = _stub("Background on transformers.")  # heuristic alone => HELPFUL

    def judge(**kwargs):  # accepts claim/abstract/resolved/... via **kwargs
        return RelevanceVerdict(
            supports_claim="supports", priority="obligatory", confidence=0.9, justification="stub"
        )

    out = fill_relevance(rec, resolver=_NullResolver(), judge=judge)
    # robust to enum-vs-value storage
    assert SupportsClaim(out.supports_claim) is SupportsClaim.SUPPORTS
    assert Priority(out.priority) is Priority.OBLIGATORY  # judge overrode the heuristic


def test_fill_relevance_without_judge_abstains():
    rec = _stub()
    out = fill_relevance(rec, resolver=_NullResolver(), judge=None)
    assert SupportsClaim(out.supports_claim) is SupportsClaim.INCONCLUSIVE


# ── batched judging ──────────────────────────────────────────────────
def test_parse_batch_verdicts_maps_by_index():
    text = '[{"i":0,"supports_claim":"supports","priority":"obligatory"},{"i":1,"supports_claim":"does not"}]'
    got = _parse_batch_verdicts(text)
    assert set(got) == {0, 1}
    assert got[0].supports_claim is SupportsClaim.SUPPORTS
    assert got[0].priority is Priority.OBLIGATORY
    assert got[1].supports_claim is SupportsClaim.DOES_NOT


def test_judge_batch_abstains_without_evidence():
    judge = LLMRelevanceJudge(settings=None)
    out = judge.judge_batch([
        {"claim": "X improves Y.", "abstract": "", "resolved": None},
        {"claim": "Z does W.", "abstract": "", "resolved": None},
    ])
    assert len(out) == 2
    assert all(v.supports_claim is SupportsClaim.INCONCLUSIVE for v in out)
    assert judge.calls == 0  # no model call when there is no evidence


def test_fill_relevance_batch_applies_verdicts():
    r1, r2 = _stub("We use method M as a baseline."), _stub("See also surveys.")

    def judge_batch(items):
        assert len(items) == 2
        return [
            RelevanceVerdict(supports_claim="supports", priority="obligatory"),
            RelevanceVerdict(supports_claim="partial", priority="helpful"),
        ]

    out = fill_relevance_batch([r1, r2], resolver=_NullResolver(), judge_batch=judge_batch)
    assert SupportsClaim(out[0].supports_claim) is SupportsClaim.SUPPORTS
    assert Priority(out[0].priority) is Priority.OBLIGATORY
    assert SupportsClaim(out[1].supports_claim) is SupportsClaim.PARTIAL


# ── table citations are not judged (in_table) ────────────────────────
def _table_stub() -> CitationRecord:
    return CitationRecord(
        paper_id="p", claim_id="ct", cite_key="ref-9",
        claim=Claim(claim_id="ct", text="Table 3 Comparison of models"),
        cited_as=CitedAs(), in_table=True,
    )


def test_fill_relevance_skips_table_citations():
    """A table/figure citation is SKIPPED (no verdict) and never reaches the judge."""
    called = []

    def judge(**kwargs):
        called.append(1)
        return RelevanceVerdict(supports_claim="supports", priority="obligatory")

    out = fill_relevance(_table_stub(), resolver=_NullResolver(), judge=judge)
    assert not called
    assert SupportsClaim(out.supports_claim) is SupportsClaim.SKIPPED


def test_fill_relevance_batch_excludes_table_citations_from_the_judge():
    table, prose = _table_stub(), _stub("We use method M as a baseline.")

    def judge_batch(items):
        assert [it["cite_key"] for it in items] == ["k1"]  # only the prose claim is judged
        return [RelevanceVerdict(supports_claim="supports", priority="obligatory")]

    out = fill_relevance_batch([table, prose], resolver=_NullResolver(), judge_batch=judge_batch)
    assert SupportsClaim(out[0].supports_claim) is SupportsClaim.SKIPPED  # table skipped, no verdict
    assert SupportsClaim(out[1].supports_claim) is SupportsClaim.SUPPORTS


def test_verdict_downgrades_clear_contradiction_to_does_not():
    from citation_verifier.backends.relevance_judge import _verdict_from_row

    v = _verdict_from_row(
        {"supports_claim": "partial", "justification": "chronologically impossible; the evidence contradicts the claim"}
    )
    assert SupportsClaim(v.supports_claim) is SupportsClaim.DOES_NOT
    # mere absence ("does not mention") is honest abstention, not a contradiction
    v2 = _verdict_from_row(
        {"supports_claim": "partial", "justification": "the evidence does not mention this specific result"}
    )
    assert SupportsClaim(v2.supports_claim) is SupportsClaim.PARTIAL


def test_group_system_reserves_inconclusive_for_insufficient_evidence():
    """The judge prompt must follow SKILL.md: inconclusive = evidence INSUFFICIENT
    (couldn't retrieve), does_not = evidence present but off-claim. Guards against
    silently reverting to the old "absence in the abstract => inconclusive" bias.
    """
    from citation_verifier.backends.relevance_judge import _GROUP_SYSTEM

    s = _GROUP_SYSTEM.lower()
    # does_not broadened beyond explicit contradiction (off-topic/narrower scope).
    assert "not only on explicit contradiction" in s
    assert "materially" in s and "different from or narrower" in s
    # inconclusive reserved for genuinely insufficient evidence, not a soft does_not.
    assert "insufficient to judge" in s
    assert "soft 'does_not'" in s
