"""
Two-stage relevance: when the abstract verdict is ``inconclusive``, the stage
fetches the cited paper's full text and re-judges on claim-relevant chunks. The
network fetch is monkeypatched, so this is fully offline.
"""

from __future__ import annotations

from citation_verifier.grounding import fulltext
from citation_verifier.schema import CitationRecord, CitedAs, Claim, Resolved, SupportsClaim
from citation_verifier.stages.relevance import RelevanceVerdict, fill_relevance_batch


class _NoResolver:
    def resolve(self, *args, **kwargs):
        return None


def _rec(claim="the model is fine-tuned on the WebText corpus"):
    return CitationRecord(
        paper_id="p",
        claim_id="c1",
        cite_key="ref-1",
        claim=Claim(claim_id="c1", text=claim),
        cited_as=CitedAs(raw="ref"),
        resolved=Resolved(
            source="s2",
            arxiv_id="1706.03762",
            abstract="A short abstract that does not mention the training corpus.",
        ),
    )


def test_inconclusive_abstract_escalates_to_full_text(monkeypatch):
    # Stage-2 full text: the Introduction (a default section) confirms the claim.
    monkeypatch.setattr(
        fulltext,
        "fetch_full_text_with_source",
        lambda *a, **k: fulltext.FullTextResult(
            r"\section{Introduction} The model is fine-tuned on the WebText corpus.",
            "arxiv_eprint",
            "https://arxiv.org/e-print/1706.03762",
        ),
    )
    calls: list = []

    def judge_batch(items):
        calls.append(items)
        out = []
        for it in items:
            if "WebText" in it["abstract"]:  # stage 2 evidence = full-text chunk
                out.append(RelevanceVerdict(SupportsClaim.SUPPORTS, justification="confirmed in §Introduction"))
            else:  # stage 1: the abstract can't tell
                out.append(RelevanceVerdict(SupportsClaim.INCONCLUSIVE, justification="not in abstract"))
        return out

    rec = _rec()
    fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge_batch)

    assert len(calls) == 2  # stage 1 (abstract) + stage 2 (full text)
    assert rec.supports_claim == SupportsClaim.SUPPORTS  # upgraded after reading full text
    assert "based on full text" in (rec.notes or "")
    ft_ev = [e for e in rec.evidence if e.kind == "full_text"]
    assert ft_ev  # full-text evidence recorded
    # provenance: the evidence is tagged with its channel + section + source URL
    assert ft_ev[0].source.startswith("arxiv_eprint §")
    assert "Introduction" in ft_ev[0].source
    assert ft_ev[0].url == "https://arxiv.org/e-print/1706.03762"


def test_no_arxiv_id_keeps_stage1_verdict(monkeypatch):
    # No arXiv id -> no full text to fetch -> the stage-1 inconclusive verdict stands.
    monkeypatch.setattr(
        fulltext,
        "fetch_full_text_with_source",
        lambda *a, **k: fulltext.FullTextResult("SHOULD NOT BE USED", "arxiv_html", "x"),
    )

    def judge_batch(items):
        return [RelevanceVerdict(SupportsClaim.INCONCLUSIVE, justification="x") for _ in items]

    rec = _rec()
    rec.resolved.arxiv_id = None
    fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge_batch)
    assert rec.supports_claim == SupportsClaim.INCONCLUSIVE


def _stage2_supports_on(marker, full_text):
    """A judge that returns SUPPORTS once the full-text marker is in the evidence."""
    def judge_batch(items):
        out = []
        for it in items:
            if marker in it["abstract"]:  # stage-2 evidence carries the marker
                out.append(RelevanceVerdict(SupportsClaim.SUPPORTS, justification="ft"))
            else:  # stage-1 abstract
                out.append(full_text(it))
        return out
    return judge_batch


def test_does_not_from_abstract_escalates_to_full_text(monkeypatch):
    # does_not is the most consequential verdict; an abstract that doesn't mention the
    # claim is NOT proof the body doesn't support it — escalate before committing.
    monkeypatch.setattr(
        fulltext, "fetch_full_text_with_source",
        lambda *a, **k: fulltext.FullTextResult(
            r"\section{Introduction} The model is fine-tuned on the WebText corpus.", "arxiv_eprint", "u"),
    )
    judge = _stage2_supports_on(
        "WebText", lambda it: RelevanceVerdict(SupportsClaim.DOES_NOT, justification="absent from abstract")
    )
    rec = _rec()
    fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge)
    assert rec.supports_claim == SupportsClaim.SUPPORTS  # body read -> false negative avoided
    assert "based on full text" in (rec.notes or "")


def test_detail_partial_escalates_but_generic_partial_does_not(monkeypatch):
    # partial escalates ONLY when the claim is about specific methods/data/results
    # (those live in the body); a generic background partial stands on the abstract.
    monkeypatch.setattr(
        fulltext, "fetch_full_text_with_source",
        lambda *a, **k: fulltext.FullTextResult(
            r"\section{Results} It reaches 95% accuracy on the benchmark.", "arxiv_eprint", "u"),
    )
    judge = _stage2_supports_on(
        "accuracy", lambda it: RelevanceVerdict(SupportsClaim.PARTIAL, justification="abstract partial")
    )
    # detail-bearing claim (accuracy/benchmark) -> escalates -> upgraded on full text
    detail = _rec("the model reaches 95% accuracy on the QA benchmark")
    fill_relevance_batch([detail], resolver=_NoResolver(), judge_batch=judge)
    assert detail.supports_claim == SupportsClaim.SUPPORTS
    assert "based on full text" in (detail.notes or "")
    # generic background claim -> partial stands, abstract-only (no escalation)
    generic = _rec("prior work has explored this direction broadly")
    fill_relevance_batch([generic], resolver=_NoResolver(), judge_batch=judge)
    assert generic.supports_claim == SupportsClaim.PARTIAL
    assert "based on abstract only" in (generic.notes or "")
