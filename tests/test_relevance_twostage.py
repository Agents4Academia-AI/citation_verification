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


def _rec():
    return CitationRecord(
        paper_id="p",
        claim_id="c1",
        cite_key="ref-1",
        claim=Claim(claim_id="c1", text="the model is fine-tuned on the WebText corpus"),
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
        "fetch_full_text",
        lambda *a, **k: r"\section{Introduction} The model is fine-tuned on the WebText corpus.",
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
    assert any(e.kind == "full_text" for e in rec.evidence)


def test_no_arxiv_id_keeps_stage1_verdict(monkeypatch):
    # No arXiv id -> no full text to fetch -> the stage-1 inconclusive verdict stands.
    monkeypatch.setattr(fulltext, "fetch_full_text", lambda *a, **k: "SHOULD NOT BE USED")

    def judge_batch(items):
        return [RelevanceVerdict(SupportsClaim.INCONCLUSIVE, justification="x") for _ in items]

    rec = _rec()
    rec.resolved.arxiv_id = None
    fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge_batch)
    assert rec.supports_claim == SupportsClaim.INCONCLUSIVE
