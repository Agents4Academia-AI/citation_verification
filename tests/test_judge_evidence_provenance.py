"""Offline regressions for whole-paper verdict / unrelated-excerpt mismatches."""
import json

import pytest

from citation_verifier.backends.relevance_judge import LLMRelevanceJudge
from citation_verifier.grounding import fulltext
from citation_verifier.schema import CitationRecord, CitedAs, Claim, Resolved
from citation_verifier.stages import relevance

DECOY = "Calibration on TreeLSTM is reported in the benchmark table."
PASSAGE = "Temperature scaling leaves the predicted class unchanged."


def run_stage(monkeypatch, quote, *, excerpts=False):
    paper = f"Introduction\n{DECOY}\nResults\n{PASSAGE}"
    monkeypatch.setattr(fulltext, "full_text_with_trail",
                        lambda _: fulltext.FullTextResult(paper, "arxiv_html", "https://example.org/paper"))
    # Reproduce the old bug: heuristic selection picks unrelated text.
    monkeypatch.setattr(fulltext, "select_evidence_chunks", lambda *a, **k: [("TreeLSTM", DECOY)])
    if excerpts:
        monkeypatch.setattr(relevance, "_WHOLE_PAPER_CHARS", 1)
    judge = LLMRelevanceJudge(settings={}, mode="messages")

    def respond(system, user, max_tokens):
        payload = json.loads(user.split("\n", 1)[1])
        full = "complete text" in payload[0]["evidence"] or "[TreeLSTM]" in payload[0]["evidence"]
        verdict = {"claim_id": "c", "supports_claim": "supports" if full else "inconclusive",
                   "justification": "Class predictions do not change." if full else "Not in abstract."}
        if full and quote is not None:
            verdict["evidence_quote"] = quote
        return json.dumps([{"cite_key": "ref", "claims": [verdict]}])

    monkeypatch.setattr(judge, "_run_messages", respond)
    monkeypatch.setattr(judge, "_assemble_evidence", lambda abstract, resolved: (abstract, "L0"))
    rec = CitationRecord(paper_id="p", claim_id="c", cite_key="ref", claim=Claim(claim_id="c", text="Predictions are invariant."),
                         cited_as=CitedAs(raw="Reference"), resolved=Resolved(source="arxiv", arxiv_id="1706.04599", abstract="Background abstract."))
    relevance.fill_relevance_batch([rec], resolver=None, judge_batch=judge.judge_batch)
    return rec


def test_reports_judges_actual_passage_not_heuristic_decoy(monkeypatch):
    rec = run_stage(monkeypatch, PASSAGE)
    evidence = [e for e in rec.evidence if e.kind == "full_text"]
    assert len(evidence) == 1
    assert evidence[0].quote == PASSAGE
    assert evidence[0].url == "https://example.org/paper"
    assert "judge-selected passage" in evidence[0].source
    assert "TreeLSTM" not in rec.notes
    assert rec.supports_claim.value == "supports"


@pytest.mark.parametrize("quote", [None, "", 42, "An invented passage not found anywhere in the cited paper.",
                                   "Temperature scaling ... unchanged.", "x" * 2001,
                                   "[complete text of the cited paper]"])
def test_missing_or_invalid_quote_is_explicit_without_substituting_evidence(monkeypatch, quote):
    rec = run_stage(monkeypatch, quote)
    assert not [e for e in rec.evidence if e.kind == "full_text"]
    assert "no validated judge-selected quotation available" in rec.notes
    # Evidence traceability does not silently change semantic labels or imply proof of absence.
    assert rec.supports_claim.value == "supports"


def test_quote_from_unseen_part_of_long_paper_is_rejected(monkeypatch):
    rec = run_stage(monkeypatch, PASSAGE, excerpts=True)
    assert not [e for e in rec.evidence if e.kind == "full_text"]
    assert "no validated" in rec.notes


def test_whitespace_only_variation_is_accepted(monkeypatch):
    rec = run_stage(monkeypatch, PASSAGE.replace(" ", "\n"))
    assert [e.quote for e in rec.evidence if e.kind == "full_text"] == [PASSAGE]
