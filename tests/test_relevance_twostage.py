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


def test_every_partial_is_checked_against_the_body(monkeypatch):
    # "Related but weaker than the claim", said on an abstract, is a statement about the
    # abstract. Escalating only claims a keyword list took for methods/data/results left
    # a third of one batch's pairs at "partial" with the paper unread.
    monkeypatch.setattr(
        fulltext, "fetch_full_text_with_source",
        lambda *a, **k: fulltext.FullTextResult(
            r"\section{Results} It reaches 95% accuracy on the benchmark. "
            r"\section{Related} Prior work has explored this direction broadly.", "arxiv_eprint", "u"),
    )
    for claim, word in (("the model reaches 95% accuracy on the QA benchmark", "accuracy"),
                        ("prior work has explored this direction broadly", "explored")):
        judge = _stage2_supports_on(
            word, lambda it: RelevanceVerdict(SupportsClaim.PARTIAL, justification="abstract partial")
        )
        rec = _rec(claim)
        fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge)
        assert rec.supports_claim == SupportsClaim.SUPPORTS, claim
        assert "based on full text" in (rec.notes or "")


def test_a_claim_the_full_text_answer_left_out_keeps_its_first_verdict(monkeypatch):
    from citation_verifier.backends.relevance_judge import _abstain

    monkeypatch.setattr(
        fulltext, "fetch_full_text_with_source",
        lambda *a, **k: fulltext.FullTextResult(r"\section{Results} Something.", "arxiv_eprint", "u"),
    )

    def judge_batch(items):
        if "Something" in items[0]["abstract"]:          # stage 2: the answer omits the claim
            return [_abstain("batch judge returned no verdict")]
        return [RelevanceVerdict(SupportsClaim.PARTIAL, justification="abstract partial")]

    rec = _rec("the model reaches 95% accuracy")
    fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge_batch)
    assert rec.supports_claim == SupportsClaim.PARTIAL and "abstract partial" in rec.notes


def test_a_paper_short_enough_is_judged_whole(monkeypatch):
    from citation_verifier.stages import relevance as R

    paper = r"\section{Intro} " + ("Background words here. " * 50) + r"\section{Results} The answer is 42."
    monkeypatch.setattr(fulltext, "fetch_full_text_with_source",
                        lambda *a, **k: fulltext.FullTextResult(paper, "arxiv_eprint", "u"))
    seen = []

    def judge_batch(items):
        seen.extend(it["abstract"] for it in items)
        return [RelevanceVerdict(SupportsClaim.PARTIAL, justification="p") for _ in items]

    fill_relevance_batch([_rec("an unrelated phrasing of the claim")], resolver=_NoResolver(), judge_batch=judge_batch)
    assert any("The answer is 42." in e and "Background words" in e for e in seen[1:])
    assert len(paper) < R._WHOLE_PAPER_CHARS


def test_judge_batches_respect_a_size_budget():
    from citation_verifier.backends.relevance_judge import _chunks_by_budget

    sizes = {"a": 100, "b": 150, "c": 300, "d": 10}
    groups = list(_chunks_by_budget(list(sizes), sizes.get, max_items=10, max_chars=260))
    assert groups == [["a", "b"], ["c"], ["d"]]
    assert list(_chunks_by_budget(list("abcde"), lambda _: 1, max_items=2, max_chars=99)) == [["a", "b"], ["c", "d"], ["e"]]


def test_the_whole_paper_verdict_is_final(monkeypatch):
    """The judge's own excerpt pass used to run after the stage's whole-paper re-judge and
    replace its verdict: "partial" on the whole paper became "inconclusive" on three short
    excerpts, still labelled as based on the full text."""
    from citation_verifier.backends.relevance_judge import LLMRelevanceJudge

    paper = (r"\section{Introduction} Background. " + "Filler about other topics. " * 40
             + r"\section{Results} Our method reaches 91 percent accuracy on the benchmark.")
    monkeypatch.setattr(fulltext, "full_text_with_trail",
                        lambda r: fulltext.FullTextResult(paper, "arxiv_eprint", "u"))
    judge = LLMRelevanceJudge(settings={}, mode="messages")
    seen = []

    def chunk(rows):
        out = {}
        for ck, grp in rows:
            whole = "complete text of the cited paper" in grp["evidence"]
            seen.append("whole" if whole else "first")
            verdict = SupportsClaim.PARTIAL if whole else SupportsClaim.INCONCLUSIVE
            out[ck] = {cid: RelevanceVerdict(verdict, justification=seen[-1]) for _i, cid, _c in grp["members"]}
        return out

    monkeypatch.setattr(judge, "_judge_citation_chunk", chunk)
    rec = _rec("the method reaches 91% accuracy on the benchmark")
    rec.resolved.abstract = "An abstract about something else, long enough to count as an abstract. " * 4
    fill_relevance_batch([rec], resolver=_NoResolver(), judge_batch=judge.judge_batch)
    assert seen == ["first", "whole"]
    assert rec.supports_claim == SupportsClaim.PARTIAL
