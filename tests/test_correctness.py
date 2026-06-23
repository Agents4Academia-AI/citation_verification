"""
Offline tests for the correctness stage's author matching. Cited references use
Vancouver "Surname Initials" while canonical records use "First Last", so an
abbreviated cited name (``Firat M``) must match its full form (``Mehmet Firat``)
without flagging — only a genuinely different surname/initial should flag.
"""

from __future__ import annotations

from citation_verifier.stages.correctness import _author_issue, _same_author, _unresolved_note


def test_same_author_matches_abbreviated_and_glued_forms():
    assert _same_author("Firat M", "Mehmet Firat")
    assert _same_author("Yang Z", "Zhengyuan Yang")
    assert _same_author("Adewumi T", "Tosin P. Adewumi")
    assert _same_author("Ferrucci DA", "David A Ferrucci")
    assert _same_author("Chen C-FR", "Chun-Fu Chen")          # 3-letter initials
    assert _same_author("YangZ", "Zhengyuan Yang")            # glued despacing artifact
    assert _same_author("V aswani A", "Ashish Vaswani")       # despace split "V aswani" -> Vaswani
    assert _same_author("Y uan H", "Hongping Yuan")           # despace split "Y uan" -> Yuan
    assert _same_author("Yang, Zhengyuan", "Zhengyuan Yang")  # "Last, First"


def test_unresolved_note_lists_the_searched_sources():
    class _Resolver:
        sources = ("crossref", "arxiv", "s2", "dblp")

    note = _unresolved_note(_Resolver())
    assert "could not retrieve" in note
    assert "Crossref" in note and "Semantic Scholar" in note and "arXiv" in note


def test_same_author_rejects_real_differences():
    assert not _same_author("Smith J", "Jones A")  # different surname
    assert not _same_author("Yang Z", "Yang Q")    # same surname, different initial


def test_author_issue_clears_abbreviations_but_keeps_real_mismatch():
    # Whole list, every name abbreviated against its full form -> no issue.
    assert (
        _author_issue(
            ["Yang Z", "Gan Z", "Wang J"],
            ["Zhengyuan Yang", "Zhe Gan", "Jianfeng Wang"],
        )
        is None
    )
    assert _author_issue(["Firat M"], ["Mehmet Firat"]) is None
    # Genuinely different first author (wrong-paper / real mismatch) -> flagged.
    assert _author_issue(["Huang J"], ["Tom B. Brown"]) is not None
    # A list that mostly disagrees -> flagged.
    assert _author_issue(["Smith J", "Jones A"], ["Adams B", "Clark C"]) is not None
