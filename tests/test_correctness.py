"""
Offline tests for the correctness stage's author matching. Cited references use
Vancouver "Surname Initials" while canonical records use "First Last", so an
abbreviated cited name (``Firat M``) must match its full form (``Mehmet Firat``)
without flagging — only a genuinely different surname/initial should flag.
"""

from __future__ import annotations

from citation_verifier.stages.correctness import (
    _author_issue,
    _name_key,
    _same_author,
    _unresolved_note,
)


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


def test_name_key_total_on_all_initials_fragment():
    # A despaced reference can orphan an all-initials fragment ("S. R." from
    # "Bowman, S. R."); _name_key must stay total (no surname to key on), never
    # IndexError. Regression: a 60-author list that mis-split this way crashed the
    # whole correctness stage ("list index out of range").
    assert _name_key("S. R.") == ("", "s")
    assert _name_key("S.") == ("s.", "")  # a single initial is harmless either way
    # The crash reproduced through the real path: _author_issue over a mangled list.
    mangled = ["Perez", "E.", "Ringer", "S.", "Bowman", "S. R.", "Askell", "A."]
    assert _author_issue(mangled, ["Ethan Perez", "Sam Ringer"]) is not None  # no crash


# ── weak fuzzy-title match abstains to unresolved (GPT-feedback precision) ──
def _fake_resolver(resolved):
    class _R:
        sources = ("crossref", "s2")

        def resolve(self, cite_key, reference):
            return resolved

    return _R()


def _rec(authors, title, year):
    from citation_verifier.schema import CitationRecord, CitedAs, Claim

    return CitationRecord(
        paper_id="p", claim_id="c", cite_key="k",
        claim=Claim(claim_id="c", text="A claim."),
        cited_as=CitedAs(raw="ref", authors=authors, title=title, year=year),
    )


def _resolved(authors, title, year, method):
    from citation_verifier.schema import Resolved

    return Resolved(
        source="s2", match_method=method, match_score=0.96,
        title=title, authors=authors, year=year, abstract="some abstract",
    )


def test_fuzzy_match_is_weak_rules():
    from citation_verifier.schema import MatchMethod
    from citation_verifier.stages.correctness import _fuzzy_match_is_weak

    cited = _rec(["Naidoo V"], "AI-powered chatbots for healthcare: A systematic review", 2021)
    # author wrong + year wrong + generic title -> weak
    both = _resolved(["Małgorzata Pieścik-Lech"], "Systematic Review", 2017, MatchMethod.FUZZY_TITLE)
    assert _fuzzy_match_is_weak(cited, both) is True
    # author wrong, year OK, but the resolved title drops the distinctive content
    # (the real Naidoo -> "Systematic review" by Mubarak collision) -> weak
    generic = _resolved(["Mubarak A"], "Systematic review", 2021, MatchMethod.FUZZY_TITLE)
    assert _fuzzy_match_is_weak(cited, generic) is True
    # author wrong but the title FULLY covers the cited title -> trust it (likely a
    # cited-author slip on the right paper), stays non-weak
    sametitle = _resolved(
        ["Mubarak A"], "AI-powered chatbots for healthcare: A systematic review", 2021,
        MatchMethod.FUZZY_TITLE,
    )
    assert _fuzzy_match_is_weak(cited, sametitle) is False
    # author matches (abbreviated form) -> always trusted, never weak
    au_ok = _resolved(["V Naidoo"], "Systematic Review", 2017, MatchMethod.FUZZY_TITLE)
    assert _fuzzy_match_is_weak(cited, au_ok) is False
    # no first author to anchor on -> cannot judge, not weak
    assert _fuzzy_match_is_weak(_rec([], "t", 2021), both) is False


def test_fill_correctness_abstains_on_weak_fuzzy_match():
    from citation_verifier.schema import Exists, MatchMethod
    from citation_verifier.stages.correctness import fill_correctness

    rec = _rec(["Naidoo V"], "AI-powered chatbots for healthcare: A systematic review", 2021)
    wrong = _resolved(["Małgorzata Pieścik-Lech"], "Systematic Review", 2017, MatchMethod.FUZZY_TITLE)
    out = fill_correctness(rec, resolver=_fake_resolver(wrong))
    assert Exists(out.exists) is Exists.UNRESOLVED
    assert out.resolved is None  # the likely-wrong candidate must not feed relevance
    assert out.metadata_issues and "abstained" in out.metadata_issues[0]


def test_fill_correctness_keeps_yes_for_id_match_despite_both_mismatches():
    # An identifier (DOI/arXiv) match is authoritative + already title-gated, so it
    # is NOT downgraded even when author and year both differ from the cited text.
    from citation_verifier.schema import Exists, MatchMethod
    from citation_verifier.stages.correctness import fill_correctness

    rec = _rec(["Naidoo V"], "AI-powered chatbots for healthcare: A systematic review", 2021)
    byid = _resolved(["Małgorzata Pieścik-Lech"], "Systematic Review", 2017, MatchMethod.DOI)
    out = fill_correctness(rec, resolver=_fake_resolver(byid))
    assert Exists(out.exists) is Exists.YES
    assert out.resolved is not None


def test_fill_correctness_keeps_yes_for_single_axis_fuzzy_mismatch():
    from citation_verifier.schema import Exists, MatchMethod
    from citation_verifier.stages.correctness import fill_correctness

    rec = _rec(["Firat M"], "Some Paper Title", 2021)
    # author matches (abbrev), only the year is off by >1 -> stays yes + a flag
    yr = _resolved(["Mehmet Firat"], "Some Paper Title", 2017, MatchMethod.FUZZY_TITLE)
    out = fill_correctness(rec, resolver=_fake_resolver(yr))
    assert Exists(out.exists) is Exists.YES
