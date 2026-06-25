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


def test_fill_correctness_verifies_live_url_as_direct_url(monkeypatch):
    import citation_verifier.grounding.url_validate as uv
    from citation_verifier.schema import Exists, MatchMethod
    from citation_verifier.stages.correctness import fill_correctness

    rec = _rec(["Ganler"], "code-r1", 2025)
    rec.cited_as.url = "https://github.com/ganler/code-r1"
    monkeypatch.setattr(
        uv, "validate_citation_url",
        lambda raw, github_token=None: uv.UrlCheck(url=raw, status="live", http=200, method="github_api"),
    )
    # No structured match -> falls back to direct-URL verification -> exists=yes.
    out = fill_correctness(rec, resolver=_fake_resolver(None))
    assert _val(out.exists) == Exists.YES.value
    assert _val(out.resolved.match_method) == MatchMethod.DIRECT_URL.value
    assert out.resolved.source == "url" and out.resolved.url_valid is True
    assert not out.metadata_issues


def test_fill_correctness_blocked_url_abstains_with_actionable_reason(monkeypatch):
    import citation_verifier.grounding.url_validate as uv
    from citation_verifier.schema import Exists
    from citation_verifier.stages.correctness import fill_correctness

    rec = _rec(["OpenAI"], "o3-o4-mini system card", 2025)
    rec.cited_as.url = "https://openai.com/index/o3-o4-mini-system-card/"
    monkeypatch.setattr(
        uv, "validate_citation_url",
        lambda raw, github_token=None: uv.UrlCheck(url=raw, status="blocked", http=403, method="fetch"),
    )
    out = fill_correctness(rec, resolver=_fake_resolver(None))
    assert _val(out.exists) == Exists.UNRESOLVED.value  # 403 is NOT a false yes
    assert "url access blocked: HTTP 403" in " ".join(out.metadata_issues)


def _val(x):
    return getattr(x, "value", x)


def test_author_issue_suppresses_org_vs_individual():
    # An organization / "* Team" author against a listed individual (or vice versa)
    # is a benign attribution discrepancy, not a metadata error — no first-author
    # mismatch warning. (Suppresses the warning only; the resolver gate is untouched.)
    assert _author_issue(["Tom Brown", "Benjamin Mann"], ["OpenAI"]) is None
    assert _author_issue(["OpenAI"], ["Tom Brown"]) is None
    assert _author_issue(["DeepSeek-AI"], ["Daya Guo"]) is None
    assert _author_issue(["An Yang", "Baosong Yang"], ["Qwen Team"]) is None
    assert _author_issue(["Gemini Team"], ["Rohan Anil"]) is None
    # but two genuinely different PEOPLE (no org on either side) still flag
    assert _author_issue(["George Kingsley Zipf"], ["Y. Chao"]) is not None


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


def test_reject_bad_fuzzy_match_rules():
    from citation_verifier.schema import MatchMethod
    from citation_verifier.stages.correctness import _reject_bad_fuzzy_match

    def reject(rec, cand):
        return _reject_bad_fuzzy_match(rec.cited_as, cand)

    cited = _rec(["Naidoo V"], "AI-powered chatbots for healthcare: A systematic review", 2021)
    # author wrong + year wrong + generic title -> reject
    both = _resolved(["Małgorzata Pieścik-Lech"], "Systematic Review", 2017, MatchMethod.FUZZY_TITLE)
    assert reject(cited, both)
    # author wrong, year OK, but the resolved title drops the distinctive content
    # (the real Naidoo -> "Systematic review" by Mubarak collision) -> reject
    generic = _resolved(["Mubarak A"], "Systematic review", 2021, MatchMethod.FUZZY_TITLE)
    assert reject(cited, generic)
    # author wrong but the title FULLY covers the cited title -> trust it (likely a
    # cited-author slip on the right paper)
    sametitle = _resolved(
        ["Mubarak A"], "AI-powered chatbots for healthcare: A systematic review", 2021,
        MatchMethod.FUZZY_TITLE,
    )
    assert reject(cited, sametitle) is None
    # author matches (abbreviated form), candidate title contained -> trusted
    au_ok = _resolved(["V Naidoo"], "Systematic Review", 2017, MatchMethod.FUZZY_TITLE)
    assert reject(cited, au_ok) is None
    # no cited first author to anchor on (candidate has authors) -> not rejected here
    assert reject(_rec([], "t", 2021), both) is None

    # Real false-positives from the 6514 / 13781 reports must now reject:
    # Sutton & Barto textbook (1998) mis-resolved to a 1992 chapter — the FIRST AUTHOR
    # MATCHES, but the 6-year gap + diverging titles ("volume" vs "challenge") mark a
    # different work. (The old author-confirmed short-circuit let this through.)
    sutton = _rec(["Richard S Sutton", "Andrew G Barto"],
                  "Reinforcement learning: An introduction, volume 1", 1998)
    chapter = _resolved(["Richard S. Sutton"],
                        "Introduction: The Challenge of Reinforcement Learning", 1992,
                        MatchMethod.FUZZY_TITLE)
    assert reject(sutton, chapter)
    # LangChain GitHub ref resolved to a Crossref record whose "title" is a bare date,
    # with no authors — reject on either signal.
    langchain = _rec(["Harrison Chase"], "Langchain, October 2022", 2022)
    date_only = _resolved([], "October 2022", 2022, MatchMethod.FUZZY_TITLE)
    assert reject(langchain, date_only)
    # GUARD: a legit preprint-vs-published gap with the SAME title must NOT reject.
    legit = _rec(["Jane Doe"], "Scaling laws for neural language models", 2020)
    published = _resolved(["Jane Doe"], "Scaling laws for neural language models", 2023,
                          MatchMethod.FUZZY_TITLE)
    assert reject(legit, published) is None


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


def test_reject_fuzzy_short_title_url_and_divergence():
    """The three severe wrong-paper false-positives from the 6-venue review."""
    from citation_verifier.schema import CitedAs, MatchMethod, Resolved
    from citation_verifier.stages.correctness import _reject_bad_fuzzy_match as R

    def C(t, a, y, url=None):
        return CitedAs(raw="x", title=t, authors=a, year=y, url=url)

    def V(t, a, y):
        return Resolved(source="s2", match_method=MatchMethod.FUZZY_TITLE, title=t, authors=a, year=y)

    # short cited title + wrong author -> coincidental substring (o3-mini in a pneumonia paper)
    assert R(C("OpenAI o3-mini", ["OpenAI"], 2025),
             V("Performance analysis of LLMs in clinical treatment of pneumonia", ["Zhiwu Lin"], 2025))
    # URL/release citation + wrong author -> the URL is the real anchor (Genie 2)
    assert R(C("Genie 2: A large-scale foundation world model", ["J Parker-Holder"], 2024, url="https://deepmind"),
             V("The Mathematics of Genie 2: A Large-Scale Foundation World Model", ["Miquel Noguer I Alonso"], 2025))
    # wrong author + diverging titles with only a 1-yr gap (wu2024 -> wrong video paper)
    assert R(C("Scaling inference computation: compute-optimal inference for problem solving with language models", ["Yangzhen Wu"], 2024),
             V("Inference Compute-Optimal Video Vision Language Models", ["Peiqi Wang"], 2025))
    # a real paper (same title, author-variant matched, no URL) is still accepted
    assert R(C("A specific multi word paper title", ["Firat M"], 2021),
             V("A specific multi word paper title", ["Mehmet Firat"], 2019)) is None


def test_strings_match_tolerates_lost_hyphen():
    from citation_verifier.stages.correctness import _strings_match

    assert _strings_match("Distinctive image features from scaleinvariant keypoints",
                          "Distinctive Image Features from Scale-Invariant Keypoints")
    assert not _strings_match("Attention is all you need", "Deep residual learning for image recognition")


def test_author_issue_suppresses_org_contributors_and_hyphen_surnames():
    # org list now covers "*-Contributors" collaborations
    assert _author_issue(["Qingwen Bu"], ["AgiBot-World-Contributors"]) is None
    # hyphenated surname vs spaced form is the same person, not a mismatch
    assert _author_issue(["Diego Perez-Liebana"], ["Diego Perez Liebana"]) is None
    # genuinely different people still flag
    assert _author_issue(["Abby O'Neill"], ["A. Padalkar"]) is not None
