"""The web as the last resort: the finder, the reading of what it found, and the stage.

Offline: the finder and every fetch are fakes.
"""

from __future__ import annotations

import pytest

from citation_verifier.backends.web_search import _chosen, web_search_tool
from citation_verifier.errors import ModelUnavailable
from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import MultiSourceResolver
from citation_verifier.grounding.web_lookup import WebLocator
from citation_verifier.schema import CitationRecord, CitedAs, Claim, Exists

LOUIZOS = (
    "Christos Louizos, Max Welling, and Diederik P. Kingma. 2018. Learning sparse neural networks "
    "through l0regularization. In International Conference on Learning Representations."
)
O3MINI = "OpenAI. 2025. OpenAI o3-mini."


@pytest.fixture(autouse=True)
def _no_cache_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", str(tmp_path / "cache"))
    for name in ("fetch_arxiv_by_id", "fetch_crossref_by_doi", "fetch_openreview_by_id", "fetch_openlibrary_by_isbn"):
        monkeypatch.setattr(paper_lookup, name, lambda *a, **k: [])
    monkeypatch.setattr(MultiSourceResolver, "_abstract_top_up", lambda self, c: "")


def _cited(ref: str, title: str, authors: list[str], year: int | None) -> CitedAs:
    return CitedAs(raw=ref, title=title, authors=authors, year=year)


def _locator(urls, *, pages=None, **kw):
    calls = []

    def search(reference):
        calls.append(reference)
        return list(urls), list(urls)

    loc = WebLocator(search, MultiSourceResolver(validate_urls=False)._match,
                     fetch_page=lambda u: (pages or {}).get(u, ""), **kw)
    return loc, calls


# ── the finder ───────────────────────────────────────────────────────────────
def test_the_tool_version_follows_the_model():
    assert web_search_tool("claude-opus-4-8", 3)["type"] == "web_search_20260209"
    assert web_search_tool("claude-sonnet-5-5", 3)["type"] == "web_search_20260209"
    assert web_search_tool("claude-haiku-4-5-20251001", 3)["type"] == "web_search_20250305"
    assert web_search_tool("claude-haiku-4-5", 2)["max_uses"] == 2


def test_only_urls_the_search_returned_are_kept():
    text = 'Here: {"urls": ["https://arxiv.org/abs/1712.01312", "https://made-up.example/paper"]}'
    seen = ["https://www.arxiv.org/abs/1712.01312/", "https://other.example/x"]
    assert _chosen(text, seen) == ["https://www.arxiv.org/abs/1712.01312/"]
    assert _chosen("no json at all", seen) == []


# ── reading what was found ───────────────────────────────────────────────────
def test_an_identifier_in_the_url_is_fetched_and_matched(monkeypatch):
    monkeypatch.setattr(paper_lookup, "fetch_arxiv_by_id", lambda i: [{
        "source": "arxiv", "title": "Learning Sparse Neural Networks through $L_0$ Regularization",
        "authors": ["Christos Louizos", "Max Welling", "Diederik P. Kingma"], "year": 2017,
        "venue": "", "doi": "", "arxiv_id": i, "url": f"https://arxiv.org/abs/{i}", "abstract": "We propose…"}])
    loc, _ = _locator(["https://arxiv.org/abs/1712.01312"])
    out = loc.locate(LOUIZOS, _cited(LOUIZOS, "Learning sparse neural networks through l0regularization",
                                     ["Christos Louizos", "Max Welling"], 2018))
    assert out.status == "record" and out.resolved.arxiv_id == "1712.01312"


def test_a_web_page_counts_when_its_title_and_owner_are_the_cited_ones():
    url = "https://openai.com/index/openai-o3-mini/"
    page = "<html><head><title>OpenAI o3-mini | OpenAI</title></head><body>Pushing the frontier</body></html>"
    loc, _ = _locator([url], pages={url: page})
    out = loc.locate(O3MINI, _cited(O3MINI, "OpenAI o3-mini", ["OpenAI"], 2025))
    assert out.status == "page"
    assert out.resolved.source == "web" and out.resolved.url == url


def test_a_page_with_the_title_but_not_the_author_does_not_count():
    url = "https://some-blog.example/easyr1-review"
    page = "<title>EasyR1: An efficient, scalable, multi-modality RL training framework</title><p>A review.</p>"
    ref = "Zheng, Y., Lu, J. Easyr1: An efficient, scalable, multi-modality rl training framework. 2025."
    loc, _ = _locator([url], pages={url: page})
    out = loc.locate(ref, _cited(ref, "Easyr1: An efficient, scalable, multi-modality rl training framework",
                                 ["Zheng, Y.", "Lu, J."], 2025))
    assert out.status == "none"


def test_an_outcome_is_cached_and_the_limit_holds():
    loc, calls = _locator([])
    cited = _cited(O3MINI, "OpenAI o3-mini", ["OpenAI"], 2025)
    assert loc.locate(O3MINI, cited).status == "none"
    assert loc.locate(O3MINI, cited).detail == "cached" and len(calls) == 1
    limited, _ = _locator([], max_lookups=1)
    assert limited.locate("A. Author. First. 2020.", cited).status == "none"
    assert limited.locate("B. Author. Second. 2021.", cited).status == "skipped"


def test_an_unreachable_model_stops_the_searching():
    def down(reference):
        raise ModelUnavailable("401 OAuth token expired")

    loc = WebLocator(down, MultiSourceResolver(validate_urls=False)._match)
    cited = _cited(O3MINI, "OpenAI o3-mini", ["OpenAI"], 2025)
    assert loc.locate(O3MINI, cited).status == "failed"
    assert loc.locate("Other. A title. 2020.", cited).status == "failed" and loc.unavailable


# ── the stage ────────────────────────────────────────────────────────────────
class _Nothing:
    """A resolver that finds nothing and heard from every source it asked."""

    sources = ("crossref",)

    def resolve(self, key, ref):
        return None

    def failed_sources(self):
        return []

    def searched_sources(self):
        return ["Crossref", "arXiv"]

    def _match(self, ref, cands):
        return MultiSourceResolver(validate_urls=False)._match(ref, cands)


def _record(ref: str, cited: CitedAs) -> CitationRecord:
    return CitationRecord(paper_id="p", claim_id="c", cite_key="k", claim=Claim(claim_id="c", text=""), cited_as=cited)


def test_the_stage_takes_a_web_page_and_says_how_it_was_found():
    from citation_verifier.stages.correctness import fill_correctness

    url = "https://openai.com/index/openai-o3-mini/"
    loc, _ = _locator([url], pages={url: "<title>OpenAI o3-mini | OpenAI</title>"})
    rec = fill_correctness(_record(O3MINI, _cited(O3MINI, "OpenAI o3-mini", ["OpenAI"], 2025)),
                           resolver=_Nothing(), web=loc)
    assert rec.exists == Exists.YES and rec.resolved.source == "web"
    assert any(e.kind == "web" for e in rec.evidence)


def test_not_found_anywhere_says_so():
    from citation_verifier.stages.correctness import fill_correctness

    ref = "Park, M. Openai o1: Self-play curricula for general-purpose reasoning. Trans. Emergent Reasoning, 2024."
    loc, _ = _locator([])
    rec = fill_correctness(_record(ref, _cited(ref, "Openai o1: Self-play curricula for general-purpose reasoning",
                                               ["Park, M."], 2024)), resolver=_Nothing(), web=loc)
    assert rec.exists == Exists.UNRESOLVED
    note = rec.metadata_issues[0]
    assert note.startswith("could not retrieve a confident match — searched Crossref, arXiv")
    assert note.endswith("not found on the web either — likely fabricated, check by hand")


def test_the_backend_runs_the_web_only_when_a_searcher_is_wired():
    from citation_verifier.backends.agentic import AgenticBackend

    assert AgenticBackend(settings=None).web_searcher is None
    assert AgenticBackend(settings={"enable_relevance_judge": False}).web_searcher is None

    class Searcher:
        def __call__(self, reference):
            return [], []

        def drain_usage(self):
            from citation_verifier.interfaces import RunUsage

            return RunUsage(backend="agentic", model="m", cost_usd=0.02, tool_calls=2)

    backend = AgenticBackend(settings={"web_searcher": Searcher()})
    web = backend._web_locator(MultiSourceResolver(validate_urls=False))
    assert web is not None
    from citation_verifier.interfaces import RunUsage, VerificationResult

    usage = RunUsage(backend="agentic")
    result = VerificationResult(paper_id="p", backend="agentic", usage=usage)
    backend._fold_web_usage(usage, web, result)
    assert usage.tool_calls == 2 and usage.by_tier["bulk"].cost_usd == pytest.approx(0.02)


BENGIO = ("Yoshua Bengio and Yves Grandvalet. Serie scientifique scientific series no unbiased estimator of the "
          "variance of k-fold cross-validation no unbiased estimator of the variance of k-fold crossvalidation.")
BENGIO_CITED = CitedAs(raw=BENGIO, authors=["Yoshua Bengio", "Yves Grandvalet"], title=(
    "Serie scientifique scientific series no unbiased estimator of the variance of k-fold cross-validation "
    "no unbiased estimator of the variance of k-fold crossvalidation"))


def test_a_pdf_result_is_read_and_a_garbled_title_is_matched_by_what_it_repeats():
    url = "https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf"
    pdf = ("\x0cJournal of Machine Learning Research 5 (2004) 1089–1105 No Unbiased Estimator of the Variance "
           "of K-Fold Cross-Validation Yoshua Bengio BENGIOY@IRO.UMONTREAL.CA Yves Grandvalet Abstract Most "
           "machine learning researchers perform quantitative experiments to estimate generalization error.")
    loc, _ = _locator([url], pages={url: pdf})
    out = loc.locate(BENGIO, BENGIO_CITED)
    assert out.status == "page" and out.resolved.url == url


def test_an_html_page_is_matched_by_what_a_garbled_title_repeats():
    url = "https://cirano.qc.ca/en/summaries/2003s-22"
    page = ("<title>No Unbiased Estimator of the Variance of K-Fold Cross-Validation | CIRANO</title>"
            "<p>Yoshua Bengio, Yves Grandvalet</p>")
    loc, _ = _locator([url], pages={url: page})
    assert loc.locate(BENGIO, BENGIO_CITED).status == "page"


def test_a_scanned_pdf_cannot_be_checked():
    url = "http://web.archive.org/web/2010/http://galton.uchicago.edu/techreports/tr348.pdf"
    ref = "Augustine Kong. A note on importance sampling using standardized weights. Technical Report 348, 1992."
    loc, _ = _locator([url], pages={url: "\x0c   "})
    out = loc.locate(ref, _cited(ref, "A note on importance sampling using standardized weights", ["Augustine Kong"], 1992))
    assert out.status == "none"


KONG = ("Augustine Kong. A note on importance sampling using standardized weights. Technical Report 348, "
        "Department of Statistics, University of Chicago, 1992.")
KONG_CITED = CitedAs(raw=KONG, authors=["Augustine Kong"], year=1992,
                     title="A note on importance sampling using standardized weights")
LISTING = ("<title>Technical Reports | Department of Statistics</title><table>"
           "<tr><td>349</td><td>Zhang, B.</td><td>August 1992</td><td>Nonparametric function estimation.</td></tr>"
           "<tr><td>348</td><td>Kong, A.</td><td>August 1992</td><td><a href='tr348.pdf'>A note on importance "
           "sampling using standardized weights</a></td></tr></table>")


def test_a_listing_on_the_cited_institutions_own_site_counts():
    """The department's technical-report list records the report; it is read even though
    the finder did not pick it (it is not the work), because it is on the cited institution's site."""
    url = "https://stat.uchicago.edu/department-library/technical-reports-2/"

    def search(reference):
        return [], ["https://arxiv.org/abs/1809.04129", url]

    loc = WebLocator(search, MultiSourceResolver(validate_urls=False)._match, fetch_page=lambda u: LISTING if u == url else "")
    out = loc.locate(KONG, KONG_CITED)
    assert out.status == "page" and out.resolved.url == url


def test_the_same_entry_in_a_citing_papers_references_does_not_count():
    url = "https://arxiv.org/html/1809.04129"
    loc, _ = _locator([url], pages={url: LISTING.replace("Technical Reports | Department of Statistics", "Rethinking ESS")})
    assert loc.locate(KONG, KONG_CITED).status == "none"


# ── a report list runs over several pages ────────────────────────────────────
DOE = ("Jane Doe. On thinning spatial point processes. Technical Report 12, Department of Statistics, "
       "Northfield University, 1987.")
DOE_CITED = CitedAs(raw=DOE, authors=["Jane Doe"], year=1987, title="On thinning spatial point processes")
_LIB = "https://stat.northfield.edu/library"


def _list_page(rows: str, *links: str) -> str:
    nav = "".join(f"<a href='{h}'>page</a>" for h in links)
    return f"<title>Technical Reports</title><nav>{nav}<a href='/people/'>People</a></nav><table>{rows}</table>"


def test_the_other_pages_of_a_report_list_are_read():
    """The search returned one page of the department's report list; the report is on
    another. The list's other pages — same site, same directory, same name but for the
    numbering — are read."""
    first = f"{_LIB}/technical-reports-3/"
    pages = {
        first: _list_page("<tr><td>40</td><td>Roe, R.</td><td>An unrelated report</td></tr>",
                          "/library/technical-reports/", "/library/technical-reports-2/", "https://elsewhere.org/technical-reports-2/"),
        f"{_LIB}/technical-reports/": _list_page("<tr><td>90</td><td>Poe, P.</td><td>Another one</td></tr>"),
        f"{_LIB}/technical-reports-2/": _list_page(
            "<tr><td>12</td><td>Doe, J.</td><td>1987</td><td>On thinning spatial point processes</td></tr>"),
    }
    fetched = []

    def fetch(u):
        fetched.append(u)
        return pages.get(u, "")

    loc = WebLocator(lambda ref: ([], [first]), MultiSourceResolver(validate_urls=False)._match, fetch_page=fetch)
    out = loc.locate(DOE, DOE_CITED)
    assert out.status == "page" and out.resolved.url == f"{_LIB}/technical-reports-2/"
    assert not any("elsewhere.org" in u or "/people" in u for u in fetched)


def test_only_pages_of_the_same_list_are_followed():
    from citation_verifier.grounding.web_lookup import _sibling_pages

    here = "https://stat.northfield.edu/library/technical-reports-2/"
    links = [
        "https://stat.northfield.edu/library/technical-report-4/",
        "https://stat.northfield.edu/library/technical-reports/",
        "https://stat.northfield.edu/library/annual-review/",
        "https://stat.northfield.edu/people/technical-reports/",
        "https://stat.northfield.edu/library/technical-reports-2/?page=3",
        "https://other.edu/library/technical-reports/",
    ]
    assert _sibling_pages(here, links) == links[:2] + [links[4]]


def test_the_search_is_told_to_look_on_the_issuing_institutions_site():
    from citation_verifier.backends.web_search import _SYSTEM

    assert "institution's own website" in _SYSTEM and "domain" in _SYSTEM


def test_a_session_out_of_turns_is_unfinished_not_failed():
    """The SDK ends a session out of turns with an error result that carries no text, then
    raises; the searches it ran are still read."""
    import asyncio
    import dataclasses
    import types

    from citation_verifier.backends.web_search import WebSearcher

    @dataclasses.dataclass
    class Options:
        system_prompt: str = ""
        tools: list = None
        allowed_tools: list = None
        setting_sources: list = None
        cwd: str = ""
        model: str = ""
        max_turns: int = 0

    class AssistantMessage: ...
    class TextBlock: ...

    class UserMessage:
        content = "results: https://stat.northfield.edu/library/technical-reports-3/ and https://example.org/x"

    class ResultMessage:
        subtype, is_error, result, usage, total_cost_usd, num_turns = "error_max_turns", True, None, None, 0.2, 9

    async def query(prompt, options):
        yield UserMessage()
        yield ResultMessage()
        raise Exception("Claude Code returned an error result: Reached maximum number of turns (9)")

    sdk = types.SimpleNamespace(AssistantMessage=AssistantMessage, TextBlock=TextBlock, UserMessage=UserMessage,
                                ResultMessage=ResultMessage, ClaudeAgentOptions=Options, query=query)
    searcher = WebSearcher({"model_bulk": "claude-haiku-4-5-20251001"})
    text, seen, finished = asyncio.run(searcher._run_query(sdk, "Reference: x"))
    assert finished is False and text == ""
    assert seen[0] == "https://stat.northfield.edu/library/technical-reports-3/"


def test_the_reference_as_printed_names_the_institution():
    """The stage hands the locator the reference rebuilt from its parsed fields, which drops
    the publication details; the institution is read from the reference as printed, and the
    finder is given that."""
    url = "https://stat.uchicago.edu/department-library/technical-reports-2/"
    asked = []

    def search(reference):
        asked.append(reference)
        return [], [url]

    rebuilt = 'Augustine Kong "A note on importance sampling using standardized weights" 1992'
    loc = WebLocator(search, MultiSourceResolver(validate_urls=False)._match, fetch_page=lambda u: LISTING if u == url else "")
    out = loc.locate(rebuilt, KONG_CITED)
    assert asked == [KONG] and out.status == "page" and out.resolved.url == url


# ── the table pass has the same last resort ──────────────────────────────────
def test_a_table_rows_citation_no_index_found_is_looked_for_on_the_web():
    from citation_verifier.tables.pipeline import _Grounder

    url = "https://stat.uchicago.edu/department-library/technical-reports-2/"
    web = WebLocator(lambda ref: ([], [url]), MultiSourceResolver(validate_urls=False)._match,
                     fetch_page=lambda u: LISTING if u == url else "")

    class NoIndex:
        def resolve(self, key, reference):
            return None

        def failed_sources(self):
            return []

        def searched_sources(self):
            return ["Crossref"]

    errors: list[str] = []
    found = _Grounder({"kong1992note": KONG_CITED}, None, NoIndex(), errors, web).resolve("kong1992note", "")
    assert found is not None and found.source == "web" and found.url == url
    assert _Grounder({"kong1992note": KONG_CITED}, None, NoIndex(), errors).resolve("kong1992note", "") is None


def test_the_table_pass_searches_the_web_only_when_it_may():
    from citation_verifier.tables.pipeline import _build_web

    class Searcher:
        def __call__(self, reference):
            return [], []

    resolver = MultiSourceResolver(validate_urls=False)
    web, searcher = _build_web({"web_searcher": Searcher(), "web_search_max_lookups": 7}, resolver)
    assert isinstance(web, WebLocator) and web.max_lookups == 7 and isinstance(searcher, Searcher)
    assert _build_web({"web_searcher": Searcher(), "enable_web_search": False}, resolver) == (None, None)
    assert _build_web({"web_searcher": Searcher()}, None) == (None, None)


# ── a page found on the web is checked like a record ─────────────────────────
GRAPHS = "J. Smith. Deep learning for graphs. NeurIPS, 2019."
GRAPHS_CITED = CitedAs(raw=GRAPHS, authors=["J. Smith"], year=2019, venue="NeurIPS", title="Deep learning for graphs")
_FILLER = " ".join(["Lecture notes on message passing, spectral methods and pooling."] * 30)


def test_a_name_elsewhere_on_a_page_is_not_its_authorship():
    """A fabricated citation borrows a real title; the page with that title is someone
    else's, and the cited surname appears only in its reading list, far from the title."""
    url = "https://course.example.edu/deep-learning-for-graphs/"
    page = (f"<title>Deep Learning for Graphs</title><h1>Deep Learning for Graphs</h1>"
            f"<p>Taught by A. Okafor, Spring 2021.</p><p>{_FILLER}</p><h2>Reading</h2><p>K. Smith, Spectral notes.</p>")
    loc, _ = _locator([url], pages={url: page})
    assert loc.locate(GRAPHS, GRAPHS_CITED).status == "none"


def test_a_page_whose_stated_authors_are_others_is_not_the_work():
    url = "https://press.example.com/books/deep-learning-for-graphs"
    page = ('<meta name="citation_title" content="Deep Learning for Graphs">'
            '<meta name="citation_author" content="Okafor, Ada"><meta name="citation_author" content="Lee, Min">'
            "<title>Deep Learning for Graphs</title><p>Deep Learning for Graphs. Foreword by J. Smith.</p>")
    loc, _ = _locator([url], pages={url: page})
    assert loc.locate(GRAPHS, GRAPHS_CITED).status == "none"


def test_what_the_page_states_is_compared_with_the_citation():
    """Authors and year are the page's, not copies of the citation's — so a wrong year
    or a co-author the page does not list is reported, as for any record."""
    from citation_verifier.stages.correctness import fill_correctness

    url = "https://smith.example.org/deep-learning-for-graphs.pdf"
    pdf = ("\x0cTechnical note, March 2016. Deep Learning for Graphs. Jane Smith, Dept. of Computing. "
           "Abstract. We survey neural networks on graphs. " + _FILLER)
    cited = CitedAs(raw=GRAPHS, authors=["J. Smith"], year=2019, venue="NeurIPS", title="Deep learning for graphs")
    loc, _ = _locator([url], pages={url: pdf})
    rec = fill_correctness(_record(GRAPHS, cited), resolver=_Nothing(), web=loc)
    assert rec.exists == Exists.YES and rec.resolved.authors == ["J. Smith"] and rec.resolved.year == 2016
    assert any(i.startswith("year mismatch: cited 2019 vs 2016") for i in rec.metadata_issues)
    web = next(e for e in rec.evidence if e.kind == "web")
    assert "title, author and year read off it" in web.quote


def test_a_page_that_gives_no_year_is_not_given_the_cited_one():
    url = "https://openai.com/index/openai-o3-mini/"
    loc, _ = _locator([url], pages={url: "<title>OpenAI o3-mini | OpenAI</title><p>Pushing the frontier.</p>"})
    out = loc.locate(O3MINI, _cited(O3MINI, "OpenAI o3-mini", ["OpenAI"], 2025))
    assert out.status == "page" and out.resolved.year is None and out.resolved.authors == ["OpenAI"]
    assert out.detail.endswith("it gives no year")


def test_a_report_list_entry_gives_the_reports_author_and_year():
    url = "https://stat.uchicago.edu/department-library/technical-reports-2/"
    loc, _ = _locator([url], pages={url: LISTING})
    out = loc.locate(KONG, KONG_CITED)
    assert out.status == "page" and out.resolved.authors == ["Augustine Kong"] and out.resolved.year == 1992
    assert "entry in the report list of stat.uchicago.edu" in out.detail
