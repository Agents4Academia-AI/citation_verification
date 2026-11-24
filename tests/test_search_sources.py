"""Finding what the scholarly indexes miss: books, workshop papers, garbled titles.

Offline: every lookup function is stubbed.
"""

from __future__ import annotations

import json

import pytest

from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import (
    MultiSourceResolver,
    _clean_title_variant,
    _likely_titles,
    _looks_like_book,
    _norm_title,
    _repeated_title,
    _should_search_openreview,
)
from citation_verifier.schema import CitationRecord, CitedAs, Claim
from citation_verifier.stages import correctness as C

_LOOKUPS = (
    "search_semantic_scholar_match", "search_crossref_by_fields", "search_arxiv_by_title",
    "search_openalex_by_title", "search_openlibrary", "search_openreview", "search_crossref",
    "search_semantic_scholar", "search_openalex", "search_dblp", "search_arxiv",
)


@pytest.fixture()
def offline(monkeypatch):
    """Every lookup answers nothing; a test overrides the ones it needs."""
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", "")
    for name in _LOOKUPS:
        monkeypatch.setattr(paper_lookup, name, lambda *a, **k: [])
    monkeypatch.setattr(MultiSourceResolver, "_abstract_top_up", lambda self, c: "")
    return monkeypatch


def _row(source, title, authors, year, **kw):
    return {"source": source, "title": title, "authors": authors, "year": year, "venue": kw.get("venue", ""),
            "type": kw.get("type", ""), "doi": "", "arxiv_id": "", "url": kw.get("url", ""),
            "abstract": kw.get("abstract", "")}


# ── titles ───────────────────────────────────────────────────────────────────
def test_a_volume_or_edition_is_not_part_of_the_title():
    assert _clean_title_variant("Deep learning, volume 1") == "Deep learning"
    assert _clean_title_variant("Reinforcement learning: An introduction, 2nd edition") == (
        "Reinforcement learning: An introduction"
    )
    assert _clean_title_variant("Attention is all you need") == "Attention is all you need"
    ref = "Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep learning, volume 1. MIT Press, 2016."
    assert _likely_titles(ref)[0] == "Deep learning"


def test_a_glued_subscript_and_maths_markup_compare_equal():
    assert _norm_title("Learning sparse neural networks through l0regularization") == _norm_title(
        "Learning Sparse Neural Networks through $L_0$ Regularization"
    )
    assert "Learning sparse neural networks through l0 regularization" in _likely_titles(
        "Christos Louizos, Max Welling, and Diederik P. Kingma. 2018. Learning sparse neural networks "
        "through l0regularization. In International Conference on Learning Representations."
    )


def test_a_title_that_repeats_itself_is_searched_as_the_phrase_it_repeats():
    garbled = (
        "Serie scientifique scientific series no unbiased estimator of the variance of k-fold "
        "cross-validation no unbiased estimator of the variance of k-fold crossvalidation"
    )
    assert _repeated_title(garbled).lower().startswith("no unbiased estimator of the variance of k-fold")
    assert _repeated_title("Attention is all you need") == ""


def test_routing_to_books_and_to_openreview():
    assert _looks_like_book("Carl Rasmussen and C. Williams. Gaussian processes for machine learning. MIT Press, 2006.")
    assert _looks_like_book("R. Sutton. Reinforcement learning: An introduction. 2nd edition, 2018.")
    assert not _looks_like_book("A. Vaswani et al. Attention is all you need. In NeurIPS, 2017.")
    assert _should_search_openreview("X. Dang. Assessing diversity collapse. In SSI-FM Workshop at ICLR 2025.")
    assert not _should_search_openreview("A. Author. A title. Nature, 2020.")


# ── the new sources ──────────────────────────────────────────────────────────
def test_openalex_is_asked_by_fields_and_needs_no_key(monkeypatch):
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        return json.dumps({"results": [{"display_name": "Deep Learning", "publication_year": 2016,
                                         "type": "book", "authorships": []}]}).encode()

    monkeypatch.setattr(paper_lookup, "_get", fake_get)
    monkeypatch.setattr(paper_lookup, "CONTACT_EMAIL", "")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    rows = paper_lookup.search_openalex_by_title("Deep learning, volume 1", "Goodfellow", 2016)
    assert rows and rows[0]["title"] == "Deep Learning" and rows[0]["type"] == "book"
    query = paper_lookup.urllib.parse.unquote_plus(seen["url"])
    assert "title.search:Deep learning volume 1" in query  # its own syntax characters removed
    assert "raw_author_name.search:Goodfellow" in query and "publication_year:2014-2018" in query


def test_open_library_gives_the_edition_nearest_the_cited_year(monkeypatch):
    doc = {"key": "/works/OL1W", "title": "Reinforcement Learning", "subtitle": "An Introduction",
           "author_name": ["Richard S. Sutton", "Andrew G. Barto"], "first_publish_year": 1998,
           "publish_year": [1998, 2017, 2018, 2019], "publisher": ["MIT Press"]}
    monkeypatch.setattr(paper_lookup, "_get", lambda url, **kw: json.dumps({"docs": [doc]}).encode())
    row = paper_lookup.search_openlibrary("Reinforcement learning", "Sutton", 2018)[0]
    assert row["title"] == "Reinforcement Learning: An Introduction" and row["year"] == 2018
    assert row["source"] == "openlibrary" and row["venue"] == "MIT Press"
    assert paper_lookup.search_openlibrary("Reinforcement learning", "Sutton", 1990)[0]["year"] == 1998


def test_openreview_returns_the_paper_with_its_abstract(monkeypatch):
    note = {"id": "AMiK", "forum": "AMiK", "pdate": 1741409069640,
            "content": {"title": {"value": "Assessing Diversity Collapse in Reasoning"},
                        "authors": {"value": ["Xingyu Dang", "Christina Baek"]},
                        "venue": {"value": "SSI-FM Poster"}, "abstract": {"value": "We identify a striking phenomenon."}}}
    monkeypatch.setattr(paper_lookup, "_get", lambda url, **kw: json.dumps({"notes": [note]}).encode())
    row = paper_lookup.search_openreview("Assessing diversity collapse in reasoning")[0]
    assert row["year"] == 2025 and row["url"] == "https://openreview.net/forum?id=AMiK"
    assert row["abstract"].startswith("We identify") and row["authors"][0] == "Xingyu Dang"


def test_a_dblp_bot_check_page_is_dblp_not_answering(monkeypatch):
    monkeypatch.setattr(paper_lookup, "_get", lambda url, **kw: b"<!doctype html><title>Making sure you're not a bot!</title>")
    paper_lookup.reset_failures()
    assert paper_lookup.search_dblp("Deep learning Goodfellow") == []
    assert "dblp.org" in paper_lookup.failures()


# ── the cascade ──────────────────────────────────────────────────────────────
def test_a_book_the_paper_indexes_lack_is_found_and_a_review_of_it_is_not_taken(offline):
    ref = "Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, volume 1. MIT Press, 2016."
    review = _row("crossref", "Ian Goodfellow, Yoshua Bengio, and Aaron Courville: Deep learning", ["Jeff Heaton"], 2017)
    offline.setattr(paper_lookup, "search_crossref", lambda *a, **k: [review])
    offline.setattr(paper_lookup, "search_openlibrary", lambda *a, **k: [
        _row("openlibrary", "Deep Learning", ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"], 2016,
             venue="MIT Press", type="book")])
    found = MultiSourceResolver(validate_urls=False).resolve("goodfellow2016deep", ref)
    assert found is not None and found.source == "openlibrary" and found.year == 2016


def test_a_workshop_paper_is_found_on_openreview_as_a_last_resort(offline):
    ref = ("Xingyu Dang, Christina Baek, J Zico Kolter, and Aditi Raghunathan. Assessing diversity "
           "collapse in reasoning. 2025.")
    offline.setattr(paper_lookup, "search_openreview", lambda *a, **k: [
        _row("openreview", "Assessing Diversity Collapse in Reasoning", ["Xingyu Dang", "Christina Baek"], 2025,
             abstract="We identify a striking phenomenon.")])
    found = MultiSourceResolver(validate_urls=False).resolve("dang2025", ref)
    assert found is not None and found.source == "openreview" and found.abstract


def test_not_found_names_the_sources_that_were_asked(offline):
    resolver = MultiSourceResolver(validate_urls=False)
    rec = CitationRecord(paper_id="p", claim_id="c", cite_key="k", claim=Claim(claim_id="c", text=""),
                         cited_as=CitedAs(raw="Augustine Kong. A note on importance sampling using "
                                              "standardized weights. Technical Report 348, 1992.",
                                          title="A note on importance sampling using standardized weights"))
    note = C.fill_correctness(rec, resolver=resolver).metadata_issues[0]
    assert note.startswith("could not retrieve a confident match — searched")
    assert "Crossref" in note and "OpenAlex" in note
    assert "DBLP" not in note  # not routed for this reference, so not asked


def test_an_acl_reference_splits_at_its_year_whatever_the_names_look_like():
    from citation_verifier.extract.pdf import _split_author_title

    authors, title = _split_author_title(
        "Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. "
        "CRITIC: Large language models can self-correct with tool-interactive critiquing. In ICLR."
    )
    assert title == "CRITIC: Large language models can self-correct with tool-interactive critiquing"
    assert authors[3] == "yelong shen" and authors[-1] == "Weizhu Chen"
    authors, title = _split_author_title(
        "Xingyu Chen, Jiahao Xu, Zhuosheng Zhang, and 1 others. 2024b. Do not think that much for 2+3=? "
        "On the overthinking of o1-like LLMs. arXiv preprint."
    )
    assert authors == ["Xingyu Chen", "Jiahao Xu", "Zhuosheng Zhang"]
    assert title.startswith("Do not think that much")
    # A title before the year is not an author list: the year rule leaves it alone.
    authors, title = _split_author_title("Smith J. Title of the paper here. 2020. Journal of Things.")
    assert title == "Title of the paper here"


def test_a_namesake_of_a_generic_book_title_is_not_taken():
    from citation_verifier.schema import MatchMethod, Resolved

    cited = CitedAs(raw="x", title="Deep learning, volume 1",
                    authors=["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"], year=2016)
    namesake = Resolved(source="s2", match_method=MatchMethod.FUZZY_TITLE, match_score=1.0,
                        title="Deep Learning", authors=["Xingbang Hao", "Guigang Zhang"], year=2016)
    assert C._reject_bad_fuzzy_match(cited, namesake)
    book = namesake.model_copy(update={"authors": ["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville"]})
    assert C._reject_bad_fuzzy_match(cited, book) is None


def test_a_placeholder_abstract_is_no_evidence(monkeypatch):
    from citation_verifier.backends import relevance_judge as rj

    judge = rj.LLMRelevanceJudge(settings=None, mode="query")
    monkeypatch.setattr(judge, "_judge_citation_chunk", lambda chunk: (_ for _ in ()).throw(
        AssertionError("a placeholder must not be sent to the model")))
    out = judge.judge_batch([{
        "cite_key": "k", "claim_id": "c", "claim": "Bengio & Grandvalet prove that no unbiased estimator exists.",
        "abstract": "This paper presents research carried out at CIRANO and aims at encouraging discussion and comment.",
        "resolved": None,
    }])
    assert out[0].supports_claim == "inconclusive"


# ── a source that keeps failing is left alone ───────────────────────────────
def test_a_failing_source_is_left_alone_after_a_few_failures(monkeypatch):
    import urllib.error

    calls = []

    def urlopen(req, timeout=0):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(paper_lookup.time, "sleep", lambda _s: None)
    monkeypatch.setattr(paper_lookup.urllib.request, "urlopen", urlopen)
    paper_lookup.reset_failures()
    with pytest.raises(urllib.error.HTTPError):
        paper_lookup._get("https://api.semanticscholar.org/graph/v1/paper/x", retries=4)
    assert len(calls) == 3                          # the breaker opened; no fourth or fifth try
    with pytest.raises(paper_lookup.SourceDown):    # the next request is not made at all
        paper_lookup._get("https://api.semanticscholar.org/graph/v1/paper/y")
    assert len(calls) == 3 and "api.semanticscholar.org" in paper_lookup.failures()

    # after the cool-down, one probe; a failure re-opens at once
    clock = [paper_lookup.time.monotonic() + 3600]
    monkeypatch.setattr(paper_lookup.time, "monotonic", lambda: clock[0])
    with pytest.raises(urllib.error.HTTPError):
        paper_lookup._get("https://api.semanticscholar.org/graph/v1/paper/z", retries=4)
    assert len(calls) == 4
    with pytest.raises(paper_lookup.SourceDown):
        paper_lookup._get("https://api.semanticscholar.org/graph/v1/paper/w")


def test_a_bot_check_page_closes_off_dblp_at_once(monkeypatch):
    monkeypatch.setattr(paper_lookup, "_get", lambda url, **kw: b"<!doctype html><title>Making sure you're not a bot!</title>")
    assert paper_lookup.search_dblp("anything") == []
    monkeypatch.undo()
    with pytest.raises(paper_lookup.SourceDown):
        paper_lookup._get("https://dblp.org/search/publ/api?q=x")


def test_a_placeholder_abstract_leaves_a_record_thin():
    from citation_verifier.grounding.resolver import _is_thin
    from citation_verifier.schema import MatchMethod, Resolved

    placeholder = Resolved(source="s2", match_method=MatchMethod.FUZZY_TITLE, title="t",
                           abstract="This paper presents research carried out at CIRANO and aims at encouraging discussion.")
    assert _is_thin(placeholder)
    assert not _is_thin(placeholder.model_copy(update={"arxiv_id": "2001.00001"}))


# ── a namesake gets a second look on the web ─────────────────────────────────
def test_a_title_namesake_by_other_authors_gets_a_second_look_on_the_web():
    from citation_verifier.grounding.web_lookup import WebLookup
    from citation_verifier.schema import MatchMethod, Resolved

    cited = CitedAs(raw="x", title="Deep learning for image recognition",
                    authors=["Ian Goodfellow", "Yoshua Bengio"], year=2016)
    namesake = Resolved(source="s2", match_method=MatchMethod.FUZZY_TITLE, match_score=1.0,
                        title="Deep Learning for Image Recognition", authors=["Wei Zhang", "Li Wang"], year=2016)
    real = namesake.model_copy(update={"source": "openlibrary", "authors": ["Ian Goodfellow", "Yoshua Bengio"]})

    class Resolver:
        def resolve(self, key, ref):
            return namesake

        def failed_sources(self):
            return []

    class Web:
        def locate(self, reference, cited_as):
            return WebLookup("record", real, ["https://example.org/book"])

    def record():
        return CitationRecord(paper_id="p", claim_id="c", cite_key="k", claim=Claim(claim_id="c", text=""), cited_as=cited)

    with_web = C.fill_correctness(record(), resolver=Resolver(), web=Web())
    assert with_web.resolved.source == "openlibrary" and with_web.exists.value == "yes"
    without = C.fill_correctness(record(), resolver=Resolver())
    assert without.resolved.source == "s2"
    assert any("first-author mismatch" in i for i in without.metadata_issues)


# ── a short abstract that is about the work is one ───────────────────────────
_SHORT_TITLE = "A note on the sampling error of the difference between correlated proportions"
_SHORT_REAL = ("Two formulas are given for judging the significance of the difference between "
               "correlated proportions.")
_SHORT_BOILERPLATE = "This paper presents research carried out at the institute and aims at encouraging discussion."


def test_a_short_abstract_on_the_works_subject_is_an_abstract():
    assert paper_lookup.substantive_abstract(_SHORT_REAL, _SHORT_TITLE)
    assert not paper_lookup.substantive_abstract(_SHORT_BOILERPLATE, _SHORT_TITLE)
    assert not paper_lookup.substantive_abstract("Correlated proportions.", _SHORT_TITLE)  # not a sentence
    assert not paper_lookup.substantive_abstract(_SHORT_REAL, None)  # no title: length decides
    assert paper_lookup.substantive_abstract("words of an abstract " * 12, None)


def test_a_short_abstract_about_the_work_reaches_the_judge(monkeypatch):
    from citation_verifier.backends import relevance_judge as rj
    from citation_verifier.schema import MatchMethod, Resolved
    from citation_verifier.stages.relevance import RelevanceVerdict

    judge = rj.LLMRelevanceJudge(settings=None, mode="query")
    judge.escalate_full_text = False
    monkeypatch.setattr(judge, "_intro_for", lambda resolved: "")
    monkeypatch.setattr(judge, "_prefetch_intros", lambda items: None)
    sent = []

    def chunk(rows):
        sent.extend(rows)
        return {ck: {cid: RelevanceVerdict(supports_claim="supports") for _i, cid, _c in g["members"]} for ck, g in rows}

    monkeypatch.setattr(judge, "_judge_citation_chunk", chunk)
    resolved = Resolved(source="s2", match_method=MatchMethod.FUZZY_TITLE, title=_SHORT_TITLE, abstract=_SHORT_REAL)
    out = judge.judge_batch([{"cite_key": "k", "claim_id": "c", "claim": "A test for correlated proportions.",
                              "abstract": _SHORT_REAL, "resolved": resolved}])
    assert sent and sent[0][1]["evidence"] == _SHORT_REAL
    assert out[0].supports_claim == "supports"


# ── a record with nothing to read does not stop the other indexes ────────────
_LONG_ABSTRACT = ("Upper bounds are derived for the probability that the sum of independent bounded random "
                  "variables exceeds its mean by a given amount. The bounds hold for every sample size and "
                  "extend to sampling without replacement and to U-statistics.")


def test_a_record_without_an_abstract_is_held_while_every_index_is_asked(offline):
    ref = 'Q. Smith "Tail bounds for sum of bounded random variables" 1963'
    asked = []

    def s2(*a, **k):
        asked.append("s2")
        return [_row("s2", "Tail bounds for sum of bounded random variables", ["Q. Smith"], 1963)]

    def crossref(*a, **k):
        asked.append("crossref")
        return [_row("crossref", "Tail Bounds for Sums of Bounded Random Variables", ["Quentin Smith"], 1963,
                     abstract=_LONG_ABSTRACT)]

    offline.setattr(paper_lookup, "search_semantic_scholar_match", s2)
    offline.setattr(paper_lookup, "search_crossref_by_fields", crossref)
    found = MultiSourceResolver(validate_urls=False).resolve("smith1963", ref)
    assert asked[:2] == ["s2", "crossref"]
    assert found is not None and found.source == "crossref" and found.abstract == _LONG_ABSTRACT


def test_a_title_repeated_with_another_spelling_is_still_its_repeat():
    garbled = ("Discussion paper series on the bias of leave-one-out estimators of risk "
               "on the bias of leaveoneout estimators of risk")
    assert _repeated_title(garbled) == "on the bias of leave-one-out estimators of risk"


def test_a_garbled_title_is_resolved_as_the_work_it_names_twice(offline):
    ref = ('Ana Ruiz, Ben Ott "Discussion paper series on the bias of leave-one-out estimators of risk '
           'on the bias of leaveoneout estimators of risk"')
    garbled = _row("s2", "Discussion Paper Series On the Bias of Leave-One-Out Estimators of Risk On the Bias "
                         "of Leave-One-Out Estimators of Risk", ["Ana Ruiz", "Ben Ott"], None)
    clean = _row("openalex", "On the Bias of Leave-One-Out Estimators of Risk", ["Ana Ruiz", "Ben Ott"], 2003,
                 abstract=_LONG_ABSTRACT)
    offline.setattr(paper_lookup, "search_semantic_scholar_match", lambda *a, **k: [garbled])
    offline.setattr(paper_lookup, "search_openalex_by_title", lambda *a, **k: [clean])
    found = MultiSourceResolver(validate_urls=False).resolve("ruiz", ref)
    assert found is not None and found.source == "openalex" and found.year == 2003


# ── arXiv's API down: the paper's own page has the record ────────────
_ABS_PAGE = (
    b'<html><head><meta name="citation_title" content="Sparsity and Heterogeneous Dropout for Continual '
    b'Learning in the Null Space of Neural Activations" />'
    b'<meta name="citation_author" content="Abbasi, Ali" /><meta name="citation_author" content="Kolouri, Soheil" />'
    b'<meta name="citation_date" content="2022/03/12" /><meta name="citation_arxiv_id" content="2203.06514" />'
    b'<meta name="citation_abstract" content="Continual learning from a non-stationary stream &amp; more." />'
    b"</head></html>"
)


def _arxiv_api_down(monkeypatch, asked):
    import urllib.error

    def fake_get(url, **kw):
        asked.append(url)
        if "export.arxiv.org" in url:
            raise urllib.error.URLError("arXiv API: 503")
        if url == "https://arxiv.org/abs/2203.06514":
            return _ABS_PAGE
        raise AssertionError(f"unexpected fetch {url}")

    monkeypatch.setattr(paper_lookup, "_get", fake_get)


def test_an_arxiv_id_is_read_off_the_papers_page_when_the_api_fails(monkeypatch):
    asked: list[str] = []
    _arxiv_api_down(monkeypatch, asked)
    [rec] = paper_lookup.fetch_arxiv_by_id("2203.06514v2")
    assert rec["arxiv_id"] == "2203.06514" and rec["year"] == 2022
    assert rec["authors"] == ["Ali Abbasi", "Soheil Kolouri"]
    assert rec["abstract"] == "Continual learning from a non-stationary stream & more."
    assert len(asked) == 2  # the API, then the page


def test_a_printed_arxiv_id_still_matches_exactly_when_the_api_fails(offline):
    from citation_verifier.schema import MatchMethod

    asked: list[str] = []
    _arxiv_api_down(offline, asked)
    ref = ('Ali Abbasi, Soheil Kolouri "Sparsity and heterogeneous dropout for continual learning in the '
           'null space of neural activations" 2022 arXiv:2203.06514')
    got = MultiSourceResolver(validate_urls=False).resolve("abbasi", ref)
    assert got.arxiv_id == "2203.06514" and got.match_method == MatchMethod.ARXIV


def test_a_record_missing_the_printed_arxiv_id_is_not_cached_through_an_outage():
    from citation_verifier.schema import Resolved

    ref = 'A. Abbasi "Sparsity and heterogeneous dropout" 2022 arXiv:2203.06514'
    record = Resolved(source="openreview", title="Sparsity and heterogeneous dropout", authors=["Ali Abbasi"],
                      year=2022, abstract="An abstract long enough to read. " * 10)
    assert MultiSourceResolver._second_best(ref, record)
    assert not MultiSourceResolver._second_best(ref, record.model_copy(update={"arxiv_id": "2203.06514"}))
