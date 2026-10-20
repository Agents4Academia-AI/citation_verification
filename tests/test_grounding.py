"""
Offline tests for the grounding helpers that don't touch the network:
title extraction for the arXiv ``ti:`` search, and the empty-input guards of the
direct id/title fetchers (so the keyless floor stays green without network).
"""

from __future__ import annotations

from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import (
    MultiSourceResolver,
    _arxiv_from_siblings,
    _dict_to_candidate,
    _likely_title,
    _likely_titles,
    _looks_like_author_clause,
    _should_search_arxiv,
)
from citation_verifier.interfaces import Candidate
from citation_verifier.schema import MatchMethod


def test_likely_title_prefers_quoted_segment():
    ref = (
        'Song, Yang, Ermon, Stefano. "Generative modeling by estimating gradients '
        'of the data distribution". Neural Information Processing Systems (NeurIPS). 2019'
    )
    assert _likely_title(ref) == "Generative modeling by estimating gradients of the data distribution"


def test_likely_title_unquoted_fallback_skips_authors_and_year():
    ref = "Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning. 2014"
    # author list + year clauses are rejected; the title clause is returned
    assert _likely_title(ref) == "Sequence to sequence learning"


def test_likely_title_prefers_title_over_long_venue_clause():
    ref = (
        "Yang Z, Gan Z, Wang J, Hu X, Lu Y, Liu Z, Wang L. "
        "An empirical study of GPT-3 for few-shot knowledge-based VQA. "
        "Proceedings of the AAAI Conference on Artificial Intelligence. 2022;36(3):3081–9."
    )
    assert _likely_title(ref) == "An empirical study of GPT-3 for few-shot knowledge-based VQA"


def test_likely_title_skips_in_venue_clause():
    ref = (
        "Huang J, Poulis A, Pappas N, Weiss R, Zoph B, Vaswani A, Le QV. "
        "Language models are few-shot learners. "
        "In: Advances in Neural Information Processing Systems. 2020. p. 1877–901."
    )
    assert _likely_title(ref) == "Language models are few-shot learners"


def test_likely_title_skips_two_author_vancouver_list():
    # 2-author Vancouver lists have ONE comma + multi-letter initial blocks
    # ("Colby KM, Hilf FD"); these used to be returned verbatim as the title,
    # so the title search queried author names and found nothing.
    assert (
        _likely_title(
            "Colby KM, Hilf FD. Parry, the paranoid computer program. "
            "In: Proceedings of the National Computer Conference. 1972."
        )
        == "Parry, the paranoid computer program"
    )
    assert (
        _likely_title(
            "Bender EM, Gebru T. The dangers of stylized language: Emergent "
            "biases and sociotechnical remedies. In: Proceedings of FAccT. 2021."
        )
        == "The dangers of stylized language: Emergent biases and sociotechnical remedies"
    )
    # Hyphenated surname must still be recognized as an author token.
    assert (
        _likely_title(
            "Wardrip-Fruin N, Mateas M. The role of non-player characters in "
            "game-based learning for k-12 education. In: Proceedings. 2020."
        )
        == "The role of non-player characters in game-based learning for k-12 education"
    )


def test_looks_like_author_clause_guards_titlecase_acronym():
    # 2-author Vancouver lists are flagged...
    assert _looks_like_author_clause("Colby KM, Hilf FD")
    assert _looks_like_author_clause("Bender EM, Gebru T")
    assert _looks_like_author_clause("Wardrip-Fruin N, Mateas M")
    # ...but real titles (lowercase function words, or a lone Title-Case phrase
    # ending in an acronym) must NOT be mistaken for an author list.
    assert not _looks_like_author_clause("Fast, accurate object detection")
    assert not _looks_like_author_clause("Attention is all you need")
    assert not _looks_like_author_clause(
        "An empirical study of GPT-3 for few-shot knowledge-based VQA"
    )


def test_likely_title_empty_when_nothing_titlelike():
    assert _likely_title("") == ""
    assert _likely_title("Smith, J. 2020.") == ""


def test_likely_titles_adds_site_suffix_stripped_variant():
    ref = 'OpenAI. "OpenAI o3 and o4-mini System Card --- openai.com". 2025'
    assert _likely_titles(ref) == [
        "OpenAI o3 and o4-mini System Card --- openai.com",
        "OpenAI o3 and o4-mini System Card",
    ]


def test_id_and_title_fetchers_guard_empty_input_without_network():
    # Empty / too-short inputs must return [] without making a request.
    assert paper_lookup.fetch_arxiv_by_id("") == []
    assert paper_lookup.fetch_crossref_by_doi("") == []
    assert paper_lookup.fetch_semantic_scholar_by_doi("") == []
    assert paper_lookup.fetch_semantic_scholar_by_arxiv("") == []
    assert paper_lookup.fetch_openalex_by_doi("") == []
    assert paper_lookup.search_arxiv_by_title("") == []
    assert paper_lookup.search_arxiv_by_title("ab") == []


def test_identifier_tier_prefers_s2_exact_and_stops(monkeypatch):
    calls: list[str] = []

    def fake_s2(doi, api_key=None):
        calls.append("s2")
        return [{
            "source": "s2",
            "title": "Exact DOI Paper",
            "authors": ["A. Author"],
            "year": 2024,
            "doi": doi,
            "abstract": "S2 abstract",
        }]

    def fake_crossref(doi):
        calls.append("crossref")
        return [{
            "source": "crossref",
            "title": "Exact DOI Paper",
            "authors": ["A. Author"],
            "year": 2024,
            "doi": doi,
        }]

    monkeypatch.setattr(paper_lookup, "fetch_semantic_scholar_by_doi", fake_s2)
    monkeypatch.setattr(paper_lookup, "fetch_crossref_by_doi", fake_crossref)

    resolved = MultiSourceResolver(validate_urls=False).resolve(
        "k", "A. Author. Exact DOI Paper. 2024. doi:10.1234/example"
    )

    assert resolved is not None
    assert resolved.source == "s2"
    assert resolved.abstract == "S2 abstract"
    assert calls == ["s2"]


def test_crossref_identifier_match_gets_openalex_abstract_top_up(monkeypatch):
    def fake_s2(doi, api_key=None):
        return []

    def fake_crossref(doi):
        return [{
            "source": "crossref",
            "title": "Exact DOI Paper",
            "authors": ["A. Author"],
            "year": 2024,
            "doi": doi,
            "abstract": "",
        }]

    def fake_openalex(doi, api_key=None):
        return [{
            "source": "openalex",
            "title": "Exact DOI Paper",
            "authors": ["A. Author"],
            "year": 2024,
            "doi": doi,
            "abstract": "OpenAlex abstract",
        }]

    monkeypatch.setattr(paper_lookup, "fetch_semantic_scholar_by_doi", fake_s2)
    monkeypatch.setattr(paper_lookup, "fetch_crossref_by_doi", fake_crossref)
    monkeypatch.setattr(paper_lookup, "fetch_openalex_by_doi", fake_openalex)

    resolved = MultiSourceResolver(validate_urls=False).resolve(
        "k", "A. Author. Exact DOI Paper. 2024. doi:10.1234/example"
    )

    assert resolved is not None
    assert resolved.source == "crossref"
    assert resolved.abstract == "OpenAlex abstract"


def test_exact_title_match_survives_noisy_author_and_year_metadata():
    ref = (
        "Huang J, Poulis A, Pappas N, Weiss R, Zoph B, Vaswani A, Le QV. "
        "Language models are few-shot learners. "
        "In: Advances in Neural Information Processing Systems. 2023. p. 1877–901."
    )
    candidate = Candidate(
        source="s2",
        title="Language Models are Few-Shot Learners",
        authors=["Tom B. Brown", "Benjamin Mann", "Nick Ryder"],
        year=2020,
        abstract="We demonstrate that scaling up language models greatly improves task-agnostic performance.",
    )

    resolved = MultiSourceResolver(validate_urls=False)._match(ref, [candidate])

    assert resolved is not None
    assert resolved.title == "Language Models are Few-Shot Learners"


def test_author_gate_tolerates_glued_surname_initials():
    ref = (
        "YangZ, GanZ, WangJ. "
        "An empirical study of GPT-3 for few-shot knowledge-based VQA. 2022."
    )
    candidate = Candidate(
        source="s2",
        title="An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA",
        authors=["Zhengyuan Yang", "Zhe Gan", "Jianfeng Wang"],
        year=2021,
    )

    assert MultiSourceResolver._gate(candidate, ref, 2022) is True


def test_uncorroborated_near_title_does_not_match():
    ref = (
        "Ram A, Fischer A, Saha S, Choudhury R, Batra D, Foulds J. "
        "Alexa prize: socialbot grand challenge 3 finals. "
        "In: Proceedings of the International Conference on Acoustics, Speech and Signal Processing. "
        "2018. p. 6294–8."
    )
    candidate = Candidate(
        source="crossref",
        title="Socialbot DREAM in Alexa Prize Challenge 2019",
        authors=["Y. M. Kuratov", "I. F. Yusupov", "D. R. Baymurzina"],
        year=2021,
    )

    assert MultiSourceResolver(validate_urls=False)._match(ref, [candidate]) is None


# ── evidence enrichment: missing-abstract top-up cascade ─────────────
def test_s2_parse_and_candidate_carry_tldr():
    # S2 returns the tldr object even when the licensed abstract is null; it must
    # survive parsing and the dict->Candidate adaptation (via Candidate.extra).
    item = {
        "title": "Language Models are Unsupervised Multitask Learners",
        "abstract": None,  # S2 withholds the licensed full abstract for GPT-2
        "tldr": {"model": "tldr@v2", "text": "LMs learn tasks without explicit supervision."},
        "externalIds": {"CorpusId": 1},  # no ArXiv id -> off-arXiv tech report
    }
    d = paper_lookup._parse_s2_item(item)
    assert d["abstract"] == "" and d["arxiv_id"] == ""
    assert d["tldr"] == "LMs learn tasks without explicit supervision."
    c = _dict_to_candidate(d)
    assert c.abstract == "" and c.extra.get("tldr") == "LMs learn tasks without explicit supervision."


def test_abstract_top_up_falls_back_to_tldr(monkeypatch):
    # GPT-2 case: no abstract, no arXiv id, no DOI -> tldr is the only evidence,
    # and it must NOT trigger any network lookup.
    def boom(*a, **k):
        raise AssertionError("no id present -> must not hit the network")

    monkeypatch.setattr(paper_lookup, "fetch_arxiv_by_id", boom)
    monkeypatch.setattr(paper_lookup, "fetch_openalex_by_doi", boom)
    c = Candidate(source="s2", title="GPT-2", extra={"tldr": "one-line summary"})
    assert MultiSourceResolver(validate_urls=False)._abstract_top_up(c) == "one-line summary"


def test_abstract_top_up_prefers_real_arxiv_abstract_over_tldr(monkeypatch):
    # Real text beats the AI summary; located by exact id, never a title search.
    monkeypatch.setattr(
        paper_lookup,
        "fetch_arxiv_by_id",
        lambda aid: [{"source": "arxiv", "title": "X", "abstract": "REAL ARXIV ABSTRACT"}],
    )
    c = Candidate(source="s2", title="X", arxiv_id="2207.12598", extra={"tldr": "summary"})
    assert MultiSourceResolver(validate_urls=False)._abstract_top_up(c) == "REAL ARXIV ABSTRACT"


def test_s2_parse_and_candidate_carry_oa_pdf():
    # S2's openAccessPdf.url (the web "View via Publisher"/PDF link) must survive
    # parsing and the dict->Candidate adaptation via Candidate.extra.
    item = {
        "title": "ELIZA—a computer program",
        "abstract": "An abstract that does not cover the specific claim.",
        "isOpenAccess": True,
        "openAccessPdf": {"url": "https://dl.acm.org/doi/pdf/10.1145/x", "status": "BRONZE"},
        "externalIds": {"DOI": "10.1145/365153.365168"},
    }
    d = paper_lookup._parse_s2_item(item)
    assert d["oa_pdf"] == "https://dl.acm.org/doi/pdf/10.1145/x"
    c = _dict_to_candidate(d)
    assert c.extra.get("oa_pdf") == "https://dl.acm.org/doi/pdf/10.1145/x"


def test_s2_parse_missing_oa_pdf_is_empty():
    d = paper_lookup._parse_s2_item({"title": "No OA paper", "externalIds": {}})
    assert d["oa_pdf"] == ""
    assert "oa_pdf" not in _dict_to_candidate(d).extra  # falsy -> not carried


def test_to_resolved_prefers_oa_pdf_as_url():
    # The OA PDF becomes resolved.url so Stage-2 relevance can fetch full text for
    # an off-arXiv paper; it wins over the candidate's landing-page URL.
    c = Candidate(
        source="s2", title="t", url="https://www.semanticscholar.org/paper/abc",
        abstract="present", extra={"oa_pdf": "https://ojs.aaai.org/.../download/2303"},
    )
    res = MultiSourceResolver(validate_urls=False)._to_resolved(c, MatchMethod.FUZZY_TITLE, 0.9)
    assert res.url == "https://ojs.aaai.org/.../download/2303"
    # without an OA PDF, the landing-page URL is kept.
    c2 = Candidate(source="s2", title="t", url="https://landing.example/p", abstract="a")
    res2 = MultiSourceResolver(validate_urls=False)._to_resolved(c2, MatchMethod.FUZZY_TITLE, 0.9)
    assert res2.url == "https://landing.example/p"


def test_present_abstract_short_circuits_all_top_up(monkeypatch):
    # "abstract 已经有就不查": a present abstract must skip every enrichment call.
    def boom(*a, **k):
        raise AssertionError("must not fetch when the abstract is already present")

    monkeypatch.setattr(paper_lookup, "fetch_arxiv_by_id", boom)
    monkeypatch.setattr(paper_lookup, "fetch_openalex_by_doi", boom)
    c = Candidate(
        source="s2", title="t", arxiv_id="2207.12598", doi="10.1/x",
        abstract="REAL ABSTRACT", extra={"tldr": "summary"},
    )
    res = MultiSourceResolver(validate_urls=False)._to_resolved(c, MatchMethod.FUZZY_TITLE, 0.9)
    assert res.abstract == "REAL ABSTRACT"


def test_likely_title_handles_apostrophe_in_double_quotes():
    # A curly apostrophe inside a double-quoted title must not end the quote
    # (else "Let's verify step by step" is unextractable -> unresolved).
    from citation_verifier.grounding.resolver import _likely_title

    ref = 'Lightman, H., Kosaraju, V. "Let’s verify step by step" 2024'
    assert _likely_title(ref) == "Let’s verify step by step"


def test_first_author_surname_helpers():
    from citation_verifier.grounding.resolver import _cand_first_surname, _ref_first_surname

    assert _ref_first_surname("Hindle, A., Barr, E. T., Su, Z.") == "hindle"

    class _C:
        authors = ["Premkumar T. Devanbu"]

    assert _cand_first_surname(_C()) == "devanbu"


def test_validate_url_only_flags_definitive_gone():
    # non-http / empty -> not flagged as dead (returns True, "nothing to disprove")
    from citation_verifier.grounding.paper_lookup import validate_url

    assert validate_url("") is True
    assert validate_url("not-a-url") is True


def test_subtitle_reference_reduces_colon_prefixed_title():
    # a colon-prefix the canonical record drops ("I Speak, You Verify: <title>") —
    # the fallback search uses the post-colon title, matched against the subtitle.
    from citation_verifier.grounding.resolver import _subtitle_reference

    ref = 'Key, D. "I Speak, You Verify: Toward trustworthy neural program synthesis" 2022'
    assert _subtitle_reference(ref) == 'Key, D. "Toward trustworthy neural program synthesis" 2022'
    assert _subtitle_reference('Foo B. "A plain title with no colon" 2020') == ""


def test_match_rejects_exact_id_with_contradicting_title():
    # an arXiv id that resolves to a DIFFERENT paper must not be accepted (-> unresolved),
    # but a matching-title candidate at the same id is accepted.
    import types

    from citation_verifier.grounding.resolver import MultiSourceResolver

    def cand(**kw):
        d = dict(source="s2", title="", authors=[], year=None, venue="", doi="", arxiv_id="", url="", abstract="x", extra={})
        d.update(kw)
        return types.SimpleNamespace(**d)

    r = MultiSourceResolver()
    ref = 'Mejia F. "Kar: Evaluating model proof generation of lambda calculus" arXiv:2102.00182'
    wrong = cand(arxiv_id="2102.00182", title="Entropic barriers as a reason for hardness", authors=["M. Bellitti"])
    assert r._match(ref, [wrong]) is None  # id points elsewhere -> rejected
    right = cand(arxiv_id="2102.00182", title="Kar: Evaluating model proof generation of lambda calculus")
    res = r._match(ref, [right])
    assert res is not None and getattr(res.match_method, "value", res.match_method) == "arxiv"


# ── open-access full-text cascade (oa_fulltext, adapted from Klara Kaleb) ──


def test_oa_html_to_text_strips_scripts_and_markup():
    from citation_verifier.grounding.oa_fulltext import _html_to_text

    html = "<html><head><style>.x{}</style></head><body><script>bad()</script><p>Hello &amp; world</p></body></html>"
    out = _html_to_text(html)
    assert "Hello" in out and "world" in out
    assert "bad()" not in out and ".x{}" not in out and "<p>" not in out


def test_oa_html_to_text_preserves_section_structure():
    # headings must survive as their own lines so split_sections() can recover
    # document structure (heading provenance + experimental-section gating).
    from citation_verifier.grounding.fulltext import split_sections
    from citation_verifier.grounding.oa_fulltext import _html_to_text

    html = "<h2>Introduction</h2><p>We propose a method.</p><h2>Conclusion</h2><p>It works well.</p>"
    out = _html_to_text(html)
    assert "Introduction\n" in out  # heading isolated on its own line
    sections = dict(split_sections(out))
    assert "Introduction" in sections and "method" in sections["Introduction"].lower()
    assert "Conclusion" in sections


def test_oa_norm_doi_strips_prefixes():
    from citation_verifier.grounding.oa_fulltext import _norm_doi

    assert _norm_doi("https://doi.org/10.1/abc") == "10.1/abc"
    assert _norm_doi("doi:10.1/abc") == "10.1/abc"
    assert _norm_doi("  10.1/abc  ") == "10.1/abc"


def test_oa_arxiv_html_text_extracts_and_guards(monkeypatch):
    from citation_verifier.grounding import oa_fulltext as oa

    body = "word " * 400  # comfortably past the _MIN_TEXT guard
    monkeypatch.setattr(oa, "_http_get_text", lambda url, timeout=30: f"<html><body>{body}</body></html>")
    assert "word" in oa.arxiv_html_text("2310.12345v2")
    # too-short pages are rejected (stub / error / landing page guard)
    monkeypatch.setattr(oa, "_http_get_text", lambda url, timeout=30: "<html><body>tiny</body></html>")
    assert oa.arxiv_html_text("2310.12345") == ""
    # empty id => no network, no text
    assert oa.arxiv_html_text("") == ""


def test_oa_arxiv_html_reports_actual_host(monkeypatch):
    # provenance: when the first host (arxiv.org/html) is empty and ar5iv answers,
    # the returned URL must be the ar5iv one — not assumed to be arxiv.org/html.
    from citation_verifier.grounding import oa_fulltext as oa

    body = "word " * 400

    def fake_get(url, timeout=30):
        return "" if "arxiv.org/html" in url else f"<html><body>{body}</body></html>"

    monkeypatch.setattr(oa, "_http_get_text", fake_get)
    text, url = oa.arxiv_html_text_with_url("2310.12345v2")
    assert "word" in text
    assert url == "https://ar5iv.org/abs/2310.12345"


def test_oa_html_to_text_unescapes_entities():
    # entities must keep their semantics (not be dropped to spaces).
    from citation_verifier.grounding.oa_fulltext import _html_to_text

    out = _html_to_text("<p>Caf&eacute; &amp; &#945;-decay</p>")
    assert "Café" in out and " & " in out and "α-decay" in out


def test_oa_unpaywall_is_env_gated_failsoft(monkeypatch):
    from citation_verifier.grounding import oa_fulltext as oa

    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    # no contact email => Unpaywall is skipped without ever touching the network
    monkeypatch.setattr(oa, "_http_get_json", lambda url, timeout=30: (_ for _ in ()).throw(AssertionError("net!")))
    assert oa._unpaywall_pdf_urls("10.1/abc", 30) == []


def test_oa_pdf_urls_orders_and_dedupes(monkeypatch):
    from citation_verifier.grounding import oa_fulltext as oa

    monkeypatch.setattr(oa, "_openalex_pdf_url", lambda doi, t: ["https://a/x.pdf"])
    monkeypatch.setattr(oa, "_unpaywall_pdf_urls", lambda doi, t: ["https://a/x.pdf", "https://b/y.pdf"])
    monkeypatch.setattr(oa, "_meta_pdf_url", lambda doi, t: ["https://c/z.pdf"])
    assert oa.oa_pdf_urls("10.1/abc") == ["https://a/x.pdf", "https://b/y.pdf", "https://c/z.pdf"]
    assert oa.oa_pdf_urls("") == []  # empty doi => no work


def test_oa_pdf_urls_failsoft_when_a_resolver_raises(monkeypatch):
    from citation_verifier.grounding import oa_fulltext as oa

    monkeypatch.setattr(oa, "_openalex_pdf_url", lambda doi, t: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(oa, "_unpaywall_pdf_urls", lambda doi, t: ["https://b/y.pdf"])
    monkeypatch.setattr(oa, "_meta_pdf_url", lambda doi, t: [])
    assert oa.oa_pdf_urls("10.1/abc") == ["https://b/y.pdf"]


def test_oa_fulltext_by_doi_returns_first_extractable(monkeypatch):
    from citation_verifier.grounding import fulltext
    from citation_verifier.grounding import oa_fulltext as oa

    monkeypatch.setattr(
        oa, "_oa_pdf_sources", lambda doi, t: [("https://a/x.pdf", "openalex_oa"), ("https://b/y.pdf", "unpaywall")]
    )
    seen = []

    def fake_extract(url, timeout=30, max_chars=200_000):
        seen.append(url)
        return "FULL TEXT" if url.endswith("y.pdf") else ""

    monkeypatch.setattr(fulltext, "fetch_full_text_from_url", fake_extract)
    # bare-text wrapper still returns the text
    assert oa.fulltext_by_doi("10.1/abc") == "FULL TEXT"
    assert seen == ["https://a/x.pdf", "https://b/y.pdf"]  # tried in order, stopped at first hit
    # provenance variant carries the channel + URL that succeeded
    res = oa.fulltext_by_doi_with_source("10.1/abc")
    assert (res.text, res.source, res.url) == ("FULL TEXT", "unpaywall", "https://b/y.pdf")


def test_the_arxiv_id_survives_when_another_source_wins_the_match():
    r"""A conference paper resolves to its publisher record, which has no arXiv id.

    Crossref answers with the ``10.1109/…`` record while Semantic Scholar returns the same
    paper WITH its arXiv id in the very same fetch. Dropping it costs the full text: the
    DOI link is a landing page, so the paper is reduced to its abstract even though a
    preprint is one request away. Gated on an exact normalised-title match — a
    near-namesake's arXiv id would send the judge to the wrong paper.
    """
    winner = Candidate(source="crossref", title="USIP: Unsupervised Stable Interest Point",
                       doi="10.1109/iccv.2019.00045")
    sibling = Candidate(source="s2", title="USIP:  unsupervised stable interest point",
                        arxiv_id="1904.00229")
    other = Candidate(source="s2", title="A different paper entirely", arxiv_id="9999.99999")
    assert _arxiv_from_siblings(winner, [sibling, other]) == "1904.00229"
    # a candidate that already has one is left alone
    assert _arxiv_from_siblings(sibling, [other]) == ""
    # and a near-namesake never contributes its id
    assert _arxiv_from_siblings(winner, [other]) == ""


def test_a_spelled_out_venue_routes_the_reference_to_arxiv():
    """A bibliography that writes the venue out rather than abbreviating it carries none of
    the acronyms, so arXiv was never queried for it even though the paper is mirrored
    there. Both spellings occur in one bibliography."""
    assert _should_search_arxiv(
        'Li, Jiaxin. "USIP". Proceedings of the IEEE/CVF international conference on '
        "computer vision. 2019"
    )
    assert _should_search_arxiv(
        "A. B. A study. Transactions of the Association for Computational Linguistics, 2023."
    )
    # a title that merely mentions the field must not route every reference to arXiv
    assert not _should_search_arxiv(
        "J. Doe. Deep learning for computer vision applications. My Book, 2019."
    )
