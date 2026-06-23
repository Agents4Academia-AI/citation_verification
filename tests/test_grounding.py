"""
Offline tests for the grounding helpers that don't touch the network:
title extraction for the arXiv ``ti:`` search, and the empty-input guards of the
direct id/title fetchers (so the keyless floor stays green without network).
"""

from __future__ import annotations

from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import (
    MultiSourceResolver,
    _dict_to_candidate,
    _likely_title,
    _likely_titles,
    _looks_like_author_clause,
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
