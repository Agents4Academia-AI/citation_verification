"""
Offline tests for the grounding helpers that don't touch the network:
title extraction for the arXiv ``ti:`` search, and the empty-input guards of the
direct id/title fetchers (so the keyless floor stays green without network).
"""

from __future__ import annotations

from citation_verifier.interfaces import Candidate
from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import MultiSourceResolver, _likely_title, _likely_titles


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
