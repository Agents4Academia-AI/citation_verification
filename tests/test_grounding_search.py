"""
Offline tests for the grounding search cascade — the pure helpers and the
empty-input guards of the field-aware / title-match search functions (which must
return ``[]`` without touching the network, keeping the keyless floor green).
"""

from __future__ import annotations

from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import (
    MultiSourceResolver,
    _first_author,
    _should_search_arxiv,
    _should_search_dblp,
    _title_tokens_contradict,
)


def test_first_author_surname_both_orders():
    # surname-first ("Vaswani A, …") and given-first ("Ashish Vaswani, …")
    assert _first_author("Vaswani A, Shazeer N, Parmar N. Attention is all you need") == "Vaswani"
    assert _first_author("Ashish Vaswani, Noam Shazeer. Attention is all you need") == "Vaswani"
    assert _first_author("Radford A, Wu J, Child R. Language models …") == "Radford"


def test_first_author_empty_when_nothing_usable():
    assert _first_author("") == ""
    assert _first_author("1996. p. 8-14.") == ""


def test_title_token_gate_rejects_missing_version_tokens():
    ref = 'OpenAI. "OpenAI o3 and o4-mini System Card --- openai.com". 2025'
    assert _title_tokens_contradict(ref, "OpenAI o1 System Card")
    assert not _title_tokens_contradict(ref, "OpenAI o3 and o4-mini System Card")


def test_title_query_steps_fall_back_to_title_only_after_author_constraint():
    ref = 'Radford, Alec, Wu, Jeffrey. "Language models are unsupervised multitask learners". 2019'
    steps = MultiSourceResolver(validate_urls=False)._title_query_steps(ref)
    names_args = [(fn.__name__, args) for fn, args in steps]

    assert (
        "search_arxiv_by_title",
        ("Language models are unsupervised multitask learners", 4, "Radford"),
    ) in names_args
    assert (
        "search_arxiv_by_title",
        ("Language models are unsupervised multitask learners", 4, None),
    ) in names_args
    assert (
        "search_crossref_by_fields",
        ("Language models are unsupervised multitask learners", None, 4),
    ) in names_args
    assert names_args.index(
        (
            "search_crossref_by_fields",
            ("Language models are unsupervised multitask learners", "Radford", 4),
        )
    ) < names_args.index(
        (
            "search_arxiv_by_title",
            ("Language models are unsupervised multitask learners", 4, "Radford"),
        )
    )


def test_source_routing_skips_slow_arxiv_for_old_non_preprint_refs():
    old_ref = "Colby KM, Hilf FD. Parry, the paranoid computer program. 1972."
    assert not _should_search_arxiv(old_ref)
    steps = MultiSourceResolver(validate_urls=False)._title_query_steps(old_ref)
    assert all(fn is not paper_lookup.search_arxiv_by_title for fn, _ in steps)


def test_source_routing_keeps_arxiv_and_dblp_for_modern_ml_refs():
    ref = "Radford A, Wu J. Language models are unsupervised multitask learners. 2019."
    assert _should_search_arxiv(ref)
    assert _should_search_dblp(ref)
    sources = MultiSourceResolver(validate_urls=False)._broad_sources(ref)
    assert paper_lookup.SOURCE_ARXIV in sources
    assert paper_lookup.SOURCE_DBLP in sources


def test_search_functions_guard_empty_input_without_network():
    # Too-short / empty titles must short-circuit to [] (no request made).
    assert paper_lookup.search_semantic_scholar_match("") == []
    assert paper_lookup.search_semantic_scholar_match("ab") == []
    assert paper_lookup.search_arxiv_by_title("", author="Vaswani") == []
    assert paper_lookup.search_crossref_by_fields("") == []
