"""
Offline tests for the grounding search cascade — the pure helpers and the
empty-input guards of the field-aware / title-match search functions (which must
return ``[]`` without touching the network, keeping the keyless floor green).
"""

from __future__ import annotations

from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import _first_author


def test_first_author_surname_both_orders():
    # surname-first ("Vaswani A, …") and given-first ("Ashish Vaswani, …")
    assert _first_author("Vaswani A, Shazeer N, Parmar N. Attention is all you need") == "Vaswani"
    assert _first_author("Ashish Vaswani, Noam Shazeer. Attention is all you need") == "Vaswani"
    assert _first_author("Radford A, Wu J, Child R. Language models …") == "Radford"


def test_first_author_empty_when_nothing_usable():
    assert _first_author("") == ""
    assert _first_author("1996. p. 8-14.") == ""


def test_search_functions_guard_empty_input_without_network():
    # Too-short / empty titles must short-circuit to [] (no request made).
    assert paper_lookup.search_semantic_scholar_match("") == []
    assert paper_lookup.search_semantic_scholar_match("ab") == []
    assert paper_lookup.search_arxiv_by_title("", author="Vaswani") == []
    assert paper_lookup.search_crossref_by_fields("") == []
