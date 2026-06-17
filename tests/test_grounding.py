"""
Offline tests for the grounding helpers that don't touch the network:
title extraction for the arXiv ``ti:`` search, and the empty-input guards of the
direct id/title fetchers (so the keyless floor stays green without network).
"""

from __future__ import annotations

from citation_verifier.grounding import paper_lookup
from citation_verifier.grounding.resolver import _likely_title


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


def test_likely_title_empty_when_nothing_titlelike():
    assert _likely_title("") == ""
    assert _likely_title("Smith, J. 2020.") == ""


def test_id_and_title_fetchers_guard_empty_input_without_network():
    # Empty / too-short inputs must return [] without making a request.
    assert paper_lookup.fetch_arxiv_by_id("") == []
    assert paper_lookup.fetch_crossref_by_doi("") == []
    assert paper_lookup.search_arxiv_by_title("") == []
    assert paper_lookup.search_arxiv_by_title("ab") == []
