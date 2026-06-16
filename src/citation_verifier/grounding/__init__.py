"""
grounding — "never decide from memory" layer.

This package fetches *canonical* metadata for a cited reference from authoritative
scholarly sources and matches it to the reference, so the verification stages can
adjudicate against retrieved facts instead of model recall.

Public surface:
    MultiSourceResolver  — resolve(cite_key, reference) -> Resolved | None
    lookup_paper         — baseline-compatible candidate fetch
    validate_url         — fail-soft URL liveness check

The whole package imports and runs without ``requests`` and without network
(optional sources return ``[]``; network functions fail soft).
"""

from __future__ import annotations

from .paper_lookup import (
    lookup_paper,
    search_arxiv,
    search_crossref,
    search_dblp,
    search_openalex,
    search_semantic_scholar,
    validate_url,
)
from .resolver import MultiSourceResolver

__all__ = [
    "MultiSourceResolver",
    "lookup_paper",
    "validate_url",
    "search_crossref",
    "search_arxiv",
    "search_dblp",
    "search_semantic_scholar",
    "search_openalex",
]
