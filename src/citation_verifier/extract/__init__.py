"""
extract — turn an ingested :class:`PaperSource` into ``CitationRecord`` stubs.

Two interchangeable extractors, both satisfying the ``Extractor`` Protocol from
the contract (:mod:`citation_verifier.interfaces`) and emitting the identical
stub shape (key + ``claim`` + ``cited_as``; judged axes at defaults):

  * :class:`~citation_verifier.extract.latex.LatexExtractor` — the preferred
    path, used when LaTeX e-print (``.bbl`` / ``.bib`` + ``\\cite`` sites) is
    available. Exact metadata and precise claim-site anchoring.
  * :class:`~citation_verifier.extract.pdf.PdfExtractor` — the fallback, used
    when only a PDF is on disk.

Public surface::

    get_extractor(source) -> Extractor      # pick latex if tex_available else pdf
    extract(source)       -> list[CitationRecord]   # convenience: dispatch + run

Import-safe with no network and no heavy deps at import (``bibtexparser`` /
``pypdf`` are imported lazily inside the extractors).
"""

from __future__ import annotations

from ..interfaces import Extractor, PaperSource
from ..schema import CitationRecord
from .latex import LatexExtractor, make_claim_id
from .pdf import PdfExtractor

__all__ = [
    "get_extractor",
    "extract",
    "LatexExtractor",
    "PdfExtractor",
    "make_claim_id",
]


def get_extractor(source: PaperSource) -> Extractor:
    """Return the right extractor for ``source``.

    LaTeX is strongly preferred (exact references + anchored claim sites); the
    PDF extractor is the fallback. The choice is purely structural —
    ``source.tex_available`` is set by ``ingest`` based on what landed on disk.

    Args:
        source: The normalized, on-disk paper view from ``ingest``.

    Returns:
        A :class:`LatexExtractor` if ``source.tex_available`` else a
        :class:`PdfExtractor`. Both satisfy the ``Extractor`` Protocol.
    """
    if source.tex_available:
        return LatexExtractor()
    return PdfExtractor()


def extract(source: PaperSource) -> list[CitationRecord]:
    """Dispatch (LaTeX-first, PDF fallback) and return record stubs.

    Convenience wrapper over :func:`get_extractor`. Degrade-not-crash: if the
    chosen extractor yields nothing (e.g. an unreadable PDF) the result is an
    empty list rather than an exception.

    Args:
        source: The ingested paper.

    Returns:
        A list of ``CitationRecord`` stubs, one per ``(claim_site, cite_key)``.
    """
    return get_extractor(source).extract(source)
