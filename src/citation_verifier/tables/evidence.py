"""
tables/evidence.py — retrieve the CITED paper's own text for cell verification.

A table cell asserts something about another paper, so the evidence has to come from
that paper. This wires the existing grounding layer into the ``evidence_for`` seam of
:func:`citation_verifier.tables.verify.verify_table`.

Why a dedicated builder rather than "just use the abstract":

  * **Abstracts are truncated.** The lookup layer caps a fetched abstract at 600
    characters, roughly half of a typical one, and the sentence that settles a property
    is as likely to be in the second half as the first.
  * **Table properties are rarely in the abstract at all.** "Needs no retraining",
    "requires access to the training data", "is SE(3)-equivariant" are method details
    that live in the body. Judging them on an abstract yields honest-but-useless
    ``unverifiable`` for most cells.
  * **The query is the COLUMN, not the row.** Excerpts must be selected against what the
    property means; searching the cited paper for its own name finds nothing useful.

Retrieval order per reference: resolve the citation, then arXiv full text (HTML → LaTeX
e-print → PDF) or, for non-arXiv work, the open-access text behind its URL/DOI. Every
step is fail-soft — with nothing retrieved the cells become ``unverifiable``, which is
the honest outcome, never a refutation.

Both external dependencies (``lookup``, ``resolver``) are injected, so this module is
testable offline.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

__all__ = ["build_evidence_provider", "compose_evidence"]

_MAX_EVIDENCE_CHARS = 9000


def _dimension_queries(dimensions: list[Any]) -> list[str]:
    """One retrieval query per column: what the column asserts, not its terse header."""
    out: list[str] = []
    for d in dimensions or []:
        gloss = (getattr(d, "gloss", "") or "").strip()
        header = (getattr(d, "header", "") or "").strip()
        q = f"{header}. {gloss}" if gloss else header
        if q.strip():
            out.append(q.strip())
    return out


def compose_evidence(
    title: str, abstract: str, chunks: list[tuple[str, str]], *, max_chars: int = _MAX_EVIDENCE_CHARS
) -> str:
    """Assemble the evidence block sent to the judge, tagged by provenance.

    Section tags are kept so the judge can say where it read something, and duplicate
    excerpts (the same passage selected by two columns) are dropped.

    The block opens with how much of the paper it covers, because that changes what
    silence means. A property missing from the full text is evidence the work does not
    claim it; the same property missing from an abstract is evidence of nothing. Without
    the distinction the judge abstained on both, and "we could not check" was reported for
    cells where the cited paper had in fact been read end to end.
    """
    coverage = (
        "COVERAGE: full text — the excerpts below were selected from the whole paper"
        if chunks else
        "COVERAGE: title and abstract only — the body was not retrieved"
    )
    parts = [coverage, f"TITLE: {title}".strip()]
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    seen: set[str] = set()
    body: list[str] = []
    for heading, text in chunks:
        key = re.sub(r"\W+", "", (text or "").lower())[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        body.append(f"[{heading or 'body'}] {text}")
    if body:
        parts.append("FULL TEXT EXCERPTS:\n" + "\n".join(body))
    out = "\n\n".join(p for p in parts if p.strip())
    return out[:max_chars]


def build_evidence_provider(
    *,
    lookup: Callable[[str | None], Any],
    resolver: Any,
    dimensions: list[Any],
    use_full_text: bool = True,
    chunks_per_query: int = 2,
    chunk_chars: int = 700,
    retries: int = 3,
    pace_seconds: float = 1.0,
) -> Callable[[str | None, str], tuple[str, str]]:
    """Build an ``evidence_for(cite_key, row_label) -> (evidence, source)`` callable.

    Args:
        lookup: ``cite_key`` (or PDF marker) -> the reference as written, with a ``raw``
            attribute. Return ``None`` when the key is not in the bibliography.
        resolver: object with ``resolve(key, reference)`` returning a canonical record
            (``title`` / ``abstract`` / ``arxiv_id`` / ``url``), or ``None``.
        dimensions: the table's columns — their glosses become the retrieval queries.
        use_full_text: fetch the cited paper's body. Off makes every cell abstract-only.
        chunks_per_query: excerpts to keep per column.
        chunk_chars: max characters per excerpt.
        retries: attempts before giving up on a reference. The metadata APIs rate-limit
            a burst of lookups and simply return nothing, which is indistinguishable
            from "this paper does not exist" — measured: the same key failed when
            queried back-to-back and resolved on every attempt once spaced out. Without
            retrying, a fifth of the rows silently become ``unverifiable``.
        pace_seconds: delay before each retry (doubled each time).

    Returns:
        A cached provider. Never raises: any failure yields ``("", key)`` and the cells
        become ``unverifiable``.
    """
    queries = _dimension_queries(dimensions)
    cache: dict[str | None, tuple[str, str]] = {}

    def evidence_for(cite_key: str | None, row_label: str) -> tuple[str, str]:
        if cite_key in cache:
            return cache[cite_key]
        result = ("", str(cite_key or row_label))
        try:
            ref = lookup(cite_key)
            reference = (getattr(ref, "raw", "") or "") if ref is not None else ""
            got = None
            if ref is not None:
                for attempt in range(max(1, retries)):
                    got = resolver.resolve(str(cite_key or ""), reference or row_label)
                    if got is not None:
                        break
                    if attempt + 1 < max(1, retries):
                        time.sleep(pace_seconds * (2**attempt))
            if got is not None:
                title = getattr(got, "title", "") or ""
                abstract = getattr(got, "abstract", "") or ""
                source = getattr(got, "url", "") or str(cite_key or "")
                chunks: list[tuple[str, str]] = []
                if use_full_text and queries:
                    chunks = _full_text_chunks(got, queries, chunks_per_query, chunk_chars)
                result = (compose_evidence(title, abstract, chunks), source)
        except Exception:  # noqa: BLE001 — retrieval is best-effort by contract
            result = ("", str(cite_key or row_label))
        cache[cite_key] = result
        return result

    return evidence_for


def _full_text_of(resolved: Any) -> str:
    """The cited paper's body, through every channel the project has.

    The same four-step cascade the prose stage uses: arXiv, then the resolved URL, then
    the DOI's open-access copy (Unpaywall / OpenAlex / publisher meta tags), then a title
    search. Stopping after the first two — as this module did — silently reduces a whole
    class of papers to their abstract: an ICCV or ICRA reference resolves to a
    ``doi.org/10.1109/…`` link that is a landing page, not a PDF, so every cell citing it
    came back "we only saw the abstract" while an open-access copy existed. Measured:
    nine of USEEK's cells, whose citations are almost all IEEE-published.
    """
    from ..grounding.fulltext import (  # noqa: PLC0415 — lazy: keeps import network-free
        fetch_full_text_from_url,
        fetch_full_text_via_search_with_source,
        fetch_full_text_with_source,
    )

    arxiv_id = getattr(resolved, "arxiv_id", None)
    if arxiv_id:
        text = getattr(fetch_full_text_with_source(arxiv_id), "text", "") or ""
        if text:
            return text
    url = (getattr(resolved, "url", "") or "").strip()
    if url:
        got = fetch_full_text_from_url(url)
        text = got if isinstance(got, str) else (getattr(got, "text", "") or "")
        if text:
            return text
    doi = (getattr(resolved, "doi", "") or "").strip()
    if doi:
        from ..grounding.oa_fulltext import fulltext_by_doi_with_source  # noqa: PLC0415

        text = getattr(fulltext_by_doi_with_source(doi), "text", "") or ""
        if text:
            return text
    title = (getattr(resolved, "title", "") or "").strip()
    if title:
        got = fetch_full_text_via_search_with_source(title, year=getattr(resolved, "year", None))
        return getattr(got, "text", "") or ""
    return ""


def _full_text_chunks(
    resolved: Any, queries: list[str], per_query: int, chunk_chars: int
) -> list[tuple[str, str]]:
    """Excerpts of the cited paper's body, selected once per column query."""
    text = _full_text_of(resolved)
    if not text:
        return []
    from ..grounding.fulltext import (  # noqa: PLC0415 — lazy: keeps import network-free
        select_evidence_chunks,
        split_sections,
    )
    sections = split_sections(text)
    out: list[tuple[str, str]] = []
    for q in queries:
        out += select_evidence_chunks(q, sections, k=per_query, max_chars=chunk_chars)
    return out
