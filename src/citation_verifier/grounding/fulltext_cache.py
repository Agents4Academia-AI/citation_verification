"""On-disk cache for retrieved full text.

Fetching one paper's body walks arXiv HTML → LaTeX e-print → PDF, downloading up to
200 KB and parsing it. The content does not change, so paying that once per paper rather
than once per paper *per run* is free speed — and it removes one source of run-to-run
variance: which channel answers first can differ between runs, and a different channel
yields different text, from which different passages are then selected. Measured: a
confirmed finding disappeared between two runs because retrieval surfaced another
paragraph of the same paper.

Only non-empty results are cached. An empty fetch is usually a timeout or a transient
403, and caching it would freeze a passing failure into a permanent one.

Shares its location and disable switch with :mod:`resolve_cache`
(``CITATION_VERIFIER_CACHE``). Every operation fails soft.
"""

from __future__ import annotations

from pathlib import Path

from ..diskcache import clear as _clear
from ..diskcache import key_for, namespace_dir, read_json, write_json

__all__ = ["cache_dir", "clear", "read", "write"]

_VERSION = "v1"  # bump when the parsing that produces the text changes materially


def cache_dir() -> Path | None:
    """Where entries are stored, or ``None`` when caching is off."""
    return namespace_dir("fulltext", _VERSION)


def _key(ident: str) -> str:
    """A stable key for this identifier."""
    return key_for(ident)


def read(ident: str) -> tuple[str, str, str] | None:
    """Cached ``(text, source, url)`` for this identifier, or ``None``."""
    data = read_json(cache_dir(), _key(ident)) if (ident or "").strip() else None
    if not isinstance(data, dict):
        return None
    text = data.get("text")
    if not isinstance(text, str) or not text:
        return None
    return text, str(data.get("source") or ""), str(data.get("url") or "")



def write(ident: str, text: str, source: str = "", url: str = "") -> None:
    """Store a non-empty retrieval. Never raises."""
    if not (ident or "").strip() or not text:
        return
    write_json(cache_dir(), _key(ident), {"text": text, "source": source, "url": url})



def clear() -> int:
    """Delete every entry; returns how many were removed."""
    return _clear(cache_dir())
