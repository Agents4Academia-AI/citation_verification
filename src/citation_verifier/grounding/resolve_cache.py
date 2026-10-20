"""On-disk cache for reference resolution.

Resolving one reference costs 2–19 seconds: the cascade tries DOI/arXiv ids, then S2,
arXiv and Crossref title matches, then a broad search, and a rate-limited source is
retried with backoff. Bibliographic records do not change, so paying that once per
reference — rather than once per reference *per run* — is free correctness-preserving
speed. Measured on the comparison-table corpus: the same forty-odd references were
re-resolved on every one of nine runs.

Only SUCCESSES are cached. A miss is very often a rate limit rather than a fact about the
world — the same key failed back-to-back and resolved once spaced out — so caching it
would freeze a transient failure into a permanent one.

Location: ``$CITATION_VERIFIER_CACHE`` if set, else ``$XDG_CACHE_HOME/citation-verifier``,
else ``~/.cache/citation-verifier``. Every operation fails soft: an unwritable or corrupt
cache degrades to no cache, never to an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..diskcache import clear as _clear
from ..diskcache import key_for, namespace_dir, read_json, write_json

__all__ = ["cache_dir", "clear", "read", "write"]

_VERSION = "v1"  # bump when the parsing that produces a Resolved changes materially


def cache_dir() -> Path | None:
    """Where entries are stored, or ``None`` when caching is off."""
    return namespace_dir("resolve", _VERSION)


def _key(ident: str) -> str:
    """A stable key for this identifier."""
    return key_for(ident)


def read(reference: str, model: Any) -> Any | None:
    """A previously cached ``Resolved`` for this reference, or ``None``."""
    data = read_json(cache_dir(), _key(reference)) if (reference or "").strip() else None
    if not isinstance(data, dict):
        return None
    try:
        return model(**data)
    except Exception:  # noqa: BLE001 — a stale shape is a miss, not a failure
        return None



def write(reference: str, resolved: Any) -> None:
    """Store a successful resolution. Never raises."""
    if resolved is None or not (reference or "").strip():
        return
    try:
        payload = resolved.model_dump(mode="json")
    except AttributeError:
        return
    write_json(cache_dir(), _key(reference), payload)



def clear() -> int:
    """Delete every entry; returns how many were removed."""
    return _clear(cache_dir())
