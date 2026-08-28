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

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

__all__ = ["cache_dir", "clear", "read", "write"]

_VERSION = "v1"  # bump when the parsing that produces a Resolved changes materially


def cache_dir() -> Path | None:
    """Where resolutions are stored, or ``None`` when no cache is usable."""
    raw = os.environ.get("CITATION_VERIFIER_CACHE")
    if raw == "":  # explicitly disabled
        return None
    base = (
        Path(raw)
        if raw
        else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "citation-verifier"
    )
    try:
        path = base / "resolve" / _VERSION
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def _key(reference: str) -> str:
    """A stable key for a reference string.

    Whitespace and case are normalised so the same reference reached through the LaTeX
    ``.bbl`` and through a PDF's reference list hits one entry.
    """
    norm = re.sub(r"\s+", " ", (reference or "")).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def read(reference: str, model: Any) -> Any | None:
    """A previously cached ``Resolved`` for this reference, or ``None``."""
    root = cache_dir()
    if root is None or not (reference or "").strip():
        return None
    path = root / f"{_key(reference)}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    try:
        return model(**data)
    except Exception:  # noqa: BLE001 — a stale shape is a miss, not a failure
        return None


def write(reference: str, resolved: Any) -> None:
    """Store a successful resolution. Never raises."""
    root = cache_dir()
    if root is None or resolved is None or not (reference or "").strip():
        return
    try:
        payload = resolved.model_dump(mode="json")
    except AttributeError:
        return
    path = root / f"{_key(reference)}.json"
    try:
        # Write-then-rename so a crashed run cannot leave a half-written entry that later
        # reads as a corrupt cache.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def clear() -> int:
    """Delete every cached resolution; returns how many were removed."""
    root = cache_dir()
    if root is None:
        return 0
    n = 0
    for p in root.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError:
            continue
    return n
