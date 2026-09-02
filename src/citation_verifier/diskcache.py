"""Shared location and disable switch for the project's on-disk caches.

Three things are worth keeping between runs — a resolved reference, a retrieved full
text, a column's gloss — and each was re-deriving the same directory logic. They share a
root and one environment variable so a deployment can point them all somewhere else, or
turn them all off, in one place.

``CITATION_VERIFIER_CACHE``: a directory to use, or the empty string to disable caching
entirely (the test suite sets it empty so a developer's cache can never satisfy a call
before the fakes are consulted).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

__all__ = ["clear", "key_for", "namespace_dir", "read_json", "write_json"]


def namespace_dir(namespace: str, version: str) -> Path | None:
    """The directory for one cache, or ``None`` when caching is off or unusable.

    Versioned: bump it when the derivation changes materially, so stale entries are
    ignored rather than silently trusted.
    """
    raw = os.environ.get("CITATION_VERIFIER_CACHE")
    if raw == "":
        return None
    base = (
        Path(raw)
        if raw
        else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "citation-verifier"
    )
    try:
        path = base / namespace / version
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def key_for(*parts: str) -> str:
    """A stable filename for the given identifying parts.

    Whitespace and case are normalised, so the same reference reached through a LaTeX
    ``.bbl`` and through a PDF's reference list lands on one entry.
    """
    norm = "\x1f".join(re.sub(r"\s+", " ", p or "").strip().lower() for p in parts)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def read_json(root: Path | None, key: str) -> Any | None:
    """A cached JSON payload, or ``None``. A missing or damaged entry is a miss."""
    if root is None or not key:
        return None
    try:
        return json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(root: Path | None, key: str, payload: Any) -> None:
    """Store a payload. Never raises.

    Written to a temporary file and renamed, so a crashed run cannot leave a half-written
    entry that later reads as a corrupt cache.
    """
    if root is None or not key:
        return
    path = root / f"{key}.json"
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def clear(root: Path | None) -> int:
    """Delete every entry in one cache; returns how many were removed."""
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
