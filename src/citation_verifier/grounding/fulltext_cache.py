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

import hashlib
import json
import os
import re
from pathlib import Path

__all__ = ["cache_dir", "clear", "read", "write"]

_VERSION = "v1"  # bump when the parsing that produces the text changes materially


def cache_dir() -> Path | None:
    """Where full texts are stored, or ``None`` when no cache is usable."""
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
        path = base / "fulltext" / _VERSION
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        return None


def _key(ident: str) -> str:
    """A stable key for an arXiv id or a URL."""
    norm = re.sub(r"\s+", "", (ident or "")).strip().lower().rstrip("/")
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def read(ident: str) -> tuple[str, str, str] | None:
    """Cached ``(text, source, url)`` for this identifier, or ``None``."""
    root = cache_dir()
    if root is None or not (ident or "").strip():
        return None
    try:
        data = json.loads((root / f"{_key(ident)}.json").read_text(encoding="utf-8"))
        text = data["text"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(text, str) or not text:
        return None
    return text, str(data.get("source") or ""), str(data.get("url") or "")


def write(ident: str, text: str, source: str = "", url: str = "") -> None:
    """Store a non-empty retrieval. Never raises."""
    root = cache_dir()
    if root is None or not (ident or "").strip() or not text:
        return
    path = root / f"{_key(ident)}.json"
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"text": text, "source": source, "url": url}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        return


def clear() -> int:
    """Delete every cached full text; returns how many were removed."""
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
