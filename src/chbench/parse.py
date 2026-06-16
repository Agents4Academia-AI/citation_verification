"""
parse.py — turn a harvested paper into references + claim sites.

This mirrors the agent's extraction seam (citation_verifier.extract) but for the
GOLD pipeline: it produces *parsed-reference* dicts whose key fields line up 1:1
with :class:`citation_verifier.schema.CitationRecord` so labelling is a direct
fill. Each parsed item carries the record key ``(paper_id, claim_id, cite_key)``
plus the ``cited_as`` (claimed metadata) and the ``claim`` (the in-text site):

    {
      "paper_id":  str,
      "claim_id":  str,        # deterministic per claim site
      "cite_key":  str,        # \\cite key / bib key
      "cited_as":  {raw, authors[], title, year, venue, doi, arxiv_id, url},
      "claim":     {claim_id, text, section, char_span},
    }

One dict per (claim-site, citation) pair — exactly the record granularity.

LaTeX is the primary path (.bbl/.bib + ``\cite`` call-sites give exact reference
+ claim anchoring); PDF is a fallback. The .bib parse uses ``bibtexparser`` when
available and degrades to a small stdlib regex parser otherwise, so this module
imports and runs without optional deps. PDF text extraction is an HONEST STUB:
without a PDF text layer it returns ``[]`` rather than fabricating claim sites.
"""

from __future__ import annotations

import re
import tarfile
from pathlib import Path
from typing import Any

# ── claim-id derivation (must match the agent extractor's convention) ─────────


def make_claim_id(paper_id: str, cite_key: str, occurrence: int) -> str:
    """Deterministic, stable claim-site id: ``<paper_id>:<cite_key>#<n>``.

    The same (paper, citation, occurrence) always yields the same id so gold and
    agent output join on ``(paper_id, claim_id, cite_key)`` across runs.
    """
    return f"{paper_id}:{cite_key}#{occurrence}"


# ── .bib / .bbl parsing ───────────────────────────────────────────────────────


def _parse_bib_entries(bib_text: str) -> dict[str, dict[str, Any]]:
    """Parse BibTeX text into ``{cite_key: cited_as-dict}``.

    Uses ``bibtexparser`` if importable; otherwise a minimal stdlib fallback that
    captures key + title/author/year/journal-or-booktitle fields. Both yield the
    ``cited_as`` shape consumed by the labeller.
    """
    try:
        import bibtexparser  # type: ignore

        db = bibtexparser.loads(bib_text)
        out: dict[str, dict[str, Any]] = {}
        for entry in db.entries:
            key = entry.get("ID")
            if not key:
                continue
            out[key] = _entry_to_cited_as(entry)
        return out
    except Exception:
        return _parse_bib_fallback(bib_text)


def _entry_to_cited_as(entry: dict[str, str]) -> dict[str, Any]:
    """Map a bibtexparser entry dict to the ``cited_as`` field shape."""
    authors = [
        a.strip()
        for a in re.split(r"\s+and\s+", entry.get("author", ""))
        if a.strip()
    ]
    year = entry.get("year")
    venue = entry.get("journal") or entry.get("booktitle") or entry.get("publisher")
    return {
        "raw": _flatten_bib_entry(entry),
        "authors": authors,
        "title": (entry.get("title") or "").strip("{} ") or None,
        "year": year,
        "venue": venue,
        "doi": entry.get("doi"),
        "arxiv_id": entry.get("eprint") if entry.get("archiveprefix", "").lower() == "arxiv" else None,
        "url": entry.get("url"),
    }


def _flatten_bib_entry(entry: dict[str, str]) -> str:
    """Reconstruct a readable one-line reference string from a bib entry."""
    parts = [entry.get("author", ""), entry.get("title", ""), entry.get("year", "")]
    venue = entry.get("journal") or entry.get("booktitle") or ""
    parts.append(venue)
    return ". ".join(p.strip("{} ") for p in parts if p).strip()


def _parse_bib_fallback(bib_text: str) -> dict[str, dict[str, Any]]:
    """Dependency-free BibTeX parser (best-effort): enough for offline import."""
    out: dict[str, dict[str, Any]] = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,]+),(.*?)\n\}", bib_text, re.DOTALL):
        key = m.group(1).strip()
        body = m.group(2)
        fields: dict[str, str] = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.+?)[}\"]\s*,?", body, re.DOTALL):
            fields[fm.group(1).lower()] = re.sub(r"\s+", " ", fm.group(2)).strip()
        out[key] = _entry_to_cited_as(fields)
    return out


def _find_tex_sources(tex_dir: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Collect concatenated ``.tex`` body and parsed bib entries from a dir.

    Returns ``(body_text, {cite_key: cited_as})``. Reads ``.bib`` and ``.bbl``
    (the rendered bibliography) for references.
    """
    body_parts: list[str] = []
    bib_text_parts: list[str] = []
    for path in sorted(tex_dir.rglob("*")):
        if path.suffix == ".tex":
            body_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        elif path.suffix in (".bib", ".bbl"):
            bib_text_parts.append(path.read_text(encoding="utf-8", errors="replace"))
    entries = _parse_bib_entries("\n".join(bib_text_parts))
    return "\n".join(body_parts), entries


# ── \cite call-site extraction ────────────────────────────────────────────────

_CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}")


def _extract_claim_sites(body: str) -> list[tuple[str, str]]:
    """Find ``\\cite{...}`` call-sites; return ``[(cite_key, sentence), ...]``.

    The "sentence" is a window of text around the citation, used as the claim it
    is attached to. Multi-key ``\\cite{a,b}`` yields one site per key.
    """
    sites: list[tuple[str, str]] = []
    for m in _CITE_RE.finditer(body):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        sentence = _sentence_around(body, m.start(), m.end())
        for key in keys:
            sites.append((key, sentence))
    return sites


def _sentence_around(text: str, start: int, end: int, *, window: int = 240) -> str:
    """Extract a cleaned text window around a citation as the claim sentence."""
    left = text.rfind(".", max(0, start - window), start)
    lo = left + 1 if left != -1 else max(0, start - window)
    right = text.find(".", end, end + window)
    hi = right + 1 if right != -1 else min(len(text), end + window)
    snippet = text[lo:hi]
    snippet = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", snippet)  # strip macros
    return re.sub(r"\s+", " ", snippet).strip()


def _extract_tex_dir(tex_archive: Path, work_dir: Path) -> Path | None:
    """Extract an arXiv e-print tar(.gz) into a dir; return it, or None on error."""
    target = work_dir / "tex"
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tex_archive) as tf:
            tf.extractall(target, filter="data")  # py3.12+ safe extraction
        return target
    except (tarfile.TarError, OSError, ValueError):
        return None


# ── public entry point ────────────────────────────────────────────────────────


def parse_paper(path: str | Path, *, paper_id: str | None = None) -> list[dict[str, Any]]:
    """Parse a harvested paper into reference+claim-site dicts (record key shape).

    Args:
        path: a LaTeX e-print archive (``.tar``/``.tar.gz``), an extracted tex
            directory, or a PDF. LaTeX is the primary path.
        paper_id: stable paper id for the record key; defaults to the file stem.

    Returns:
        One dict per (claim-site, citation) pair with ``paper_id``, ``claim_id``,
        ``cite_key``, ``cited_as`` and ``claim`` populated (the rest of the
        CitationRecord is filled by :mod:`chbench.label`). Empty list if the
        paper could not be parsed (fail-soft; e.g. a PDF without a text layer).
    """
    p = Path(path)
    pid = paper_id or p.stem.split(".tar")[0]

    tex_dir: Path | None = None
    if p.is_dir():
        tex_dir = p
    elif p.suffix in (".gz", ".tar") or p.name.endswith(".tar.gz"):
        tex_dir = _extract_tex_dir(p, p.parent)
    elif p.suffix == ".pdf":
        # HONEST STUB: PDF text extraction needs a PDF text layer (pdfminer /
        # the agent's pdf extractor). The gold pipeline prefers LaTeX; rather
        # than fabricate references, return nothing and let the caller record
        # this paper as unparsed.
        return []

    if tex_dir is None:
        return []

    body, entries = _find_tex_sources(tex_dir)
    if not entries and not body:
        return []

    parsed: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for cite_key, sentence in _extract_claim_sites(body):
        counts[cite_key] = counts.get(cite_key, 0) + 1
        claim_id = make_claim_id(pid, cite_key, counts[cite_key])
        cited_as = entries.get(cite_key) or {"raw": cite_key}
        parsed.append(
            {
                "paper_id": pid,
                "claim_id": claim_id,
                "cite_key": cite_key,
                "cited_as": cited_as,
                "claim": {
                    "claim_id": claim_id,
                    "text": sentence,
                    "section": None,
                    "char_span": None,
                },
            }
        )
    return parsed
