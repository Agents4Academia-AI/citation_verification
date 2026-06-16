"""
harvest.py — fetch the raw papers behind the seed descriptors.

Given seed descriptors from :mod:`chbench.sources`, download each paper's
artifacts (arXiv LaTeX e-print preferred, PDF fallback) into the dataset work
dir and emit *paper descriptors* — the harvested-state records the parse stage
consumes:

    {
      "paper_id":      str,          # stable id (arxiv id, or hash of url/title)
      "venue":         str | None,
      "title":         str | None,
      "arxiv_id":      str | None,
      "pdf_path":      str | None,   # local path if a PDF was fetched
      "tex_path":      str | None,   # local path to e-print tar/dir if fetched
      "source_kind":   "arxiv_latex" | "arxiv_pdf" | "pdf" | "unresolved",
      "seed":          {...},        # the originating seed descriptor (provenance)
      "harvest_error": str | None,   # set on fail-soft degrade
    }

Resumable: a paper whose target file already exists on disk is not re-downloaded;
the descriptor is reconstructed from disk. Fail-soft: a paper that cannot be
fetched is emitted with ``source_kind="unresolved"`` and a ``harvest_error`` so
the run continues and the failure is auditable.

Network is stdlib ``urllib`` only and lazy. Offline (no ``pdf_url``/``arxiv_id``,
or download fails) yields ``unresolved`` descriptors rather than crashing.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_USER_AGENT = "chbench/0.1 (CitationHallucinationBench)"
_TIMEOUT = 60


def _paper_id_for(seed: dict[str, Any]) -> str:
    """Derive a stable ``paper_id`` for a seed descriptor.

    Prefers the arXiv id; otherwise a short content hash of the most stable
    identifying field available (url > title > the whole seed).
    """
    if seed.get("arxiv_id"):
        return str(seed["arxiv_id"])
    basis = seed.get("paper_url") or seed.get("paper_title") or json.dumps(seed, sort_keys=True)
    return "h" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]  # noqa: S324 (id only)


def _download(url: str, dest: Path, *, timeout: int = _TIMEOUT) -> bool:
    """Download ``url`` to ``dest`` (stdlib). Returns True on success, else False.

    Fail-soft: any network/IO error returns False; partial files are removed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            data = resp.read()
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        if dest.exists():
            try:
                dest.unlink()
            except OSError:
                pass
        return False


def _harvest_one(seed: dict[str, Any], out_dir: Path, *, fetch: bool) -> dict[str, Any]:
    """Harvest a single seed into ``out_dir`` and return its paper descriptor."""
    paper_id = _paper_id_for(seed)
    paper_dir = out_dir / paper_id
    descriptor: dict[str, Any] = {
        "paper_id": paper_id,
        "venue": seed.get("venue"),
        "title": seed.get("paper_title"),
        "arxiv_id": seed.get("arxiv_id"),
        "pdf_path": None,
        "tex_path": None,
        "source_kind": "unresolved",
        "seed": seed,
        "harvest_error": None,
    }

    pdf_path = paper_dir / f"{paper_id}.pdf"
    tex_path = paper_dir / f"{paper_id}.tar.gz"

    # Resume: reuse anything already on disk.
    if pdf_path.exists():
        descriptor.update(pdf_path=str(pdf_path), source_kind="arxiv_pdf")
        return descriptor
    if tex_path.exists():
        descriptor.update(tex_path=str(tex_path), source_kind="arxiv_latex")
        return descriptor

    if not fetch:
        descriptor["harvest_error"] = "offline: fetch disabled"
        return descriptor

    # Preferred path: arXiv LaTeX e-print (gives .bbl/.bib + \cite call-sites).
    arxiv_id = seed.get("arxiv_id")
    if arxiv_id:
        eprint_url = f"https://arxiv.org/e-print/{arxiv_id}"
        if _download(eprint_url, tex_path):
            descriptor.update(tex_path=str(tex_path), source_kind="arxiv_latex")
            return descriptor

    # Fallback: PDF (from the seed's pdf_url, or arXiv pdf).
    pdf_url = seed.get("pdf_url") or (
        f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None
    )
    if pdf_url and _download(pdf_url, pdf_path):
        kind = "arxiv_pdf" if (arxiv_id and "arxiv.org" in pdf_url) else "pdf"
        descriptor.update(pdf_path=str(pdf_path), source_kind=kind)
        return descriptor

    descriptor["harvest_error"] = "no fetchable arxiv_id/pdf_url, or download failed"
    return descriptor


def harvest(
    seeds: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    fetch: bool = False,
) -> list[dict[str, Any]]:
    """Harvest paper artifacts for ``seeds`` into ``out_dir``; return descriptors.

    Args:
        seeds: seed descriptors from :mod:`chbench.sources`.
        out_dir: dataset work dir; one subdir per ``paper_id`` is created.
        fetch: when False (default) nothing is downloaded — already-present files
            are reused and missing ones yield ``unresolved`` descriptors (offline,
            resumable). When True, missing artifacts are downloaded (fail-soft).

    Returns:
        One paper descriptor per seed (see module docstring). Also writes a
        ``harvest.json`` manifest under ``out_dir`` for the next stage / resume.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    descriptors = [_harvest_one(seed, out, fetch=fetch) for seed in seeds]

    manifest = out / "harvest.json"
    manifest.write_text(
        json.dumps(descriptors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return descriptors
