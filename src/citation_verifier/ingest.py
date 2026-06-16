"""
ingest.py — normalize an input into an on-disk :class:`PaperSource`.

Accepts any of:
  - a local ``.pdf`` path,
  - a bare arXiv id (``1706.03762``, ``2310.06825v2``),
  - an arXiv abs/pdf URL (``https://arxiv.org/abs/2310.06825``).

It computes a stable ``paper_id``, makes the per-paper artifact dir
``papers/<paper_id>/``, and downloads the PDF there when the input is an arXiv
reference. Whether a LaTeX e-print is *available* is probed cheaply (a HEAD-ish
GET) but the actual e-print download/extraction is the extract layer's job — we
only set ``tex_available`` so the orchestrator can pick the LaTeX extractor.

Pure stdlib (``urllib``) and fail-soft: a download/probe failure never raises;
it degrades (no pdf_path / tex_available=False) so the run can continue.
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from pathlib import Path

from .config import Settings, load_settings
from .interfaces import PaperSource

__all__ = ["ingest", "parse_arxiv_id", "ARXIV_ID_RE"]

# Matches 1706.03762, 2310.06825v2, etc. (post-2007 arXiv id scheme).
ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")

_USER_AGENT = "citation-verifier/0.1 (+https://github.com/Agents4Academia/citation_verification)"
_TIMEOUT = 60


def parse_arxiv_id(value: str) -> str | None:
    """Extract an arXiv id from a bare id or an abs/pdf URL, else ``None``.

    The version suffix (``v2``) is preserved when present.
    """
    m = ARXIV_ID_RE.search(value)
    if not m:
        return None
    return m.group(1) + (m.group(2) or "")


def _safe_paper_id(value: str) -> str:
    """Derive a filesystem-safe ``paper_id`` for a local file input.

    Uses the file stem when it is already safe; otherwise a short content hash
    of the path so two different files never collide.
    """
    stem = Path(value).stem
    if stem and re.fullmatch(r"[A-Za-z0-9._-]+", stem):
        return stem
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"pdf-{digest}"


def _download(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` (fail-soft). Return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read()
        if not data:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _probe_tex_available(arxiv_id: str) -> bool:
    """Cheaply probe whether an arXiv e-print (LaTeX source) seems available.

    arXiv serves source at ``/e-print/<id>``. We do a tiny ranged GET and treat
    any successful, non-empty response as "available". Fail-soft: returns False
    on any error so the orchestrator falls back to the PDF extractor.
    """
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            status = getattr(resp, "status", 200)
            # 200 (full) or 206 (partial) both indicate the resource exists.
            return status in (200, 206)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def ingest(source: str, *, settings: Settings | None = None) -> PaperSource:
    """Normalize ``source`` into an on-disk :class:`PaperSource`.

    Args:
        source: A local PDF path, a bare arXiv id, or an arXiv abs/pdf URL.
        settings: Resolved settings (for ``papers_dir``); loaded if omitted.

    Returns:
        A :class:`PaperSource` with ``paper_id``, ``work_dir`` and — when
        resolvable — ``pdf_path``, ``arxiv_id`` and ``tex_available`` set.

    Raises:
        ValueError: only when ``source`` is neither an existing PDF nor a
            parseable arXiv reference (a genuine input error, not a soft failure).
    """
    settings = settings or load_settings()
    papers_dir = Path(settings.papers_dir)

    candidate = Path(source)
    is_local_pdf = source.lower().endswith(".pdf") and candidate.exists()

    if is_local_pdf:
        paper_id = _safe_paper_id(source)
        work_dir = papers_dir / paper_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return PaperSource(
            paper_id=paper_id,
            kind="pdf",
            pdf_path=str(candidate.resolve()),
            tex_available=False,
            work_dir=str(work_dir),
        )

    arxiv_id = parse_arxiv_id(source)
    if arxiv_id is None:
        raise ValueError(
            f"Could not parse an arXiv id and no local PDF exists at: {source!r}"
        )

    paper_id = arxiv_id
    work_dir = papers_dir / paper_id
    work_dir.mkdir(parents=True, exist_ok=True)

    pdf_dest = work_dir / f"{arxiv_id}.pdf"
    pdf_path: str | None = str(pdf_dest) if pdf_dest.exists() else None
    if pdf_path is None:
        if _download(f"https://arxiv.org/pdf/{arxiv_id}", pdf_dest):
            pdf_path = str(pdf_dest)

    tex_available = _probe_tex_available(arxiv_id)

    return PaperSource(
        paper_id=paper_id,
        kind="arxiv_latex" if tex_available else "arxiv_pdf",
        pdf_path=pdf_path,
        tex_available=tex_available,
        arxiv_id=arxiv_id,
        work_dir=str(work_dir),
    )
