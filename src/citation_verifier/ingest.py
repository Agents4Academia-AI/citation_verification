"""
ingest.py — normalize an input into an on-disk :class:`PaperSource`.

Accepts any of:
  - a local ``.pdf`` path,
  - a local LaTeX/source archive (``.zip`` / ``.tar.gz``) — unzipped into
    ``work_dir/tex`` so the more-accurate LaTeX extractor reads its ``.bbl``/``.bib``,
  - a bare arXiv id (``1706.03762``, ``2310.06825v2``) or an arXiv abs/pdf URL,
  - any other http(s) URL — downloaded, then routed by content (a PDF to the PDF
    extractor; a LaTeX source archive to the LaTeX extractor).

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


_ARCHIVE_EXTS = (".zip", ".tar.gz", ".tgz", ".tar")


def _is_http_url(value: str) -> bool:
    return bool(re.match(r"https?://", value.strip(), re.IGNORECASE))


def _extract_archive(archive: Path, dest: Path) -> bool:
    """Extract a ``.zip`` or ``.tar(.gz)`` into ``dest``, skipping path-traversal
    members (zip-slip guard). Fail-soft: returns ``True`` on success, else ``False``."""
    import tarfile
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    droot = dest.resolve()
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for name in zf.namelist():
                    if (droot / name).resolve().is_relative_to(droot):
                        zf.extract(name, dest)
            return True
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                for m in tf.getmembers():
                    if (droot / m.name).resolve().is_relative_to(droot):
                        tf.extract(m, dest)  # noqa: S202 — guarded by is_relative_to
            return True
    except Exception:  # noqa: BLE001 — corrupt archive / OS error: degrade
        return False
    return False


def _ingest_arxiv(arxiv_id: str, papers_dir: Path) -> PaperSource:
    """Download the arXiv PDF + probe LaTeX availability (the original arXiv path)."""
    work_dir = papers_dir / arxiv_id
    work_dir.mkdir(parents=True, exist_ok=True)
    pdf_dest = work_dir / f"{arxiv_id}.pdf"
    pdf_path: str | None = str(pdf_dest) if pdf_dest.exists() else None
    if pdf_path is None and _download(f"https://arxiv.org/pdf/{arxiv_id}", pdf_dest):
        pdf_path = str(pdf_dest)
    tex_available = _probe_tex_available(arxiv_id)
    return PaperSource(
        paper_id=arxiv_id,
        kind="arxiv_latex" if tex_available else "arxiv_pdf",
        pdf_path=pdf_path,
        tex_available=tex_available,
        arxiv_id=arxiv_id,
        work_dir=str(work_dir),
    )


def _ingest_archive(archive: Path, papers_dir: Path) -> PaperSource:
    """A local LaTeX/source archive (.zip/.tar.gz): unzip into ``work_dir/tex`` so
    the (more accurate) LaTeX extractor reads its ``.bbl``/``.bib`` + ``\\cite`` sites."""
    paper_id = _safe_paper_id(str(archive))
    work_dir = papers_dir / paper_id
    tex_dir = work_dir / "tex"
    has_tex = _extract_archive(archive, tex_dir) and any(tex_dir.rglob("*.tex"))
    return PaperSource(
        paper_id=paper_id,
        kind="latex" if has_tex else "pdf",
        tex_dir=str(tex_dir) if has_tex else None,
        tex_available=has_tex,
        work_dir=str(work_dir),
    )


def _ingest_url(url: str, papers_dir: Path) -> PaperSource:
    """A generic http(s) URL: download, then route by content — a PDF goes to the PDF
    extractor; a LaTeX source archive (zip/tar) goes to the LaTeX extractor."""
    paper_id = "url-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    work_dir = papers_dir / paper_id
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / "download"
    if not _download(url, dest):
        raise ValueError(f"could not download: {url!r}")
    with open(dest, "rb") as fh:
        head = fh.read(5)
    if head[:4] == b"%PDF":
        pdf = work_dir / f"{paper_id}.pdf"
        dest.replace(pdf)
        return PaperSource(
            paper_id=paper_id, kind="pdf", pdf_path=str(pdf),
            tex_available=False, work_dir=str(work_dir),
        )
    tex_dir = work_dir / "tex"
    if _extract_archive(dest, tex_dir) and any(tex_dir.rglob("*.tex")):
        return PaperSource(
            paper_id=paper_id, kind="latex", tex_dir=str(tex_dir),
            tex_available=True, work_dir=str(work_dir),
        )
    raise ValueError(f"URL did not return a PDF or a LaTeX source archive: {url!r}")


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
    src = source.strip()
    low = src.lower()
    candidate = Path(src)

    # 1) Local PDF.
    if low.endswith(".pdf") and candidate.exists():
        paper_id = _safe_paper_id(src)
        work_dir = papers_dir / paper_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return PaperSource(
            paper_id=paper_id, kind="pdf", pdf_path=str(candidate.resolve()),
            tex_available=False, work_dir=str(work_dir),
        )

    # 2) Local LaTeX/source archive (.zip/.tar.gz) -> the more accurate LaTeX path.
    if candidate.exists() and low.endswith(_ARCHIVE_EXTS):
        return _ingest_archive(candidate, papers_dir)

    # 3) arXiv: a bare id, or any arxiv.org URL (checked before the generic-URL
    #    branch so an arXiv link still gets the LaTeX-preferred path).
    if not _is_http_url(src) or "arxiv.org" in low:
        arxiv_id = parse_arxiv_id(src)
        if arxiv_id is not None:
            return _ingest_arxiv(arxiv_id, papers_dir)

    # 4) Any other http(s) URL -> download and route by content (PDF / LaTeX archive).
    if _is_http_url(src):
        return _ingest_url(src, papers_dir)

    raise ValueError(
        f"Unrecognized input: {source!r} — expected a local PDF or .zip/.tar.gz, "
        "an arXiv id/URL, or an http(s) PDF URL"
    )
