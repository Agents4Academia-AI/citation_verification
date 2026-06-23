"""
grounding/fulltext.py — Stage-2 evidence for two-stage relevance.

Two-stage relevance escalates here ONLY when the abstract/TLDR can't decide a
claim (see :mod:`citation_verifier.stages.relevance`). Rather than feed the judge
a whole paper, we split the fetched full text into sections and retrieve the few
chunks most relevant to the claim — so the judge reads a handful of high-signal
passages, not the entire document.

This module is the **pure, offline core**: ``split_sections`` and
``select_evidence_chunks`` do no network and no heavy imports, so they are unit
testable on a fixture. Fetching the text (arXiv LaTeX source preferred, PDF
fallback) is a separate, network, fail-soft concern layered on top.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile

__all__ = ["split_sections", "select_evidence_chunks", "fetch_full_text"]

# Sections a relevance judge consults by default; experimental ones are added
# only when the claim itself is about methods/data/results (see _needs_experimental).
_DEFAULT_SECTIONS = ("abstract", "introduction", "conclusion", "discussion", "summary")
_EXPERIMENTAL_SECTIONS = (
    "method", "approach", "model", "architecture", "experiment", "evaluation",
    "result", "dataset", "data", "setup", "implementation", "analysis", "ablation",
)
# Claim words that signal the answer lives in a methods/experiments section.
_EXPERIMENTAL_HINTS = {
    "dataset", "datasets", "benchmark", "accuracy", "f1", "score", "metric", "metrics",
    "outperform", "outperforms", "sota", "state-of-the-art", "experiment", "experiments",
    "ablation", "trained", "training", "fine-tuned", "parameters", "method", "approach",
    "result", "results", "baseline", "baselines", "evaluation", "evaluated", "precision",
    "recall", "epochs", "hyperparameter", "architecture",
}
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "it", "its", "from", "at", "into", "using", "use", "used", "can", "could", "may",
    "we", "our", "their", "they", "which", "such", "than", "then", "also", "but",
}

# LaTeX sectioning command -> the heading title in its braces.
_LATEX_SECTION_RE = re.compile(r"\\(?:sub){0,2}section\*?\s*\{([^}]{1,120})\}")
# A plain-text heading line: short, title/upper-case, no terminal period. Tolerates
# a leading number ("1 Introduction", "2.1 Method").
_PLAIN_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z][^.\n]{1,60})\s*$"
)


def _tokens(text: str) -> set[str]:
    """Content tokens of a string, lowercased, stopwords/short words dropped."""
    return {
        t for t in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
        if len(t) > 2 and t not in _STOPWORDS
    }


def _heading_key(heading: str) -> str:
    """Normalize a heading to a lowercase keyword for section-class matching."""
    return re.sub(r"[^a-z]+", " ", (heading or "").lower()).strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split paper text into ``[(heading, body), …]`` in document order.

    Prefers LaTeX ``\\section{…}`` boundaries (arXiv source); falls back to
    plain-text heading lines. Any text before the first heading is returned under
    a leading ``""`` heading so nothing is dropped. Fails soft to a single
    ``[("", text)]`` block when no structure is found.
    """
    text = (text or "").strip()
    if not text:
        return []

    latex = list(_LATEX_SECTION_RE.finditer(text))
    if latex:
        return _split_at(text, [(m.start(), m.end(), m.group(1).strip()) for m in latex])

    # Plain-text fallback: heading-like lines become section boundaries.
    bounds: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        m = _PLAIN_HEADING_RE.match(line)
        if m and len(_tokens(line)) <= 6:
            bounds.append((offset, offset + len(line), m.group(1).strip()))
        offset += len(line)
    if bounds:
        return _split_at(text, bounds)
    return [("", text)]


def _split_at(text: str, bounds: list[tuple[int, int, str]]) -> list[tuple[str, str]]:
    """Cut ``text`` into (heading, body) using (start, end_of_heading, title) marks."""
    out: list[tuple[str, str]] = []
    pre = text[: bounds[0][0]].strip()
    if pre:
        out.append(("", pre))
    for i, (_, h_end, title) in enumerate(bounds):
        body_end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        body = text[h_end:body_end].strip()
        if body:
            out.append((title, body))
    return out


def _needs_experimental(claim: str) -> bool:
    """True when the claim is about methods/data/results (answer is past the abstract)."""
    return bool(_tokens(claim) & _EXPERIMENTAL_HINTS)


def _section_in_scope(heading: str, want_experimental: bool) -> bool:
    """Is this section one the judge should read for this claim?"""
    key = _heading_key(heading)
    if not key:
        return True  # pre-heading / abstract-ish block: always in scope
    if any(s in key for s in _DEFAULT_SECTIONS):
        return True
    if want_experimental and any(s in key for s in _EXPERIMENTAL_SECTIONS):
        return True
    return False


def _chunks(body: str, max_chars: int) -> list[str]:
    """Paragraph-ish chunks of a section body, each <= ``max_chars``."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paras:
        paras = [body.strip()]
    out: list[str] = []
    for p in paras:
        p = re.sub(r"\s+", " ", p)
        if len(p) <= max_chars:
            out.append(p)
        else:  # window a long paragraph on sentence boundaries
            cur = ""
            for sent in re.split(r"(?<=[.!?])\s+", p):
                if len(cur) + len(sent) + 1 > max_chars and cur:
                    out.append(cur.strip())
                    cur = sent
                else:
                    cur = f"{cur} {sent}".strip()
            if cur:
                out.append(cur.strip())
    return out


def select_evidence_chunks(
    claim: str,
    sections: list[tuple[str, str]],
    *,
    k: int = 3,
    max_chars: int = 600,
) -> list[tuple[str, str]]:
    """Return up to ``k`` ``(section_heading, chunk)`` most relevant to ``claim``.

    Scope is the default sections (abstract/intro/conclusion/discussion) plus the
    experimental sections **only when the claim is about methods/data/results**.
    Chunks are scored by lexical overlap with the claim (token-set ratio); ties and
    empty-overlap fall back to document order so the judge still gets the leading
    (most summary-like) passages. Pure + deterministic; no network.
    """
    claim_toks = _tokens(claim)
    want_exp = _needs_experimental(claim)
    scored: list[tuple[float, int, str, str]] = []
    order = 0
    for heading, body in sections:
        if not _section_in_scope(heading, want_exp):
            continue
        for chunk in _chunks(body, max_chars):
            ct = _tokens(chunk)
            overlap = len(claim_toks & ct) / len(claim_toks) if claim_toks and ct else 0.0
            scored.append((overlap, order, heading, chunk))
            order += 1
    # Highest overlap first; document order breaks ties (earlier = more summary-like).
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(h, c) for _, _, h, c in scored[:k]]


# ───────────────────────────────────────────────────────────────
# Fetching full text (network, fail-soft) — arXiv LaTeX source, then PDF
# ───────────────────────────────────────────────────────────────
_ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{stem}"
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{stem}"
_TEX_MIN_CHARS = 500  # below this the LaTeX source is too thin; try the PDF


def _arxiv_stem(arxiv_id: str | None) -> str:
    """Normalize an arXiv id to its version-less stem ('1706.03762v5' -> '1706.03762')."""
    m = re.search(r"(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})", (arxiv_id or "").lower())
    return m.group(1).split("v")[0] if m else ""


def _extract_tex_from_eprint(data: bytes) -> str:
    """Pure: turn arXiv e-print bytes into concatenated LaTeX source.

    The e-print is usually a gzipped tar of the paper's sources, sometimes a
    single gzipped ``.tex``. Concatenate every ``.tex`` member (main file first,
    detected by ``\\documentclass``) so :func:`split_sections` sees all sections.
    Returns ``""`` on anything it can't parse.
    """
    if not data:
        return ""
    try:  # tarfile handles .tar and .tar.gz transparently with 'r:*'
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            texts: list[tuple[bool, str]] = []
            for member in tar.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".tex"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                body = f.read().decode("utf-8", "replace")
                texts.append(("\\documentclass" in body, body))
            if texts:
                texts.sort(key=lambda t: (not t[0]))  # main file (has documentclass) first
                return "\n\n".join(b for _, b in texts)
    except (tarfile.TarError, OSError, EOFError):
        pass
    try:  # not a tar — maybe a single gzipped .tex
        return gzip.decompress(data).decode("utf-8", "replace")
    except (OSError, EOFError):
        return ""


def fetch_full_text(arxiv_id: str | None, *, timeout: int = 30, max_chars: int = 200_000) -> str:
    """Best-effort full text of an arXiv paper: LaTeX source first, PDF fallback.

    Network + fail-soft: returns ``""`` for a non-arXiv id or on ANY fetch/parse
    failure (the relevance stage then keeps its abstract-only verdict). Capped at
    ``max_chars`` so a huge paper can't blow up downstream chunking.
    """
    stem = _arxiv_stem(arxiv_id)
    if not stem:
        return ""
    tex = _extract_tex_from_eprint(_http_get_bytes(_ARXIV_EPRINT_URL.format(stem=stem), timeout))
    if len(tex) >= _TEX_MIN_CHARS:
        return tex[:max_chars]
    pdf_text = _fetch_arxiv_pdf_text(stem, timeout)
    return (pdf_text or tex)[:max_chars]


def _fetch_arxiv_pdf_text(stem: str, timeout: int) -> str:
    """Download the arXiv PDF and extract its text (needs a PDF extractor). ``""`` on failure."""
    data = _http_get_bytes(_ARXIV_PDF_URL.format(stem=stem), timeout)
    if not data:
        return ""
    import tempfile
    from pathlib import Path

    try:
        from ..extract.pdf import extract_pdf_text
    except Exception:  # noqa: BLE001 — PDF extractor is optional
        return ""
    tmp = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(data)
            tmp = fh.name
        return extract_pdf_text(Path(tmp))
    except Exception:  # noqa: BLE001
        return ""
    finally:
        if tmp:
            Path(tmp).unlink(missing_ok=True)


def _http_get_bytes(url: str, timeout: int) -> bytes:
    """GET raw bytes with a UA header; ``b""`` on any error (network is optional)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "citation-verifier/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — arXiv https
            return resp.read()
    except Exception:  # noqa: BLE001 — fail soft (offline / 404 / timeout)
        return b""
