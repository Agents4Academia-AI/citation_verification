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

A record's link is often a web page, not a PDF: a publisher's landing page, or an
institution's list of its reports. Such a page is read for the link to the work's PDF —
the page's ``citation_pdf_url``, a link whose text is the work's title, or a PDF link
beside the title — and the PDF is accepted only when it carries the title itself. A PDF
whose pages are images (a scan) has no text layer; runs that use a model install a reader
for it (:func:`reading_scans`), and the transcription is labelled as such.
"""

from __future__ import annotations

import gzip
import html as _html
import io
import re
import tarfile
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from . import fulltext_cache

__all__ = [
    "FullTextResult",
    "full_text_of",
    "full_text_with_trail",
    "reading_scans",
    "introduction_of",
    "split_sections",
    "select_evidence_chunks",
    "fetch_full_text",
    "fetch_full_text_with_source",
    "fetch_full_text_from_url",
    "fetch_full_text_via_search",
    "fetch_full_text_via_search_with_source",
]


@dataclass(frozen=True)
class FullTextResult:
    """Retrieved full text plus its provenance — which channel and URL produced it.

    ``source`` is a coarse channel label for debugging judge misfires
    (``arxiv_html`` | ``arxiv_eprint`` | ``arxiv_pdf`` | ``resolved_url`` |
    ``linked_pdf`` | ``openalex_oa`` | ``unpaywall`` | ``meta_pdf`` | ``web_search``,
    with :data:`SCAN_LABEL` appended when a model read the text off a scan); ``url`` is
    the document actually fetched. Truthy iff non-empty text — so callers keep using
    ``if not result:`` / ``if result:`` exactly as with the bare-string fetchers.

    ``tried`` says, when there is no text, what was tried and why each attempt failed —
    a few words each, for the reader of the report.
    """

    text: str
    source: str = ""
    url: str = ""
    tried: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.text)

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
    """Is this section one the judge should read for this claim?

    Default (summary-ish) sections are always in scope; recognized experimental
    sections only when the claim is about methods/data/results. An *unknown*
    heading is kept in scope: flat PDF text often misparses author/affiliation
    lines into bogus headings, so excluding the unknown would drop real body
    prose (the relevant passage frequently lives there).
    """
    key = _heading_key(heading)
    if not key:
        return True  # pre-heading / abstract-ish block: always in scope
    if any(s in key for s in _DEFAULT_SECTIONS):
        return True
    if any(s in key for s in _EXPERIMENTAL_SECTIONS):
        return want_experimental  # gate recognized methods/results sections
    return True  # unknown heading (e.g. misparsed PDF line) -> keep in scope


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


def _is_heading_like(chunk: str) -> bool:
    """True when a chunk names a topic rather than saying anything about it.

    Scoring is overlap-with-the-CLAIM, normalised by the claim's length, so a chunk is
    never penalised for being short — and a heading is the densest possible match, being
    nothing but signal words. "E.2 High-order Features" was returned as evidence for
    whether a method composes higher-order features; the judge could only reply that it
    had been shown a section heading. The section's actual paragraphs are what carry the
    answer, and they are separate chunks that rank behind it.

    A passage states something and therefore ends a sentence somewhere; a heading does
    not.
    """
    text = (chunk or "").strip()
    if not text:
        return True
    if len(text.split()) > 12 or re.search(r"[.!?][\"'\u201d\u2019)\]]*(?:\s|$)", text):
        return False
    # Short and unpunctuated is not enough — a table row or a clipped sentence is short too
    # and does assert something. A heading also LOOKS like one: a section number, a
    # trailing colon, or title case.
    return bool(
        re.match(r"^\(?(?:[A-Z]|\d+)(?:\.\d+)*\)?[.\s]", text)      # "E.2 …", "4.1 …", "3 …"
        or text.endswith(":")
        or _is_title_case(text)
    )


def _is_title_case(text: str) -> bool:
    """True when most content words are capitalised, as in a heading."""
    words = [w for w in re.findall(r"[A-Za-z][\w'-]*", text) if len(w) > 3]
    if len(words) < 2:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) >= 0.6


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
            if _is_heading_like(chunk):
                continue
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


def fetch_full_text_with_source(
    arxiv_id: str | None, *, timeout: int = 30, max_chars: int = 200_000
) -> FullTextResult:
    """Best-effort arXiv full text **with provenance** (HTML → LaTeX e-print → PDF).

    Network + fail-soft: returns an empty :class:`FullTextResult` for a non-arXiv id
    or on ANY fetch/parse failure (the relevance stage then keeps its abstract-only
    verdict). Capped at ``max_chars`` so a huge paper can't blow up chunking.
    """
    stem = _arxiv_stem(arxiv_id)
    if not stem:
        return FullTextResult("")
    # Served from disk when we have fetched this paper before: the content does not
    # change, and re-fetching it is also a source of run-to-run variance — which channel
    # answers first can differ, and a different channel yields different text, from which
    # different passages are then selected. Set CITATION_VERIFIER_CACHE="" to disable.
    hit = fulltext_cache.read(f"arxiv:{stem}")
    if hit is not None:
        return FullTextResult(hit[0], hit[1], hit[2])
    got = _fetch_arxiv_full_text(stem, timeout=timeout, max_chars=max_chars)
    if got.text:
        fulltext_cache.write(f"arxiv:{stem}", got.text, got.source, got.url)
    return got


def _fetch_arxiv_full_text(stem: str, *, timeout: int, max_chars: int) -> FullTextResult:
    """:func:`fetch_full_text_with_source` without the disk cache — the fetch cascade."""
    # arXiv's HTML rendering first — clean text, no LaTeX/PDF parsing (adapted from
    # Klara Kaleb's prior/fulltext.py); fall back to the LaTeX e-print, then PDF.
    from .oa_fulltext import arxiv_html_text_with_url

    html, html_url = arxiv_html_text_with_url(stem, timeout=timeout, max_chars=max_chars)
    if html:
        return FullTextResult(html, "arxiv_html", html_url)
    tex = _extract_tex_from_eprint(_http_get_bytes(_ARXIV_EPRINT_URL.format(stem=stem), timeout))
    if len(tex) >= _TEX_MIN_CHARS:
        return FullTextResult(tex[:max_chars], "arxiv_eprint", _ARXIV_EPRINT_URL.format(stem=stem))
    pdf_text = _fetch_arxiv_pdf_text(stem, timeout)
    if pdf_text:
        return FullTextResult(pdf_text[:max_chars], "arxiv_pdf", f"https://arxiv.org/pdf/{stem}")
    return FullTextResult(tex[:max_chars], "arxiv_eprint", _ARXIV_EPRINT_URL.format(stem=stem)) if tex else FullTextResult("")


def full_text_of(resolved: Any) -> str:
    """The cited paper's body, through every channel the project has; ``""`` when none
    answers. See :func:`full_text_with_trail`."""
    return full_text_with_trail(resolved).text


def full_text_with_trail(resolved: Any) -> FullTextResult:
    """The cited paper's body with its provenance — or, when there is none, why.

    arXiv first (HTML, LaTeX e-print, PDF), then the resolved URL (a PDF, or a page that
    links the work's PDF), then the DOI's open-access copy (Unpaywall / OpenAlex /
    publisher meta tags), then a title search (when a search backend is wired — none is
    by default). Stopping after the first two silently reduces a whole class of papers to
    their abstract: an IEEE or Springer reference resolves to a ``doi.org`` landing page,
    not a PDF, while an open-access copy exists. Every channel is cached on disk and
    fails soft. An empty result's ``tried`` names each channel asked and what it gave.
    """
    from .oa_fulltext import fulltext_by_doi_with_source  # noqa: PLC0415 — sibling, lazy

    tried: list[str] = []
    title = (getattr(resolved, "title", "") or "").strip()
    arxiv_id = getattr(resolved, "arxiv_id", None)
    if arxiv_id:
        got = fetch_full_text_with_source(arxiv_id)
        if getattr(got, "text", ""):
            return got
        tried.append(f"arXiv {arxiv_id}: no text could be fetched")
    url = (getattr(resolved, "url", "") or "").strip()
    if url:
        report: dict = {}
        got = fetch_full_text_from_url(url, title=title or None, report=report)
        text = got if isinstance(got, str) else (getattr(got, "text", "") or "")
        if text:
            return FullTextResult(text, report.get("source") or "resolved_url", report.get("url") or url)
        tried.append(report.get("note") or f"{_host(url)}: nothing could be read")
    doi = (getattr(resolved, "doi", "") or "").strip()
    if doi:
        got = fulltext_by_doi_with_source(doi, title=title or None)
        if getattr(got, "text", ""):
            return got
        tried += list(getattr(got, "tried", ()) or ()) or [f"DOI {doi}: no open-access copy found"]
    if title:
        got = fetch_full_text_via_search_with_source(title, year=getattr(resolved, "year", None))
        if getattr(got, "text", ""):
            return got
    if not tried:
        tried.append("no arXiv id, link or DOI to fetch it by")
    return FullTextResult("", tried=tuple(tried))


def introduction_of(text: str, max_chars: int = 2500) -> str:
    """The Introduction of a paper's full text, capped; ``""`` when it has none.

    Read from the section structure, not by searching for the word: a table of contents
    ("1 Introduction 1.1 Conformal inference 2 Methods …") at the top of an HTML
    rendering made "the text after 'Introduction' up to the next numbered heading" a
    single line of the contents — 23 characters where the section has thousands.
    """
    def capped(body: str) -> str:
        body = re.sub(r"\s+", " ", body).strip()
        return body[:max_chars] + ("…" if len(body) > max_chars else "")

    for heading, body in split_sections(text or ""):
        if "introduction" in _heading_key(heading) and _is_prose(body):
            return capped(body)
    # The section split can miss it — a rendering's contents list names "1 Introduction"
    # first and the split keeps that one; a journal prints "1 | INTRODUCTION". Every
    # heading line that says Introduction is tried, and the first followed by prose wins.
    for m in _INTRO_HEADING_RE.finditer(text or ""):
        rest = text[m.end() : m.end() + max_chars * 3]
        stop = _NEXT_HEADING_RE.search(rest, 1)
        body = rest[: stop.start()] if stop else rest
        if _is_prose(body):
            return capped(body)
    return ""


_INTRO_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+|[IVX]+)[ \t]*[.|:]?[ \t]*)?introduction[ \t]*$"
)
_NEXT_HEADING_RE = re.compile(r"(?m)^[ \t]*(?:\d+|[IVX]+)[ \t]*[.|]?[ \t]+[A-Z][^\n]{0,80}$")


def _is_prose(body: str) -> bool:
    """Running text, not a contents list: most of it sits in long lines."""
    lines = [ln.strip() for ln in (body or "").splitlines() if ln.strip()]
    total = sum(len(ln) for ln in lines)
    return total > 400 and sum(len(ln) for ln in lines if len(ln) > 60) >= 0.5 * total


def fetch_full_text(arxiv_id: str | None, *, timeout: int = 30, max_chars: int = 200_000) -> str:
    """Bare-text arXiv full text (HTML → LaTeX → PDF). See
    :func:`fetch_full_text_with_source` for provenance; this is the back-compat
    string wrapper. ``""`` for a non-arXiv id or on any failure."""
    return fetch_full_text_with_source(arxiv_id, timeout=timeout, max_chars=max_chars).text


def fetch_full_text_from_url(
    url: str | None,
    *,
    timeout: int = 30,
    max_chars: int = 200_000,
    title: str | None = None,
    report: dict | None = None,
    _depth: int = 0,
) -> str:
    """Best-effort full text from a URL: a PDF, or a web page that links the work's PDF.

    The bytes are content-sniffed for the ``%PDF`` magic. A PDF is extracted (a scan is
    handed to the installed reader, see :func:`reading_scans`). A web page — a landing
    page, an institution's list of its reports — is read for the work's PDF when
    ``title`` is given: its ``citation_pdf_url``, a link whose text is the title, a PDF link
    on the work's own page or beside the title in a list (:func:`_pdf_links_for`). A
    linked PDF is accepted only when it carries the title itself: a list page links many
    works. Network + fail-soft: ``""`` for an empty url or ANY error.

    ``report``, when given, is filled with ``url`` (the document read) and ``source``
    (``resolved_url`` | ``linked_pdf``, plus :data:`PREVIEW_LABEL` for a publisher's
    preview and :data:`SCAN_LABEL` for a scan), or, when nothing was read, ``note`` — why,
    in a few words — and ``reason``, the innermost cause on its own.
    """
    report = {} if report is None else report
    url = (url or "").strip()
    if not url:
        return ""
    keys = [f"url:{url}"] + ([_linked_key(url, title)] if title else [])
    for key in keys:
        hit = fulltext_cache.read(key)
        if hit is not None:
            report.update(url=hit[2] or url, source=hit[1] or "resolved_url")
            return hit[0][:max_chars]
    data = _http_get_bytes(url, timeout)
    host = _host(url)

    def fail(reason: str, note: str = "") -> str:
        report.update(reason=reason, note=note or reason)
        return ""

    if not data:
        return fail(f"{host}: no response")
    if data[:5] == b"%PDF-":
        text, how, why = _pdf_text(data, title or "")
        text = text[:max_chars]
        if not text:
            return fail(f"{host}: {why or 'no text could be extracted from the PDF'}")
        source = _labelled("resolved_url", preview=_is_preview(url), scan=how == "scan")
        fulltext_cache.write(f"url:{url}", text, source, url)
        report.update(url=url, source=source)
        return text
    if _depth and (not title or _depth >= _MAX_LINK_DEPTH):
        return fail(f"{host}: the link returns a web page, not the PDF")
    if not title:
        return fail(f"{host}: a web page, not a PDF")
    page = data[:_MAX_PAGE_BYTES].decode("utf-8", errors="replace")
    here = _host(_page_base(page, url))  # a doi.org link lands on the publisher's page
    links = _pdf_link_candidates(page, url, title)
    if not links:
        if _depth:
            return fail(f"{host}: the link returns a web page, not the PDF")
        return fail(f"{here}: a web page that links no PDF of the work")
    reason = ""
    for link, preview in links:
        sub: dict = {}
        text = fetch_full_text_from_url(
            link, timeout=timeout, max_chars=max_chars, title=title, report=sub, _depth=_depth + 1
        )
        if text and _text_matches_paper(text, title):
            got = sub.get("source") or ""
            source = _labelled(
                "linked_pdf", preview=preview or PREVIEW_LABEL in got, scan=SCAN_LABEL in got
            )
            fulltext_cache.write(_linked_key(url, title), text, source, sub.get("url") or link)
            report.update(url=sub.get("url") or link, source=source)
            return text
        reason = reason or (
            f"{_host(link)}: the linked PDF is another work" if text else sub.get("reason", "")
        )
    reason = reason or "nothing could be read from it"
    # One clause per page would repeat itself down a chain of pages ("doi.org links the
    # work's PDF, but cambridge.org links the work's PDF, but …"): the page the record
    # points to and the innermost cause say it all.
    shown = reason[len(here) + 2 :] if reason.startswith(f"{here}: ") else reason
    return fail(reason, f"{here} links the work's PDF, but {shown}")


def _labelled(channel: str, *, preview: bool = False, scan: bool = False) -> str:
    """A full-text source label with what qualifies the text appended."""
    return channel + (PREVIEW_LABEL if preview else "") + (SCAN_LABEL if scan else "")


def source_notes(source: str | None) -> str:
    """The labels in a full-text source that qualify its text — a publisher preview, a
    transcription of a scan — or ``""``."""
    return "".join(label for label in (PREVIEW_LABEL, SCAN_LABEL) if label in (source or ""))


# A publisher's preview is the work's first page or pages: it is the work, so its text is
# evidence, but not all of it — what it does not say, the rest of the work may.
PREVIEW_LABEL = " (publisher preview: first pages only)"
_PREVIEW_RE = re.compile(r"(?i)\bpreview\b|\bfirst[ -]pages?\b|\bsample[ -]pages?\b|\blook[ -]inside\b")


def _is_preview(url: str, label: str = "") -> bool:
    """True when a link says it leads to a preview — in its address or in its text."""
    parts = urllib.parse.urlparse(_html.unescape(url or ""))
    return bool(_PREVIEW_RE.search(f"{parts.netloc} {parts.path}") or _PREVIEW_RE.search(label or ""))


# ── Following a web page to the work's PDF ──────────────────────────
_MAX_LINK_DEPTH = 2  # a list page -> the work's own page -> its PDF
_MAX_LINKS = 3  # candidate links fetched from one page
_MAX_PAGE_BYTES = 3_000_000
_AROUND = 600  # at most this much markup either side of a PDF link is its entry
_ANCHOR_RE = re.compile(r"(?is)<a\b([^>]*)>(.*?)</a\s*>")
_HREF_RE = re.compile(r"""(?is)\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""")
_PDF_WORDS_RE = re.compile(r"(?i)\b(?:pdf|full[ -]?text|download)\b")
_META_PDF_RE = (
    re.compile(r'(?is)<meta[^>]*?name=["\']citation_pdf_url["\'][^>]*?content=["\']([^"\']+)'),
    re.compile(r'(?is)<meta[^>]*?content=["\']([^"\']+)["\'][^>]*?name=["\']citation_pdf_url["\']'),
)
_PAGE_TITLE_RES = (
    re.compile(r"(?is)<title[^>]*>(.*?)</title>"),
    re.compile(r'(?is)<meta[^>]*?(?:name|property)=["\'](?:citation_title|og:title|dc\.title)["\'][^>]*?content=["\']([^"\']+)'),
    re.compile(r"(?is)<h1[^>]*>(.*?)</h1>"),
)
_BASE_RES = (
    re.compile(r'(?is)<base[^>]*?href=["\']([^"\']+)'),
    re.compile(r'(?is)<link[^>]*?rel=["\']canonical["\'][^>]*?href=["\']([^"\']+)'),
    re.compile(r'(?is)<meta[^>]*?property=["\']og:url["\'][^>]*?content=["\']([^"\']+)'),
)


def _pdf_links_for(page: str, page_url: str, title: str) -> list[str]:
    """Links on a web page that may lead to the work's PDF, most likely first (see
    :func:`_pdf_link_candidates`)."""
    return [link for link, _preview in _pdf_link_candidates(page, page_url, title)]


def _pdf_link_candidates(page: str, page_url: str, title: str) -> list[tuple[str, bool]]:
    """``(link, is_preview)`` for links on a web page that may lead to the work's PDF.

    In order: the page's ``citation_pdf_url`` (a landing page declares its work's PDF); a
    link whose text is the title (a list of reports links each title to its file); on the
    work's own page — its title is the page's — any link to a PDF; elsewhere, a PDF link
    with the title beside it. A link to a preview comes after every other: a paywalled
    work's page offers the first pages when the whole is out of reach. The caller checks
    the fetched PDF for the title, so a neighbour's link that slips in costs a fetch, not
    a wrong answer.
    """
    base = _page_base(page, page_url)
    own_page = _text_matches_paper(_page_title(page), title)
    ranked: list[tuple[int, int, str, bool]] = []
    for rx in _META_PDF_RE:
        ranked += [(0, m.start(), m.group(1), _is_preview(m.group(1))) for m in rx.finditer(page)]
    for m in _ANCHOR_RE.finditer(page):
        href = _href(m.group(1))
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        label = _plain(m.group(2))
        preview = _is_preview(href, label)
        rank = None
        if _text_matches_paper(label, title):
            rank = 1
        elif _looks_like_pdf_link(href, label):
            if own_page:
                rank = 2
            elif _text_matches_paper(_plain(_entry_around(page, m.start(), m.end())), title):
                rank = 3
        if rank is not None:
            ranked.append((rank, m.start(), href, preview))
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    here = page_url.split("#", 1)[0]
    for _rank, _pos, href, preview in sorted(ranked, key=lambda r: (r[3], r[0], r[1])):
        link = urllib.parse.urljoin(base, _html.unescape(href).strip()).split("#", 1)[0]
        if link.startswith(("http://", "https://")) and link != here and link not in seen:
            seen.add(link)
            out.append((link, preview))
    return out[:_MAX_LINKS]


# The entry a link sits in — a table row, a list item, a definition — so a PDF link is
# matched with the title in its own entry, not with the next one's: in a list where
# some entries have no file, the title of one sits a few hundred characters from the
# link of another.
_ROW_START_RE = re.compile(r"(?is)<(?:tr|li|dt|article)\b")
_ROW_END_RE = re.compile(r"(?is)</(?:tr|li|dd|article)\s*>")
_PARA_START_RE = re.compile(r"(?is)<p\b")
_PARA_END_RE = re.compile(r"(?is)</p\s*>")


def _entry_around(page: str, start: int, end: int) -> str:
    """The markup of the entry holding ``page[start:end]``, at most :data:`_AROUND`
    characters either side."""
    lo, hi = max(0, start - _AROUND), min(len(page), end + _AROUND)
    a, b = lo, hi
    for rx in (_ROW_START_RE, _PARA_START_RE):
        opens = list(rx.finditer(page, lo, start))
        if opens:
            a = opens[-1].start()
            break
    for rx in (_ROW_END_RE, _PARA_END_RE):
        close = rx.search(page, end, hi)
        if close:
            b = close.end()
            break
    return page[a:b]


def _looks_like_pdf_link(href: str, label: str) -> bool:
    path = urllib.parse.urlparse(_html.unescape(href)).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path or bool(_PDF_WORDS_RE.search(label))


def _href(attrs: str) -> str:
    m = _HREF_RE.search(attrs or "")
    return (m.group(1) or m.group(2) or m.group(3) or "").strip() if m else ""


def _plain(fragment: str) -> str:
    """Markup to plain text: tags dropped, entities decoded, whitespace collapsed."""
    text = re.sub(r"(?s)<[^>]+>", " ", fragment or "")
    return re.sub(r"\s+", " ", _html.unescape(text)).strip()


def _page_title(page: str) -> str:
    """What the page says it is: its <title>, title metadata and first heading."""
    return " ".join(_plain(m.group(1)) for rx in _PAGE_TITLE_RES for m in [rx.search(page)] if m)


def _page_base(page: str, page_url: str) -> str:
    """The address relative links resolve against: the page's own, when it states it (a
    ``doi.org`` link lands on the publisher's page, whose links are relative to it)."""
    for rx in _BASE_RES:
        m = rx.search(page)
        if m and m.group(1).strip().startswith(("http://", "https://")):
            return _html.unescape(m.group(1).strip())
    return page_url


def _linked_key(url: str, title: str | None) -> str:
    """Cache key for the PDF a page links for one work: a list page links many."""
    return f"url:{url}|title:{' '.join(sorted(_tokens(title or '')))}"


def _host(url: str) -> str:
    host = urllib.parse.urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else (host or (url or "")[:60])


# ── Scanned PDFs ────────────────────────────────────────────────────
# A PDF whose pages are images has no text layer. Runs that use a model install a reader
# for it (backends/scan_reader.py); without one, a scan stays unread and says so.
SCAN_LABEL = " (scan, read by a model)"
_SCAN_CHARS_PER_PAGE = 100  # less extractable text than this per page: the pages are images
_SCAN_READER: Any = None


@contextmanager
def reading_scans(reader: Any):
    """Within the block, a PDF with no text layer is read by ``reader(data, title=…)``.

    ``reader`` returns the text it read off the pages (``""`` when it could not). With
    ``reader=None`` the block changes nothing.
    """
    global _SCAN_READER
    if reader is None:
        yield
        return
    before, _SCAN_READER = _SCAN_READER, reader
    try:
        yield
    finally:
        _SCAN_READER = before


def _pdf_text(data: bytes, title: str = "") -> tuple[str, str, str]:
    """``(text, how, why)`` for PDF bytes: ``how`` is ``"scan"`` when a model read the
    pages; ``why`` says, when there is no text, what stood in the way."""
    text = _pdf_bytes_to_text(data)
    if not _looks_scanned(data, text):
        return text, "", ""
    reader = _SCAN_READER
    if reader is None:
        return "", "", "the PDF is a scan with no text layer, and no model in this run reads scans"
    try:
        read = (reader(data, title=title) or "").strip()
    except Exception as exc:  # noqa: BLE001 — fail soft: an unread scan is still an answer
        return "", "", f"the PDF is a scan; reading it failed ({' '.join(str(exc).split())[:120]})"
    if not read:
        return "", "", "the PDF is a scan, and nothing could be read from it"
    return read, "scan", ""


def _looks_scanned(data: bytes, text: str) -> bool:
    """True when a PDF's pages are images: no text at all, or (almost) none while most
    pages carry an image. A PDF that does not open is judged by its text alone."""
    chars = len((text or "").strip())
    if not chars:
        return True
    pages, with_images = _page_stats(data)
    if not pages:
        return False
    return chars < _SCAN_CHARS_PER_PAGE * pages and with_images * 2 >= pages


def _page_stats(data: bytes) -> tuple[int, int]:
    """``(pages, pages_with_an_image)`` of a PDF; ``(0, 0)`` when it does not open."""
    try:
        import fitz  # noqa: PLC0415 — PyMuPDF, optional

        with fitz.open(stream=data, filetype="pdf") as doc:
            return doc.page_count, sum(1 for page in doc if page.get_images())
    except Exception:  # noqa: BLE001
        pass
    try:
        from pypdf import PdfReader  # noqa: PLC0415

        pages = PdfReader(io.BytesIO(data)).pages
        return len(pages), sum(1 for page in pages if len(page.images))
    except Exception:  # noqa: BLE001
        return 0, 0


# Title-token overlap required between the cited title and a fetched PDF before we
# trust it (the title-search namesake gate). Tuned so the GPT-2 cite
# "… Unsupervised Multitask Learners" rejects the 2024 "… Supervised Multitask
# Learners": every content token of the cited title must appear, with a 1-token
# slack only for long (>=7-token) titles to tolerate PDF-extraction noise.
def _text_matches_paper(text: str, title: str) -> bool:
    """True when ``text`` (a fetched PDF) is plausibly the paper named by ``title``.

    A title web-search can surface a same-title namesake, so a fetched PDF is only
    trusted when (nearly) every content token of the cited title is present in its
    head. This is the precision gate that makes a title-based fetch safe.
    """
    want = {t for t in _tokens(title) if len(t) > 3} or _tokens(title)
    if not want:
        return False
    missing = want - _tokens((text or "")[:6000])
    tol = 1 if len(want) >= 7 else 0  # extraction slack for long titles only
    return len(missing) <= tol


def _search_query(title: str, year: int | None) -> str:
    """Build a web-search query biased toward a downloadable PDF of the paper."""
    q = f'"{title}" filetype:pdf'
    return f"{q} {year}" if year else q


def _default_web_search(query: str, *, max_results: int = 5) -> list[str]:
    """No search engine is wired for full text: ``[]``.

    The two this used to call are gone — Google's Custom Search JSON API takes no new
    customers and shuts down on 2027-01-01, and DuckDuckGo's HTML endpoint blocks scripted
    queries — so the tier returned nothing anyway. Finding a REFERENCE on the web is done
    by a model with web search (:mod:`citation_verifier.backends.web_search`); pass
    ``search=`` here to use a search backend for full text.
    """
    return []


def fetch_full_text_via_search_with_source(
    title: str | None,
    *,
    year: int | None = None,
    search=None,
    timeout: int = 30,
    max_chars: int = 200_000,
    max_results: int = 5,
) -> FullTextResult:
    """Last-resort Stage-2 full text **with provenance**: web-search a free PDF,
    fetch + title-verify it, and report the matched URL (``source="web_search"``).

    For off-arXiv papers with no fetchable ``openAccessPdf`` (e.g. CEUR / OpenAI
    tech reports), search the web for a downloadable PDF, fetch each hit, and return
    the first whose text passes :func:`_text_matches_paper` — the title-namesake
    gate. ``search`` is injectable (default: :func:`_default_web_search`, which has no
    engine and returns nothing) for tests and alternate backends. Network + fail-soft:
    empty for a too-short title, no hit, or ANY error.
    """
    title = (title or "").strip()
    if len(title) < 6:
        return FullTextResult("")
    search = search or _default_web_search
    try:
        urls = search(_search_query(title, year), max_results=max_results)
    except Exception:  # noqa: BLE001 — fail soft
        return FullTextResult("")
    for url in (urls or [])[:max_results]:
        text = fetch_full_text_from_url(url, timeout=timeout, max_chars=max_chars)
        if text and _text_matches_paper(text, title):
            return FullTextResult(text, "web_search", url)
    return FullTextResult("")


def fetch_full_text_via_search(
    title: str | None,
    *,
    year: int | None = None,
    search=None,
    timeout: int = 30,
    max_chars: int = 200_000,
    max_results: int = 5,
) -> str:
    """Bare-text web-search full text. See :func:`fetch_full_text_via_search_with_source`
    for provenance; this is the back-compat string wrapper."""
    return fetch_full_text_via_search_with_source(
        title, year=year, search=search, timeout=timeout, max_chars=max_chars, max_results=max_results
    ).text


def _fetch_arxiv_pdf_text(stem: str, timeout: int) -> str:
    """Download the arXiv PDF and extract its text. ``""`` on failure."""
    return _pdf_bytes_to_text(_http_get_bytes(_ARXIV_PDF_URL.format(stem=stem), timeout))


def _pdf_bytes_to_text(data: bytes) -> str:
    """Pure-ish: extract text from in-memory PDF bytes (needs a PDF extractor).

    Writes to a temp file (the extractor takes a path) and always cleans it up.
    Returns ``""`` when the extractor is unavailable or the bytes don't parse.
    """
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
