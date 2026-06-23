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

__all__ = [
    "split_sections",
    "select_evidence_chunks",
    "fetch_full_text",
    "fetch_full_text_from_url",
    "fetch_full_text_via_search",
]

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


def fetch_full_text_from_url(
    url: str | None, *, timeout: int = 30, max_chars: int = 200_000
) -> str:
    """Best-effort full text from a direct PDF URL (e.g. S2's ``openAccessPdf``).

    Stage-2 source for **off-arXiv** papers: when the resolved record has no arXiv
    id but carries an open-access PDF link, fetch it and extract the text. The
    bytes are content-sniffed for the ``%PDF`` magic so a landing page / HTML
    response (or a paywall interstitial) fails soft to ``""`` instead of feeding
    the judge garbage. Network + fail-soft: ``""`` for an empty url or ANY error.

    No title verification is needed here because the URL is the curated OA link for
    an already-matched paper — it carries no namesake risk (contrast a title search).
    """
    if not (url or "").strip():
        return ""
    data = _http_get_bytes(url, timeout)
    if data[:5] != b"%PDF-":  # not a PDF (HTML landing page, paywall, error) -> skip
        return ""
    return _pdf_bytes_to_text(data)[:max_chars]


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


def _google_cse_search(query: str, *, max_results: int = 5, timeout: int = 20) -> list[str]:
    """Google Programmable Search (Custom Search JSON API) -> result URLs.

    OPTIONAL + fail-soft + key-gated, mirroring the S2/OpenAlex convention: returns
    ``[]`` unless BOTH ``GOOGLE_API_KEY`` and ``GOOGLE_CSE_ID`` are set, and on any
    error. This is the only ToS-clean way to query google.com from code; absent a
    key the web-search tier is simply a no-op.
    """
    import json
    import os
    import urllib.parse
    import urllib.request

    key = os.environ.get("GOOGLE_API_KEY", "")
    cx = os.environ.get("GOOGLE_CSE_ID", "")
    if not (key and cx):
        return []
    params = urllib.parse.urlencode(
        {"key": key, "cx": cx, "q": query, "num": min(max(max_results, 1), 10)}
    )
    req = urllib.request.Request(
        f"https://www.googleapis.com/customsearch/v1?{params}",
        headers={"User-Agent": "citation-verifier/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — googleapis https
            data = json.loads(resp.read())
    except Exception:  # noqa: BLE001 — fail soft (offline / quota / bad key)
        return []
    return [it["link"] for it in (data.get("items") or []) if it.get("link")]


# A browser-ish UA: DuckDuckGo's HTML endpoint serves empty/blocked pages to the
# default library UA. Used ONLY for the search request, not the PDF fetch.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _parse_ddg_results(html: str, max_results: int) -> list[str]:
    """Pure: extract de-duplicated result URLs from a DuckDuckGo HTML page.

    Robust to both DDG link shapes: the redirect wrapper
    ``…/l/?uddg=<percent-encoded-url>`` and a direct ``href`` on the
    ``result__a`` anchor. Order-preserving, http(s)-only, DDG-internal links
    dropped, capped at ``max_results``.
    """
    import urllib.parse

    def _add(url: str) -> None:
        if url.startswith("//"):
            url = "https:" + url
        if "uddg=" in url:  # redirect wrapper -> decode the real target
            m = re.search(r"uddg=([^\"'&]+)", url)
            url = urllib.parse.unquote(m.group(1)) if m else ""
        if url.startswith("http") and "duckduckgo.com" not in url and url not in seen:
            seen.add(url)
            out.append(url)

    out: list[str] = []
    seen: set[str] = set()
    # 1) hrefs on result anchors (current format may be a direct external link).
    for tag in re.findall(r"<a\b[^>]*class=\"[^\"]*result__a[^\"]*\"[^>]*>", html or ""):
        m = re.search(r'href="([^"]+)"', tag)
        if m:
            _add(m.group(1))
        if len(out) >= max_results:
            return out
    # 2) fallback: any uddg= wrapper anywhere (older/lite format).
    for enc in re.findall(r"uddg=([^\"'&]+)", html or ""):
        _add("uddg=" + enc)
        if len(out) >= max_results:
            break
    return out


def _ddg_search(query: str, *, max_results: int = 5, timeout: int = 20) -> list[str]:
    """DuckDuckGo HTML search -> result URLs. Keyless, fail-soft.

    The no-key fallback backend: it scrapes ``html.duckduckgo.com`` (no official
    API), so it is best-effort and ToS-gray — fine for local runs, but a deployer
    should set ``GOOGLE_API_KEY``/``GOOGLE_CSE_ID`` for the reliable path. ``[]`` on
    any error / block.
    """
    import urllib.parse
    import urllib.request

    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=urllib.parse.urlencode({"q": query}).encode(),
        headers={
            "User-Agent": _BROWSER_UA,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — ddg https
            html = resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — fail soft (offline / blocked / 429)
        return []
    return _parse_ddg_results(html, max_results)


def _default_web_search(query: str, *, max_results: int = 5) -> list[str]:
    """Default backend: Google Custom Search when keyed, else DuckDuckGo (keyless)."""
    return _google_cse_search(query, max_results=max_results) or _ddg_search(
        query, max_results=max_results
    )


def fetch_full_text_via_search(
    title: str | None,
    *,
    year: int | None = None,
    search=None,
    timeout: int = 30,
    max_chars: int = 200_000,
    max_results: int = 5,
) -> str:
    """Last-resort Stage-2 full text: web-search for a free PDF, fetch + verify it.

    For off-arXiv papers with no fetchable ``openAccessPdf`` (e.g. CEUR / OpenAI
    tech reports), search the web for a downloadable PDF, fetch each hit, and return
    the first whose text passes :func:`_text_matches_paper` — the title-namesake
    gate. ``search`` is injectable (default: :func:`_default_web_search`, i.e.
    Google Custom Search when keyed, else DuckDuckGo keyless) for tests and alternate
    backends. Network + fail-soft: ``""`` for a too-short title, no hit, or ANY error.
    """
    title = (title or "").strip()
    if len(title) < 6:
        return ""
    search = search or _default_web_search
    try:
        urls = search(_search_query(title, year), max_results=max_results)
    except Exception:  # noqa: BLE001 — fail soft
        return ""
    for url in (urls or [])[:max_results]:
        text = fetch_full_text_from_url(url, timeout=timeout, max_chars=max_chars)
        if text and _text_matches_paper(text, title):
            return text
    return ""


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
