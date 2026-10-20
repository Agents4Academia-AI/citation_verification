"""
grounding/oa_fulltext.py — open-access full-text sources (keyless-first cascade).

Adapted from /home/yyq/prior (author: **Klara Kaleb**) — specifically her
``src/prior/fulltext.py`` open-access cascade. Adapted, NOT copied verbatim:

  * The persistent on-disk full-text cache from the original is intentionally
    **not** adapted (out of scope here).
  * Credentialed sources read their secret from the environment and fail soft
    when it is unset — **you supply your own**, nothing is hard-coded:
      - ``OPENALEX_API_KEY``   (OpenAlex; also enables the polite pool)
      - ``UNPAYWALL_EMAIL``    (Unpaywall requires a contact email)
      - ``CONTACT_EMAIL``      (polite User-Agent / OpenAlex ``mailto`` fallback)
    Klara's original also wires Elsevier / Springer / Wiley TDM keys, EZproxy
    institutional cookies, and a Playwright browser path; those are left out here
    (they need per-publisher credentials / a browser) — see her module to add them.

This module only RESOLVES open-access locations; PDF byte→text extraction reuses
:func:`citation_verifier.grounding.fulltext.fetch_full_text_from_url` (which
already content-sniffs ``%PDF`` and falls back pypdf↔PyMuPDF). Network + fail-soft
throughout: every function returns ``""`` / ``[]`` on any error or missing key.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import urllib.parse
import urllib.request

__all__ = [
    "arxiv_html_text",
    "arxiv_html_text_with_url",
    "fulltext_by_doi",
    "fulltext_by_doi_with_source",
    "oa_pdf_urls",
]

_TIMEOUT = 30
_MIN_TEXT = 1000  # guard against stub / error / landing pages
# arXiv's HTML renderings — clean text, no PDF/LaTeX parsing (Klara's tier 1).
_ARXIV_HTML_HOSTS = ("https://arxiv.org/html/{id}", "https://ar5iv.org/abs/{id}")
_CITATION_PDF_RE = (
    re.compile(r'name=["\']citation_pdf_url["\'][^>]*?content=["\']([^"\']+)', re.I | re.S),
    re.compile(r'content=["\']([^"\']+)["\'][^>]*?name=["\']citation_pdf_url["\']', re.I | re.S),
)


def _contact_email() -> str:
    return os.environ.get("CONTACT_EMAIL") or os.environ.get("UNPAYWALL_EMAIL") or ""


def _user_agent() -> str:
    email = _contact_email()
    return f"citation-verifier/0.1 ({'mailto:' + email if email else 'https://github.com/Agents4Academia-AI'})"


def _http_get_text(url: str, timeout: int = _TIMEOUT) -> str:
    """GET text (follows redirects); ``""`` on any error. Network is optional."""
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except Exception:  # noqa: BLE001 — fail soft
        return ""


def _http_get_json(url: str, timeout: int = _TIMEOUT) -> dict:
    raw = _http_get_text(url, timeout)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


# LaTeXML emits `<math alttext="y^{\\prime}=g(\\epsilon,y)" …>`; the attribute is the
# original TeX, which reads perfectly well as evidence.
_MATH_ALTTEXT_RE = re.compile(
    r"""(?isx)
    <math\b [^>]*? \balttext=      # the attribute, in either quoting style
    (?: "(?P<dq>[^"]*)" | '(?P<sq>[^']*)' )
    [^>]* > .*? </math\s*>
    """
)


def _html_to_text(html: str) -> str:
    """Strip scripts/markup → plain text, **keeping heading/paragraph boundaries**.

    Adapted from Klara's ``_html_to_text``; the boundary-preservation is ours, so
    that downstream :func:`split_sections` can still recover document structure
    (heading provenance + experimental-section gating) from the HTML rendering —
    a flat space-collapsed string would degrade to one unsectioned block.
    """
    # Recover the maths before dropping the element. arXiv's HTML is LaTeXML output, which
    # carries the original expression in `alttext`; deleting the whole <math> node takes it
    # with it and leaves a mutilated sentence — "Here, , where is a permutation sampled
    # uniformly from all permutations ." A paper that defines its terms in equations then
    # reads as though it defines nothing, and a column whose criterion IS an equation
    # cannot be checked against it at all. Measured: eight of one table's twelve cells
    # came back unverifiable on a paper whose full text we had.
    html = _MATH_ALTTEXT_RE.sub(
        lambda m: " " + _html.unescape(m.group("dq") or m.group("sq") or "") + " ", html
    )
    html = re.sub(r"(?is)<(script|style|math|svg|noscript).*?</\1>", " ", html)
    # Headings → their own line (so _PLAIN_HEADING_RE picks them up); block-level
    # elements → a blank line (so _chunks sees paragraph boundaries).
    html = re.sub(r"(?is)</?h[1-6][^>]*>", "\n", html)
    html = re.sub(r"(?is)<(?:/?(?:p|div|section|article|li|tr|br|figcaption)\b)[^>]*>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)  # remaining inline tags → space
    html = _html.unescape(html)  # entities → real chars (&amp;→&, Greek letters, …) — keep semantics
    html = re.sub(r"[ \t\xa0]+", " ", html)  # collapse intra-line whitespace incl. nbsp (keep \n)
    html = re.sub(r" *\n *", "\n", html)  # trim around line breaks
    html = re.sub(r"\n{3,}", "\n\n", html)  # cap blank-line runs
    return html.strip()


def arxiv_html_text_with_url(
    arxiv_id: str, *, timeout: int = _TIMEOUT, max_chars: int = 200_000
) -> tuple[str, str]:
    """arXiv's HTML rendering as ``(clean_text, source_url)``; ``("", "")`` if none.

    Adapted from Klara's tier 1: the HTML rendering needs no PDF/LaTeX parsing, so
    it is the cleanest full-text source for a modern arXiv paper. Returns the host
    that actually answered (``arxiv.org/html`` *or* ``ar5iv.org``) so callers record
    accurate provenance instead of assuming the first host.
    """
    stem = re.sub(r"v\d+$", "", (arxiv_id or "").strip().split(":")[-1])
    if not stem:
        return "", ""
    for tmpl in _ARXIV_HTML_HOSTS:
        url = tmpl.format(id=stem)
        html = _http_get_text(url, timeout)
        if "<html" in html[:2000].lower():
            text = _html_to_text(html)
            if len(text) > _MIN_TEXT:
                return text[:max_chars], url
    return "", ""


def arxiv_html_text(arxiv_id: str, *, timeout: int = _TIMEOUT, max_chars: int = 200_000) -> str:
    """Bare-text arXiv HTML rendering (arxiv.org/html → ar5iv); ``""`` if none. See
    :func:`arxiv_html_text_with_url` for the source URL."""
    return arxiv_html_text_with_url(arxiv_id, timeout=timeout, max_chars=max_chars)[0]


def _norm_doi(doi: str) -> str:
    return (
        (doi or "")
        .strip()
        .replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("doi:", "")
    )


def _unpaywall_pdf_urls(doi: str, timeout: int) -> list[str]:
    """Legal OA PDF URLs for a DOI via Unpaywall. Requires ``UNPAYWALL_EMAIL``."""
    email = os.environ.get("UNPAYWALL_EMAIL") or _contact_email()
    if not email:
        return []
    data = _http_get_json(
        f"https://api.unpaywall.org/v2/{doi}?{urllib.parse.urlencode({'email': email})}", timeout
    )
    urls: list[str] = []
    for loc in [data.get("best_oa_location"), *(data.get("oa_locations") or [])]:
        if loc:
            urls += [u for u in (loc.get("url_for_pdf"), loc.get("url")) if u]
    return urls


def _openalex_pdf_url(doi: str, timeout: int) -> list[str]:
    """OA PDF URL for a DOI via OpenAlex ``best_oa_location`` (Klara's tier 2 source).

    Uses ``OPENALEX_API_KEY`` when set (and the ``mailto`` polite pool); works
    keyless too, just rate-limited.
    """
    params = {}
    if key := os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = key
    if email := _contact_email():
        params["mailto"] = email
    qs = ("?" + urllib.parse.urlencode(params)) if params else ""
    data = _http_get_json(f"https://api.openalex.org/works/doi:{doi}{qs}", timeout)
    out = []
    for loc in (data.get("best_oa_location"), data.get("primary_location")):
        if loc and loc.get("pdf_url"):
            out.append(loc["pdf_url"])
    return out


def _meta_pdf_url(doi: str, timeout: int) -> list[str]:
    """Resolve doi.org → landing page → ``<meta citation_pdf_url>`` (Klara's generic
    OA resolver — covers preprint servers + OA journals that Unpaywall/OpenAlex miss)."""
    html = _http_get_text(f"https://doi.org/{doi}", timeout)
    if not html:
        return []
    out = []
    for rx in _CITATION_PDF_RE:
        if m := rx.search(html):
            out.append(m.group(1).replace("&amp;", "&"))
    return out


def _oa_pdf_sources(doi: str, timeout: int) -> list[tuple[str, str]]:
    """Candidate ``(pdf_url, channel)`` pairs for a DOI, best-first, de-duplicated.

    Channel is the resolver that produced the URL (``openalex_oa`` | ``unpaywall``
    | ``meta_pdf``) — kept so callers can record full-text provenance. Order is
    Klara's cascade, keyless-first; each resolver fails soft. The resolvers are
    looked up by name at call time, so each is independently monkeypatchable.
    """
    doi = _norm_doi(doi)
    if not doi:
        return []
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for resolver, channel in (
        (_openalex_pdf_url, "openalex_oa"),
        (_unpaywall_pdf_urls, "unpaywall"),
        (_meta_pdf_url, "meta_pdf"),
    ):
        try:
            found = resolver(doi, timeout)
        except Exception:  # noqa: BLE001 — each source fails soft
            found = []
        for u in found:
            if u and u not in seen:
                seen.add(u)
                out.append((u, channel))
    return out


def oa_pdf_urls(doi: str, *, timeout: int = _TIMEOUT) -> list[str]:
    """Candidate open-access PDF URLs for a DOI, best-first, de-duplicated.

    Order (Klara's cascade, keyless-first): OpenAlex best_oa_location → Unpaywall →
    the landing page's citation_pdf_url. See :func:`_oa_pdf_sources` for the
    channel each URL came from.
    """
    return [u for u, _ in _oa_pdf_sources(doi, timeout)]


def fulltext_by_doi_with_source(doi: str, *, timeout: int = _TIMEOUT, max_chars: int = 200_000):
    """Full text for a DOI **with provenance** via the open-access cascade.

    Resolves OA PDF URLs (OpenAlex / Unpaywall / landing page) and extracts the
    first that yields real text, reusing the project's ``%PDF``-checked extractor;
    the returned :class:`~citation_verifier.grounding.fulltext.FullTextResult`
    carries the channel + URL that succeeded. Empty result if nothing fetchable.
    """
    from .fulltext import FullTextResult, fetch_full_text_from_url

    for url, channel in _oa_pdf_sources(doi, timeout):
        if text := fetch_full_text_from_url(url, timeout=timeout, max_chars=max_chars):
            return FullTextResult(text, channel, url)
    return FullTextResult("")


def fulltext_by_doi(doi: str, *, timeout: int = _TIMEOUT, max_chars: int = 200_000) -> str:
    """Bare-text DOI open-access full text. See :func:`fulltext_by_doi_with_source`
    for provenance; this is the back-compat string wrapper."""
    return fulltext_by_doi_with_source(doi, timeout=timeout, max_chars=max_chars).text
