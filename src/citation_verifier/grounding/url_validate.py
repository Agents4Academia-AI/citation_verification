"""
grounding/url_validate.py — verify a citation that points at a web / software /
system-card URL (not a scholarly paper).

When the structured cascade (Crossref / S2 / arXiv / OpenAlex) finds nothing, a
citation may still be a real object reachable by URL: a GitHub repo, a Notion
page, a model system card. This normalizes the common citation artifacts off the
URL ("https: //x ,2025." -> "https://x"), then validates it directly and reports
``live`` / ``blocked`` / ``dead`` so the correctness stage can mark it
``exists=yes`` as a **web/software object** (``match_method="direct_url"``, NOT a
scholarly match) — or abstain with an *actionable* reason when access is blocked.

Import is network-free; all HTTP is stdlib-only (no ``requests`` dependency),
lazy, and fail-soft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A browser-like UA — some hosts (GitHub, vendor pages) 403 a bare urllib agent.
_UA = (
    "Mozilla/5.0 (compatible; cverify/0.1; "
    "+https://github.com/Agents4Academia-AI/citation_verification)"
)
_GITHUB_RE = re.compile(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


@dataclass(frozen=True)
class UrlCheck:
    """Outcome of validating a citation URL."""

    url: str  # the normalized URL
    status: str  # "live" | "blocked" | "dead" | "error"
    http: int | None = None
    method: str = ""  # "github_api" | "fetch"


def normalize_url(raw: str) -> str:
    """Strip citation artifacts off a URL.

    Fixes the despaced scheme ("https: //" / "https:// x"), removes a trailing
    citation year (",2025"), and trailing punctuation/brackets — so a URL that is
    actually reachable isn't failed by "...langchain,2025." or "https: //github...".
    """
    u = (raw or "").strip()
    u = re.sub(r"^\s*(https?)\s*:\s*/\s*/\s*", r"\1://", u, flags=re.IGNORECASE)
    u = re.sub(r",\s*(?:19|20)\d{2}[a-z]?\b", "", u)  # trailing/inner citation year
    u = re.sub(r"\s+", "", u)  # any remaining internal whitespace
    return re.sub(r"[\s,;.)\]\}>]+$", "", u)  # trailing punctuation / brackets


def _http_status(url: str, headers: dict[str, str], timeout: float = 8.0) -> int | None:
    """Return the HTTP status for ``url`` (or ``None`` on a network error). Body unread."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — http(s) only
            return getattr(resp, "status", None) or resp.getcode()
    except urllib.error.HTTPError as exc:
        return exc.code
    except Exception:  # noqa: BLE001 — DNS / TLS / timeout / refused: treat as unknown
        return None


def _classify(code: int | None) -> str:
    if code is None:
        return "error"
    if 200 <= code < 400:
        return "live"
    if code in (401, 403, 429):  # access gated / bot-blocked — NOT proof of absence
        return "blocked"
    return "dead"  # 404/410/5xx


def validate_citation_url(raw: str, *, github_token: str | None = None) -> UrlCheck | None:
    """Validate a citation URL. ``None`` when ``raw`` isn't an http(s) URL.

    GitHub URLs go through the API (a clean 200/404, and far fewer bot-blocks than
    scraping the HTML page); a token, if configured, raises the rate limit.
    Everything else is a browser-UA GET. Never raises.
    """
    url = normalize_url(raw)
    if not re.match(r"https?://", url, re.IGNORECASE):
        return None
    m = _GITHUB_RE.match(url)
    if m:
        owner, repo = m.group(1), m.group(2).removesuffix(".git")
        api = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {"User-Agent": _UA, "Accept": "application/vnd.github+json"}
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"
        code = _http_status(api, headers)
        return UrlCheck(url=url, status=_classify(code), http=code, method="github_api")
    code = _http_status(url, {"User-Agent": _UA})
    return UrlCheck(url=url, status=_classify(code), http=code, method="fetch")
