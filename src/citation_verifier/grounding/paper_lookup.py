"""
paper_lookup — the "grounding" floor for the citation verifier.

This is the one piece that stops the agent from *guessing*. An LLM asked
"is this citation real and correct?" will happily hallucinate an answer.
Instead we look the reference up in authoritative metadata sources and hand the
caller the *canonical* record(s) to compare against:

  - Crossref  (api.crossref.org)        -> journals, conferences, books, DOIs
  - arXiv     (export.arxiv.org)        -> preprints
  - DBLP      (dblp.org/search/publ/api)-> CS publications (strong author/venue)
  - Semantic Scholar (api.semanticscholar.org)  -> abstracts, optional, keyed
  - OpenAlex  (api.openalex.org)        -> broad coverage, optional, keyed

Design contract (frozen by the architect):

  - The **stdlib floor** (Crossref + arXiv + DBLP) uses ONLY ``urllib``/``xml``
    from the standard library, so it imports and runs without ``requests`` and
    degrades to ``{found: 0, ...}`` offline.
  - The **optional sources** (Semantic Scholar, OpenAlex) speak HTTP via the
    optional ``requests`` dependency *and* require an API key from the
    environment. Without either they return ``[]`` — they never break the floor.
  - Every network function is **fail-soft**: any exception (DNS, timeout,
    rate-limit, parse error) yields an empty/zero result, never a raised
    exception. This keeps the degrade-not-crash contract of the orchestrator.

This layer is deliberately "dumb": it fetches candidates only. The *matching*
and *comparison* live in :mod:`citation_verifier.grounding.resolver` and the
:mod:`citation_verifier.stages`.

CLI smoke test::

    python -m citation_verifier.grounding.paper_lookup "Attention is all you need"
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

# Crossref asks API users to identify themselves; doing so puts you in the
# faster "polite pool". Set CROSSREF_MAILTO in the environment to opt in.
CONTACT_EMAIL = os.environ.get("CROSSREF_MAILTO", "")
USER_AGENT = (
    f"citation-verifier/0.1 (mailto:{CONTACT_EMAIL})"
    if CONTACT_EMAIL
    else "citation-verifier/0.1"
)
TIMEOUT = 20

# HTTP statuses worth retrying: rate-limiting + transient server/gateway errors.
# These are exactly what a burst of concurrent lookups provokes; a short backoff
# turns a spurious "unresolved" into a real match.
_RETRY_CODES = frozenset({429, 500, 502, 503, 504})


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default`` on junk."""
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# Minimum seconds between requests to a host, enforced across ALL threads. arXiv's
# export API asks for ~1 request / 3 s on a single connection; bursting it (which
# the concurrent correctness pass does) returns 429 and silently drops real
# matches. Spacing arXiv here lets the other sources stay fully parallel. Tune the
# arXiv interval with ARXIV_MIN_INTERVAL_SECONDS (0 disables). Unlisted hosts are
# never throttled.
_HOST_MIN_INTERVAL: dict[str, float] = {
    "export.arxiv.org": _env_float("ARXIV_MIN_INTERVAL_SECONDS", 3.0),
    # Semantic Scholar's keyless pool is ~1 req/s shared; pace it so a batch of
    # title-match lookups doesn't 429. A key (S2_API_KEY) raises the real limit.
    "api.semanticscholar.org": _env_float("S2_MIN_INTERVAL_SECONDS", 1.1),
    # DBLP throttles aggressively (429 + temporary IP blocks) under bursts; pace it
    # so the concurrent correctness pass fetches every CS ref instead of dropping
    # the tail. OpenAlex's polite pool is lenient (~10 req/s) — a gentle pace only.
    "dblp.org": _env_float("DBLP_MIN_INTERVAL_SECONDS", 1.5),
    "api.openalex.org": _env_float("OPENALEX_MIN_INTERVAL_SECONDS", 0.2),
}
_RETRY_AFTER_CAP = 30.0  # ignore absurd Retry-After values

# Shared throttle state: the next time each host may be hit. Slots are reserved
# under the lock (advancing _next_allowed by the interval) and slept on OUTSIDE
# it, so concurrent callers queue distinct slots instead of serializing on sleep.
_throttle_lock = threading.Lock()
_next_allowed: dict[str, float] = {}


def _throttle(host: str) -> None:
    """Block until this thread's reserved slot for ``host`` (no-op if untracked).

    Paces requests to rate-limited hosts (arXiv) across every worker thread so a
    burst of concurrent lookups cannot trip 429s.
    """
    interval = _HOST_MIN_INTERVAL.get(host, 0.0)
    if interval <= 0:
        return
    with _throttle_lock:
        now = time.monotonic()
        slot = max(now, _next_allowed.get(host, 0.0))
        _next_allowed[host] = slot + interval
    wait = slot - now
    if wait > 0:
        time.sleep(wait)


def _parse_retry_after(value: str | None) -> float | None:
    """Numeric ``Retry-After`` (seconds), capped; ``None`` if absent or HTTP-date.

    The HTTP-date form is ignored (callers fall back to exponential backoff) to
    keep this dependency-free. Shared by the urllib and ``requests`` paths.
    """
    if not value:
        return None
    try:
        return min(max(float(value), 0.0), _RETRY_AFTER_CAP)
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float | None:
    """``Retry-After`` off a urllib 429/503 response, honoring the server's hint."""
    try:
        return _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
    except Exception:  # noqa: BLE001 — header access is best-effort
        return None

# Source identifiers used in the returned candidate dicts and in
# ``schema.Resolved.source`` — keep these in sync with the schema docstring.
SOURCE_CROSSREF = "crossref"
SOURCE_ARXIV = "arxiv"
SOURCE_DBLP = "dblp"
SOURCE_S2 = "s2"
SOURCE_OPENALEX = "openalex"


# ───────────────────────────────────────────────────────────────
# Low-level helpers (stdlib only)
# ───────────────────────────────────────────────────────────────
def _get(
    url: str,
    *,
    accept: str | None = None,
    timeout: int = TIMEOUT,
    retries: int = 4,
    headers: dict[str, str] | None = None,
) -> bytes:
    """HTTP GET via stdlib urllib, throttled per host and retried on transient errors.

    Two layers keep a burst of concurrent lookups from turning real matches into
    spurious "unresolved" results:

      - **Throttle:** :func:`_throttle` spaces requests to rate-limited hosts
        (arXiv) across all threads *before* each attempt, so concurrency never
        exceeds the host's limit in the first place.
      - **Retry:** on 429 / 5xx / timeout we back off and retry, honoring a
        numeric ``Retry-After`` when the server sends one, else exponential
        backoff with jitter (capped). A permanent error (e.g. 404) raises at once.

    After exhausting retries it raises, so callers still fail soft.
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    hdrs = {"User-Agent": USER_AGENT}
    if accept:
        hdrs["Accept"] = accept
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    delay = 1.0
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(retries + 1):
        _throttle(host)  # space rate-limited hosts across threads (also paces retries)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in _RETRY_CODES or attempt == retries:
                raise
            wait = _retry_after_seconds(exc)
            wait = wait if wait is not None else delay + random.uniform(0.0, 0.5)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                raise
            wait = delay + random.uniform(0.0, 0.5)
        time.sleep(wait)
        delay = min(delay * 2, 16.0)
    raise last_exc  # pragma: no cover — loop returns or raises above


def _clean(text: str | None, limit: int = 600) -> str:
    """Strip JATS/HTML markup + whitespace from a title/abstract and truncate."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # JATS / HTML tags
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _coerce_year(value: Any) -> int | None:
    """Best-effort parse of a 4-digit year from arbitrary metadata."""
    if value in (None, ""):
        return None
    try:
        return int(str(value)[:4])
    except (TypeError, ValueError):
        return None


# ───────────────────────────────────────────────────────────────
# Crossref (stdlib floor)
# ───────────────────────────────────────────────────────────────
def search_crossref(query: str, rows: int = 4) -> list[dict]:
    """Look a reference up in Crossref. Best for published venues + DOIs.

    Returns a list of candidate dicts; an empty list on any failure.
    """
    params = urllib.parse.urlencode({"query.bibliographic": query, "rows": rows})
    if CONTACT_EMAIL:
        params += "&mailto=" + urllib.parse.quote(CONTACT_EMAIL)
    try:
        data = json.loads(_get(f"https://api.crossref.org/works?{params}"))
    except Exception:  # network / rate-limit / parse — fail soft  # noqa: BLE001
        return []
    return [_parse_crossref_item(it) for it in data.get("message", {}).get("items", [])]


def search_crossref_by_fields(
    title: str, author: str | None = None, rows: int = 4
) -> list[dict]:
    """Search Crossref by the TITLE field (+ optional first-author), not the raw
    bibliographic string. ``query.title`` + ``query.author`` is far more precise
    for a published paper than dumping the whole noisy reference into
    ``query.bibliographic``. Returns ``[]`` for a too-short title or on failure.
    """
    title = (title or "").strip()
    if len(title) < 6:
        return []
    p: dict[str, Any] = {"query.title": title, "rows": rows}
    if author and author.strip():
        p["query.author"] = author.strip()
    params = urllib.parse.urlencode(p)
    if CONTACT_EMAIL:
        params += "&mailto=" + urllib.parse.quote(CONTACT_EMAIL)
    try:
        data = json.loads(_get(f"https://api.crossref.org/works?{params}"))
    except Exception:  # noqa: BLE001 — fail soft
        return []
    return [_parse_crossref_item(it) for it in data.get("message", {}).get("items", [])]


def _parse_crossref_item(it: dict) -> dict:
    """Map one Crossref work object to a candidate dict."""
    authors = [
        " ".join(p for p in (a.get("given"), a.get("family")) if p)
        for a in it.get("author", [])
    ]
    year = None
    parts = (it.get("issued") or {}).get("date-parts") or [[None]]
    if parts and parts[0]:
        year = _coerce_year(parts[0][0])
    return {
        "source": SOURCE_CROSSREF,
        "title": _clean((it.get("title") or [""])[0], limit=400),
        "authors": authors,
        "year": year,
        "venue": (it.get("container-title") or [""])[0],
        "type": it.get("type", ""),
        "doi": (it.get("DOI") or "").lower(),
        "arxiv_id": "",
        "url": it.get("URL", ""),
        "abstract": _clean(it.get("abstract")),
    }


def fetch_crossref_by_doi(doi: str) -> list[dict]:
    """Fetch the exact Crossref work for ``doi`` (``/works/{doi}``).

    A direct DOI lookup guarantees the cited record is a candidate even when a
    noisy bibliographic search would not surface it. Returns ``[]`` on any
    failure or when the DOI is empty.
    """
    doi = (doi or "").strip().lower()
    if not doi:
        return []
    try:
        data = json.loads(_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))
    except Exception:  # noqa: BLE001 — fail soft (bad DOI / network / 404)
        return []
    msg = data.get("message")
    return [_parse_crossref_item(msg)] if isinstance(msg, dict) and msg.get("title") else []


# ───────────────────────────────────────────────────────────────
# arXiv (stdlib floor)
# ───────────────────────────────────────────────────────────────
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    """Look a reference up on arXiv. Best for preprints.

    Returns a list of candidate dicts; an empty list on any failure.
    """
    q = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query?search_query=all:{q}"
        f"&start=0&max_results={max_results}"
    )
    try:
        root = ET.fromstring(_get(url))
    except Exception:  # noqa: BLE001
        return []
    return [_parse_arxiv_entry(e) for e in root.findall(f"{_ATOM}entry")]


def _parse_arxiv_entry(e: ET.Element) -> dict:
    """Map one arXiv Atom <entry> to a candidate dict."""

    def text(tag: str) -> str:
        el = e.find(tag)
        return el.text.strip() if el is not None and el.text else ""

    arxiv_url = text(f"{_ATOM}id")
    return {
        "source": SOURCE_ARXIV,
        "title": _clean(text(f"{_ATOM}title"), limit=400),
        "authors": [a.findtext(f"{_ATOM}name", "").strip() for a in e.findall(f"{_ATOM}author")],
        "year": _coerce_year(text(f"{_ATOM}published")),
        "venue": text(f"{_ARXIV}journal_ref") or "arXiv (preprint)",
        "type": "preprint",
        "doi": (text(f"{_ARXIV}doi") or "").lower(),
        "url": arxiv_url,
        "arxiv_id": arxiv_url.rsplit("/", 1)[-1] if arxiv_url else "",
        "abstract": _clean(text(f"{_ATOM}summary")),
    }


def fetch_arxiv_by_id(arxiv_id: str) -> list[dict]:
    """Fetch the exact arXiv paper for ``arxiv_id`` (the API ``id_list`` query).

    A direct id lookup guarantees the cited record is a candidate even when a
    noisy ``all:`` title search would not surface it. The version suffix is
    stripped (``2207.12598v3`` -> ``2207.12598``). Returns ``[]`` on any failure
    or when the id is empty. A bad id can't produce a false match: the resolver's
    arXiv-exact step still compares ids before accepting.
    """
    stem = (arxiv_id or "").split("v")[0].strip()
    if not stem:
        return []
    url = f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(stem)}&max_results=1"
    try:
        root = ET.fromstring(_get(url))
    except Exception:  # noqa: BLE001 — fail soft (bad id / network)
        return []
    return [_parse_arxiv_entry(e) for e in root.findall(f"{_ATOM}entry")]


def search_arxiv_by_title(
    title: str, max_results: int = 3, author: str | None = None
) -> list[dict]:
    """Search arXiv by the TITLE field (``ti:``), optionally ANDed with ``au:``.

    A title-field phrase query surfaces a known preprint far more reliably than
    an ``all:`` search over a noisy full reference (author names + venue drown
    out the title). Adding the first-author surname (``au:``) disambiguates a
    reused title from its namesakes. Best for venue-cited preprints that carry no
    arXiv id. Returns ``[]`` for an empty/too-short title or on any failure.
    """
    title = (title or "").strip()
    if len(title) < 6:
        return []
    q = "ti:" + urllib.parse.quote(f'"{title}"')
    if author and author.strip():
        q += "+AND+au:" + urllib.parse.quote(author.strip())
    url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results={max_results}"
    try:
        root = ET.fromstring(_get(url))
    except Exception:  # noqa: BLE001 — fail soft
        return []
    return [_parse_arxiv_entry(e) for e in root.findall(f"{_ATOM}entry")]


# ───────────────────────────────────────────────────────────────
# DBLP (stdlib floor — CS bibliography)
# ───────────────────────────────────────────────────────────────
def search_dblp(query: str, max_results: int = 4) -> list[dict]:
    """Look a reference up on DBLP. Strong author/venue signal for CS papers.

    Uses the public publication search API
    (https://dblp.org/search/publ/api). Returns candidate dicts; ``[]`` on
    any failure.
    """
    params = urllib.parse.urlencode({"q": query, "format": "json", "h": max_results})
    try:
        data = json.loads(_get(f"https://dblp.org/search/publ/api?{params}"))
    except Exception:  # noqa: BLE001
        return []

    hits = (((data or {}).get("result") or {}).get("hits") or {}).get("hit") or []
    out: list[dict] = []
    for h in hits:
        info = h.get("info") or {}
        # DBLP gives authors as {"author": {...}} or {"author": [{...}, ...]}.
        raw_authors = (info.get("authors") or {}).get("author")
        authors: list[str] = []
        if isinstance(raw_authors, list):
            authors = [a.get("text", "") if isinstance(a, dict) else str(a) for a in raw_authors]
        elif isinstance(raw_authors, dict):
            authors = [raw_authors.get("text", "")]
        elif isinstance(raw_authors, str):
            authors = [raw_authors]
        authors = [re.sub(r"\s+\d{4}$", "", a).strip() for a in authors if a]

        out.append(
            {
                "source": SOURCE_DBLP,
                "title": _clean(info.get("title"), limit=400),
                "authors": authors,
                "year": _coerce_year(info.get("year")),
                "venue": info.get("venue", "") or "",
                "type": info.get("type", ""),
                "doi": (info.get("doi") or "").lower(),
                "arxiv_id": "",
                "url": info.get("ee") or info.get("url") or "",
                "abstract": "",  # DBLP carries no abstracts
            }
        )
    return out


# ───────────────────────────────────────────────────────────────
# Semantic Scholar (OPTIONAL — needs `requests` + S2_API_KEY)
# ───────────────────────────────────────────────────────────────
def _requests_or_none():  # -> module | None
    """Lazy import of the optional ``requests`` dependency. Never raises."""
    try:
        import requests  # type: ignore

        return requests
    except Exception:  # noqa: BLE001
        return None


def _requests_get_json(
    requests: Any,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = TIMEOUT,
    retries: int = 4,
) -> Any:
    """``requests.get(url).json()`` with the SAME per-host throttle + ``Retry-After``
    backoff as :func:`_get`.

    The keyed S2/OpenAlex sources speak HTTP via ``requests`` rather than urllib, so
    without this they bypass the pacing/retry the stdlib floor gets — and a burst of
    concurrent lookups 429-drops real matches. Routing them here paces each host
    (``_HOST_MIN_INTERVAL``) across all worker threads and retries transient 429/5xx
    honoring the server's ``Retry-After``. Raises on final failure (callers fail soft).
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    delay = 1.0
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(retries + 1):
        _throttle(host)  # pace rate-limited hosts across threads (also paces retries)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — network/connection error
            last_exc = exc
            if attempt == retries:
                raise
            time.sleep(delay + random.uniform(0.0, 0.5))
            delay = min(delay * 2, 16.0)
            continue
        if resp.status_code in _RETRY_CODES and attempt < retries:
            wait = _parse_retry_after(resp.headers.get("Retry-After"))
            time.sleep(wait if wait is not None else delay + random.uniform(0.0, 0.5))
            delay = min(delay * 2, 16.0)
            continue
        resp.raise_for_status()
        return resp.json()
    raise last_exc  # pragma: no cover — loop returns or raises above


def search_semantic_scholar(
    query: str, max_results: int = 4, api_key: str | None = None
) -> list[dict]:
    """Look a reference up on Semantic Scholar (abstracts + citation intent).

    OPTIONAL: returns ``[]`` unless BOTH the ``requests`` dependency is
    installed AND an API key is available (argument or ``S2_API_KEY`` env).
    Fail-soft on any error.
    """
    key = api_key or os.environ.get("S2_API_KEY", "")
    requests = _requests_or_none()
    if not key or requests is None:
        return []

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,year,venue,abstract,externalIds,url,tldr"
    try:
        data = _requests_get_json(
            requests,
            url,
            params={"query": query, "limit": max_results, "fields": fields},
            headers={"x-api-key": key, "User-Agent": USER_AGENT},
        )
    except Exception:  # noqa: BLE001
        return []

    return [_parse_s2_item(it) for it in (data.get("data") or [])]


_S2_FIELDS = "title,authors,year,venue,abstract,externalIds,url,tldr"


def _parse_s2_item(it: dict) -> dict:
    """Map one Semantic Scholar paper object to a candidate dict."""
    ext = it.get("externalIds") or {}
    tldr = it.get("tldr") or {}
    tldr_text = tldr.get("text") if isinstance(tldr, dict) else tldr
    return {
        "source": SOURCE_S2,
        "title": _clean(it.get("title"), limit=400),
        "authors": [a.get("name", "") for a in (it.get("authors") or [])],
        "year": _coerce_year(it.get("year")),
        "venue": it.get("venue", "") or "",
        "type": "",
        "doi": (ext.get("DOI") or "").lower(),
        "arxiv_id": ext.get("ArXiv", "") or "",
        "url": it.get("url", "") or "",
        "abstract": _clean(it.get("abstract")),
        # S2's one-line AI summary. The API exposes it even when the licensed full
        # abstract is withheld (e.g. the GPT-2/GPT-1 tech reports), so it is the
        # only API-accessible evidence for those — carried through Candidate.extra
        # and used ONLY as a last-resort abstract fallback (see _abstract_top_up).
        "tldr": _clean(tldr_text),
    }


def search_semantic_scholar_match(title: str, api_key: str | None = None) -> list[dict]:
    """Resolve a TITLE to its single best Semantic Scholar match.

    Uses the ``/paper/search/match`` endpoint, which returns the one paper whose
    title best matches — far more precise than a relevance search (it surfaces the
    canonical paper where a plain search returns same-title namesakes) and it
    carries the arXiv id + abstract. **Keyless-capable**: routed through the
    throttled/retried :func:`_get` (S2's keyless pool is paced + 429-retried); an
    ``S2_API_KEY`` (argument or env) just raises the rate limit. Returns ``[]`` for
    a too-short title, a "no match" (404), or any error (fail-soft).
    """
    title = (title or "").strip()
    if len(title) < 6:
        return []
    key = api_key or os.environ.get("S2_API_KEY", "")
    params = urllib.parse.urlencode({"query": title, "fields": _S2_FIELDS})
    url = f"https://api.semanticscholar.org/graph/v1/paper/search/match?{params}"
    try:
        data = json.loads(_get(url, headers={"x-api-key": key} if key else None))
    except Exception:  # noqa: BLE001 — fail soft (404 "no match" / 429 / parse)
        return []
    rows = data.get("data") if isinstance(data, dict) else None
    return [_parse_s2_item(it) for it in (rows or [])]


def fetch_semantic_scholar_by_doi(doi: str, api_key: str | None = None) -> list[dict]:
    """Fetch Semantic Scholar metadata by exact DOI, including abstract if known."""
    return _fetch_semantic_scholar_by_id("DOI", doi, api_key)


def fetch_semantic_scholar_by_arxiv(arxiv_id: str, api_key: str | None = None) -> list[dict]:
    """Fetch Semantic Scholar metadata by exact arXiv id, including abstract if known."""
    stem = (arxiv_id or "").split("v")[0].strip()
    return _fetch_semantic_scholar_by_id("ARXIV", stem, api_key)


def _fetch_semantic_scholar_by_id(
    prefix: str, value: str, api_key: str | None = None
) -> list[dict]:
    """Exact-id S2 lookup for abstract-bearing identifier matches."""
    value = (value or "").strip()
    if not value:
        return []
    key = api_key or os.environ.get("S2_API_KEY", "")
    paper_id = f"{prefix}:{value}"
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{urllib.parse.quote(paper_id, safe=':/')}?"
        f"{urllib.parse.urlencode({'fields': _S2_FIELDS})}"
    )
    try:
        data = json.loads(_get(url, headers={"x-api-key": key} if key else None))
    except Exception:  # noqa: BLE001 — fail soft (404 / 429 / parse)
        return []
    return [_parse_s2_item(data)] if isinstance(data, dict) and data.get("title") else []


# ───────────────────────────────────────────────────────────────
# OpenAlex (OPTIONAL — needs `requests`; key optional but used as mailto/key)
# ───────────────────────────────────────────────────────────────
def search_openalex(
    query: str, max_results: int = 4, api_key: str | None = None
) -> list[dict]:
    """Look a reference up on OpenAlex (broad scholarly coverage).

    OPTIONAL: returns ``[]`` unless the ``requests`` dependency is installed
    AND a credential is available (argument, ``OPENALEX_API_KEY``, or a
    ``CROSSREF_MAILTO`` to join OpenAlex's polite pool). Fail-soft on error.
    """
    key = api_key or os.environ.get("OPENALEX_API_KEY", "") or CONTACT_EMAIL
    requests = _requests_or_none()
    if not key or requests is None:
        return []

    params: dict[str, Any] = {"search": query, "per-page": max_results}
    # OpenAlex uses mailto for the polite pool; an explicit api_key uses api_key.
    if "@" in str(key):
        params["mailto"] = key
    else:
        params["api_key"] = key
    try:
        data = _requests_get_json(
            requests,
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": USER_AGENT},
        )
    except Exception:  # noqa: BLE001
        return []

    return [_parse_openalex_item(it) for it in data.get("results", []) or []]


def fetch_openalex_by_doi(doi: str, api_key: str | None = None) -> list[dict]:
    """Fetch the exact OpenAlex work for a DOI; useful as an abstract top-up."""
    doi = (doi or "").strip().lower()
    if not doi:
        return []
    params: dict[str, Any] = {}
    key = api_key or os.environ.get("OPENALEX_API_KEY", "") or CONTACT_EMAIL
    if key:
        if "@" in str(key):
            params["mailto"] = key
        else:
            params["api_key"] = key
    suffix = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    url = f"https://api.openalex.org/works/{suffix}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        data = json.loads(_get(url))
    except Exception:  # noqa: BLE001 — fail soft (404 / rate-limit / parse)
        return []
    return [_parse_openalex_item(data)] if isinstance(data, dict) and data.get("display_name") else []


def _parse_openalex_item(it: dict) -> dict:
    """Map one OpenAlex work object to a candidate dict."""
    authorships = it.get("authorships") or []
    authors = [((a.get("author") or {}).get("display_name") or "") for a in authorships]
    ids = it.get("ids") or {}
    doi = (it.get("doi") or "").replace("https://doi.org/", "").lower()
    arxiv_id = ""
    for v in ids.values():
        if isinstance(v, str) and "arxiv" in v.lower():
            arxiv_id = v.rstrip("/").rsplit("/", 1)[-1]
            break
    return {
        "source": SOURCE_OPENALEX,
        "title": _clean(it.get("display_name"), limit=400),
        "authors": authors,
        "year": _coerce_year(it.get("publication_year")),
        "venue": (((it.get("primary_location") or {}).get("source") or {}).get("display_name") or ""),
        "type": it.get("type", "") or "",
        "doi": doi,
        "arxiv_id": arxiv_id,
        "url": it.get("doi") or (it.get("primary_location") or {}).get("landing_page_url") or "",
        "abstract": _reconstruct_openalex_abstract(it.get("abstract_inverted_index")),
    }


def _reconstruct_openalex_abstract(inverted: dict | None) -> str:
    """Rebuild plain abstract text from OpenAlex's inverted index, if present."""
    if not inverted or not isinstance(inverted, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda p: p[0])
    return _clean(" ".join(w for _, w in positions))


# ───────────────────────────────────────────────────────────────
# URL validation helper (fail-soft HEAD/GET)
# ───────────────────────────────────────────────────────────────
def validate_url(url: str, timeout: int = 15) -> bool:
    """Return True iff ``url`` resolves to a live resource (2xx/3xx).

    Tries a cheap HEAD first, falls back to a ranged GET (some scholarly hosts
    reject HEAD). Stdlib only; fail-soft -> ``False`` on any error or non-2xx/3xx.
    """
    if not url or not str(url).lower().startswith(("http://", "https://")):
        return False

    def _try(method: str) -> bool:
        req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
        if method == "GET":
            req.add_header("Range", "bytes=0-0")  # ask for 1 byte, not the body
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return 200 <= getattr(resp, "status", 200) < 400
        except urllib.error.HTTPError as e:
            # A 405 (method not allowed) on HEAD is inconclusive, not a failure.
            return False if e.code >= 400 else True
        except Exception:  # noqa: BLE001
            return False

    return _try("HEAD") or _try("GET")


# ───────────────────────────────────────────────────────────────
# Combined entry point (baseline-compatible signature)
# ───────────────────────────────────────────────────────────────
def _candidate_to_dict(c: Any) -> dict:
    """Serialize a resolver Candidate without importing its class at module load."""
    return {
        "source": getattr(c, "source", ""),
        "title": getattr(c, "title", ""),
        "authors": list(getattr(c, "authors", []) or []),
        "year": getattr(c, "year", None),
        "venue": getattr(c, "venue", ""),
        "type": (getattr(c, "extra", {}) or {}).get("type", ""),
        "doi": getattr(c, "doi", ""),
        "arxiv_id": getattr(c, "arxiv_id", ""),
        "url": getattr(c, "url", ""),
        "abstract": getattr(c, "abstract", ""),
    }


def lookup_paper(query: str, source: str = "auto", max_results: int = 4) -> dict:
    """Return canonical metadata candidates for a cited reference.

    Baseline-compatible: returns ``{query, found, candidates, note}``.

    Args:
        query: the reference string or title as written in the bibliography.
        source: ``"crossref"`` | ``"arxiv"`` | ``"dblp"`` | ``"s2"`` |
            ``"openalex"`` | ``"auto"``. ``"auto"`` queries the stdlib floor
            (Crossref + arXiv + DBLP) and, if credentials are present, the
            optional sources too. Optional sources return ``[]`` without a key
            or ``requests``, so ``"auto"`` is always safe offline.
        max_results: per-source candidate cap.

    The caller compares these candidates against what the draft *claims* and
    decides existence / metadata correctness; this function never adjudicates.
    """
    candidates: list[dict] = []
    if source == "auto":
        try:
            # Keep the CLI/tool path aligned with the production resolver: direct
            # DOI/arXiv lookup + precise title queries first, broad search last.
            from .resolver import MultiSourceResolver

            resolver = MultiSourceResolver(validate_urls=False)
            candidates = [
                _candidate_to_dict(c) for c in resolver.candidates(query, max_results)
            ]
            return _lookup_response(query, candidates)
        except Exception:  # noqa: BLE001 — fall through to the legacy broad fan-out
            candidates = []

    if source in (SOURCE_CROSSREF, "auto"):
        candidates += search_crossref(query, max_results)
    if source in (SOURCE_ARXIV, "auto"):
        candidates += search_arxiv(query, max_results)
    if source in (SOURCE_DBLP, "auto"):
        candidates += search_dblp(query, max_results)
    if source in (SOURCE_S2, "auto"):
        candidates += search_semantic_scholar(query, max_results)
    if source in (SOURCE_OPENALEX, "auto"):
        candidates += search_openalex(query, max_results)

    return _lookup_response(query, candidates)


def _lookup_response(query: str, candidates: list[dict]) -> dict:
    return {
        "query": query,
        "found": len(candidates),
        "candidates": candidates,
        "note": (
            "No candidates found — the paper may not exist, may be mis-titled, "
            "or may live in a source not covered here (a book, thesis, or "
            "non-indexed workshop). Treat as a red flag, not proof of fabrication."
            if not candidates
            else "Compare each candidate's title/authors/year/venue against what "
            "the draft claims. A strong title match with mismatched authors/year/"
            "venue is a metadata error; no strong title match anywhere is a red flag."
        ),
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Attention is all you need"
    print(f"Looking up: {q!r}\n")
    print(json.dumps(lookup_paper(q), indent=2, ensure_ascii=False))
