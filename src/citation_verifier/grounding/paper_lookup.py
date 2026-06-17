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
import re
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
# turns a spurious "unverified" into a real match.
_RETRY_CODES = frozenset({429, 500, 502, 503, 504})

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
    url: str, *, accept: str | None = None, timeout: int = TIMEOUT, retries: int = 2
) -> bytes:
    """HTTP GET via stdlib urllib, with backoff on transient errors.

    Retries a few times on rate-limit / 5xx / timeout — the failure modes a burst
    of concurrent lookups provokes — then raises so callers still fail soft. A
    permanent error (e.g. 404) is raised immediately without retrying.
    """
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    delay = 0.5
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in _RETRY_CODES or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                raise
        time.sleep(delay)
        delay *= 2
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


def search_arxiv_by_title(title: str, max_results: int = 3) -> list[dict]:
    """Search arXiv by the TITLE field (``ti:``).

    A title-field phrase query surfaces a known preprint far more reliably than
    an ``all:`` search over a noisy full reference (author names + venue drown
    out the title). Best for venue-cited preprints that carry no arXiv id in the
    reference. Returns ``[]`` for an empty/too-short title or on any failure.
    """
    title = (title or "").strip()
    if len(title) < 6:
        return []
    q = "ti:" + urllib.parse.quote(f'"{title}"')
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
    fields = "title,authors,year,venue,abstract,externalIds,url"
    try:
        resp = requests.get(
            url,
            params={"query": query, "limit": max_results, "fields": fields},
            headers={"x-api-key": key, "User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []

    out: list[dict] = []
    for it in data.get("data", []) or []:
        ext = it.get("externalIds") or {}
        out.append(
            {
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
            }
        )
    return out


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
        resp = requests.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []

    out: list[dict] = []
    for it in data.get("results", []) or []:
        authorships = it.get("authorships") or []
        authors = [
            ((a.get("author") or {}).get("display_name") or "") for a in authorships
        ]
        ids = it.get("ids") or {}
        doi = (it.get("doi") or "").replace("https://doi.org/", "").lower()
        arxiv_id = ""
        for v in ids.values():
            if isinstance(v, str) and "arxiv" in v.lower():
                arxiv_id = v.rstrip("/").rsplit("/", 1)[-1]
                break
        out.append(
            {
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
        )
    return out


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
