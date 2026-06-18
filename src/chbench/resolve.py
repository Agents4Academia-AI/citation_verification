"""
resolve.py — the GOLD resolver (deliberately DIFFERENT from the agent's).

ANTI-CIRCULARITY (decisions-phy.md, frozen): the gold oracle must NOT reuse the
agent's grounding code or its judge model, or correctness P/R would measure
self-agreement instead of accuracy. Therefore this module:

  * MUST NOT import the agent's grounding package.
  * MUST NOT call the agent's LLM judge.
  * Uses an INDEPENDENT source order — DBLP first (the agent floor is
    Crossref+arXiv) — and records full provenance so the gold's origin is
    auditable.

It is a thin, stdlib-only HTTP client (lazy ``urllib``) returning a ``Resolved``
sub-record (imported from the contract schema). Network is optional and
fail-soft: with ``fetch=False`` (default) or on any error it returns ``None``,
so the pipeline runs offline. The match cascade (DOI > arXiv-id > fuzzy-title)
is intentionally simple and human-auditable; the human adjudicator overrides it
where needed (provenance distinguishes the two).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

# Contract schema import ONLY — never the agent's grounding code (anti-circularity).
from citation_verifier.schema import MatchMethod, Resolved

_USER_AGENT = "chbench-gold-resolver/0.1 (CitationHallucinationBench; independent-of-agent)"
_TIMEOUT = 20

# Independent source order — DBLP first to diverge from the agent's Crossref/arXiv
# floor. Each source's provenance is stamped onto the Resolved record.
_GOLD_SOURCE_ORDER = ("dblp", "crossref", "arxiv")


def _http_get(url: str, *, timeout: int = _TIMEOUT) -> bytes | None:
    """Stdlib GET, fail-soft (returns None on any error)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _norm_title(title: str) -> str:
    """Normalize a title for comparison: lowercase, alnum-only, collapsed."""
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _title_similarity(a: str, b: str) -> float:
    """Title similarity in [0,1]. Uses rapidfuzz if present, else token Jaccard."""
    if not a or not b:
        return 0.0
    na, nb = _norm_title(a), _norm_title(b)
    try:
        from rapidfuzz import fuzz  # type: ignore

        return fuzz.token_sort_ratio(na, nb) / 100.0
    except Exception:
        ta, tb = set(na.split()), set(nb.split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


class GoldResolver:
    """Resolve a cited reference to a canonical record for GOLD labelling.

    Deliberately independent of the agent's resolver (anti-circularity). Records
    ``match_method``, ``match_score`` and ``source`` provenance on every result.

    Args:
        fetch: when False (default) the resolver is offline and
            :meth:`resolve` always returns ``None`` (used for import-safe,
            networkless dry-runs). When True it queries the independent sources.
        min_title_score: fuzzy-title acceptance threshold (gated additionally by
            year proximity when a year is parseable from the reference).
    """

    def __init__(self, *, fetch: bool = False, min_title_score: float = 0.82) -> None:
        self.fetch = fetch
        self.min_title_score = min_title_score

    # ── public API ────────────────────────────────────────────────────────────

    def resolve(self, reference: str) -> dict | None:
        """Resolve a free-text ``reference`` to a canonical record.

        Returns a :class:`citation_verifier.schema.Resolved` dumped to a ``dict``
        (so callers stay decoupled from the model), or ``None`` if no confident
        match was found / offline. The returned dict's ``match_method`` and
        ``source`` record how/where the match was made (gold provenance).
        """
        resolved = self.resolve_model(reference)
        return resolved.model_dump(mode="json") if resolved is not None else None

    def resolve_model(self, reference: str) -> Resolved | None:
        """Same as :meth:`resolve` but returns a typed ``Resolved`` (or None)."""
        if not self.fetch or not reference.strip():
            return None

        doi = _extract_doi(reference)
        arxiv_id = _extract_arxiv_id(reference)
        ref_year = _extract_year(reference)

        for source in _GOLD_SOURCE_ORDER:
            candidates = self._query(source, reference)
            if not candidates:
                continue
            match = self._best_match(candidates, reference, doi, arxiv_id, ref_year)
            if match is not None:
                return match
        return None

    # ── matching cascade ────────────────────────────────────────────────────────

    def _best_match(
        self,
        candidates: list[dict[str, Any]],
        reference: str,
        doi: str | None,
        arxiv_id: str | None,
        ref_year: int | None,
    ) -> Resolved | None:
        """Apply DOI > arXiv-id > fuzzy-title (year-gated) over candidates."""
        # 1) DOI exact.
        if doi:
            for c in candidates:
                if c.get("doi") and c["doi"].lower() == doi.lower():
                    return _to_resolved(c, MatchMethod.DOI, 1.0)
        # 2) arXiv-id exact.
        if arxiv_id:
            for c in candidates:
                if c.get("arxiv_id") and arxiv_id in c["arxiv_id"]:
                    return _to_resolved(c, MatchMethod.ARXIV, 1.0)
        # 3) Fuzzy title, gated by year proximity (±1) when both years known.
        best: tuple[float, dict[str, Any]] | None = None
        for c in candidates:
            score = _title_similarity(reference, c.get("title", ""))
            if ref_year and c.get("year") and abs(int(c["year"]) - ref_year) > 1:
                continue
            if best is None or score > best[0]:
                best = (score, c)
        if best and best[0] >= self.min_title_score:
            return _to_resolved(best[1], MatchMethod.FUZZY_TITLE, round(best[0], 3))
        return None

    # ── per-source query (independent of the agent) ─────────────────────────────

    def _query(self, source: str, reference: str) -> list[dict[str, Any]]:
        """Dispatch a candidate query to one independent source. Fail-soft."""
        if source == "dblp":
            return self._query_dblp(reference)
        if source == "crossref":
            return self._query_crossref(reference)
        if source == "arxiv":
            return self._query_arxiv(reference)
        return []

    def _query_dblp(self, reference: str) -> list[dict[str, Any]]:
        """Query DBLP's publication API (independent gold source)."""
        q = urllib.parse.quote(reference[:300])
        raw = _http_get(f"https://dblp.org/search/publ/api?q={q}&format=json&h=5")
        if raw is None:
            return []
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        hits = ((data.get("result") or {}).get("hits") or {}).get("hit") or []
        out: list[dict[str, Any]] = []
        for h in hits:
            info = h.get("info", {}) or {}
            authors = info.get("authors", {}).get("author", []) if info.get("authors") else []
            if isinstance(authors, dict):
                authors = [authors]
            names = [a.get("text", "") if isinstance(a, dict) else str(a) for a in authors]
            out.append(
                {
                    "source": "dblp",
                    "title": info.get("title", ""),
                    "authors": names,
                    "year": int(info["year"]) if info.get("year", "").isdigit() else None,
                    "venue": info.get("venue", ""),
                    "doi": info.get("doi", ""),
                    "arxiv_id": "",
                    "url": info.get("ee", "") or info.get("url", ""),
                    "abstract": "",
                }
            )
        return out

    def _query_crossref(self, reference: str) -> list[dict[str, Any]]:
        """Query Crossref (used by gold AFTER dblp; independent client code)."""
        params = urllib.parse.urlencode({"query.bibliographic": reference[:300], "rows": 5})
        raw = _http_get(f"https://api.crossref.org/works?{params}")
        if raw is None:
            return []
        try:
            data = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return []
        out: list[dict[str, Any]] = []
        for it in (data.get("message", {}) or {}).get("items", []):
            authors = [
                " ".join(p for p in (a.get("given"), a.get("family")) if p)
                for a in it.get("author", [])
            ]
            parts = (it.get("issued") or {}).get("date-parts") or [[None]]
            year = parts[0][0] if parts and parts[0] else None
            out.append(
                {
                    "source": "crossref",
                    "title": (it.get("title") or [""])[0],
                    "authors": authors,
                    "year": year,
                    "venue": (it.get("container-title") or [""])[0],
                    "doi": it.get("DOI", ""),
                    "arxiv_id": "",
                    "url": it.get("URL", ""),
                    "abstract": _strip_markup(it.get("abstract", "")),
                }
            )
        return out

    def _query_arxiv(self, reference: str) -> list[dict[str, Any]]:
        """Query arXiv (last independent source). Stdlib XML parse."""
        q = urllib.parse.quote(reference[:300])
        raw = _http_get(
            f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results=5"
        )
        if raw is None:
            return []
        atom = "{http://www.w3.org/2005/Atom}"
        arx = "{http://arxiv.org/schemas/atom}"
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return []
        out: list[dict[str, Any]] = []
        for e in root.findall(f"{atom}entry"):

            def _text(tag: str, _e: ET.Element = e) -> str:
                el = _e.find(tag)
                return el.text.strip() if el is not None and el.text else ""

            url = _text(f"{atom}id")
            out.append(
                {
                    "source": "arxiv",
                    "title": re.sub(r"\s+", " ", _text(f"{atom}title")),
                    "authors": [
                        a.findtext(f"{atom}name", "").strip()
                        for a in e.findall(f"{atom}author")
                    ],
                    "year": int(_text(f"{atom}published")[:4]) if _text(f"{atom}published") else None,
                    "venue": _text(f"{arx}journal_ref") or "arXiv (preprint)",
                    "doi": _text(f"{arx}doi"),
                    "arxiv_id": url.rsplit("/", 1)[-1],
                    "url": url,
                    "abstract": _strip_markup(_text(f"{atom}summary")),
                }
            )
        return out


# ── helpers ───────────────────────────────────────────────────────────────────


def _to_resolved(candidate: dict[str, Any], method: MatchMethod, score: float) -> Resolved:
    """Map a raw candidate dict to a contract ``Resolved`` with provenance."""
    return Resolved(
        source=candidate.get("source"),
        match_method=method,
        match_score=score,
        title=candidate.get("title") or None,
        authors=list(candidate.get("authors") or []),
        year=candidate.get("year"),
        venue=candidate.get("venue") or None,
        doi=candidate.get("doi") or None,
        arxiv_id=candidate.get("arxiv_id") or None,
        url=candidate.get("url") or None,
        abstract=candidate.get("abstract") or None,
    )


def _extract_doi(reference: str) -> str | None:
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", reference)
    return m.group(0).rstrip(".") if m else None


def _extract_arxiv_id(reference: str) -> str | None:
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", reference)
    return m.group(1) if m else None


def _extract_year(reference: str) -> int | None:
    years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", reference)]
    return years[-1] if years else None


def _strip_markup(text: str, limit: int = 600) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]
