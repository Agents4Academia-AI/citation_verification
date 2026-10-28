"""
grounding/web_lookup.py — the web, as the last place a reference is looked for.

Every index has been asked and none returned the cited work. A finder (a model with web
search — see :mod:`citation_verifier.backends.web_search`) names the result URLs that are
the work's own pages, and this module reads them. It decides; the finder does not.

* **A record behind the URL.** An identifier in it — an arXiv id, a DOI (also inside a
  publisher URL), an ACL Anthology id, an OpenReview id, an ISBN — or in the page's
  citation tags is fetched from its index, and the record is matched to the reference by
  the resolver's own title / author / year gates. The record, not the web page, is the
  evidence: a web hit alone never makes a reference exist.
* **The page itself**, when no record stands behind it: a blog post, a repository, a model
  card, a leaderboard, a report. It is checked like a record: its title must match the
  cited title, and its authors and year are what the page itself states — its citation
  tags, or the byline and date printed beside the title (an organisation: the site it
  owns). A page naming none of the cited authors there is not the work; what it does name
  is compared with the citation, so a wrong year or co-author is reported.

* **A listing on the issuing institution's own site**: a department's technical-report
  list names the report, its number and its author, and is often all that is online of a
  report from before the web. A list runs over several pages, and the search returns
  whichever page it returns — so the list's other pages are read too.

Outcomes are cached on disk (``web/v2``), including a search that found nothing, so a
re-run does not pay for the same fabricated reference twice; an outcome reached while an
index failed is not cached. The number of lookups per run is capped.
"""

from __future__ import annotations

import html
import re
import threading
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from ..diskcache import key_for, namespace_dir, read_json, write_json
from ..errors import ModelUnavailable
from ..schema import CitedAs, MatchMethod, Resolved
from . import paper_lookup
from .resolver import (
    _dict_to_candidate,
    _fuzzy_title_score,
    _likely_title,
    _norm_title,
    _repeated_title,
    _title_tokens_contradict,
    clean_title,
)

__all__ = ["WebLocator", "WebLookup", "locator_for"]

_VERSION = "v3"
_ON_HOST_URLS = 6  # pages of the cited institution's site read per lookup
_SIBLING_PAGES = 8  # other pages of a report list followed per lookup
_PAGE_BYTES = 800_000
_PDF_BYTES = 8_000_000
_TITLE_SCORE = 90.0

_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s?#&\"'<>]+)")
_DOI_TAIL_RE = re.compile(r"(?:\.pdf|/(?:full|abstract|pdf|epdf|html|meta|summary))+$", re.I)
_ACL_URL_RE = re.compile(r"aclanthology\.org/((?:\d{4}\.[\w-]+\.\d+)|(?:[A-Z]\d{2}-\d{4}))", re.I)
_OPENREVIEW_URL_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([\w-]+)")
_ISBN_RE = re.compile(r"(?:isbn[=/:]?|/isbn/)\s*(97[89][\d-]{10,14}|\d[\d-]{8,11}[\dX])", re.I)


@dataclass
class WebLookup:
    """What a web lookup came to.

    ``status``: ``record`` (an index record found through the web), ``page`` (the work's
    own web page), ``none`` (searched, not found), ``failed`` or ``skipped``.
    """

    status: str
    resolved: Resolved | None = None
    urls: list[str] = field(default_factory=list)
    detail: str = ""


class WebLocator:
    """Find a reference on the web and read what is found.

    Args:
        search: ``reference -> (urls_chosen, urls_seen)`` — the finder.
        match: ``(reference, candidates) -> Resolved | None`` — the resolver's matcher.
        max_lookups: lookups allowed for this locator's lifetime (one run).
        fetch_page: ``url -> html`` (injectable for tests).
    """

    def __init__(
        self,
        search: Any,
        match: Any,
        *,
        max_lookups: int = 30,
        concurrency: int = 3,
        fetch_page: Any = None,
    ) -> None:
        self.search = search
        self.match = match
        self.max_lookups = max_lookups
        self.fetch_page = fetch_page or _fetch_page
        self.lookups = 0
        self.unavailable = ""
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max(1, concurrency))

    def locate(self, reference: str, cited: CitedAs) -> WebLookup:
        """Look for the work ``reference`` cites. Never raises."""
        reference = (reference or "").strip()
        if not reference:
            return WebLookup("skipped", detail="no reference text")
        root = namespace_dir("web", _VERSION)
        key = key_for("web", reference)
        cached = read_json(root, key)
        if isinstance(cached, dict) and cached.get("status") in ("record", "page", "none"):
            try:
                resolved = Resolved(**cached["resolved"]) if cached.get("resolved") else None
            except Exception:  # noqa: BLE001 — a stale shape is a miss
                resolved = None
            if resolved is not None or cached["status"] == "none":
                return WebLookup(cached["status"], resolved, list(cached.get("urls") or []),
                                 cached.get("detail") or "cached")
        with self._lock:
            if self.unavailable:
                return WebLookup("failed", detail=f"the search model is unavailable ({self.unavailable})")
            if self.lookups >= self.max_lookups:
                return WebLookup("skipped", detail=f"the limit of {self.max_lookups} web lookups per paper was reached")
            self.lookups += 1
        # The finder is given the reference as printed: the string the indexes are asked
        # with is rebuilt from the parsed fields and drops what follows the title — "Technical
        # Report 348, Department of Statistics, University of Chicago" — which is what finds
        # a report, and what says whose site its listing is on.
        printed = (cited.raw or "").strip() or reference
        with self._slots:
            try:
                chosen, seen = self.search(printed)
            except ModelUnavailable as exc:
                with self._lock:
                    self.unavailable = str(exc)[:200]
                return WebLookup("failed", detail=f"the search model is unavailable ({exc})")
            except Exception as exc:  # noqa: BLE001 — a failed search is not a finding
                return WebLookup("failed", detail=f"web search failed ({type(exc).__name__})")
            paper_lookup.reset_failures()
            urls = list(chosen or [])
            # Pages of the institution or publisher the citation names, among what the
            # searches returned, are read too: a department's report list or a publisher's
            # catalogue is not the work, so the model may not pick it, yet it records it.
            # Pages whose address reads as a list of reports or publications go first.
            known = {_address(u) for u in urls}
            on_host = []
            for u in _on_cited_hosts(seen or [], reference, cited):
                if _address(u) not in known:
                    known.add(_address(u))
                    on_host.append(u)
            on_host.sort(key=lambda u: not _LISTING_PATH_RE.search(urllib.parse.urlsplit(u).path))
            urls += on_host[:_ON_HOST_URLS]
            outcome = self._read(reference, cited, urls)
            index_failed = bool(paper_lookup.failures())
        if not index_failed:
            write_json(root, key, {
                "status": outcome.status,
                "resolved": outcome.resolved.model_dump(mode="json") if outcome.resolved else None,
                "urls": outcome.urls,
                "detail": outcome.detail,
            })
        return outcome

    # ── reading what was found ─────────────────────────────────────
    def _read(self, reference: str, cited: CitedAs, urls: list[str]) -> WebLookup:
        rows: list[dict] = []
        for url in urls:
            rows += _records_behind(url, cited.year)
        found = self.match(reference, [_dict_to_candidate(r) for r in rows]) if rows else None
        if found is not None:
            return WebLookup("record", found, urls)
        listings: list[tuple[str, dict]] = []
        for url in urls:
            page = _read_page(self.fetch_page, url)
            if not page:
                continue
            ids = [page.get("doi"), page.get("arxiv_id")]
            extra = _records_for_ids(*ids, year=cited.year) if any(ids) else []
            if extra:
                found = self.match(reference, [_dict_to_candidate(r) for r in extra])
                if found is not None:
                    return WebLookup("record", found, urls)
            work = _page_as_work(reference, cited, page, url)
            if work is not None:
                return WebLookup("page", work[0], urls, work[1])
            if not page.get("pdf") and _hosted_by_cited(page["host"], reference, cited):
                listings.append((url, page))
        # The other pages of a list on the cited institution's site: the search returned
        # one page of a department's report list, the report is on another.
        visited = {_address(u) for u in urls}
        followed = 0
        for url, page in listings:
            for link in _sibling_pages(url, page.get("links") or []):
                if _address(link) in visited or followed >= _SIBLING_PAGES:
                    continue
                visited.add(_address(link))
                followed += 1
                sibling = _read_page(self.fetch_page, link)
                work = _page_as_work(reference, cited, sibling, link) if sibling else None
                if work is not None:
                    return WebLookup("page", work[0], [*urls, link], work[1])
        return WebLookup("none", urls=urls)


def locator_for(search: Any, resolver: Any, settings: Any = None) -> WebLocator | None:
    """A locator over ``resolver``'s own matcher, with the run's lookup cap
    (``web_search_max_lookups``); ``None`` without a finder or a matcher."""
    match = getattr(resolver, "_match", None)
    if search is None or not callable(match):
        return None
    raw = settings.get("web_search_max_lookups") if isinstance(settings, dict) else getattr(
        settings, "web_search_max_lookups", None
    )
    return WebLocator(search, match, max_lookups=int(raw or 50))


# ── identifiers in URLs ──────────────────────────────────────────────────────
def _records_behind(url: str, year: int | None) -> list[dict]:
    """Index records for the identifiers a URL carries."""
    arxiv = _ARXIV_URL_RE.search(url or "")
    doi = _doi_in(url)
    acl = _ACL_URL_RE.search(url or "")
    if acl and not doi:
        doi = f"10.18653/v1/{acl.group(1)}"
    rows = _records_for_ids(doi, arxiv.group(1) if arxiv else "", year=year)
    openreview = _OPENREVIEW_URL_RE.search(url or "")
    if openreview:
        rows += paper_lookup.fetch_openreview_by_id(openreview.group(1))
    isbn = _ISBN_RE.search(url or "")
    if isbn:
        rows += paper_lookup.fetch_openlibrary_by_isbn(isbn.group(1), year)
    return rows


def _records_for_ids(doi: str | None, arxiv_id: str | None, *, year: int | None = None) -> list[dict]:
    rows: list[dict] = []
    if arxiv_id:
        rows += paper_lookup.fetch_arxiv_by_id(arxiv_id)
    if doi:
        rows += paper_lookup.fetch_crossref_by_doi(doi)
    return rows


def _doi_in(text: str) -> str:
    m = _DOI_RE.search(urllib.parse.unquote(text or ""))
    return _DOI_TAIL_RE.sub("", m.group(1)).rstrip(".,;)") if m else ""


# ── the page itself ──────────────────────────────────────────────────────────
def _fetch_page(url: str) -> str:
    """A page's HTML (the first few hundred KB), or the text of a PDF's first two pages
    marked with a leading ``\x0c``, or ``""``. Never raises.

    Many works' own page IS a PDF — a journal's article file, a workshop paper, a technical
    report — so a search result that is one is read too. A scanned PDF has no text and
    cannot be checked.
    """
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": paper_lookup.USER_AGENT, "Accept": "text/html,application/pdf"}
        )
        with urllib.request.urlopen(req, timeout=25) as resp:  # noqa: S310 — a search result URL
            kind = (resp.headers.get("Content-Type") or "").lower()
            if "pdf" in kind or (not kind and url.lower().endswith(".pdf")):
                return "\x0c" + _pdf_head_text(resp.read(_PDF_BYTES))
            if "html" not in (kind or "html"):
                return ""
            return resp.read(_PAGE_BYTES).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _pdf_head_text(data: bytes) -> str:
    """The text of a PDF's first two pages (PyMuPDF, else pypdf), or ``""``."""
    try:
        import fitz  # noqa: PLC0415 — optional

        with fitz.open(stream=data, filetype="pdf") as doc:
            return " ".join(doc[i].get_text() for i in range(min(2, len(doc))))
    except Exception:  # noqa: BLE001
        pass
    try:
        import io

        from pypdf import PdfReader  # noqa: PLC0415 — optional

        reader = PdfReader(io.BytesIO(data))
        return " ".join((reader.pages[i].extract_text() or "") for i in range(min(2, len(reader.pages))))
    except Exception:  # noqa: BLE001
        return ""


_META_RE = re.compile(r"<meta\s+[^>]*?(?:name|property)\s*=\s*[\"']([^\"']+)[\"'][^>]*?content\s*=\s*[\"']([^\"']*)[\"']", re.I)
_META_REV_RE = re.compile(r"<meta\s+[^>]*?content\s*=\s*[\"']([^\"']*)[\"'][^>]*?(?:name|property)\s*=\s*[\"']([^\"']+)[\"']", re.I)
_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_HREF_RE = re.compile(r"<a\s[^>]*?href\s*=\s*[\"']([^\"'#][^\"']*)[\"']", re.I)
_MAX_LINKS = 1000
_LISTING_PATH_RE = re.compile(r"(?i)report|paper|thes[ie]s|dissertation|publication|librar|series|archive")
_PAGE_PARAM_RE = re.compile(r"(?i)(?:^|&)(?:page|pg|p|start|offset)=\d+")


def _read_page(fetch: Any, url: str) -> dict | None:
    """Title, authors, year, identifiers and text of a page, from its HTML — or, for a
    PDF, its first pages' text (``titles`` empty: the title is looked for in the text)."""
    raw = fetch(url) or ""
    if not raw:
        return None
    if raw.startswith("\x0c"):
        text = " ".join(raw[1:].split())
        if len(text) < 40:
            return None  # a scan, or an empty file: nothing to check
        return {"titles": [], "authors": [], "year": None, "venue": "", "doi": _doi_in(text[:3000]),
                "arxiv_id": "", "description": "", "host": urllib.parse.urlsplit(url).netloc.lower(),
                "path": urllib.parse.urlsplit(url).path.lower(), "text": text[:20_000], "pdf": True}
    meta: dict[str, list[str]] = {}
    for name, content in _META_RE.findall(raw):
        meta.setdefault(name.lower(), []).append(html.unescape(content).strip())
    for content, name in _META_REV_RE.findall(raw):
        meta.setdefault(name.lower(), []).append(html.unescape(content).strip())
    links = list(dict.fromkeys(
        urllib.parse.urljoin(url, html.unescape(h)).split("#", 1)[0] for h in _HREF_RE.findall(raw)
    ))[:_MAX_LINKS]
    title_tag = _TITLE_TAG_RE.search(raw)
    titles = [t for t in (
        *(meta.get("citation_title") or []), *(meta.get("og:title") or []),
        html.unescape(title_tag.group(1)).strip() if title_tag else "",
    ) if t]
    date = next(iter(meta.get("citation_publication_date") or meta.get("citation_date") or meta.get("citation_year") or []), "")
    year = re.search(r"\b(19|20)\d{2}\b", date)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    host = urllib.parse.urlsplit(url).netloc.lower()
    return {
        "titles": titles,
        "authors": meta.get("citation_author") or [],
        "year": int(year.group(0)) if year else None,
        "venue": next(iter(meta.get("citation_conference_title") or meta.get("citation_journal_title") or []), ""),
        "doi": _doi_in(" ".join(meta.get("citation_doi") or meta.get("dc.identifier") or [])),
        "arxiv_id": next(iter(meta.get("citation_arxiv_id") or []), ""),
        "description": next(iter(meta.get("citation_abstract") or meta.get("description") or meta.get("og:description") or []), ""),
        "host": host,
        "path": urllib.parse.urlsplit(url).path.lower(),
        "text": " ".join(text.split())[:200_000],
        "links": links,
    }


def _page_as_work(reference: str, cited: CitedAs, page: dict, url: str) -> tuple[Resolved, str] | None:
    """The page as the cited work, with what was read off it — or ``None``.

    Found on the web, a work is checked as one found in an index is: the title must be the
    cited title, and the authorship and the date are what the PAGE says, never what the
    citation says. The authors are the page's own (its citation tags), else the cited
    authors whose names stand next to the title — the byline of a paper's first page, the
    entry in a report list; a name somewhere else on the page (a reading list, a reference
    section) is no evidence of authorship. The year is the page's own, else one printed
    beside the title. A page that names none of the cited authors is not the work; what it
    does name is then compared with the citation like any record, so a wrong year or a
    wrong co-author is reported.
    """
    wanted = clean_title(cited.title or _likely_title(reference))
    # A title copied from a broken record says its subject twice; it is that subject.
    wanted = _repeated_title(wanted) or wanted
    if not wanted:
        return None
    probe = f"{wanted}."  # the title alone, as a reference the contradiction gate can read
    best, score = "", 0.0
    if page.get("pdf"):
        head = _norm_title(page["text"][:2500])
        s = float(fuzz.partial_ratio(_norm_title(wanted), head)) if len(_norm_title(wanted)) >= 12 else 0.0
        best, score = wanted, s
    for t in page["titles"]:
        s = _fuzzy_title_score(wanted, t)
        if s > score and not _title_tokens_contradict(probe, t):
            best, score = t, s
    what = "the work's own page"
    authors: list[str] = []
    year: int | None = None
    if score >= _TITLE_SCORE:
        byline = _near_title(page["text"][:6000] if page.get("pdf") else page["text"], wanted,
                             _BYLINE_BEFORE, _BYLINE_AFTER)
        authors = _authors_on_page(cited, page, byline)
        year = page["year"] or _year_near_title(byline, cited.year)
    if not authors:
        entry = _catalogue_entry(reference, cited, page, wanted)
        if entry is None:
            return None
        score, window = entry
        best, what = wanted, f"an entry in the report list of {page['host']}"
        authors = _cited_names_in(cited, window)
        year = _year_near_title(window, cited.year)
    read = "title, author" + ("s" if len(authors) > 1 else "") + (" and year" if year else "")
    detail = f"found by web search: {what} — {read} read off it" + ("" if year else "; it gives no year")
    resolved = Resolved(
        source="web",
        match_method=MatchMethod.DIRECT_URL,
        match_score=round(score / 100.0, 3),
        title=best,
        authors=authors,
        year=year,
        venue=page["venue"] or None,
        url=url,
        url_valid=True,
        abstract=page["description"][:4000] or None,
    )
    return resolved, detail


# Where a work's page, or its first page, names its authors and date: just before the title
# (a journal's running head, a listing's author column) or after it (the byline).
_BYLINE_BEFORE, _BYLINE_AFTER = 300, 600
# One entry of a report list: number, author, date, then the title.
_ENTRY_BEFORE, _ENTRY_AFTER = 150, 100
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _near_title(text: str, wanted: str, before: int, after: int) -> str:
    """The (normalised, accent-folded) text around where ``wanted`` is printed in ``text``,
    or ``""`` when it is not printed there."""
    norm_wanted, norm = _norm_title(wanted), _norm_title(text)
    if len(norm_wanted) < 12 or not norm:
        return ""
    hit = fuzz.partial_ratio_alignment(norm_wanted, norm)
    if hit is None or hit.score < _TITLE_SCORE:
        return ""
    return _fold(norm[max(0, hit.dest_start - before) : hit.dest_end + after])


def _authors_on_page(cited: CitedAs, page: dict, byline: str) -> list[str]:
    """The page's authors when it states them and a cited author is among them; else the
    cited authors named beside the title (or, for an organisation, owning the site)."""
    stated = [a for a in page["authors"] if (a or "").strip()]
    if stated:
        theirs = {_surname(a) for a in stated}
        return stated if any(_surname(a) in theirs for a in cited.authors or []) else []
    host_words = set(re.split(r"[.\-]", _fold(page["host"])))
    return [
        a for a in cited.authors or []
        if (n := _surname(a)) and len(n) >= 3
        and (re.search(rf"\b{re.escape(n)}\b", byline) or re.sub(r"\W", "", _fold(a)) in host_words)
    ]


def _cited_names_in(cited: CitedAs, window: str) -> list[str]:
    return [
        a for a in cited.authors or []
        if (n := _surname(a)) and len(n) >= 3 and re.search(rf"\b{re.escape(n)}\b", window)
    ]


def _year_near_title(window: str, cited_year: int | None) -> int | None:
    """A year printed beside the title: the cited one if it is among them, else the latest
    (a page's date line, a journal's volume year) — ``None`` when none is."""
    years = [int(y) for y in _YEAR_RE.findall(window or "")]
    if not years:
        return None
    return cited_year if cited_year in years else max(years)


def _catalogue_entry(reference: str, cited: CitedAs, page: dict, wanted: str) -> tuple[float, str] | None:
    """A listing of the work on the site of the institution or publisher the citation
    names — a department's technical-report list, a publisher's catalogue — as the score of
    the title found there and the entry around it; ``None`` when there is no such entry.

    Three things must hold: the page is on that institution's site (its host carries a
    distinctive word of the citation's publication details: "University of Chicago" ->
    ``stat.uchicago.edu``); the cited title is printed on it; and a cited author's surname
    stands right beside it, in the same entry. The host is the guard: a citing paper's
    reference list carries the same title and author, but not on the cited institution's
    own site.
    """
    if page.get("pdf") or not _hosted_by_cited(page["host"], reference, cited):
        return None
    norm_wanted, text = _norm_title(wanted), _norm_title(page["text"])
    if len(norm_wanted) < 12:
        return None
    hit = fuzz.partial_ratio_alignment(norm_wanted, text)
    if hit is None or hit.score < 95:
        return None
    window = _fold(text[max(0, hit.dest_start - _ENTRY_BEFORE) : hit.dest_end + _ENTRY_AFTER])
    if not _cited_names_in(cited, window):
        return None
    return float(hit.score), window


def _sibling_pages(url: str, links: list[str]) -> list[str]:
    """The links among ``links`` to other pages of the same list as ``url``.

    Another page of a list is on the same site, in the same directory, under the same name
    but for its numbering ("technical-reports", "technical-reports-2", "technical-report-4"),
    or is the same address with another page number ("?page=2").
    """
    here = urllib.parse.urlsplit(url)
    host = here.netloc.lower().removeprefix("www.")
    parent, stem = _path_stem(here.path)
    out: list[str] = []
    for link in links:
        there = urllib.parse.urlsplit(link)
        if there.scheme not in ("http", "https") or there.netloc.lower().removeprefix("www.") != host:
            continue
        if there.path.rstrip("/") == here.path.rstrip("/"):
            if there.query != here.query and _PAGE_PARAM_RE.search(there.query):
                out.append(link)
            continue
        if stem and _path_stem(there.path) == (parent, stem):
            out.append(link)
    return list(dict.fromkeys(out))


def _address(url: str) -> str:
    """A URL compared loosely: scheme, ``www.``, a trailing slash and the fragment dropped."""
    u = re.sub(r"^https?://(?:www\.)?", "", (url or "").strip().lower())
    return u.split("#", 1)[0].rstrip("/")


def _path_stem(path: str) -> tuple[str, str]:
    """``(directory, name without numbering)`` of a URL path: "/lib/technical-reports-2/"
    -> ("/lib", "technicalreport"). The name is empty when nothing but numbering is left."""
    parts = (path or "/").rstrip("/").rsplit("/", 1)
    parent, last = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
    last = re.sub(r"\.\w{2,5}$", "", last.lower())
    stem = re.sub(r"[\W\d_]+", "", last)
    stem = stem[:-1] if stem.endswith("s") else stem
    return parent, stem if len(stem) >= 4 else ""


def _publication_words(reference: str, cited: CitedAs) -> set[str]:
    """Distinctive words of the citation's publication details (what follows the title):
    "Technical Report 348, Department of Statistics, University of Chicago" -> {"chicago"}."""
    raw = cited.raw or reference or ""  # as printed: the parsed form drops the publication details
    title = cited.title or ""
    tail = raw[raw.lower().find(title.lower()) + len(title):] if title and title.lower() in raw.lower() else raw
    tail += " " + (cited.venue or "")
    words = {_fold(w) for w in re.findall(r"[A-Z][A-Za-z]{2,}", tail)}
    return {w for w in words if w not in _GENERIC_PUBLICATION_WORDS}


_GENERIC_PUBLICATION_WORDS = frozenset(
    "technical report reports department dept university univ statistics statistical institute "
    "school college faculty laboratory lab press proceedings journal conference workshop symposium "
    "volume vol pages series working paper papers thesis dissertation phd master arxiv preprint inc "
    "ltd eds editor editors edition international annual national transactions letters review "
    "science sciences computer engineering research center centre and the for".split()
)


def _hosted_by_cited(host: str, reference: str, cited: CitedAs) -> bool:
    host = _fold(host or "")
    return any(w in host for w in _publication_words(reference, cited))


def _on_cited_hosts(urls: list[str], reference: str, cited: CitedAs) -> list[str]:
    """The URLs among ``urls`` that are on a site the citation names."""
    words = _publication_words(reference, cited)
    if not words:
        return []
    return [u for u in urls if any(w in _fold(urllib.parse.urlsplit(u).netloc) for w in words)]


def _surname(author: str) -> str:
    """'Bengio, Yoshua' / 'Yoshua Bengio' / 'Colby KM' -> the surname; 'OpenAI' -> itself."""
    a = (author or "").strip()
    if "," in a:
        a = a.split(",", 1)[0]
    words = [w for w in re.findall(r"[^\W\d_][^\W\d_'’-]*", a) if w.lower() not in {"and", "et", "al", "others"}]
    names = [w for w in words if not (w.isupper() and len(w) <= 3)] or words  # "KM" is initials; "IBM" alone is a name
    return _fold(names[-1]) if names else ""


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()

