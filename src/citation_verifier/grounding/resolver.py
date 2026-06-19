"""
resolver — turn one cited reference into a canonical :class:`Resolved` record.

:class:`MultiSourceResolver` satisfies the :class:`citation_verifier.interfaces.Resolver`
Protocol. It fans out to the grounding sources in :mod:`paper_lookup`, gathers
:class:`Candidate` rows, and applies a deterministic **match cascade**:

    1. DOI         — an exact DOI in the reference that a candidate also carries.
    2. arXiv id    — an exact arXiv id in the reference that a candidate carries.
    3. fuzzy title — rapidfuzz title similarity, **gated** by author overlap and
                     year agreement (±1) so a near-title-collision can't pass on
                     its own.

The resolver does NOT decide ``exists`` / ``metadata_issues`` / relevance — that
is the job of :mod:`citation_verifier.stages`. It only returns the best canonical
match plus *provenance* (``match_method``, ``match_score``, ``url_valid``) so the
stages can adjudicate against retrieved fields rather than model memory.

Import-safe and network-free at import time: the only hard dependency is
``rapidfuzz`` (a core dep); all network access is lazy and fail-soft.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..interfaces import Candidate
from ..schema import MatchMethod, Resolved
from . import paper_lookup

if TYPE_CHECKING:  # avoid importing config eagerly; it is an optional sibling
    from ..config import Settings

# Default source order for the cascade (stdlib floor first, optional last).
DEFAULT_SOURCES: tuple[str, ...] = (
    paper_lookup.SOURCE_CROSSREF,
    paper_lookup.SOURCE_ARXIV,
    paper_lookup.SOURCE_DBLP,
    paper_lookup.SOURCE_S2,
    paper_lookup.SOURCE_OPENALEX,
)

# Fuzzy-title acceptance threshold (rapidfuzz token_sort_ratio, 0..100).
FUZZY_TITLE_THRESHOLD = 85.0
# Minimum acceptance when the gate (author/year) is satisfied but title is softer.
FUZZY_TITLE_GATED_THRESHOLD = 78.0

# Patterns to pull a DOI / arXiv id out of a free-text reference string.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_ARXIV_RE = re.compile(
    r"arxiv[:\s]*((?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


# ───────────────────────────────────────────────────────────────
# Small text-normalisation helpers
# ───────────────────────────────────────────────────────────────
def _norm_title(s: str | None) -> str:
    """Lowercase, strip punctuation/whitespace for title comparison."""
    if not s:
        return ""
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _last_name(author: str) -> str:
    """Best-effort surname extraction ('First Last' or 'Last, First')."""
    author = author.strip()
    if not author:
        return ""
    if "," in author:
        return author.split(",", 1)[0].strip().lower()
    return author.split()[-1].lower()


def _author_surnames(authors: list[str]) -> set[str]:
    return {_last_name(a) for a in authors if a.strip()}


def _author_overlap(claimed: list[str], canonical: list[str]) -> float:
    """Fraction of claimed surnames that appear among canonical surnames."""
    a = _author_surnames(claimed)
    b = _author_surnames(canonical)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def _extract_doi(reference: str) -> str:
    m = _DOI_RE.search(reference or "")
    return m.group(1).rstrip(".").lower() if m else ""


def _extract_arxiv_id(reference: str) -> str:
    m = _ARXIV_RE.search(reference or "")
    return m.group(1).lower() if m else ""


# A quoted segment in a reference is almost always the title (e.g.
# `Authors. "Title". Venue. Year`). Straight or curly quotes, >= 8 chars.
_QUOTED_TITLE_RE = re.compile(r"[\"“‘]([^\"“”‘’]{8,})[\"”’]")


def _likely_title(reference: str) -> str:
    """Best-effort extraction of the cited work's TITLE for a ``ti:`` search.

    Prefers a quoted segment (the common ``Authors. "Title". Venue`` form); else
    falls back to the longest dotted clause that is not an author list, a venue,
    a year, or an id/url. Returns ``""`` when nothing title-like is found — the
    caller then simply skips the title search (no regression).
    """
    ref = (reference or "").strip()
    m = _QUOTED_TITLE_RE.search(ref)
    if m:
        return m.group(1).strip()
    head = re.split(r"(?i)\b(?:arxiv|doi|https?://)", ref)[0]
    best = ""
    for clause in re.split(r"\.\s+", head):
        c = clause.strip(" .,")
        if len(c) < 12 or re.search(r"\b(19|20)\d{2}\b", c):
            continue  # too short, or a year/venue tail
        if " and " in f" {c.lower()} " or c.count(",") >= 2:
            continue  # looks like an author list
        if len(c) > len(best):
            best = c
    return best


def _first_author(reference: str) -> str:
    """Best-effort first-author SURNAME from the start of a reference.

    Disambiguates a reused title (e.g. "… is all you need") from its namesakes
    when added to a title-field query. Handles both "Vaswani A, …" (surname-first)
    and "Ashish Vaswani, …" (given-first) by taking the longest alphabetic token
    in the first author chunk. Returns ``""`` when nothing usable is found.
    """
    head = re.split(r"[.;]", (reference or "").strip(), maxsplit=1)[0]
    if "," in head and head.index(",") < 40:
        head = head.split(",", 1)[0]
    toks = [t for t in re.findall(r"[^\W\d_]{3,}", head[:60]) if t.lower() != "and"]
    return max(toks, key=len) if toks else ""


def _fuzzy_title_score(reference: str, candidate_title: str) -> float:
    """Similarity (0..100) between a candidate *title* and a *reference* string.

    The reference often embeds the title amid authors/year/venue, so a plain
    ``token_sort_ratio`` of the full reference against a bare title is diluted.
    We take the max of:

      - ``token_sort_ratio`` (order-insensitive full match), and
      - ``token_set_ratio`` (rewards the title being a token-subset of the ref),

    which lets an exact embedded title score ~100 while the author/year gate in
    :meth:`MultiSourceResolver._gate` guards against subset false positives.
    """
    na, nb = _norm_title(reference), _norm_title(candidate_title)
    if not na or not nb:
        return 0.0
    from rapidfuzz import fuzz  # local import keeps module import cheap

    return max(
        float(fuzz.token_sort_ratio(na, nb)),
        float(fuzz.token_set_ratio(na, nb)),
    )


# ───────────────────────────────────────────────────────────────
# The resolver
# ───────────────────────────────────────────────────────────────
class MultiSourceResolver:
    """Resolve a cited reference to a canonical record via a match cascade.

    Satisfies the :class:`citation_verifier.interfaces.Resolver` Protocol::

        resolve(cite_key, reference) -> Resolved | None
        candidates(reference, max_results=4) -> list[Candidate]

    Args:
        settings: optional :class:`Settings`; used to read source enable-flags,
            ``CROSSREF_MAILTO``, and S2/OpenAlex keys. ``None`` => keyless floor.
        sources: explicit source order to query; ``None`` => :data:`DEFAULT_SOURCES`.
        validate_urls: whether ``resolve`` should HEAD/GET-check the matched URL
            (network, fail-soft). Defaults to True.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        sources: list[str] | None = None,
        validate_urls: bool = True,
    ) -> None:
        self.settings = settings
        self.sources: tuple[str, ...] = tuple(sources) if sources else DEFAULT_SOURCES
        self.validate_urls = validate_urls

    # ── source fan-out ────────────────────────────────────────
    def candidates(self, reference: str, /, max_results: int = 4) -> list[Candidate]:
        """Union of every tier's candidates (the Protocol's gather method).

        :meth:`resolve` drives the tiers directly with short-circuit-on-match; this
        returns their union for external callers (e.g. ``lookup_paper``). Fail-soft.
        """
        return (
            self._id_candidates(reference)
            + self._title_candidates(reference, max_results)
            + self._broad_candidates(reference, max_results)
        )

    def _id_candidates(self, reference: str) -> list[Candidate]:
        """Tier 1: direct lookup by the reference's own DOI / arXiv id (no search)."""
        rows: list[Candidate] = []
        ref_arxiv = _extract_arxiv_id(reference)
        if ref_arxiv:
            rows += _fetch(paper_lookup.fetch_arxiv_by_id, ref_arxiv)
        ref_doi = _extract_doi(reference)
        if ref_doi:
            rows += _fetch(paper_lookup.fetch_crossref_by_doi, ref_doi)
        return rows

    def _title_candidates(self, reference: str, max_results: int = 4) -> list[Candidate]:
        """Tier 2: precise title-field queries — S2 match, arXiv ti:+au:, Crossref.

        Querying the extracted TITLE (+ first author), not the noisy full reference,
        surfaces the canonical paper where a reused title ("Attention is all you
        need") otherwise returns same-title namesakes.
        """
        title = _likely_title(reference)
        if not title:
            return []
        author = _first_author(reference) or None
        return (
            _fetch(paper_lookup.search_semantic_scholar_match, title)
            + _fetch(paper_lookup.search_arxiv_by_title, title, max_results, author)
            + _fetch(paper_lookup.search_crossref_by_fields, title, author, max_results)
        )

    def _broad_candidates(self, reference: str, max_results: int = 4) -> list[Candidate]:
        """Tier 3: broad full-reference search across every enabled source."""
        rows: list[Candidate] = []
        for src in self.sources:
            try:
                raw = self._search_source(src, reference, max_results)
            except Exception:  # noqa: BLE001 — defense in depth; sources fail soft
                raw = []
            rows += [_dict_to_candidate(d) for d in raw]
        return rows

    def _search_source(self, src: str, reference: str, max_results: int) -> list[dict]:
        """Dispatch one source by name (keeps optional keys threaded through)."""
        s2_key = getattr(self.settings, "s2_api_key", None)
        oa_key = getattr(self.settings, "openalex_api_key", None)
        if src == paper_lookup.SOURCE_CROSSREF:
            return paper_lookup.search_crossref(reference, max_results)
        if src == paper_lookup.SOURCE_ARXIV:
            return paper_lookup.search_arxiv(reference, max_results)
        if src == paper_lookup.SOURCE_DBLP:
            return paper_lookup.search_dblp(reference, max_results)
        if src == paper_lookup.SOURCE_S2:
            return paper_lookup.search_semantic_scholar(reference, max_results, s2_key)
        if src == paper_lookup.SOURCE_OPENALEX:
            return paper_lookup.search_openalex(reference, max_results, oa_key)
        return []

    # ── the cascade ───────────────────────────────────────────
    def resolve(self, cite_key: str, reference: str, /) -> Resolved | None:
        """Resolve ``reference`` to its best canonical match, or ``None``.

        Identifier-first cascade that short-circuits on the first tier whose
        candidates actually MATCH (not merely return rows), so the precise S2
        title-match resolves most references in a single call and the broad,
        namesake-prone search only runs when the precise tiers come up empty:

          1. DOI / arXiv id  -> direct lookup, matched by exact id.
          2. title (+ author) -> S2 ``/paper/search/match``, then arXiv ``ti:+au:``,
             then Crossref ``query.title`` — each matched immediately; first hit wins.
          3. broad full-reference search -> fuzzy-title gated by author/year.

        ``cite_key`` is accepted for Protocol symmetry; matching is driven by
        ``reference`` content.
        """
        # 1) identifier.
        match = self._match(reference, self._id_candidates(reference))
        if match is not None:
            return match

        # 2) title-field, most-precise-first; stop the moment one matches.
        title = _likely_title(reference)
        if title:
            author = _first_author(reference) or None
            for fn, args in (
                (paper_lookup.search_semantic_scholar_match, (title,)),
                (paper_lookup.search_arxiv_by_title, (title, 4, author)),
                (paper_lookup.search_crossref_by_fields, (title, author, 4)),
            ):
                match = self._match(reference, _fetch(fn, *args))
                if match is not None:
                    return match

        # 3) broad fallback.
        return self._match(reference, self._broad_candidates(reference))

    def _match(self, reference: str, cands: list[Candidate]) -> Resolved | None:
        """Run the DOI -> arXiv-id -> fuzzy-title-gated cascade over ``cands``."""
        if not cands:
            return None
        ref_doi = _extract_doi(reference)
        ref_arxiv = _extract_arxiv_id(reference)
        ref_year = _extract_year(reference)

        # 1) DOI exact match.
        if ref_doi:
            for c in cands:
                if c.doi and c.doi.lower() == ref_doi:
                    return self._to_resolved(c, MatchMethod.DOI, 1.0)

        # 2) arXiv-id exact match.
        if ref_arxiv:
            stem = ref_arxiv.split("v")[0]
            for c in cands:
                if c.arxiv_id and c.arxiv_id.lower().split("v")[0] == stem:
                    return self._to_resolved(c, MatchMethod.ARXIV, 1.0)

        # 3) Fuzzy title, gated by author overlap + year (±1). Consider candidates
        #    in descending title-score order and SKIP (not abort on) ones the gate
        #    contradicts — a scholarly query routinely returns several same-title
        #    rows (e.g. a mirror/repost DOI with a much later year alongside the
        #    real record), so a gate-failing top hit must not hide a valid one.
        ranked = sorted(
            ((_fuzzy_title_score(reference, c.title), c) for c in cands),
            key=lambda t: t[0],
            reverse=True,
        )
        for score, c in ranked:
            if score < FUZZY_TITLE_GATED_THRESHOLD:
                break  # ranked descending: no later candidate can clear the bar
            gate = self._gate(c, reference, ref_year)
            if gate is False:
                continue  # author/year contradicts THIS candidate; try the next
            # gate is True  -> corroborated, accept at the lower threshold.
            # gate is None  -> no signal to check, require the stricter threshold.
            threshold = FUZZY_TITLE_GATED_THRESHOLD if gate is True else FUZZY_TITLE_THRESHOLD
            if score >= threshold:
                return self._to_resolved(c, MatchMethod.FUZZY_TITLE, round(score / 100.0, 3))

        return None

    # ── gate + assembly ───────────────────────────────────────
    @staticmethod
    def _gate(c: Candidate, reference: str, ref_year: int | None) -> bool | None:
        """Tri-state author/year corroboration for a fuzzy-title match.

        The author check looks for the CANDIDATE's surnames (cleanly parsed from
        API metadata) in the raw reference text — rather than parsing the
        reference's own author list, which mangles ``Last, First`` order and middle
        initials (``Diederik P. Kingma`` -> ``p``) and used to veto perfect title
        matches. The year guard is unchanged. Together they still block the
        token-subset false positives that ``_fuzzy_title_score`` can over-score.

        Returns:
            True  — the reference names the cited work's authors (>=2, or >=half),
                    or the year agrees (±1); and nothing contradicts.
            False — a checkable signal CONTRADICTS: the candidate has >=2 authors
                    and NONE appear in the reference (a different work), or the year
                    is off by > 1. Hard reject.
            None  — neither author nor year was checkable.
        """
        cand_surnames = {s for s in _author_surnames(c.authors) if s}
        ref_tokens = set(_norm_title(reference).split())

        author_state: bool | None = None
        if cand_surnames and ref_tokens:
            present = sum(1 for s in cand_surnames if s in ref_tokens)
            if present >= 2 or (present and present / len(cand_surnames) >= 0.5):
                author_state = True
            elif present == 0 and len(cand_surnames) >= 2:
                author_state = False  # the cited work's authors are absent -> different work

        year_state: bool | None = None
        if ref_year is not None and c.year is not None:
            year_state = abs(c.year - ref_year) <= 1

        if author_state is False or year_state is False:
            return False
        if author_state is True or year_state is True:
            return True
        return None

    def _to_resolved(self, c: Candidate, method: MatchMethod, score: float) -> Resolved:
        url_valid: bool | None = None
        if self.validate_urls and c.url:
            url_valid = paper_lookup.validate_url(c.url)
        return Resolved(
            source=c.source,
            match_method=method,
            match_score=score,
            title=c.title or None,
            authors=list(c.authors),
            year=c.year,
            venue=c.venue or None,
            doi=c.doi or None,
            arxiv_id=c.arxiv_id or None,
            url=c.url or None,
            url_valid=url_valid,
            abstract=c.abstract or None,
        )


# ───────────────────────────────────────────────────────────────
# Module-level reference parsers (shared with stages via the resolver)
# ───────────────────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _extract_year(reference: str) -> int | None:
    """Pull the most plausible publication year from a reference string."""
    matches = _YEAR_RE.findall(reference or "")
    if not matches:
        return None
    return max(int(m) for m in matches)  # references often end with the pub year


def _dict_to_candidate(d: dict) -> Candidate:
    """Adapt a :mod:`paper_lookup` candidate dict to a :class:`Candidate`."""
    return Candidate(
        source=str(d.get("source", "")),
        title=str(d.get("title", "") or ""),
        authors=list(d.get("authors", []) or []),
        year=d.get("year"),
        venue=str(d.get("venue", "") or ""),
        doi=str(d.get("doi", "") or ""),
        arxiv_id=str(d.get("arxiv_id", "") or ""),
        url=str(d.get("url", "") or ""),
        abstract=str(d.get("abstract", "") or ""),
        extra={k: v for k, v in d.items() if k == "type"},
    )


def _fetch(fn: Any, *args: Any) -> list[Candidate]:
    """Call a :mod:`paper_lookup` search/fetch fn, wrap its dicts as Candidates.

    Returns ``[]`` on any error so every source stays fail-soft.
    """
    try:
        return [_dict_to_candidate(d) for d in fn(*args)]
    except Exception:  # noqa: BLE001 — every source fails soft
        return []
