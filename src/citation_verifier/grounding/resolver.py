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
import threading
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

# Fuzzy-title acceptance threshold with no author/year corroboration.
FUZZY_TITLE_THRESHOLD = 95.0
# Minimum acceptance when the gate (author/year) is satisfied but title is softer.
FUZZY_TITLE_GATED_THRESHOLD = 78.0
# Publication year is corroborative only; metadata often differs across arXiv,
# online-first, proceedings, and journal versions.
YEAR_CORROBORATION_WINDOW = 2

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


def _surname_matches_token(surname: str, token: str) -> bool:
    """Loose surname match for PDF glue like ``YangZ`` vs ``Yang``."""
    surname = surname.lower().strip()
    token = token.lower().strip()
    if not surname or not token:
        return False
    if surname == token:
        return True
    if (
        min(len(surname), len(token)) >= 4
        and (surname.startswith(token) or token.startswith(surname))
    ):
        return True
    return False


def _extract_doi(reference: str) -> str:
    m = _DOI_RE.search(reference or "")
    return m.group(1).rstrip(".").lower() if m else ""


def _extract_arxiv_id(reference: str) -> str:
    m = _ARXIV_RE.search(reference or "")
    return m.group(1).lower() if m else ""


# A quoted segment in a reference is almost always the title (e.g.
# `Authors. "Title". Venue. Year`). Straight or curly quotes, >= 8 chars.
# A quoted title. Double-quoted form allows apostrophes inside (a curly ' / U+2019
# in "Let's verify step by step" must NOT be read as a closing quote — that left
# apostrophe-bearing titles unextractable → unresolved). Single-quoted form is the
# secondary alternative.
_QUOTED_TITLE_RE = re.compile(r"[\"“]([^\"“”]{8,})[\"”]|‘([^‘’]{8,})’")
_SITE_SUFFIX_RE = re.compile(r"\s+-{2,}\s*[\w.-]+\.[a-z]{2,}\s*$", re.IGNORECASE)
_VENUE_CLAUSE_RE = re.compile(
    r"^(?:"
    r"in:?\s+|"
    r"proceedings\s+of(?:\s+the)?\s+|"
    r"proc\.?\s+|"
    r"journal\s+of\s+|"
    r"advances\s+in\s+neural\s+information\s+processing\s+systems\b"
    r")",
    re.IGNORECASE,
)
_TITLE_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "toward",
    "towards",
    "via",
    "with",
    "without",
}
_ARXIV_HINTS = {
    "arxiv",
    "preprint",
    "attention",
    "bert",
    "chatgpt",
    "diffusion",
    "embedding",
    "gpt",
    "language",
    "llm",
    "model",
    "models",
    "neural",
    "transformer",
}
_DBLP_HINTS = {
    "aaai",
    "acl",
    "acm",
    "algorithm",
    "artificial",
    "chatbot",
    "comput",
    "computer",
    "conference",
    "cvf",
    "cvpr",
    "emnlp",
    "ieee",
    "ijcai",
    "intelligence",
    "language",
    "neurips",
    "nlp",
    "proceedings",
    "sigir",
    "transformer",
}


# Vancouver-style author token: a (possibly hyphenated/multi-word) surname
# followed by 1-4 given-name initials with no separating periods — "Colby KM",
# "Gebru T", "Wardrip-Fruin N", "Lee J-S". Used to spot 2-author lists that the
# comma / spaced-single-initial heuristics below miss.
_VANCOUVER_AUTHOR_RE = re.compile(
    r"[A-Z][A-Za-z'’.\-]+"               # surname (hyphen/apostrophe ok)
    r"(?:\s+[A-Z][A-Za-z'’.\-]+){0,2}"   # optional extra surname words (Van Der Berg)
    r"\s+[A-Z](?:[-.]?[A-Z]){0,3}\.?"    # 1-4 initials: KM, FD, T, J-S, J.S.
)


def _looks_like_author_clause(clause: str) -> bool:
    """Return True for the leading author-list segment of a reference."""
    c = clause.strip()
    if not c:
        return False
    if c.count(",") >= 2:
        return True
    # Surname-initial lists often have several one-letter initials before title.
    initials = re.findall(r"\b[A-Z]\b", c)
    if len(initials) >= 3 and len(c.split()) <= 14:
        return True
    # Vancouver lists with exactly 2 authors have only ONE comma and multi-letter
    # initial blocks ("Colby KM, Hilf FD"), so the checks above miss them and the
    # author string gets mistaken for the title. Flag the clause when EVERY
    # comma-separated segment is a short surname+initials group. Require >=2
    # segments so a lone Title-Case title ending in an acronym ("… VQA") is never
    # misread as an author list.
    segs = [s.strip() for s in c.split(",") if s.strip()]
    if len(segs) >= 2 and all(
        len(s.split()) <= 4 and _VANCOUVER_AUTHOR_RE.fullmatch(s) for s in segs
    ):
        return True
    return False


def _looks_like_venue_clause(clause: str) -> bool:
    """Return True for common venue/proceedings clauses, not paper titles."""
    c = clause.strip(" .,")
    return bool(_VENUE_CLAUSE_RE.search(c))


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
        return (m.group(1) or m.group(2)).strip()
    head = re.split(r"(?i)\b(?:arxiv|doi|https?://)", ref)[0]
    for clause in re.split(r"\.\s+", head):
        c = clause.strip(" .,")
        if len(c) < 12 or re.search(r"\b(19|20)\d{2}\b", c):
            continue  # too short, or a year/venue tail
        if _looks_like_author_clause(c):
            continue  # looks like an author list
        if _looks_like_venue_clause(c):
            continue
        return c
    return ""


def _clean_title_variant(title: str) -> str:
    """Normalize a title for fielded metadata queries without changing meaning."""
    title = re.sub(r"\s+", " ", (title or "")).strip(" .")
    title = _SITE_SUFFIX_RE.sub("", title).strip(" .")
    return title


def _likely_titles(reference: str) -> list[str]:
    """Return queryable title variants, best first, deduped by normalized text."""
    title = _likely_title(reference)
    if not title:
        return []
    variants = [title, _clean_title_variant(title)]
    # A cited title may carry a colon-prefix the canonical record drops/changes
    # ("I Speak, You Verify: Toward trustworthy neural program synthesis"), so also
    # query the substantive part after the first colon.
    if ":" in title:
        after = title.split(":", 1)[1].strip()
        if len(after.split()) >= 3:
            variants.append(after)
    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        variant = variant.strip()
        key = _norm_title(variant)
        if len(variant) >= 6 and key and key not in seen:
            seen.add(key)
            out.append(variant)
    return out


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


def _title_tokens(title: str) -> set[str]:
    """Discriminative-ish tokens for guarding fuzzy title subset matches."""
    return {
        t
        for t in _norm_title(_clean_title_variant(title)).split()
        if len(t) > 1 and t not in _TITLE_TOKEN_STOPWORDS
    }


def _tokens(text: str) -> set[str]:
    return set(_norm_title(text).split())


def _should_search_arxiv(reference: str) -> bool:
    """Route arXiv title/broad search to likely preprint or modern ML/CS refs."""
    if _extract_arxiv_id(reference):
        return True
    ref = _norm_title(reference)
    if "arxiv" in ref or "preprint" in ref:
        return True
    year = _extract_year(reference)
    if year is not None and year < 2007:
        return False
    toks = _tokens(reference)
    return bool(toks & _ARXIV_HINTS)


def _should_search_dblp(reference: str) -> bool:
    """Route DBLP broad search to likely CS/NLP/conference references."""
    toks = _tokens(reference)
    return bool(toks & _DBLP_HINTS)


def _has_digit(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _title_tokens_contradict(reference: str, candidate_title: str) -> bool:
    """True when the candidate is missing too much of the cited title.

    ``token_set_ratio`` is intentionally recall-heavy, but it can over-score a
    generic subset ("OpenAI system card") while missing version-bearing tokens
    ("o3", "o4-mini"). This gate uses the extracted title only, not the whole
    reference, and rejects those subset false positives before author/year gates.
    """
    ref_titles = _likely_titles(reference)
    if not ref_titles:
        return False
    ref_tokens = _title_tokens(ref_titles[0])
    cand_tokens = _title_tokens(candidate_title)
    if not ref_tokens or not cand_tokens:
        return False

    numbered = {t for t in ref_tokens if _has_digit(t)}
    if numbered and not (numbered & cand_tokens):
        return True

    overlap = len(ref_tokens & cand_tokens) / len(ref_tokens)
    if len(ref_tokens) <= 2:
        return overlap < 1.0
    if len(ref_tokens) <= 4:
        return overlap < (2 / 3)
    return overlap < 0.60


def _ref_first_surname(reference: str) -> str:
    """The reference's leading surname, folded — 'Hindle, A., …' -> 'hindle'."""
    m = re.match(r"\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]{1,})", reference or "")
    return m.group(1).lower() if m else ""


def _cand_first_surname(c) -> str:
    """A candidate's first-author surname, folded — 'Premkumar T. Devanbu' -> 'devanbu'."""
    authors = getattr(c, "authors", None) or []
    if not authors:
        return ""
    toks = re.sub(r"[^A-Za-z\s'’-]", " ", authors[0]).split()
    return toks[-1].lower() if toks else ""


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
        self._fetch_cache: dict[tuple[str, tuple[Any, ...]], list[Candidate]] = {}
        self._fetch_cache_lock = threading.RLock()

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

    def _fetch(self, fn: Any, *args: Any) -> list[Candidate]:
        """Fetch candidates once per resolver instance for identical source args."""
        key = (getattr(fn, "__name__", repr(fn)), args)
        with self._fetch_cache_lock:
            cached = self._fetch_cache.get(key)
        if cached is not None:
            return list(cached)

        rows = _fetch(fn, *args)
        with self._fetch_cache_lock:
            self._fetch_cache[key] = rows
        return list(rows)

    def _id_candidates(self, reference: str) -> list[Candidate]:
        """Tier 1: direct lookup by the reference's own DOI / arXiv id (no search)."""
        rows: list[Candidate] = []
        for fn, args in self._id_query_steps(reference):
            rows += self._fetch(fn, *args)
        return rows

    def _id_query_steps(self, reference: str) -> list[tuple[Any, tuple[Any, ...]]]:
        """Exact identifier lookups, ordered by abstract-bearing source first."""
        rows: list[tuple[Any, tuple[Any, ...]]] = []
        s2_key = getattr(self.settings, "s2_api_key", None)
        ref_arxiv = _extract_arxiv_id(reference)
        if ref_arxiv:
            rows.append((paper_lookup.fetch_semantic_scholar_by_arxiv, (ref_arxiv, s2_key)))
            rows.append((paper_lookup.fetch_arxiv_by_id, (ref_arxiv,)))
        ref_doi = _extract_doi(reference)
        if ref_doi:
            rows.append((paper_lookup.fetch_semantic_scholar_by_doi, (ref_doi, s2_key)))
            rows.append((paper_lookup.fetch_crossref_by_doi, (ref_doi,)))
        return rows

    def _title_candidates(self, reference: str, max_results: int = 4) -> list[Candidate]:
        """Tier 2: precise title-field queries — S2 match, arXiv ti:+au:, Crossref.

        Querying the extracted TITLE (+ first author), not the noisy full reference,
        surfaces the canonical paper where a reused title ("Attention is all you
        need") otherwise returns same-title namesakes.
        """
        rows: list[Candidate] = []
        for fn, args in self._title_query_steps(reference, max_results):
            rows += self._fetch(fn, *args)
        return _dedupe_candidates(rows)

    def _title_query_steps(
        self, reference: str, max_results: int = 4
    ) -> list[tuple[Any, tuple[Any, ...]]]:
        """Precise title-query steps, with author-constrained then title-only forms."""
        titles = _likely_titles(reference)
        if not titles:
            return []
        author = _first_author(reference) or None

        steps: list[tuple[Any, tuple[Any, ...]]] = []
        seen: set[tuple[str, tuple[Any, ...]]] = set()

        def add(fn: Any, *args: Any) -> None:
            key = (getattr(fn, "__name__", repr(fn)), args)
            if key not in seen:
                seen.add(key)
                steps.append((fn, args))

        use_arxiv = _should_search_arxiv(reference)
        for title in titles:
            add(paper_lookup.search_semantic_scholar_match, title)
            add(paper_lookup.search_crossref_by_fields, title, author, max_results)
            if use_arxiv:
                add(paper_lookup.search_arxiv_by_title, title, max_results, author)
            if author:
                add(paper_lookup.search_crossref_by_fields, title, None, max_results)
                if use_arxiv:
                    add(paper_lookup.search_arxiv_by_title, title, max_results, None)
        return steps

    def _broad_candidates(self, reference: str, max_results: int = 4) -> list[Candidate]:
        """Tier 3: broad full-reference search across every enabled source."""
        rows: list[Candidate] = []
        for src in self._broad_sources(reference):
            try:
                raw = self._search_source(src, reference, max_results)
            except Exception:  # noqa: BLE001 — defense in depth; sources fail soft
                raw = []
            rows += [_dict_to_candidate(d) for d in raw]
        return rows

    def _broad_sources(self, reference: str) -> tuple[str, ...]:
        """Source routing for the expensive broad full-reference fallback."""
        out: list[str] = []
        for src in self.sources:
            if src == paper_lookup.SOURCE_ARXIV and not _should_search_arxiv(reference):
                continue
            if src == paper_lookup.SOURCE_DBLP and not _should_search_dblp(reference):
                continue
            out.append(src)
        return tuple(out)

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
        # 1) identifier, abstract-bearing first; stop at the first exact match.
        for fn, args in self._id_query_steps(reference):
            match = self._match(reference, self._fetch(fn, *args))
            if match is not None:
                return match

        # 2) title-field, most-precise-first; stop the moment one matches.
        for fn, args in self._title_query_steps(reference, 4):
            match = self._match(reference, self._fetch(fn, *args))
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

        # An exact DOI/arXiv id is accepted ONLY if the title doesn't strongly
        # contradict the cited one — a real-but-wrong id (typo, or a fabricated
        # citation borrowing a live id) points at a different paper, which must be
        # 'unresolved', not a confident 'yes'.
        # 1) DOI exact match.
        if ref_doi:
            for c in cands:
                if c.doi and c.doi.lower() == ref_doi and not _title_tokens_contradict(reference, c.title):
                    return self._to_resolved(c, MatchMethod.DOI, 1.0)

        # 2) arXiv-id exact match.
        if ref_arxiv:
            stem = ref_arxiv.split("v")[0]
            for c in cands:
                if (
                    c.arxiv_id
                    and c.arxiv_id.lower().split("v")[0] == stem
                    and not _title_tokens_contradict(reference, c.title)
                ):
                    return self._to_resolved(c, MatchMethod.ARXIV, 1.0)

        # 3) Fuzzy title, corroborated (not vetoed) by author overlap + year.
        #    Bibliography author/year fields are often abbreviated, stale, or
        #    version-specific (arXiv vs proceedings vs journal), so title-token
        #    contradiction is the hard precision gate; author/year just lower the
        #    acceptance threshold when they agree.
        # Rank by fuzzy title, then by first-author agreement with the reference
        # (so a same-title record by a different lead author loses to the right
        # one when both are returned), then by having an abstract.
        ref_first = _ref_first_surname(reference)
        ranked = sorted(
            (
                (_fuzzy_title_score(reference, c.title), _cand_first_surname(c) == ref_first, bool(c.abstract), c)
                for c in cands
            ),
            key=lambda t: (t[0], t[1], t[2]),
            reverse=True,
        )
        for score, _first_ok, _has_abs, c in ranked:
            if score < FUZZY_TITLE_GATED_THRESHOLD:
                break  # ranked descending: no later candidate can clear the bar
            if _title_tokens_contradict(reference, c.title):
                continue
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
        initials (``Diederik P. Kingma`` -> ``p``). Author/year agreement is
        positive evidence only: metadata sources disagree on arXiv, online,
        conference, and journal years, and parsed author lists can be noisy.
        Token-subset false positives are blocked by ``_title_tokens_contradict``.

        Returns:
            True  — the reference names the cited work's first author / enough
                    authors, or the year agrees within the configured window.
            False — reserved for future hard metadata contradictions.
            None  — no positive corroboration; a missing author overlap alone is
                    too noisy to veto a strong title match.
        """
        cand_surnames = {s for s in _author_surnames(c.authors) if s}
        ref_tokens = set(_norm_title(reference).split())

        author_state: bool | None = None
        if cand_surnames and ref_tokens:
            first = _last_name(c.authors[0]) if c.authors else ""
            present = sum(
                1
                for s in cand_surnames
                if any(_surname_matches_token(s, t) for t in ref_tokens)
            )
            if (
                present >= 2
                or (first and any(_surname_matches_token(first, t) for t in ref_tokens))
                or (present and present / len(cand_surnames) >= 0.5)
            ):
                author_state = True
            elif present == 0 and len(cand_surnames) >= 2:
                author_state = False  # soft negative; do not veto title/year matches

        year_state: bool | None = None
        if ref_year is not None and c.year is not None:
            year_state = abs(c.year - ref_year) <= YEAR_CORROBORATION_WINDOW

        if author_state is True or year_state is True:
            return True
        return None

    def _to_resolved(self, c: Candidate, method: MatchMethod, score: float) -> Resolved:
        # Prefer S2's curated open-access PDF as the resolved URL: it points at the
        # actual paper text (so Stage-2 relevance can fetch full text for off-arXiv
        # papers, see stages/relevance.py) and is a strictly more useful "source"
        # link than a landing page. Falls back to the candidate's own URL.
        url = (c.extra or {}).get("oa_pdf") or c.url or None
        url_valid: bool | None = None
        if self.validate_urls and url:
            url_valid = paper_lookup.validate_url(url)
        abstract = c.abstract or self._abstract_top_up(c)
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
            url=url,
            url_valid=url_valid,
            abstract=abstract or None,
        )

    def _abstract_top_up(self, c: Candidate) -> str:
        """Fill a missing abstract for an already-resolved candidate.

        Only ever reached when ``c.abstract`` is empty (see :meth:`_to_resolved`),
        so it never re-fetches an abstract we already have. The cascade prefers the
        paper's real text over an AI summary, and locates the paper by **exact
        identifier only** — never a title search, which would surface a near-namesake
        (e.g. GPT-2's "… Unsupervised Multitask Learners" vs the 2024
        "Instruction Pre-Training: … Supervised Multitask Learners") and feed the
        wrong paper's evidence to the judge:

          1. arXiv id  -> the real arXiv abstract, by exact id.
          2. DOI       -> OpenAlex's reconstructed abstract, by exact DOI.
          3. S2 tldr   -> S2's one-line AI summary (carried in ``extra``), the only
                          API-accessible evidence for off-arXiv reports whose
                          licensed abstract S2 withholds (GPT-2 / GPT-1).
        """
        # 1) arXiv abstract by exact id (S2's ArXiv externalId or a cited arXiv id).
        if c.arxiv_id:
            for cand in self._fetch(paper_lookup.fetch_arxiv_by_id, c.arxiv_id):
                if cand.abstract:
                    return cand.abstract
        # 2) OpenAlex abstract by exact DOI.
        if c.doi:
            oa_key = getattr(self.settings, "openalex_api_key", None)
            for cand in self._fetch(paper_lookup.fetch_openalex_by_doi, c.doi, oa_key):
                if cand.abstract and cand.doi.lower() == c.doi.lower():
                    return cand.abstract
        # 3) S2 tldr fallback — already fetched during resolution, so no extra call.
        return (c.extra or {}).get("tldr") or ""


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
        extra={
            k: v
            for k, v in d.items()
            if k == "type" or (k in ("tldr", "oa_pdf") and v)
        },
    )


def _fetch(fn: Any, *args: Any) -> list[Candidate]:
    """Call a :mod:`paper_lookup` search/fetch fn, wrap its dicts as Candidates.

    Returns ``[]`` on any error so every source stays fail-soft.
    """
    try:
        return [_dict_to_candidate(d) for d in fn(*args)]
    except Exception:  # noqa: BLE001 — every source fails soft
        return []


def _dedupe_candidates(cands: list[Candidate]) -> list[Candidate]:
    """Keep source fan-out from returning the same canonical row several times."""
    out: list[Candidate] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for c in cands:
        key = (
            c.source,
            _norm_title(c.title),
            c.doi.lower(),
            c.arxiv_id.lower().split("v")[0],
            c.url,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
