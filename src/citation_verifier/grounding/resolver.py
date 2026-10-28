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
import unicodedata
from typing import TYPE_CHECKING, Any

from ..interfaces import Candidate
from ..schema import MatchMethod, Resolved
from . import paper_lookup, resolve_cache

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
    """Lowercase, strip punctuation/whitespace for title comparison.

    Maths markup is dropped rather than spaced ("$L_0$" -> "l0"), and a token whose
    subscript was glued to the next word by the PDF ("l0regularization") is split, so both
    spellings of "Learning sparse neural networks through L0 regularization" agree.
    """
    if not s:
        return ""
    s = re.sub(r"[_$\\{}]", "", s.lower())
    s = _GLUED_SUBSCRIPT_RE.sub(r"\1 \2", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# A letter run and a digit glued to a following word: "l0regularization", "gpt4turbo".
_GLUED_SUBSCRIPT_RE = re.compile(r"\b([A-Za-z]+\d+)([a-z]{4,})\b")


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
    # Venue names. Papers at these venues are overwhelmingly mirrored on arXiv, but a
    # reference that cites the proceedings ("… ICML. 2021") mentions neither "arxiv" nor
    # a topic keyword, so arXiv was never queried for them. Measured on one paper's
    # bibliography: 41 of 59 references skipped arXiv entirely. That is invisible while
    # Semantic Scholar answers, and collapses resolution the moment S2 rate-limits.
    "icml", "iclr", "neurips", "nips", "aistats", "uai", "colt", "tmlr", "jmlr",
    "cvpr", "iccv", "eccv", "wacv", "bmvc", "siggraph",
    "acl", "emnlp", "naacl", "eacl", "coling", "conll", "tacl",
    "aaai", "ijcai", "kdd", "sigir", "www", "wsdm", "recsys",
    "icra", "iros", "rss", "corl",
    "usenix", "ndss", "ccs", "oakland", "asplos", "osdi", "sosp", "nsdi", "sigcomm",
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
    # "Given Surname and Given Surname" — an ACL/ICLR two-author list has NO comma, no
    # single-letter initials and one segment, so every check above misses it and the
    # author names are taken for the title ("Shahriar Golchin and Mihai Surdeanu"), which
    # makes the reference unresolvable. Each name group must be 2-4 capitalised words so
    # a genuine short title joined by "and" ("Vision and Language") is not swallowed.
    groups = [g.strip() for g in re.split(r"\s*(?:,|\band\b|&)\s*", c) if g.strip()]
    if 2 <= len(groups) <= 6 and all(
        2 <= len(g.split()) <= 4 and all(w[:1].isupper() for w in g.split() if w[:1].isalpha())
        for g in groups
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
    """Normalize a title for fielded metadata queries without changing meaning.

    A volume or edition printed after a book's title ("Deep learning, volume 1",
    "…, 2nd edition") is not part of it; searched with it, the indexes returned
    proceedings volumes named "…, Volume 1".
    """
    title = re.sub(r"\s+", " ", (title or "")).strip(" .")
    title = _SITE_SUFFIX_RE.sub("", title).strip(" .")
    while True:
        cut = _TITLE_LOCATOR_RE.sub("", title).strip(" .,;")
        if cut == title or len(cut) < 6:
            return title
        title = cut


def clean_title(title: str | None) -> str:
    """A cited title without its volume / edition locator ("Deep learning, volume 1" ->
    "Deep learning"), as the indexes print it."""
    return _clean_title_variant(title or "")


# A volume / edition locator at the end of a title.
_TITLE_LOCATOR_RE = re.compile(
    r"[,;(]?\s*(?:vol(?:ume)?\.?\s*\d+|\d+(?:st|nd|rd|th)\s+ed(?:ition|\.)?|"
    r"(?:second|third|fourth|fifth|revised)\s+edition|ed\.\s*\d+)\)?\s*$",
    re.IGNORECASE,
)


def _repeated_title(title: str) -> str:
    """The phrase a garbled title repeats, or ``""``: a title copied from a broken record
    ("Série scientifique … no unbiased estimator of the variance of k-fold
    cross-validation no unbiased estimator of the variance of k-fold cross-validation")
    is searched as the work it names twice."""
    # Words compared without their punctuation: the two copies need not be spelt alike
    # ("cross-validation" in one, "crossvalidation" in the other).
    words = title.split()
    norm = [re.sub(r"[\W_]+", "", w.lower()) for w in words]
    for size in range(len(words) // 2, 3, -1):
        for i in range(len(words) - 2 * size + 1):
            phrase = norm[i : i + size]
            for j in range(i + size, len(words) - size + 1):
                if norm[j : j + size] == phrase:
                    return " ".join(words[i : i + size])
    return ""


def _repeated_title_reference(reference: str) -> str:
    """``reference`` with its title replaced by the phrase the title repeats, or ``""``
    when the title does not repeat one."""
    title = _likely_title(reference)
    phrase = _repeated_title(_clean_title_variant(title)) if title else ""
    if not phrase or title not in reference:
        return ""
    return reference.replace(title, phrase, 1)


def _likely_titles(reference: str) -> list[str]:
    """Return queryable title variants, best first, deduped by normalized text."""
    title = _likely_title(reference)
    if not title:
        return []
    cleaned = _clean_title_variant(title)
    variants = [cleaned, title, _GLUED_SUBSCRIPT_RE.sub(r"\1 \2", cleaned), _repeated_title(cleaned)]
    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        variant = variant.strip()
        # Deduped on the words as an index reads them: "l0regularization" and
        # "l0 regularization" compare equal once matched, but are different searches.
        key = " ".join(re.sub(r"[^\w\s]", " ", variant.lower()).split())
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
    if toks & _ARXIV_HINTS:
        return True
    # A bibliography that spells the venue out rather than abbreviating it —
    # "Proceedings of the IEEE/CVF International Conference on Computer Vision" instead of
    # "ICCV" — carries none of the acronyms above, so arXiv was never queried for it even
    # though the paper is mirrored there. Both spellings are common in one bibliography.
    if _SPELLED_OUT_VENUE_RE.search(reference or ""):
        return True
    # …and a third spelling is the abbreviated one a journal house style imposes ("Adv
    # Neural Inf Proces Syst", "Proc IEEE Conf Comput Vis Pattern Recognit"), which
    # shares no whole word with either list above. The venue alias table reads all three.
    return _cites_known_conference(reference)


def _cites_known_conference(reference: str) -> bool:
    """True when some clause of the reference names a conference the alias table knows."""
    from .. import venues  # noqa: PLC0415 — pure, but keeps this module's import light

    for clause in re.split(r"[.;]\s+|\bIn:?\s+", reference or ""):
        clause = clause.strip(" .,;:")
        if not (1 <= len(clause.split()) <= 16):
            continue  # an author list or a long title, not a venue clause
        if any(venues.venue_kind(v) == "conference" for v in venues.venue_ids(clause)):
            return True
    return False


# Venue names written out in full. Deliberately anchored on the conference/journal words
# together with a field word, so a title that merely mentions "computer vision" does not
# route every reference to arXiv.
_SPELLED_OUT_VENUE_RE = re.compile(
    r"\b(?:conference|symposium|workshop|proceedings|transactions|journal)\b[^.]{0,80}?"
    r"\b(?:machine\s+learning|computer\s+vision|pattern\s+recognition|artificial\s+"
    r"intelligence|computational\s+linguistics|natural\s+language|neural\s+information|"
    r"learning\s+representations|robotics|automation|knowledge\s+discovery|data\s+mining|"
    r"information\s+retrieval|robot\s+learning|empirical\s+methods)\b",
    re.IGNORECASE,
)


def _looks_like_book(reference: str) -> bool:
    """A reference to a book: a publisher, an ISBN, an edition, a volume of a press."""
    return bool(_BOOK_RE.search(reference or ""))


_BOOK_RE = re.compile(
    r"\b(?:[A-Z][\w&.-]*\s+)?(?:Press|Publishers?|Publishing)\b|\bISBN\b|\b\d+(?:st|nd|rd|th)\s+ed(?:ition|\.)|"
    r"\bedition\b|\b(?:Springer|Wiley|Elsevier|O'Reilly|Addison[- ]Wesley|Prentice[- ]Hall|McGraw[- ]Hill|"
    r"Pearson|Morgan\s+Kaufmann|CRC|Chapman\s+(?:and|&)\s+Hall)\b",
    re.IGNORECASE,
)


def _should_search_openreview(reference: str) -> bool:
    """A reference to a venue whose papers live on OpenReview (workshops above all)."""
    return bool(_OPENREVIEW_RE.search(reference or ""))


_OPENREVIEW_RE = re.compile(
    r"\b(?:openreview|workshop|ICLR|TMLR|Transactions\s+on\s+Machine\s+Learning\s+Research|"
    r"International\s+Conference\s+on\s+Learning\s+Representations|NeurIPS|COLM|CoRL)\b",
    re.IGNORECASE,
)


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


def _subtitle_reference(reference: str) -> str:
    """A reference variant whose colon-prefixed title is reduced to its post-colon
    part ("I Speak, You Verify: Toward …" -> "Toward …"), for a fallback search.

    Returns ``""`` when the title has no colon or the subtitle is too short to be
    a reliable query.
    """
    title = _likely_title(reference)
    if not title or ":" not in title:
        return ""
    subtitle = title.split(":", 1)[1].strip()
    if len(subtitle.split()) < 3:
        return ""
    return reference.replace(title, subtitle, 1)


def _fold(text: str) -> str:
    """Lowercase with diacritics removed — "Müller" and "Muller" are one surname."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _ref_first_surname(reference: str) -> str:
    """The reference's leading surname, folded — 'Hindle, A., …' -> 'hindle'."""
    m = re.match(r"\s*([^\W\d_][^\W\d_'’-]{1,})", reference or "")
    return _fold(m.group(1)) if m else ""


def _cand_first_surname(c) -> str:
    """A candidate's first-author surname, folded — 'Premkumar T. Devanbu' -> 'devanbu'.

    Handles the ``Last, First`` form some sources return, and keeps accented letters (an
    ASCII-only filter read "Müller" as "ller").
    """
    authors = getattr(c, "authors", None) or []
    if not authors:
        return ""
    name = authors[0]
    if "," in name:
        name = name.split(",", 1)[0]
    toks = re.sub(r"[^\w\s'’-]|[\d_]", " ", name).split()
    return _fold(toks[-1]) if toks else ""


def _ref_lead_author_tokens(reference: str) -> set[str]:
    """The name tokens of a reference's FIRST author, folded.

    References open with the first author in whatever order the style prints it —
    "Ashish Vaswani, …", "A. Vaswani, …", "Vaswani, A., …", "Vaswani A, …" — so the
    surname is not reliably the first word (:func:`_ref_first_surname` assumes it is,
    and for every given-name-first bibliography it returned the given name). The surname
    is, however, always inside the first name chunk; a chunk of bare initials means the
    surname follows the comma, so the next chunk is taken too.
    """
    head = re.split(r"[\"“‘]", reference or "", maxsplit=1)[0][:160]
    chunks = [c for c in re.split(r"\s*(?:,|;|\band\b|&)\s*", head) if c.strip()]
    tokens: set[str] = set()
    for chunk in chunks[:2]:
        words = [_fold(w) for w in re.findall(r"[^\W\d_]{2,}", chunk)][:4]
        tokens.update(words)
        if any(len(w) > 2 for w in words) and chunk is chunks[0] and len(words) >= 2:
            break  # a full name ("Ashish Vaswani"): the surname is in it
    return tokens


def _first_author_agrees(reference: str, c) -> bool | None:
    """Does the candidate's first author open the reference? ``None`` when unknowable."""
    surname = _cand_first_surname(c)
    lead = _ref_lead_author_tokens(reference)
    if not surname or not lead:
        return None
    return any(_surname_matches_token(surname, t) for t in lead)


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


_HOST_NAMES = {
    "api.crossref.org": "Crossref",
    "export.arxiv.org": "arXiv",
    "arxiv.org": "arXiv (paper pages)",
    "dblp.org": "DBLP",
    "api.semanticscholar.org": "Semantic Scholar",
    "api.openalex.org": "OpenAlex",
    "openlibrary.org": "Open Library",
    "api2.openreview.net": "OpenReview",
}

# Which source a lookup function asks, for the record of what a resolution searched.
_FN_SOURCE = (
    ("openalex", "OpenAlex"),
    ("openlibrary", "Open Library"),
    ("openreview", "OpenReview"),
    ("semantic_scholar", "Semantic Scholar"),
    ("crossref", "Crossref"),
    ("arxiv", "arXiv"),
    ("dblp", "DBLP"),
)
_SRC_NAMES = {
    paper_lookup.SOURCE_CROSSREF: "Crossref",
    paper_lookup.SOURCE_ARXIV: "arXiv",
    paper_lookup.SOURCE_DBLP: "DBLP",
    paper_lookup.SOURCE_S2: "Semantic Scholar",
    paper_lookup.SOURCE_OPENALEX: "OpenAlex",
}


# ───────────────────────────────────────────────────────────────
# The resolver
# ───────────────────────────────────────────────────────────────
def _arxiv_from_siblings(winner: Candidate, siblings: list[Candidate] | None) -> str:
    """The winner's arXiv id, recovered from another source that returned the same paper.

    A conference paper resolves to its publisher record — Crossref's ``10.1109/…`` — which
    carries a DOI and no arXiv id, while Semantic Scholar returns the same paper WITH its
    arXiv id in the very same fetch. Dropping it costs the full text: the DOI link is a
    landing page, so the paper is reduced to its abstract even though a preprint is one
    request away. Measured: nine cells of one table, whose citations are almost all
    IEEE-published, came back "we only saw the abstract".

    Gated on an exact normalised-title match, never a fuzzy one — a near-namesake's arXiv
    id would send the reader, and the judge, to the wrong paper.
    """
    if winner.arxiv_id or not siblings:
        return ""
    want = _norm_title(winner.title)
    if not want:
        return ""
    for other in siblings:
        if other is winner or not other.arxiv_id:
            continue
        if _norm_title(other.title) == want:
            return other.arxiv_id
    return ""


def _abstract_from_siblings(winner: Candidate, siblings: list[Candidate] | None) -> str:
    """The abstract of a same-title record returned beside ``winner``, when it has none.

    Same gate as :func:`_arxiv_from_siblings`: the normalised titles are equal.
    """
    if _readable(winner.abstract, winner.title) or not siblings:
        return ""
    want = _norm_title(winner.title)
    if not want:
        return ""
    for other in siblings:
        if other is not winner and _readable(other.abstract, other.title) and _norm_title(other.title) == want:
            return other.abstract
    return ""


def _is_thin(m: Resolved) -> bool:
    """A record with nothing to read: no abstract (a placeholder line is none), and no arXiv
    id to fetch the text by."""
    return not (_readable(m.abstract, m.title) or m.arxiv_id)


def _readable(abstract: str | None, title: str | None = None) -> bool:
    """An abstract of the work, not a placeholder (see
    :func:`paper_lookup.substantive_abstract`)."""
    return paper_lookup.substantive_abstract(abstract, title)


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
        self._failed = threading.local()  # per-thread: resolve() runs on worker threads

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
        name = getattr(fn, "__name__", repr(fn))
        self._asked(next((src for tag, src in _FN_SOURCE if tag in name), ""))
        key = (name, args)
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
        """Exact identifier lookups: the identifier's own registry first (arXiv for an arXiv
        id, Crossref for a DOI), then Semantic Scholar.

        S2 used to be asked first, for its abstracts. It is the index most often rate-limited,
        and the registries hold the record outright; a registry record without an abstract
        has one filled in afterwards (:meth:`_abstract_top_up`), S2 among the sources.
        """
        rows: list[tuple[Any, tuple[Any, ...]]] = []
        s2_key = getattr(self.settings, "s2_api_key", None)
        ref_arxiv = _extract_arxiv_id(reference)
        if ref_arxiv:
            rows.append((paper_lookup.fetch_arxiv_by_id, (ref_arxiv,)))
            rows.append((paper_lookup.fetch_semantic_scholar_by_arxiv, (ref_arxiv, s2_key)))
        ref_doi = _extract_doi(reference)
        if ref_doi:
            rows.append((paper_lookup.fetch_crossref_by_doi, (ref_doi,)))
            rows.append((paper_lookup.fetch_semantic_scholar_by_doi, (ref_doi, s2_key)))
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
        year = _extract_year(reference)
        oa_key = getattr(self.settings, "openalex_api_key", None)
        for i, title in enumerate(titles):
            add(paper_lookup.search_semantic_scholar_match, title)
            add(paper_lookup.search_crossref_by_fields, title, author, max_results)
            if use_arxiv:
                add(paper_lookup.search_arxiv_by_title, title, max_results, author)
            if author:
                add(paper_lookup.search_crossref_by_fields, title, None, max_results)
                if use_arxiv:
                    add(paper_lookup.search_arxiv_by_title, title, max_results, None)
            # Fielded OpenAlex (it holds books, working papers and journals the others
            # miss) for every wording — the one a garbled title repeats is often the only
            # one it can find — then the indexes for what is neither a paper nor a preprint.
            add(paper_lookup.search_openalex_by_title, title, author, year, max_results, oa_key)
            if i == 0:
                if _looks_like_book(reference):
                    add(paper_lookup.search_openlibrary, title, author, year, max_results)
                if _should_search_openreview(reference):
                    add(paper_lookup.search_openreview, title, max_results)
        return steps

    def _asked(self, source: str) -> None:
        asked = getattr(self._failed, "asked", None)
        if asked is not None and source and source not in asked:
            asked.append(source)

    def searched_sources(self) -> list[str]:
        """Sources this thread's most recent :meth:`resolve` asked (cache hits ask none)."""
        return list(getattr(self._failed, "asked", []))

    def _broad_candidates(self, reference: str, max_results: int = 4) -> list[Candidate]:
        """Tier 3: broad full-reference search across every enabled source."""
        rows: list[Candidate] = []
        for src in self._broad_sources(reference):
            self._asked(_SRC_NAMES.get(src, src))
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

        Successful resolutions are cached on disk (see
        :mod:`citation_verifier.grounding.resolve_cache`) — the cascade costs seconds per
        reference and bibliographic records do not change. Set
        ``CITATION_VERIFIER_CACHE=""`` to disable.
        """
        self._failed.hosts = []
        self._failed.asked = []
        cached = resolve_cache.read(reference, Resolved)
        if cached is not None:
            return cached
        paper_lookup.reset_failures()
        found = self._resolve_uncached(reference)
        self._failed.hosts = paper_lookup.failures()
        if found is not None and not (self._failed.hosts and self._second_best(reference, found)):
            resolve_cache.write(reference, found)
        return found

    @staticmethod
    def _second_best(reference: str, found: Resolved) -> bool:
        """A hit that the index which failed might well have bettered: nothing to read,
        dated away from the citation, or by another first author. Cached, it would outlive
        the outage that caused it — every later run would be handed the same abstract-less
        record, the same repost, or the same namesake (a book's title found as another
        group's paper while the index holding the book was down). A miss is not cached for
        that reason; neither is this. Nor is a record without the arXiv id the reference
        prints: the arXiv record, which the failed source would have returned, carries the
        full text."""
        return (
            _is_thin(found)
            or _year_rank(_extract_year(reference), found.year) == 0
            or _first_author_agrees(reference, found) is False
            or (bool(_extract_arxiv_id(reference)) and not found.arxiv_id)
        )

    def failed_sources(self) -> list[str]:
        """Sources that did not answer during this thread's most recent :meth:`resolve`.

        Empty means every source that was asked replied — so an unresolved reference
        was looked for and not found. Non-empty means the lookup was incomplete, and
        "not found" would be a claim about the sources that the run cannot back.
        """
        return [_HOST_NAMES.get(h, h) for h in getattr(self._failed, "hosts", [])]

    def _resolve_uncached(self, reference: str) -> Resolved | None:
        """:meth:`resolve` without the disk cache — the network cascade itself."""
        # A title copied from a broken record says its subject twice, after a prefix
        # that is no part of it. Matched as printed, the clean record of the work reads
        # as a contradiction (it lacks the prefix's words); so the work the title names
        # twice is looked for first, and the title as printed only if that finds nothing.
        named = _repeated_title_reference(reference)
        if named:
            found = self._cascade(named)
            if found is not None:
                return found
        return self._cascade(reference)

    def _cascade(self, reference: str) -> Resolved | None:
        """The identifier -> title -> broad -> fallback tiers for one reference string."""
        # 1) identifier, abstract-bearing first; stop at the first exact match.
        for fn, args in self._id_query_steps(reference):
            match = self._match(reference, self._fetch(fn, *args))
            if match is not None:
                return match

        # 2) title-field, most-precise-first. Short-circuit on a match whose FIRST
        #    author also agrees with the reference; otherwise keep searching (a
        #    same-title record by a different lead author — e.g. S2 returning
        #    "Devanbu" for the Hindle paper — must not win over the right one), and
        #    fall back to it only if nothing better turns up.
        #    Agreement is "the candidate's first-author surname is in the reference's
        #    first name", not "equals its first word": for a given-name-first
        #    bibliography the first word is a given name, the test never passed, and
        #    every such reference ran the whole tier — five lookups where one had
        #    already found the paper — only to return the first match anyway.
        #    The same goes for the YEAR: a record dated well away from the year the
        #    reference prints is a reprint, a repost or another edition of the cited
        #    work. It is kept in hand while the remaining steps look for the edition
        #    that was cited — with the index that usually has it rate-limited, the
        #    first record to turn up for a 2017 paper was a 2025 repost.
        ref_year = _extract_year(reference)

        def standing(m: Resolved) -> tuple[bool, bool]:
            return (
                _first_author_agrees(reference, m) is not False,
                _year_rank(ref_year, m.year) > 0,
            )

        #    And for what the record CARRIES: a record with no abstract and no arXiv id
        #    is the right work with nothing to read. With the index that usually answers
        #    first rate-limited, twenty of one paper's sixty citations were resolved to
        #    such records and came back "no abstract or introduction to judge from" —
        #    while the same papers sat on arXiv, one step further down this list. Such a
        #    match is held while the REST of the tier is asked — every source, not arXiv
        #    alone: a 1960s journal article's abstract is in Crossref and OpenAlex, a
        #    working paper's in OpenAlex, never on arXiv — and arXiv also when the routing
        #    had left it out, since the routing reads the title's vocabulary and half of
        #    those twenty titles carry none of its keywords. The first readable match
        #    that stands wins; else the answer is chosen among everything returned.
        fallback = None
        thin: Resolved | None = None
        pool: list[Candidate] = []
        asked_arxiv = False
        for fn, args in self._title_query_steps(reference, 4):
            cands = self._fetch(fn, *args)
            pool += cands
            asked_arxiv = asked_arxiv or fn is paper_lookup.search_arxiv_by_title
            match = self._match(reference, cands)
            if match is None:
                continue
            if all(standing(match)):
                if not _is_thin(match):
                    return match
                thin = thin or match
                continue
            if fallback is None or standing(match) > standing(fallback):
                fallback = match
        if thin is not None:
            if not asked_arxiv:
                for title in _likely_titles(reference)[:1]:
                    pool += self._fetch(
                        paper_lookup.search_arxiv_by_title, title, 4, _first_author(reference) or None
                    )
            # Over the whole pool a record that can be read is preferred — one step's
            # winner can be the unreadable twin of a readable record that step ranked
            # second ("sum" as cited, "Sums" with the abstract). Else the ranking runs
            # over everything, and a record that still wins takes the arXiv id (and
            # abstract) of a same-title record beside it.
            pool = _dedupe_candidates(pool)
            readable = [c for c in pool if c.arxiv_id or _readable(c.abstract, c.title)]
            best = self._match(reference, readable) if readable else None
            if best is not None and all(standing(best)):
                return best
            best = self._match(reference, pool)
            return best if best is not None and all(standing(best)) else thin
        if fallback is not None:
            if standing(fallback)[1]:
                return fallback  # the year stands; only the lead author is in doubt
            # Ask what has not been asked yet before settling for a copy: arXiv by title
            # when the routing skipped it (it holds the original of most papers that get
            # reposted), then the broad search.
            extra: list[Candidate] = []
            if not _should_search_arxiv(reference):
                for title in _likely_titles(reference)[:1]:
                    extra += self._fetch(
                        paper_lookup.search_arxiv_by_title, title, 4, _first_author(reference) or None
                    )
            other = self._match(reference, extra + self._broad_candidates(reference))
            if other is not None and standing(other) > standing(fallback):
                return other
            return fallback

        # 3) broad fallback.
        match = self._match(reference, self._broad_candidates(reference))
        if match is not None:
            return match

        # 4) colon-subtitle fallback: a cited title may carry a tagline prefix the
        #    canonical record drops ("I Speak, You Verify: <real title>"). Retry the
        #    title tier against the post-colon title alone — matched against the
        #    subtitle, so the dropped prefix is not read as a title contradiction.
        sub_ref = _subtitle_reference(reference)
        if sub_ref:
            for fn, args in self._title_query_steps(sub_ref, 4):
                match = self._match(sub_ref, self._fetch(fn, *args))
                if match is not None:
                    return match

        # 5) OpenReview, when the routing had not asked it: a workshop paper is cited with
        #    a venue as often as without one, and nothing else indexes it.
        titles = _likely_titles(reference)
        if titles and not _should_search_openreview(reference) and (ref_year is None or ref_year >= 2017):
            match = self._match(reference, self._fetch(paper_lookup.search_openreview, titles[0], 4))
            if match is not None:
                return match
        return None

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
                    return self._to_resolved(c, MatchMethod.DOI, 1.0, cands)

        # 2) arXiv-id exact match.
        if ref_arxiv:
            stem = ref_arxiv.split("v")[0]
            for c in cands:
                if (
                    c.arxiv_id
                    and c.arxiv_id.lower().split("v")[0] == stem
                    and not _title_tokens_contradict(reference, c.title)
                ):
                    return self._to_resolved(c, MatchMethod.ARXIV, 1.0, cands)

        # 3) Fuzzy title, corroborated (not vetoed) by author overlap + year.
        #    Bibliography author/year fields are often abbreviated, stale, or
        #    version-specific (arXiv vs proceedings vs journal), so title-token
        #    contradiction is the hard precision gate; author/year just lower the
        #    acceptance threshold when they agree.
        # Rank by fuzzy title, then by first-author agreement with the reference
        # (so a same-title record by a different lead author loses to the right
        # one when both are returned), then by the YEAR the reference prints, then by
        # having an abstract. One work has several records — the 1966 article and its
        # 1983 anniversary reprint, a 2017 paper and a 2025 repost under a new DOI — all
        # with the same title and authors; without the year the order among them was
        # whatever the index returned, and the reprint's year was then reported as a
        # year mismatch in the citation.
        ranked = sorted(
            (
                (
                    _fuzzy_title_score(reference, c.title),
                    _first_author_agrees(reference, c) is True,
                    _year_rank(ref_year, c.year),
                    _readable(c.abstract, c.title),
                    c,
                )
                for c in cands
            ),
            key=lambda t: (t[0], t[1], t[2], t[3]),
            reverse=True,
        )
        for score, _first_ok, _year_ok, _has_abs, c in ranked:
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
                return self._to_resolved(c, MatchMethod.FUZZY_TITLE, round(score / 100.0, 3), cands)

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

    def _to_resolved(self, c: Candidate, method: MatchMethod, score: float, siblings: list[Candidate] | None = None) -> Resolved:
        # Prefer S2's curated open-access PDF as the resolved URL: it points at the
        # actual paper text (so Stage-2 relevance can fetch full text for off-arXiv
        # papers, see stages/relevance.py) and is a strictly more useful "source"
        # link than a landing page. Falls back to the candidate's own URL.
        url = (c.extra or {}).get("oa_pdf") or c.url or None
        url_valid: bool | None = None
        if self.validate_urls and url:
            url_valid = paper_lookup.validate_url(url)
        arxiv_id = c.arxiv_id or _arxiv_from_siblings(c, siblings)
        abstract = c.abstract if _readable(c.abstract, c.title) else ""
        abstract = abstract or _abstract_from_siblings(c, siblings) or self._abstract_top_up(c) or c.abstract
        return Resolved(
            source=c.source,
            match_method=method,
            match_score=score,
            title=c.title or None,
            authors=list(c.authors),
            year=c.year,
            venue=c.venue or None,
            doi=c.doi or None,
            arxiv_id=arxiv_id or None,
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
                if _readable(cand.abstract, cand.title) and cand.doi.lower() == c.doi.lower():
                    return cand.abstract
        # 3) S2 by exact DOI — its abstract, else its one-line summary (a registry record,
        #    now asked first, carries neither).
        if c.doi and c.source != paper_lookup.SOURCE_S2:
            s2_key = getattr(self.settings, "s2_api_key", None)
            for cand in self._fetch(paper_lookup.fetch_semantic_scholar_by_doi, c.doi, s2_key):
                if (cand.doi or "").lower() == c.doi.lower():
                    if _readable(cand.abstract, cand.title):
                        return cand.abstract
                    if (cand.extra or {}).get("tldr"):
                        return cand.extra["tldr"]
        # 4) S2 tldr fallback — already fetched during resolution, so no extra call.
        return (c.extra or {}).get("tldr") or ""


# ───────────────────────────────────────────────────────────────
# Module-level reference parsers (shared with stages via the resolver)
# ───────────────────────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _year_rank(ref_year: int | None, year: int | None) -> int:
    """2 when a record's year agrees with the reference's (within the window), 1 when
    either is unknown, 0 when they disagree — for ranking, never for rejecting."""
    if ref_year is None or year is None:
        return 1
    return 2 if abs(int(year) - ref_year) <= YEAR_CORROBORATION_WINDOW else 0


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
