"""
correctness — STEP 1 (Reality & Accuracy).

:func:`fill_correctness` resolves a citation's reference against the grounding
layer and fills the *correctness* axes of a :class:`CitationRecord`:

    record.exists           : yes | no | unresolved
    record.resolved         : the canonical match (or None)
    record.metadata_issues  : [] of human-readable mismatches (authors/title/year/venue/url)
    record.evidence         : the retrieved metadata record backing the verdict

It satisfies :class:`citation_verifier.interfaces.StageFn` and follows two rules
from docs/decisions-phy.md:

  - **Abstain beats guessing.** If the resolver returns nothing, ``exists`` is
    ``unresolved`` (a red flag for the LLM to escalate via gated web search),
    NOT ``no``. Only a *strong* contradiction would justify ``no`` — and a
    bare absence is not strong, so this stage never sets ``no`` on its own. A
    fabrication verdict (``no``) is left to a downstream judge with web evidence.
  - **Degrade, never crash.** Any resolver/network error sets ``record.error``
    and leaves the axes at their ``unresolved`` default; the run continues.

This stage mutates ONLY the correctness fields (+ ``error``). LLM/network access
is entirely inside the injected ``resolver`` and is lazy + fail-soft there.
"""

from __future__ import annotations

import re
import unicodedata

from ..interfaces import Resolver
from ..schema import CitationRecord, CitedAs, Evidence, Exists, MatchMethod, Resolved

# rapidfuzz threshold above which two author/venue/title strings are "the same".
_SAME_THRESHOLD = 88.0


def fill_correctness(record: CitationRecord, *, resolver: Resolver) -> CitationRecord:
    """Fill ``exists`` / ``resolved`` / ``metadata_issues`` / ``evidence``.

    Args:
        record: a record stub (key + ``claim`` + ``cited_as`` populated).
        resolver: any object satisfying the :class:`Resolver` Protocol
            (typically :class:`MultiSourceResolver`).

    Returns:
        The same record, mutated in place (and also returned for chaining).
        On resolver failure the record is returned with ``error`` set and
        ``exists`` left at ``unresolved``.
    """
    reference = _reference_string(record)
    if not reference.strip():
        record.exists = Exists.UNRESOLVED
        record.metadata_issues = ["no reference string available to resolve"]
        return record

    try:
        resolved = resolver.resolve(record.cite_key, reference)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash
        record.error = f"correctness: resolver failed: {exc!r}"
        record.exists = Exists.UNRESOLVED
        return record

    if resolved is None or resolved.match_method is MatchMethod.NONE:
        # Absence is a red flag, not proof of fabrication: abstain.
        record.exists = Exists.UNRESOLVED
        record.resolved = resolved
        record.metadata_issues = [_unresolved_note(resolver)]
        return record

    issues = _diff_metadata(record, resolved)

    # A FUZZY_TITLE match contradicted on BOTH author and year is most likely the
    # wrong paper (a generic/partial-title collision, e.g. "…A systematic review"
    # colliding with a paper titled "Systematic Review"). Abstain to `unresolved`
    # rather than claim a false `yes`, and drop the candidate so its abstract can't
    # feed the relevance judge. Identifier (DOI/arXiv) matches are NOT downgraded —
    # the id is authoritative and already title-gated by the resolver.
    # ``match_method`` is stored as its string value (use_enum_values), so compare
    # with ``==`` (str-enum equality), not ``is``.
    if resolved.match_method == MatchMethod.FUZZY_TITLE:
        reject = _reject_bad_fuzzy_match(record.cited_as, resolved)
        if reject:
            record.exists = Exists.UNRESOLVED
            record.resolved = None
            record.metadata_issues = [f"abstained from a weak fuzzy-title match — {reject}"]
            return record

    record.resolved = resolved
    record.exists = Exists.YES
    record.metadata_issues = issues
    record.evidence = _dedupe_evidence(record.evidence + [_metadata_evidence(resolved)])
    return record


_MONTHS = frozenset(
    "january february march april may june july august september october november december".split()
)


def _is_nontitle(title: str | None) -> bool:
    """True when a 'title' is really a date / number stub ('October 2022', '2022')."""
    toks = _norm(title or "").split()
    return not toks or all(
        t in _MONTHS or re.fullmatch(r"(?:19|20)?\d{1,4}", t) for t in toks
    )


def _titles_diverge(a: str | None, b: str | None) -> bool:
    """True when two titles are different works — neither is ~contained in the other.

    A near-identical pair (one a subtitle-extended form of the other) leaves one side
    almost fully covered; two genuinely different works each keep distinctive content
    tokens (cited 'Reinforcement learning: An introduction, volume' vs candidate
    'Introduction: The Challenge of Reinforcement Learning' — 'volume' vs 'challenge').
    """
    at, bt = _content_tokens(a), _content_tokens(b)
    if len(at) < 2 or len(bt) < 2:
        return False
    inter = len(at & bt)
    return max(inter / len(at), inter / len(bt)) < 0.85


def _reject_bad_fuzzy_match(cited: CitedAs, candidate: Resolved) -> str | None:
    """Reason to REJECT a FUZZY_TITLE candidate as the wrong work, or ``None`` to accept.

    A wrong ``exists=yes`` is worse than ``unresolved``: it sends the relevance judge
    to the wrong paper. This is the single, high-precision veto over fuzzy (non-id)
    candidates — id/exact matches are authoritative and never reach here. Abstain
    beats guessing, so any of these "looks similar but is a different work" signals
    drops the match to ``unresolved``.
    """
    ctitle = candidate.title or ""
    # 1) The candidate "title" is a bare date / number ("October 2022") — not a title.
    if _is_nontitle(ctitle):
        return f"closest candidate is not a real title ('{_short(ctitle)}')"
    # 2) No author corroboration at all on a fuzzy match — the minimum bar isn't met.
    if not candidate.authors:
        return f"closest candidate has no authors to corroborate ('{_short(ctitle)}')"
    author_ok = bool(
        cited.authors and candidate.authors and _same_author(cited.authors[0], candidate.authors[0])
    )
    year_gap = abs(cited.year - candidate.year) if (cited.year and candidate.year) else 0
    # 3) A large year gap with diverging titles is a different work — even when the
    #    first author matches (a prolific author has many works: a 1998 textbook vs a
    #    1992 chapter that merely shares generic words).
    if year_gap > 2 and _titles_diverge(cited.title, ctitle):
        return (
            f"closest candidate is a different work — cited {cited.year} vs {candidate.year}, "
            f"titles diverge ('{_short(ctitle)}')"
        )
    # 4) Wrong first author confirmed by a second failing axis: year off >1, or the
    #    resolved title drops the cited title's distinctive content (generic collision).
    if cited.authors and candidate.authors and not author_ok:
        if year_gap > 1:
            return f"closest candidate's author and year both contradict ('{_short(ctitle)}')"
        if _title_poorly_covered(cited.title, ctitle):
            return f"closest candidate's author and title both contradict ('{_short(ctitle)}')"
    return None


# Tiny stopword set for title-coverage (the function words a generic match shares).
_TITLE_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by", "as"}


def _content_tokens(title: str | None) -> set[str]:
    """Distinctive content tokens of a title (lowercased, >3 chars, stopwords out)."""
    return {t for t in _norm(title or "").split() if len(t) > 3 and t not in _TITLE_STOP}


def _title_poorly_covered(cited_title: str | None, resolved_title: str | None, thresh: float = 0.6) -> bool:
    """True when the resolved title omits MOST of the cited title's content tokens.

    Catches a short/generic match that dropped the cited title's specific topic
    (the failure ``_strings_match``'s substring rule lets through). Abstains from
    judging when there is too little title to compare.
    """
    ct = _content_tokens(cited_title)
    rt = _content_tokens(resolved_title)
    if len(ct) < 3 or not rt:
        return False
    return len(ct & rt) / len(ct) < thresh


# Human-readable names for the grounding sources, for the unresolved note.
_SOURCE_NAMES = {
    "crossref": "Crossref",
    "arxiv": "arXiv",
    "dblp": "DBLP",
    "s2": "Semantic Scholar",
    "openalex": "OpenAlex",
}


def _unresolved_note(resolver: Resolver) -> str:
    """Explain an unresolved citation: which sources were searched without a match."""
    sources = getattr(resolver, "sources", ()) or ()
    where = ", ".join(_SOURCE_NAMES.get(str(s), str(s)) for s in sources)
    return (
        f"could not retrieve a confident match — searched {where}"
        if where
        else "could not retrieve a confident match in structured sources"
    )


# ───────────────────────────────────────────────────────────────
# Metadata comparison
# ───────────────────────────────────────────────────────────────
def _diff_metadata(record: CitationRecord, resolved: Resolved) -> list[str]:
    """List human-readable mismatches between ``cited_as`` and the canonical record."""
    issues: list[str] = []
    cited = record.cited_as

    # Title
    if cited.title and resolved.title and not _strings_match(cited.title, resolved.title):
        issues.append(f"title mismatch: cited '{_short(cited.title)}' vs '{_short(resolved.title)}'")

    # Year (the canonical/published year may differ from an arXiv year by ~1).
    if cited.year and resolved.year and abs(cited.year - resolved.year) > 1:
        issues.append(f"year mismatch: cited {cited.year} vs {resolved.year}")

    # Venue
    if cited.venue and resolved.venue and not _strings_match(cited.venue, resolved.venue):
        issues.append(f"venue mismatch: cited '{_short(cited.venue)}' vs '{_short(resolved.venue)}'")

    # Authors (first-author + overlap heuristic).
    author_issue = _author_issue(cited.authors, resolved.authors)
    if author_issue:
        issues.append(author_issue)

    # URL liveness (only when the resolver actually checked it).
    if resolved.url_valid is False:
        issues.append(f"cited/resolved URL did not validate: {resolved.url}")

    return issues


# Organization "authors": a paper credited to an org / "* Team" against a citation
# that lists individuals (or vice versa) is a benign, common attribution discrepancy,
# NOT a metadata error. This suppresses the WARNING only — the resolver's match gate
# is untouched, so it never loosens what counts as a match.
_ORG_AUTHOR_RE = re.compile(
    r"(?i)\b(?:open\s?ai|deepseek|qwen|google|deepmind|meta|essential\s?ai|anthropic|"
    r"microsoft|mistral|cohere|nvidia|hugging\s?face|allen\s?ai|ai2|stability|"
    r"alibaba|baidu|tencent|bytedance)\b"
)
_ORG_SUFFIX_RE = re.compile(r"(?i)\b(?:team|teams|labs?|inc|institute|consortium|collaboration)\b")


def _is_org_author(name: str) -> bool:
    """True when an author string is an organization, not a person ('OpenAI', 'Qwen Team')."""
    n = (name or "").strip()
    return bool(n) and bool(_ORG_AUTHOR_RE.search(n) or _ORG_SUFFIX_RE.search(n))


def _author_issue(claimed: list[str], canonical: list[str]) -> str | None:
    """Flag a wrong first author or a list that genuinely differs.

    Authors are matched by surname + given-name initial across the whole list, so
    an abbreviated cited name (``Yang Z``) matches its canonical full form
    (``Zhengyuan Yang``) — only a different surname or initial is reported. An
    org-vs-individual first author (``OpenAI`` / ``Qwen Team`` against a listed
    person) is suppressed: it always "mismatches" by surname but is not an error.
    """
    if not claimed or not canonical:
        return None
    if not _same_author(claimed[0], canonical[0]):
        if _is_org_author(claimed[0]) or _is_org_author(canonical[0]):
            return None
        return f"first-author mismatch: cited '{claimed[0]}' vs '{canonical[0]}'"
    matched = sum(1 for c in claimed if any(_same_author(c, k) for k in canonical))
    if matched / len(claimed) < 0.5:
        return "author list differs substantially from the canonical record"
    return None


# ───────────────────────────────────────────────────────────────
# Evidence + small helpers
# ───────────────────────────────────────────────────────────────
def _metadata_evidence(resolved: Resolved) -> Evidence:
    """Package the canonical record as retrieved evidence (NOT model memory)."""
    bits = [b for b in (resolved.title, _join(resolved.authors), str(resolved.year or "")) if b]
    src = resolved.doi or resolved.arxiv_id or resolved.url or (resolved.source or "structured")
    return Evidence(
        kind="metadata",
        source=str(src),
        quote=" — ".join(bits)[:600],
        url=resolved.url,
    )


def _dedupe_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Evidence] = []
    for e in items:
        k = (e.kind, e.source, e.quote)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _reference_string(record: CitationRecord) -> str:
    """Best available reference text for grounding.

    Prefer the cleanly-parsed structured fields (authors + *quoted* title + venue +
    year) over the raw bibliography string. PDF extraction can leave running
    header/footer contamination in ``raw`` (e.g. "Cognitive Computation (2024)
    16:248") that corrupts the resolver's year/title extraction — so a correct,
    exact-title candidate gets vetoed by the year gate. The structured fields are
    clean and carry a single year; quoting the title also lets the resolver's
    title-field search lock onto the right title (it prefers a quoted segment).
    Falls back to ``raw`` when no title was parsed.

    Either way the structured identifiers the extractor parsed (``arxiv_id`` /
    ``doi``) are appended so the resolver's exact-match cascade can fire — still
    *grounded*: the resolver only accepts an id a fetched candidate also carries.
    """
    c = record.cited_as
    if c.title:
        base = " ".join(
            p for p in (_join(c.authors), f'"{c.title}"', c.venue or "", str(c.year or "")) if p
        ).strip()
    else:
        raw = (c.raw or "").strip()
        base = raw or " ".join(
            p for p in (_join(c.authors), str(c.year or ""), c.venue or "") if p
        ).strip()
    parts = [base]
    base_lower = base.lower()
    if c.arxiv_id and "arxiv" not in base_lower:
        parts.append(f"arXiv:{c.arxiv_id}")
    if c.doi and c.doi.lower() not in base_lower:
        parts.append(f"doi:{c.doi}")
    return " ".join(p for p in parts if p).strip()


def _strings_match(a: str, b: str, thresh: float = _SAME_THRESHOLD) -> bool:
    a2, b2 = _norm(a), _norm(b)
    if not a2 or not b2:
        return True  # nothing to contradict
    if a2 in b2 or b2 in a2:
        return True
    from rapidfuzz import fuzz

    return fuzz.token_sort_ratio(a2, b2) >= thresh


_INITIALS_TOKEN_RE = re.compile(r"^[A-Z](?:[.\-]?[A-Z]){0,3}\.?$")


def _fold(s: str) -> str:
    """Lowercase + strip diacritics, for robust surname/initial comparison."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _name_key(author: str) -> tuple[str, str]:
    """Reduce an author to ``(surname, given-initial)``.

    Cited references are ``Surname Initials`` (Vancouver); canonical records are
    ``First Last`` or ``Last, First``. All forms collapse to the same key, so an
    abbreviated given name (``Firat M``) matches its full form (``Mehmet Firat``).
    """
    a = (author or "").strip()
    if not a:
        return ("", "")
    # Heal a despacing-split surname: a lone capital + a lowercase run ("V aswani"
    # -> "Vaswani", "Y uan" -> "Yuan"). A capital before a *capitalized* word is
    # given-name-first ("M Jadeja") and is left as initial + surname.
    a = re.sub(r"^([A-Z]) (?=[a-z])", r"\1", a)
    if "," in a:  # "Last, First M"
        last, rest = (p.strip() for p in a.split(",", 1))
        return (_fold(last.split()[-1]) if last.split() else "", _fold(rest[:1]))
    toks = a.split()
    if len(toks) == 1:  # single token, possibly glued ("YangZ" -> Yang, Z)
        m = re.match(r"^(.*[a-z])([A-Z]{1,3})$", toks[0])
        return (_fold(m.group(1)), _fold(m.group(2)[:1])) if m else (_fold(toks[0]), "")
    words = [t for t in toks if not _INITIALS_TOKEN_RE.match(t)]
    initials = [t for t in toks if _INITIALS_TOKEN_RE.match(t)]
    if not words:  # an all-initials fragment ("S. R.") — no surname to key on
        return ("", _fold(initials[0][:1]) if initials else "")
    if len(words) == 1:  # "Surname I[I]" — surname-first (Vancouver)
        return (_fold(words[0]), _fold(initials[0][:1]) if initials else "")
    return (_fold(words[-1]), _fold(words[0][:1]))  # "First [M] Last" — given-first


def _surname_close(a: str, b: str) -> bool:
    """Surnames equal, or one a prefix of the other (PDF glue / abbreviation)."""
    if not a or not b:
        return False
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def _same_author(cited: str, canonical: str) -> bool:
    """Same person: surnames match and any given-name initials agree."""
    cs, ci = _name_key(cited)
    ks, ki = _name_key(canonical)
    if not _surname_close(cs, ks):
        return False
    return ci == ki if (ci and ki) else True


def _norm(s: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _join(authors: list[str]) -> str:
    return ", ".join(a for a in authors if a)


def _short(s: str, n: int = 60) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
