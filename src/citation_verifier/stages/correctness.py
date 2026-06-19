"""
correctness — STEP 1 (Reality & Accuracy).

:func:`fill_correctness` resolves a citation's reference against the grounding
layer and fills the *correctness* axes of a :class:`CitationRecord`:

    record.exists           : yes | no | unverified
    record.resolved         : the canonical match (or None)
    record.metadata_issues  : [] of human-readable mismatches (authors/title/year/venue/url)
    record.evidence         : the retrieved metadata record backing the verdict

It satisfies :class:`citation_verifier.interfaces.StageFn` and follows two rules
from docs/decisions-phy.md:

  - **Abstain beats guessing.** If the resolver returns nothing, ``exists`` is
    ``unverified`` (a red flag for the LLM to escalate via gated web search),
    NOT ``no``. Only a *strong* contradiction would justify ``no`` — and a
    bare absence is not strong, so this stage never sets ``no`` on its own. A
    fabrication verdict (``no``) is left to a downstream judge with web evidence.
  - **Degrade, never crash.** Any resolver/network error sets ``record.error``
    and leaves the axes at their ``unverified`` default; the run continues.

This stage mutates ONLY the correctness fields (+ ``error``). LLM/network access
is entirely inside the injected ``resolver`` and is lazy + fail-soft there.
"""

from __future__ import annotations

from ..interfaces import Resolver
from ..schema import CitationRecord, Evidence, Exists, MatchMethod, Resolved

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
        ``exists`` left at ``unverified``.
    """
    reference = _reference_string(record)
    if not reference.strip():
        record.exists = Exists.UNVERIFIED
        record.metadata_issues = ["no reference string available to resolve"]
        return record

    try:
        resolved = resolver.resolve(record.cite_key, reference)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash
        record.error = f"correctness: resolver failed: {exc!r}"
        record.exists = Exists.UNVERIFIED
        return record

    if resolved is None or resolved.match_method is MatchMethod.NONE:
        # Absence is a red flag, not proof of fabrication: abstain.
        record.exists = Exists.UNVERIFIED
        record.resolved = resolved
        record.metadata_issues = ["no confident match in structured sources"]
        return record

    record.resolved = resolved
    record.exists = Exists.YES
    record.metadata_issues = _diff_metadata(record, resolved)
    record.evidence = _dedupe_evidence(record.evidence + [_metadata_evidence(resolved)])
    return record


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


def _author_issue(claimed: list[str], canonical: list[str]) -> str | None:
    """Flag a wrong first author or a low author overlap."""
    if not claimed or not canonical:
        return None
    if not _strings_match(_surname(claimed[0]), _surname(canonical[0]), thresh=92.0):
        return f"first-author mismatch: cited '{claimed[0]}' vs '{canonical[0]}'"
    claimed_s = {_surname(a) for a in claimed if a}
    canon_s = {_surname(a) for a in canonical if a}
    if claimed_s and (len(claimed_s & canon_s) / len(claimed_s)) < 0.5:
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


def _surname(author: str) -> str:
    author = (author or "").strip()
    if "," in author:
        return author.split(",", 1)[0].strip()
    return author.split()[-1] if author.split() else ""


def _norm(s: str) -> str:
    import re

    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def _join(authors: list[str]) -> str:
    return ", ".join(a for a in authors if a)


def _short(s: str, n: int = 60) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
