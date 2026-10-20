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

import os
import re
import unicodedata
from collections import Counter

from .. import venues
from ..interfaces import Resolver
from ..schema import CitationRecord, CitedAs, Evidence, Exists, MatchMethod, Resolved

# rapidfuzz threshold above which two author/venue/title strings are "the same".
_SAME_THRESHOLD = 88.0


def fill_correctness(record: CitationRecord, *, resolver: Resolver) -> CitationRecord:
    """Fill ``exists`` / ``resolved`` / ``metadata_issues`` / ``evidence``.

    Defects of the citation itself — an author listed twice, no year, a title that
    repeats itself — are reported whatever the lookup found: they need no record to be
    seen, and a citation too broken to resolve is where they matter most.

    Args:
        record: a record stub (key + ``claim`` + ``cited_as`` populated).
        resolver: any object satisfying the :class:`Resolver` Protocol
            (typically :class:`MultiSourceResolver`).

    Returns:
        The same record, mutated in place (and also returned for chaining).
        On resolver failure the record is returned with ``error`` set and
        ``exists`` left at ``unresolved``.
    """
    _resolve_and_compare(record, resolver)
    canonical = list(record.resolved.authors) if record.resolved else None
    defects = [
        d for d in _citation_defects(record.cited_as, canonical) if d not in record.metadata_issues
    ]
    if defects:
        record.metadata_issues = list(record.metadata_issues) + defects
    return record


def _resolve_and_compare(record: CitationRecord, resolver: Resolver) -> CitationRecord:
    """:func:`fill_correctness` without the citation's own defects."""
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
        # No structured match — but a citation can still be a verifiable web/software/
        # system-card object (GitHub repo, model system card). Try the cited URL
        # directly before abstaining.
        if _try_direct_url(record):
            return record
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
            # The scholarly candidate was wrong, but the cited URL may still be a
            # live web/software object (e.g. LangChain's GitHub repo) — try it.
            if _try_direct_url(record):
                return record
            record.exists = Exists.UNRESOLVED
            record.resolved = None
            record.metadata_issues = [f"abstained from a weak fuzzy-title match — {reject}"]
            return record

    record.resolved = resolved
    record.exists = Exists.YES
    record.metadata_issues = issues
    record.evidence = _dedupe_evidence(record.evidence + [_metadata_evidence(resolved)])
    return record


def _citation_defects(cited: CitedAs, canonical: list[str] | None = None) -> list[str]:
    """What is wrong with a citation on its face.

    Generated bibliographies carry these: one batch of ten machine-written papers listed
    "Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio" in four of them,
    and cited a report by authors alone, with no year and a title that says its own
    second half twice.

    Two people can share a name: one paper is by two "Chen, D."s (Dong and Dongdong),
    another by two Kun Xus listed side by side. So a repeated name counts only when it
    is written in full, the two are not neighbours, and the record (``canonical``) does
    not repeat it too.
    """
    out: list[str] = []
    seen: dict[str, int] = {}
    canon = Counter(_letters(a) for a in (canonical or []))
    for pos, name in enumerate(_clean_cited_authors(cited.authors)):
        given = name.split(",")[1] if "," in name else " ".join(name.split()[:-1])
        if len(re.sub(r"[^A-Za-z]", "", given)) < 2 or re.fullmatch(r"(?:\s*[A-Z]\.?)+\s*", given):
            continue  # initials only: "Chen, D." twice may be Dong and Dongdong
        key = _letters(name)
        if key in seen and pos - seen[key] > 1 and canon[key] < 2:
            out.append(f"author listed twice: '{name}'")
        seen.setdefault(key, pos)
    raw = (cited.raw or "").strip()
    # A reference cut off mid-phrase ("… In Proceedings of the 62nd Annual Meeting of")
    # lost its year to extraction, not to its author.
    truncated = bool(re.search(r"(?i)\b(?:of|the|and|in|for|on|at|to|by|with)$|,$", raw))
    if (
        (cited.title or cited.authors) and not cited.year and not truncated
        and not re.search(r"(?<!\d)(?:1[89]|20)\d{2}(?!\d)", raw)
    ):
        out.append("the citation gives no year")
    words = _norm(cited.title or "").split()
    for n in range(min(12, len(words) // 2), 4, -1):  # a run of 5+ words said twice
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        repeated = next((g for i, g in enumerate(grams) if g in grams[i + n :]), None)
        if repeated:
            out.append(f"the title repeats itself ('{_short(repeated, 50)}')")
            break
    return out


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


def _same_title(a: str | None, b: str | None) -> bool:
    """True when two titles are one title: equal once normalised, or one a clean
    extension of the other (a subtitle the other source drops). Content words that
    differ — in either direction — make them two titles, however similar the rest."""
    na, nb = _norm(a or ""), _norm(b or "")
    if not na or not nb:
        return True  # nothing to tell apart
    if na == nb or na.replace(" ", "") == nb.replace(" ", ""):
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if long_.startswith(short) or long_.endswith(short):
        return True  # "Llemma" vs "Llemma: An open language model for mathematics"
    return _content_tokens(a) == _content_tokens(b)


def _first_author_agrees(cited: CitedAs, canonical_first: str) -> bool:
    """Is the canonical first author the one the citation names first?

    Besides the name-to-name comparison this accepts a hyphenated surname written with a
    space and, as a last resort, the canonical surname appearing in the reference AS
    PRINTED: the parsed author list is the least reliable field a PDF yields, and a
    mis-split name must not turn a correct match into a rejection. An organisation is NOT
    waved through here as it is for the mismatch warning — "OpenAI" matched to a paper by
    an unrelated individual is exactly the wrong-paper case this guards.
    """
    first = cited.authors[0] if cited.authors else ""
    if not first:
        return True  # the citation names nobody to disagree with
    if not (canonical_first or "").strip():
        return False  # a record with a blank author corroborates nothing
    if _same_author(first, canonical_first):
        return True
    if _same_author(first.replace("-", " "), canonical_first.replace("-", " ")):
        return True
    surname = _name_key(canonical_first)[0]
    return len(surname) >= 4 and surname in _fold(cited.raw or "")


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
    #    Blank entries are not authors: an index returned ["", "A Isakov", ""] for a
    #    report that shared three title words with the citation, the blank first author
    #    "agreed" with anyone, and the wrong work was accepted.
    authors = [a for a in (candidate.authors or []) if (a or "").strip()]
    if not authors:
        return f"closest candidate has no authors to corroborate ('{_short(ctitle)}')"
    author_ok = bool(cited.authors and _first_author_agrees(cited, authors[0]))
    year_gap = abs(cited.year - candidate.year) if (cited.year and candidate.year) else 0

    # 3) A WRONG first author is the strongest single "different work" signal for a
    #    title-only match. Pair it with any corroborating signal and reject — a
    #    confident wrong match (which then mis-feeds the relevance judge) is worse
    #    than abstaining.
    if cited.authors and not author_ok:
        # (a) the citation carries a URL (a release/blog/repo/system card, e.g. Genie 2
        #     → deepmind.google): a fuzzy SCHOLARLY match by a different author is almost
        #     always the wrong paper — prefer the URL path / unresolved.
        if (cited.url or "").strip():
            return f"different author + the citation is a URL/release, not this paper ('{_short(ctitle)}')"
        # (b) a SHORT cited title (≤2 distinctive tokens — "OpenAI o3-mini", "Genie 2")
        #     matches longer unrelated titles as a coincidental substring (o3-mini was
        #     matched to a pneumonia paper that merely mentions it).
        if len(_content_tokens(cited.title)) <= 2:
            return f"different author on a short cited title (coincidental match) ('{_short(ctitle)}')"
        # (c) a second failing axis: year off >1, OR the titles genuinely diverge.
        if year_gap > 1 or _title_poorly_covered(cited.title, ctitle) or _titles_diverge(cited.title, ctitle):
            return f"closest candidate's author and title/year contradict ('{_short(ctitle)}')"
        # (d) the titles are not the SAME title. A wrong first author on the exact title
        #     is a finding about the citation (a real paper credited to the wrong people —
        #     kept, and reported as an author mismatch). A wrong first author on a title
        #     that is merely close is a different paper: cited "Deep Residual Learning for
        #     Image Recognition" was accepted as "Deep Learning for Image Recognition" by
        #     other authors, one content word apart and inside every looser threshold
        #     above, and the relevance judge was then handed the wrong paper.
        if not _same_title(cited.title, ctitle):
            return f"different first author and a different title ('{_short(ctitle)}')"

    # 4) Even with a matching/unknown author: a large year gap + diverging titles is a
    #    different work (a 1998 textbook vs a 1992 chapter sharing only generic words).
    if year_gap > 2 and _titles_diverge(cited.title, ctitle):
        return (
            f"closest candidate is a different work — cited {cited.year} vs {candidate.year}, "
            f"titles diverge ('{_short(ctitle)}')"
        )
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
    """Explain an unresolved citation: which sources were searched without a match.

    Says so when the search was INCOMPLETE. "Searched five sources, found nothing" is the
    strongest hint a reference was invented; the same wording over a run in which two of
    the five were rate-limiting is a hint the run cannot back, and it read identically.
    """
    sources = getattr(resolver, "sources", ()) or ()
    where = ", ".join(_SOURCE_NAMES.get(str(s), str(s)) for s in sources)
    failed_fn = getattr(resolver, "failed_sources", None)
    failed = list(failed_fn()) if callable(failed_fn) else []
    if failed:
        return (
            f"lookup incomplete — {', '.join(failed)} did not respond (rate limit / "
            f"network), so this reference was NOT fully searched; re-run before reading "
            f"anything into it"
        )
    return (
        f"could not retrieve a confident match — searched {where}"
        if where
        else "could not retrieve a confident match in structured sources"
    )


def _try_direct_url(record: CitationRecord) -> bool:
    """Verify a citation that points at a web/software/system-card URL.

    Fallback when the structured cascade found nothing (or rejected a wrong fuzzy
    match): a GitHub repo / model system card / Notion page can still be a real
    object reachable by URL. On a LIVE url -> ``exists=yes`` as a ``DIRECT_URL``
    match (a web/software object, NOT a scholarly title match), backfilling
    title/authors/year from the cited reference. On a BLOCKED (gated/bot-blocked)
    url -> abstain with an *actionable* reason instead of a false ``yes`` — a later
    run with GITHUB_TOKEN / S2_API_KEY / a search key can recover it. Returns
    ``True`` when it set the record's verdict. Network is lazy + fail-soft.
    """
    raw = (record.cited_as.url or "").strip()
    if not raw:
        return False
    from ..grounding import url_validate  # lazy: network helper, keeps import light

    chk = url_validate.validate_citation_url(raw, github_token=os.environ.get("GITHUB_TOKEN"))
    if chk is None:
        return False
    cited = record.cited_as
    if chk.status == "live":
        record.exists = Exists.YES
        record.resolved = Resolved(
            source="url",
            match_method=MatchMethod.DIRECT_URL,
            match_score=1.0,
            url=chk.url,
            url_valid=True,
            title=cited.title,
            authors=list(cited.authors or []),
            year=cited.year,
        )
        record.metadata_issues = []
        record.evidence = _dedupe_evidence(
            record.evidence
            + [
                Evidence(
                    kind="url",
                    source=chk.method,
                    quote=f"cited URL verified live (HTTP {chk.http}) — web/software/system-card object",
                    url=chk.url,
                )
            ]
        )
        return True
    if chk.status == "blocked":
        record.exists = Exists.UNRESOLVED
        record.resolved = None
        record.metadata_issues = [
            f"url access blocked: HTTP {chk.http} from direct fetch ({chk.url}); no configured "
            "API/search fallback validated — set GITHUB_TOKEN / S2_API_KEY or a search key and re-run"
        ]
        return True
    return False  # dead / network error -> let the normal abstain proceed


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
    venue_issue = _venue_issue(cited.venue or "", resolved.venue or "")
    if venue_issue:
        issues.append(venue_issue)

    # Authors (first-author + overlap heuristic).
    author_issue = _author_issue(cited.authors, resolved.authors)
    if author_issue:
        issues.append(author_issue)

    # URL liveness (only when the resolver actually checked it).
    if resolved.url_valid is False:
        issues.append(f"cited/resolved URL did not validate: {resolved.url}")

    return issues


# Words that do not count towards a venue's initials ("Conference ON Computer Vision AND …").
_VENUE_STOP = frozenset("of on in the and for a an to at de la le du des et".split())


def _venue_acronyms(venue: str) -> set[str]:
    """Acronym-shaped tokens of a venue, as lowercase letter runs.

    "NeurIPS" -> "nips" (its capitals), "ICML 2021" -> "icml", "NAACL-HLT" -> "naacl",
    "hlt". A token counts when it carries at least two capitals and is short.
    """
    out: set[str] = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z&]*", venue or ""):
        caps = "".join(ch for ch in tok if ch.isupper())
        if len(caps) >= 2 and len(tok) <= 10:
            out.add(caps.lower())
    return out


def _venue_initials(venue: str) -> str:
    """First letters of a venue name's content words, in order — "icml" for
    "International Conference on Machine Learning"."""
    words = [w for w in _norm(venue).split() if w not in _VENUE_STOP and not w.isdigit()]
    return "".join(w[0] for w in words)


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def _unlisted_venues_match(cited: str, canonical: str) -> bool:
    """The general tests, for a venue the alias table does not list.

    Without the table there is only the shape of the two strings to go on: they are close
    as strings; one is the other abbreviated word by word ("Comput. Educ. Artif. Intell." /
    "Computers and Education: Artificial Intelligence"); or one is an acronym whose
    letters are, in order, among the initials of the other's words. Permissive on purpose —
    with nothing known about either venue, a miss costs one unreported detail and a false
    report is an accusation.
    """
    if _strings_match(cited, canonical) or venues.is_abbreviation_of(cited, canonical):
        return True
    for acronyms, other in ((_venue_acronyms(cited), canonical), (_venue_acronyms(canonical), cited)):
        other_tokens = set(_norm(other).split()) | _venue_acronyms(other)
        initials = _venue_initials(other)
        for acr in acronyms:
            if acr in other_tokens or (len(acr) >= 3 and _is_subsequence(acr, initials)):
                return True
    return False


def _venue_issue(cited: str, canonical: str) -> str | None:
    """The finding for a cited venue against the venue of the record, or ``None``.

    A venue is written a dozen ways — "NeurIPS", "NIPS", "Advances in Neural Information
    Processing Systems 35", "Adv. Neural Inf. Process. Syst." — and compared as strings the
    spellings share almost nothing, so the commonest correct citation there is was reported
    as a mismatch. The alias table (:mod:`citation_verifier.venues`) decides whenever it
    knows both venues; the string tests above decide the rest.

    Two things a record cannot contradict, however different the strings look:

    * a **preprint** record. It says where a copy lives, not that the work was never
      published, so "cited as ICLR, record from arXiv" is not a finding. (The reverse is:
      cited as a preprint while the record gives a venue.)
    * a **proceedings series** ("PMLR", "Lecture Notes in Computer Science"), which names
      the volume and not the conference.
    """
    if not cited or not canonical:
        return None
    verdict = venues.compare_venues(cited, canonical)
    if verdict in (venues.SAME, venues.UNCOMPARABLE):
        return None
    if verdict == venues.UNKNOWN and _unlisted_venues_match(cited, canonical):
        return None
    if venues.is_preprint(cited):
        return f"cited as a preprint, but the record gives a venue: '{_short(canonical)}'"
    note = ""
    if venues.is_workshop(canonical) and not venues.is_workshop(cited):
        if venues.venue_ids(cited) & venues.venue_ids(canonical):
            note = " (a workshop at that conference, not the conference itself)"
    return f"venue mismatch: cited '{_short(cited)}' vs '{_short(canonical)}'{note}"


def _venues_match(cited: str, canonical: str) -> bool:
    """True unless the two strings name DIFFERENT venues (see :func:`_venue_issue`)."""
    return _venue_issue(cited, canonical) is None


# Organization "authors": a paper credited to an org / "* Team" against a citation
# that lists individuals (or vice versa) is a benign, common attribution discrepancy,
# NOT a metadata error. This suppresses the WARNING only — the resolver's match gate
# is untouched, so it never loosens what counts as a match.
_ORG_AUTHOR_RE = re.compile(
    r"(?i)\b(?:open\s?ai|deepseek|qwen|google|deepmind|meta|essential\s?ai|anthropic|"
    r"microsoft|mistral|cohere|nvidia|hugging\s?face|allen\s?ai|ai2|stability|"
    r"alibaba|baidu|tencent|bytedance)\b"
)
_ORG_SUFFIX_RE = re.compile(
    r"(?i)\b(?:team|teams|labs?|inc|institute|consortium|collaboration|contributors?|group)\b"
)


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
    claimed = _clean_cited_authors(claimed)
    canonical = [a for a in (canonical or []) if (a or "").strip()]
    if not claimed or not canonical:
        return None
    if not _same_author(claimed[0], canonical[0]):
        if _is_org_author(claimed[0]) or _is_org_author(canonical[0]):
            return None
        # Hyphenated / multi-part surname variants are the same person:
        # "Perez-Liebana" vs "Perez Liebana", "Zamfirescu-Pereira" vs "Zamfirescu Pereira".
        # Re-check with hyphens as spaces — warning-only, so resolution is untouched.
        if _same_author(claimed[0].replace("-", " "), canonical[0].replace("-", " ")):
            return None
        return f"first-author mismatch: cited '{claimed[0]}' vs '{canonical[0]}'"
    # Exact surname matching is not fit to say a person is missing from a paper: the two
    # lists spell one author as "Krompaß" / "Krompass", "Köpken" / "Koepken", "Mrkšić" /
    # "Mrkvsi'c" (an accent command a source failed to decode), "Brooks, T." /
    # "†. TimBrooks". Replayed over 744 published references, eight of the nine
    # "invented co-authors" this check named were that. Only a name with no near spelling
    # anywhere on the canonical list counts.
    unmatched = [
        c for c in claimed
        if not _is_org_author(c) and _name_key(c)[0]
        and not any(_same_author(c, k) for k in canonical)
        and not _on_author_list(c, canonical)
    ]
    matched = len(claimed) - len(unmatched)
    if matched / len(claimed) < 0.5:
        return "author list differs substantially from the canonical record"
    # Right first author, invented co-authors: the shape a fabricated or half-remembered
    # reference most often has, and the one a first-author check plus a 50% overlap bar
    # lets through (two real names followed by two that are not on the paper pass both).
    # Reported only when the canonical list is at least as long as the cited one — a
    # record that carries fewer authors than the citation is truncated, and every name
    # past its end would "not be on the paper" — and only from two names up, since a
    # single one is as often a transliteration or a changed surname as an error.
    if len(unmatched) >= 2 and len(canonical) >= len(claimed):
        names = ", ".join(f"'{n}'" for n in unmatched[:3]) + (" …" if len(unmatched) > 3 else "")
        return f"{len(unmatched)} cited authors are not on the canonical author list: {names}"
    return None


# What an author field carries besides the name when a reference was read from a PDF:
# the "and" before the last author, the year that follows the list ("and Jeff Dean. 2017"),
# footnote marks ("*Azalia Mirhoseini"), and the "et al." that closes a shortened list.
_ET_AL_RE = re.compile(r"(?i)^(?:and\s+)?(?:et\.?\s*al\.?|others|and\s+others)$")
_AUTHOR_LEAD_RE = re.compile(r"(?i)^(?:and|&)\s+")
_AUTHOR_YEAR_TAIL_RE = re.compile(r"[\s.,;:(\[]*\b(?:19|20)\d{2}[a-z]?\b.*$")
_AUTHOR_MARK_RE = re.compile(r"[*†‡§¶∗⋆#\d]+")


def _clean_cited_authors(names: list[str]) -> list[str]:
    """The cited author list with everything that is not a name taken out."""
    out: list[str] = []
    for raw in names or []:
        name = _AUTHOR_YEAR_TAIL_RE.sub("", str(raw or ""))
        name = _AUTHOR_MARK_RE.sub("", name).strip(" .,;:")
        name = _AUTHOR_LEAD_RE.sub("", name)
        name = re.sub(r"-\s+", "-", name)                  # "Pérez- Cruz": a line break
        # "Ł ukasz": a capital the PDF set apart from the rest of its word
        name = re.sub(r"^([^\W\d_]) (?=[a-z])", lambda m: m.group(1) if m.group(1).isupper() else m.group(0), name)
        name = re.sub(r"\s+", " ", name).strip(" .,;:")
        if name and not _ET_AL_RE.match(name) and sum(ch.isalpha() for ch in name) >= 2:
            out.append(name)
    return out


def _letters(name: str) -> str:
    return re.sub(r"[^a-z]", "", _fold(name))


def _on_author_list(cited: str, canonical: list[str]) -> bool:
    """True when ``cited`` has a near spelling somewhere on the canonical list.

    Loose about SPELLING, because it only ever withdraws the claim that a named person is
    not an author: a surname within a letter or two of one on the list (a transliterated
    umlaut, an undecoded accent), or found at the end of a name the record ran together
    ("†. TimBrooks", "JoyceLee").

    Not loose about WHO: where the record gives the name properly, a contradicting
    given-name initial is another person. "Li Y" is not "John Li" — and with "Sorelle A",
    "Zettlemoyer L" and "Weld DS" beside it on a five-author paper, that is exactly the
    half-invented author list this check exists for.
    """
    from rapidfuzz import fuzz  # a core dependency; imported here to keep import cheap

    surname, initial = _name_key(cited)
    surname = _letters(surname)
    if not surname:
        return True  # nothing to tell two people apart by
    for record in canonical:
        record_surname, record_initial = _name_key(record)
        record_surname = _letters(record_surname)
        if not (record_initial.isascii() and record_initial.isalpha()):
            record_initial = ""  # "†. TimBrooks": a footnote mark, not an initial
        if surname == record_surname or (
            len(surname) >= 4 and fuzz.ratio(surname, record_surname) >= 80
        ):
            if not (initial and record_initial and initial != record_initial):
                return True
            continue  # this surname, another person
        whole = _letters(record)
        if (len(surname) >= 3 and whole.endswith(surname)) or (len(surname) >= 4 and surname in whole):
            return True
    return False


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
    # Tolerate a lost hyphen/space ("Scale-Invariant" vs PDF-glued "scaleinvariant"):
    # compare with all spaces removed before flagging a mismatch.
    a3, b3 = a2.replace(" ", ""), b2.replace(" ", "")
    if a3 in b3 or b3 in a3:
        return True
    from rapidfuzz import fuzz

    return fuzz.token_sort_ratio(a2, b2) >= thresh


_INITIALS_TOKEN_RE = re.compile(r"^[A-Z](?:[.\-]?[A-Z]){0,3}\.?$")


# Letters that are not a base letter plus an accent, so NFKD leaves them as they are —
# and a record that writes "Lukasz" / "Krompass" for "Łukasz" / "Krompaß" never matches.
_LETTER_FOLD = str.maketrans({
    "ß": "ss", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe",
    "Œ": "OE", "đ": "d", "Đ": "D", "ı": "i", "þ": "th", "Þ": "Th", "ð": "d",
})


def _fold(s: str) -> str:
    """Lowercase + strip diacritics, for robust surname/initial comparison."""
    s = unicodedata.normalize("NFKD", (s or "").translate(_LETTER_FOLD))
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


_NAME_SUFFIX_RE = re.compile(r"[\s,]+(?:jr|sr|ii|iii|iv)\.?$", re.IGNORECASE)


def _name_tokens(author: str) -> frozenset[str]:
    """The words of a name, folded — for comparing two full names regardless of order."""
    return frozenset(t for t in re.findall(r"[^\W\d_]{2,}", _fold(author or "")))


def _same_author(cited: str, canonical: str) -> bool:
    """Same person: surnames match and any given-name initials agree.

    Two full names made of the same words in a different order are also one person:
    sources disagree on whether a family name leads ("Zhang Wei" / "Wei Zhang"), and
    reading that as a different surname reported a correct citation as a wrong author.
    """
    cited = _NAME_SUFFIX_RE.sub("", (cited or "").strip())
    canonical = _NAME_SUFFIX_RE.sub("", (canonical or "").strip())
    cs, ci = _name_key(cited)
    ks, ki = _name_key(canonical)
    if _surname_close(cs, ks):
        return ci == ki if (ci and ki) else True
    ct, kt = _name_tokens(cited), _name_tokens(canonical)
    return len(ct) >= 2 and ct == kt


def _norm(s: str) -> str:
    """Lowercase, diacritics folded, punctuation to spaces — for comparing titles/venues.

    Folding matters: bibliographies write "Schrodinger" or "Schr\\"odinger" for a title
    the record spells "Schrödinger", and unfolded those are different strings.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", _fold(s or ""))).strip()


def _join(authors: list[str]) -> str:
    return ", ".join(a for a in authors if a)


def _short(s: str, n: int = 60) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
