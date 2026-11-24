"""
relevance — STEP 2 (Relevance & Justification + Priority).

:func:`fill_relevance` decides whether the cited work supports the *specific*
claim it is attached to, sets the citation's priority (obligatory vs helpful),
and attaches the evidence + justification backing the verdict. It fills:

    record.supports_claim : supports | partial | does_not | inconclusive
    record.priority       : obligatory | helpful
    record.evidence       : the abstract/snippet the verdict rests on
    record.confidence     : 0..1
    record.model_tier     : which routed model produced the judgement

This is the stage where the **LLM / verify-citations skill** does its work. To
keep core import-safe and offline-runnable, the LLM is injected as an optional
``judge`` callable (a clean injection point — the agentic backend wires a real,
two-tier-routed judge here). Without a judge, a deterministic fallback runs:

  - **Priority** is inferred from cue words in the claim sentence (a method/
    baseline/dataset/"X showed that" => obligatory; survey/"see also"/background
    => helpful). This is a coarse but honest heuristic, not a hidden LLM.
  - **Supports-claim** ABSTAINS to ``inconclusive`` unless a retrieved abstract is
    available; with an abstract present but no judge, it still abstains rather
    than guess (the skill's "never decide relevance without reading" rule). The
    abstract is recorded as evidence so a later judge pass can adjudicate.

Per docs/decisions-phy.md: a relevance verdict must rest on *retrieved* evidence
(the resolved abstract), never on model memory; absent evidence => ``inconclusive``.

Degrade-not-crash: any judge/processing error sets ``record.error`` and leaves
``supports_claim`` at ``inconclusive``.

NOTE on the "construct-a-skill / justification" direction (SKILL.md): the judge
callable is the seam for the relevance-justification skill — it receives the
claim + the retrieved abstract and returns a verdict with a quoted justification,
which we store in ``record.notes`` and ``record.evidence``.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ..interfaces import Resolver
from ..schema import (
    CitationRecord,
    Evidence,
    ModelTier,
    Priority,
    SupportsClaim,
)


@runtime_checkable
class RelevanceJudge(Protocol):
    """Injection point for the LLM / verify-citations relevance skill.

    A judge takes the claim sentence and the retrieved abstract (+ optional
    cited/canonical metadata) and returns a structured verdict. Implementations
    live in the backends layer (lazy SDK import); this module never imports one.
    """

    def __call__(
        self,
        *,
        claim: str,
        abstract: str,
        cited_title: str = "",
        resolved_title: str = "",
        resolved: object | None = None,
    ) -> RelevanceVerdict: ...


class RelevanceVerdict:
    """Plain result object a :class:`RelevanceJudge` returns.

    Kept as a small concrete class (not a dataclass import dance) so backends can
    construct it without importing pydantic; all fields are optional with sane
    defaults so a partial judge response still degrades cleanly.
    """

    __slots__ = ("supports_claim", "priority", "confidence", "justification", "model_tier", "basis", "missing")

    def __init__(
        self,
        supports_claim: SupportsClaim | str = SupportsClaim.INCONCLUSIVE,
        priority: Priority | str | None = None,
        confidence: float | None = None,
        justification: str = "",
        model_tier: ModelTier | str = ModelTier.JUDGE,
    ) -> None:
        self.supports_claim = SupportsClaim(supports_claim)
        self.priority = Priority(priority) if priority is not None else None
        self.confidence = confidence
        self.justification = justification
        self.model_tier = ModelTier(model_tier)
        # What the judge was shown, for the report's "(based on …)" label; None = abstract.
        self.basis: str | None = None
        # True when no judgement was made at all (the judge's answer left this claim out,
        # or there was nothing to judge from) — a placeholder, not a verdict.
        self.missing = False


# Cue words for the deterministic priority heuristic (lowercased substrings).
_OBLIGATORY_CUES = (
    "we use",
    "we adopt",
    "we extend",
    "we build on",
    "based on",
    "baseline",
    "compared to",
    "compared with",
    "outperform",
    "state-of-the-art",
    "we follow",
    "dataset",
    "benchmark",
    "showed that",
    "demonstrated that",
    "proved",
    "as shown in",
    "introduced by",
    "proposed by",
)
_HELPFUL_CUES = (
    "see also",
    "for a survey",
    "survey",
    "overview",
    "background",
    "for example",
    "e.g.",
    "among others",
    "inter alia",
    "for instance",
)


def fill_relevance(
    record: CitationRecord, *, resolver: Resolver, judge: RelevanceJudge | None = None
) -> CitationRecord:
    """Fill ``supports_claim`` / ``priority`` / ``evidence`` / ``confidence``.

    Args:
        record: a record that has (ideally) already passed
            :func:`citation_verifier.stages.correctness.fill_correctness`, so
            ``record.resolved.abstract`` may be available as evidence.
        resolver: the :class:`Resolver` Protocol object (used as a fallback to
            retrieve an abstract if correctness did not populate one).
        judge: optional LLM relevance judge (the skill seam). ``None`` => the
            deterministic fallback runs (priority heuristic + abstain on support).

    Returns:
        The same record, mutated in place. On any error, ``record.error`` is set
        and ``supports_claim`` is left at ``inconclusive`` (degrade-not-crash).
    """
    try:
        return _fill_relevance(record, resolver=resolver, judge=judge)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash
        record.error = f"relevance: {exc!r}"
        record.supports_claim = SupportsClaim.INCONCLUSIVE
        return record


def _fill_relevance(
    record: CitationRecord, *, resolver: Resolver, judge: RelevanceJudge | None
) -> CitationRecord:
    # Always set a priority (coarse heuristic; a judge may override below).
    record.priority = _infer_priority(record.claim.text)

    # A table/figure citation is a table cell/caption, not a prose claim — do not judge
    # its relevance. SKIPPED (not "inconclusive"): no verdict, excluded from the
    # distribution and listed in a separate report appendix.
    if record.in_table:
        record.supports_claim = SupportsClaim.SKIPPED
        return record

    abstract = _retrieve_abstract(record, resolver)

    if judge is not None:
        verdict = judge(
            claim=record.claim.text,
            abstract=abstract,
            cited_title=record.cited_as.title or "",
            resolved_title=(record.resolved.title if record.resolved else "") or "",
            resolved=record.resolved,
        )
        record.supports_claim = verdict.supports_claim
        if verdict.priority is not None:
            record.priority = verdict.priority
        record.confidence = verdict.confidence
        record.model_tier = verdict.model_tier
        if verdict.justification:
            record.notes = (record.notes + "\n" if record.notes else "") + verdict.justification
        if abstract:
            record.evidence = _add_evidence(record.evidence, _abstract_evidence(record, abstract))
        return record

    # Deterministic fallback: no judge => abstain on support, but keep evidence.
    record.model_tier = ModelTier.NONE
    if abstract:
        record.evidence = _add_evidence(record.evidence, _abstract_evidence(record, abstract))
    # Honest abstain: support requires reading, which only the judge does.
    record.supports_claim = SupportsClaim.INCONCLUSIVE
    record.confidence = None
    return record


def fill_relevance_batch(
    records: list[CitationRecord], *, resolver: Resolver, judge_batch
) -> list[CitationRecord]:
    """Batched STEP 2: judge many records in one go (the judge chunks internally).

    For each record we set the priority heuristic and retrieve its abstract, then
    hand the whole list to ``judge_batch`` — a callable taking
    ``[{"claim","abstract","resolved"}]`` and returning a list of
    :class:`RelevanceVerdict` aligned to the input. Verdicts are applied back in
    place. Degrade-not-crash: on a batch error every record abstains to
    ``inconclusive``; a missing/`None` verdict abstains that one record.

    This exists so an LLM judge pays the per-call SDK/session overhead once per
    chunk instead of once per citation (see backends/relevance_judge.py).
    """
    if not records:
        return records

    # Table citations are table cells/captions, not prose claims — never judge
    # them; leave inconclusive (their extraction note already explains why). Judge
    # only the rest, and keep all records for the return.
    judged = [rec for rec in records if not rec.in_table]
    for rec in records:
        if rec.in_table:
            rec.priority = _infer_priority(rec.claim.text)
            rec.supports_claim = SupportsClaim.SKIPPED  # table/figure: no verdict, see appendix
    if not judged:
        return records

    items: list[dict] = []
    for rec in judged:
        rec.priority = _infer_priority(rec.claim.text)
        abstract = _retrieve_abstract(rec, resolver)
        # cite_key + claim_id let a batched judge group claim-sites by citation and
        # send each citation's evidence once (see backends/relevance_judge.py).
        items.append(
            {
                "cite_key": rec.cite_key,
                "claim_id": rec.claim_id,
                "claim": rec.claim.text,
                "abstract": abstract,
                "resolved": rec.resolved,
            }
        )

    try:
        verdicts = judge_batch(items)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash
        for rec in judged:
            rec.error = (rec.error + "; " if rec.error else "") + f"relevance batch: {exc!r}"
            rec.supports_claim = SupportsClaim.INCONCLUSIVE
        return records

    # Stage 2: re-judge the abstract-uncertain verdicts against full-text chunks.
    verdicts, escalated = _escalate_to_full_text(judged, verdicts, judge_batch)

    for pos, (rec, item, verdict) in enumerate(zip(judged, items, verdicts, strict=False)):
        if verdict is None:
            rec.supports_claim = SupportsClaim.INCONCLUSIVE
            continue
        rec.supports_claim = verdict.supports_claim
        if verdict.priority is not None:
            rec.priority = verdict.priority
        rec.confidence = verdict.confidence
        rec.model_tier = verdict.model_tier
        if verdict.justification:
            if pos in escalated:
                from ..grounding.fulltext import PREVIEW_LABEL  # noqa: PLC0415

                sec, notes = escalated[pos]
                what = "the cited work's first pages" if PREVIEW_LABEL in notes else "full text"
                basis = (f"{what} §{sec}" if sec else what) + notes
            else:
                basis = getattr(verdict, "basis", None) or "abstract only"
            note = f"(based on {basis}) {verdict.justification}"
            rec.notes = (rec.notes + "\n" if rec.notes else "") + note
        if item["abstract"]:
            rec.evidence = _add_evidence(rec.evidence, _abstract_evidence(rec, item["abstract"]))
    return records


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _infer_priority(claim_text: str) -> Priority:
    """Coarse obligatory/helpful inference from claim-sentence cue words."""
    text = (claim_text or "").lower()
    if not text:
        return Priority.HELPFUL
    if any(cue in text for cue in _HELPFUL_CUES) and not any(c in text for c in _OBLIGATORY_CUES):
        return Priority.HELPFUL
    if any(cue in text for cue in _OBLIGATORY_CUES):
        return Priority.OBLIGATORY
    return Priority.HELPFUL


def _retrieve_abstract(record: CitationRecord, resolver: Resolver) -> str:
    """Prefer an abstract already on ``record.resolved``; else try the resolver."""
    if record.resolved and record.resolved.abstract:
        return record.resolved.abstract
    reference = (record.cited_as.raw or record.cited_as.title or "").strip()
    if not reference:
        return ""
    try:
        resolved = resolver.resolve(record.cite_key, reference)
    except Exception:  # noqa: BLE001
        return ""
    if resolved and resolved.abstract:
        # Backfill resolved if correctness skipped it, but do not clobber a match.
        if record.resolved is None:
            record.resolved = resolved
        return resolved.abstract
    return ""


def _abstract_evidence(record: CitationRecord, abstract: str) -> Evidence:
    src = ""
    if record.resolved:
        src = record.resolved.doi or record.resolved.arxiv_id or record.resolved.url or (
            record.resolved.source or ""
        )
    return Evidence(
        kind="abstract",
        source=str(src or "structured"),
        quote=_snippet(abstract),
        url=record.resolved.url if record.resolved else None,
    )


_FULLTEXT_CHUNKS = 8  # claim-relevant chunks per escalated citation, for a paper too long to send whole
# Up to this length the WHOLE paper is the evidence. Chunks picked by word overlap with
# the claim missed the one sentence that settled it — "temperature scaling does not change
# the argmax, so accuracy is unaffected" shares few words with "leaves top-1 predictions
# unchanged" — and five such claims stayed "inconclusive" with the paper in hand.
_WHOLE_PAPER_CHARS = 100_000


def _escalate_to_full_text(
    records: list[CitationRecord], verdicts: list, judge_batch
) -> tuple[list, dict[int, tuple[str, str]]]:
    """Stage 2: re-judge abstract-uncertain records against the cited paper's full text.

    Escalates the verdicts the abstract can't be trusted on — ``inconclusive``,
    ``does_not`` (the body often supports a claim the abstract omits), and
    ``partial`` on a methods/data/results claim — by fetching the cited paper's full
    text, selecting the few claim-relevant chunks, and re-judging them in one batched
    call. A clear abstract-level ``supports`` is terminal and is not escalated.
    Returns the (mutated) verdict list plus ``{pos: (section_heading, source_notes)}`` for
    every record re-judged on full text (for the basis label + provenance). Fail-soft: a record with
    no id / no fetchable text / a judge error keeps its Stage-1 verdict.
    """
    from ..grounding.fulltext import (
        PREVIEW_LABEL,
        FullTextResult,
        full_text_with_trail,
        select_evidence_chunks,
        source_notes,
        split_sections,
    )

    # fetch each cited paper at most once per run; value carries text + provenance
    fulltext_cache: dict[str, FullTextResult] = {}
    esc_items: list[dict] = []
    esc_meta: list[tuple[int, list[tuple[str, str]], str, str]] = []
    for pos, (rec, verdict) in enumerate(zip(records, verdicts, strict=False)):
        if verdict is None:
            continue
        sc = getattr(verdict.supports_claim, "value", verdict.supports_claim)
        # Escalate to full text when the abstract is not enough to be confident:
        #   - inconclusive: the abstract couldn't decide;
        #   - does_not: the most consequential verdict — the body often supports a
        #     claim the abstract never mentions, so confirm before accusing of miscite;
        #   - partial: "related but weaker than the claim" said on an abstract and an
        #     introduction is a statement about what those two parts contain. It was
        #     escalated only for claims a keyword list took for methods/data/results, and
        #     a third of one batch's pairs stopped at "partial" with the body unread.
        # A clear abstract-level `supports` is terminal and skips this.
        escalate = sc in (
            SupportsClaim.INCONCLUSIVE.value, SupportsClaim.DOES_NOT.value, SupportsClaim.PARTIAL.value
        )
        if not escalate:
            continue
        # Full-text source, by precision/cost (each cited paper fetched once), through
        # the same cascade the judge reads the introduction from:
        #   1. arXiv e-print/HTML/PDF by id;
        #   2. the record's link — a PDF, or a page that links the work's PDF;
        #   3. the open-access cascade by DOI (OpenAlex / Unpaywall / landing page);
        #   4. a web search for a free, title-verified PDF (when a search is wired).
        # A scan is read by a model when the run installed a reader.
        res = rec.resolved
        if not res:
            continue
        cited_title = res.title or rec.cited_as.title or ""
        key = (
            res.arxiv_id
            or res.url
            or (f"doi:{res.doi}" if res.doi else "")
            or (f"search:{cited_title}" if cited_title else "")
        )
        if not key:
            continue
        if key not in fulltext_cache:
            try:
                fulltext_cache[key] = full_text_with_trail(res)
            except Exception:  # noqa: BLE001 — fail soft: the stage-1 verdict stands
                fulltext_cache[key] = FullTextResult("")
        ftr = fulltext_cache[key]
        if not ftr:
            continue
        sections = split_sections(ftr.text)
        if len(ftr.text) <= _WHOLE_PAPER_CHARS:
            chunks = select_evidence_chunks(rec.claim.text, sections, k=1)  # for the label
            # A publisher's preview is the first pages: called the complete text, what it
            # does not say would read as what the work does not say.
            head = (
                "[the first pages of the cited work only — a publisher preview; the rest "
                "of the work was not available]"
                if PREVIEW_LABEL in (ftr.source or "")
                else "[complete text of the cited paper]"
            )
            evidence = f"{head}\n{ftr.text}"
        else:
            chunks = select_evidence_chunks(rec.claim.text, sections, k=_FULLTEXT_CHUNKS)
            if not chunks:
                continue
            evidence = "\n\n".join(f"[{h or 'body'}] {c}" for h, c in chunks)
        esc_items.append(
            {
                "cite_key": rec.cite_key,
                "claim_id": rec.claim_id,
                "claim": rec.claim.text,
                "abstract": evidence,
                "resolved": rec.resolved,
            }
        )
        esc_meta.append((pos, chunks, ftr.source, ftr.url))

    if not esc_items:
        return verdicts, {}
    try:
        esc_verdicts = judge_batch(esc_items)
    except Exception:  # noqa: BLE001 — degrade-not-crash: keep the Stage-1 verdicts
        return verdicts, {}

    escalated: dict[int, tuple[str, str]] = {}  # pos -> (top section heading, source notes)
    for (pos, chunks, source, url), v2 in zip(esc_meta, esc_verdicts, strict=False):
        # A claim the full-text answer left out keeps its first verdict: "the judge
        # returned no verdict" replaced a real one in an earlier run.
        if v2 is None or getattr(v2, "missing", False):
            continue
        verdicts[pos] = v2
        # A preview is not the whole work and a transcription is not the printed text:
        # the basis says so.
        escalated[pos] = ((chunks[0][0] if chunks else "") or "", source_notes(source))
        records[pos].evidence = _add_evidence(
            records[pos].evidence, _fulltext_evidence(records[pos], chunks, source=source, url=url)
        )
    return verdicts, escalated


def _fulltext_evidence(
    record: CitationRecord, chunks: list[tuple[str, str]], *, source: str = "", url: str = ""
) -> Evidence:
    """Package retrieved full-text chunks as Evidence, tagged with channel + section.

    ``source`` is the full-text channel (``arxiv_html``/``unpaywall``/… ); falls
    back to the resolved id/url when a caller didn't supply one. The section heading
    comes from the top-ranked chunk, so provenance reads e.g. ``unpaywall §Introduction``.
    """
    heading = (chunks[0][0] if chunks else "") or "full text"
    channel = source
    if not channel and record.resolved:
        channel = record.resolved.arxiv_id or record.resolved.url or ""
    return Evidence(
        kind="full_text",
        source=f"{channel or 'arxiv'} §{heading}",
        quote=_snippet("  ".join(c for _, c in chunks)),
        url=url or (record.resolved.url if record.resolved else None),
    )


def _add_evidence(items: list[Evidence], new: Evidence) -> list[Evidence]:
    for e in items:
        if e.kind == new.kind and e.source == new.source and e.quote == new.quote:
            return items
    return items + [new]


def _snippet(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit] + ("…" if len(text) > limit else "")
