"""
relevance — STEP 2 (Relevance & Justification + Priority).

:func:`fill_relevance` decides whether the cited work supports the *specific*
claim it is attached to, sets the citation's priority (obligatory vs helpful),
and attaches the evidence + justification backing the verdict. It fills:

    record.supports_claim : supports | partial | does_not | unverified
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
  - **Supports-claim** ABSTAINS to ``unverified`` unless a retrieved abstract is
    available; with an abstract present but no judge, it still abstains rather
    than guess (the skill's "never decide relevance without reading" rule). The
    abstract is recorded as evidence so a later judge pass can adjudicate.

Per docs/decisions-phy.md: a relevance verdict must rest on *retrieved* evidence
(the resolved abstract), never on model memory; absent evidence => ``unverified``.

Degrade-not-crash: any judge/processing error sets ``record.error`` and leaves
``supports_claim`` at ``unverified``.

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

    __slots__ = ("supports_claim", "priority", "confidence", "justification", "model_tier")

    def __init__(
        self,
        supports_claim: SupportsClaim | str = SupportsClaim.UNVERIFIED,
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
        and ``supports_claim`` is left at ``unverified`` (degrade-not-crash).
    """
    try:
        return _fill_relevance(record, resolver=resolver, judge=judge)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash
        record.error = f"relevance: {exc!r}"
        record.supports_claim = SupportsClaim.UNVERIFIED
        return record


def _fill_relevance(
    record: CitationRecord, *, resolver: Resolver, judge: RelevanceJudge | None
) -> CitationRecord:
    # Always set a priority (coarse heuristic; a judge may override below).
    record.priority = _infer_priority(record.claim.text)

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
    record.supports_claim = SupportsClaim.UNVERIFIED
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
    ``unverified``; a missing/`None` verdict abstains that one record.

    This exists so an LLM judge pays the per-call SDK/session overhead once per
    chunk instead of once per citation (see backends/relevance_judge.py).
    """
    if not records:
        return records

    items: list[dict] = []
    for rec in records:
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
        for rec in records:
            rec.error = (rec.error + "; " if rec.error else "") + f"relevance batch: {exc!r}"
            rec.supports_claim = SupportsClaim.UNVERIFIED
        return records

    for rec, item, verdict in zip(records, items, verdicts, strict=False):
        if verdict is None:
            rec.supports_claim = SupportsClaim.UNVERIFIED
            continue
        rec.supports_claim = verdict.supports_claim
        if verdict.priority is not None:
            rec.priority = verdict.priority
        rec.confidence = verdict.confidence
        rec.model_tier = verdict.model_tier
        if verdict.justification:
            rec.notes = (rec.notes + "\n" if rec.notes else "") + verdict.justification
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


def _add_evidence(items: list[Evidence], new: Evidence) -> list[Evidence]:
    for e in items:
        if e.kind == new.kind and e.source == new.source and e.quote == new.quote:
            return items
    return items + [new]


def _snippet(text: str, limit: int = 600) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit] + ("…" if len(text) > limit else "")
