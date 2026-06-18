"""
agentic.py — the "agentic" backend: an explicit, staged pipeline.

This is the deterministic counterpart to the skill-driven ``claude_code``
backend. Instead of handing the whole job to the model, it walks each
record stub through fixed stages in a fixed order:

    extract stubs  ->  fill_correctness(record, resolver=...)
                       └─ if exists != no  ->  fill_relevance(record, resolver=..., judge=...)
                          └─ severity derived deterministically from the axes

Properties (per docs/decisions-phy.md):
  - **degrade-not-crash**: a stage that raises is caught per record; the record
    gets ``error`` set and keeps ``unverified`` axes, and the run continues.
  - **deterministic severity**: never judged here — always
    :func:`citation_verifier.schema.derive_severity` from the three axes.
  - **usage accounting**: every stage that "spends" is recorded into a per-tier
    :class:`RunUsage` (bulk for correctness, judge for relevance) so this
    backend is comparable to ``claude_code`` on tokens and cost.

The stage functions and the resolver come from sibling modules
(``citation_verifier.stages`` and ``citation_verifier.grounding``) via their
public API only — imported lazily inside :meth:`verify` so this module imports
even before those siblings land and without the SDK.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import PaperSource, RunUsage, VerificationResult
from ..schema import (
    CitationRecord,
    Exists,
    ModelTier,
    Priority,
    SupportsClaim,
    derive_severity,
)
from .base import BaseBackend, register
from .usage import estimate_tokens, record_tier_usage


@register
class AgenticBackend(BaseBackend):
    """Explicit staged pipeline backend (``name='agentic'``).

    No LLM and no SDK are required to *import* or to run the deterministic spine;
    the stages may internally call a judge model (relevance), but their failure
    is contained per record. Construction accepts an optional ``settings`` for
    model routing / pricing; everything has a safe default.
    """

    name = "agentic"

    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings
        self.bulk_model = _setting(settings, "model_bulk", "claude-haiku-4-5-20251001")
        self.judge_model = _setting(settings, "model_judge", "claude-opus-4-6")
        self.cost_ceiling = float(_setting(settings, "cost_ceiling_usd", 0.0) or 0.0)
        self.pricing = _setting(settings, "pricing", None)
        # The relevance seam. The deterministic baseline runs WITHOUT a judge and
        # honestly abstains on relevance. A judge fills supports_claim:
        #   - an explicitly injected `relevance_judge` wins (tests / custom), else
        #   - if ENABLE_RELEVANCE_JUDGE is on, build the LLM judge (Claude Code
        #     subscription; STEP 2 via abstract[+intro]). SDK absent => None => abstain.
        self.judge = _setting(settings, "relevance_judge", None)
        if self.judge is None and _setting(settings, "enable_relevance_judge", False):
            try:
                from .relevance_judge import build_relevance_judge
                self.judge = build_relevance_judge(settings)
            except Exception:  # noqa: BLE001 — never let judge wiring break the run
                self.judge = None

    # ──────────────────────────────────────────────────────────────
    def verify(
        self, source: PaperSource, stubs: list[CitationRecord]
    ) -> VerificationResult:
        """Two passes: correctness per record, then BATCHED relevance.

        Correctness is per-record (deterministic grounding). Relevance is a single
        pass over the records that resolved, so a batched LLM judge pays the
        per-call SDK/session overhead once per chunk instead of once per citation.
        """
        result = self._empty_result(source)
        usage: RunUsage = result.usage
        usage.model = f"bulk={self.bulk_model};judge={self.judge_model}"

        fill_correctness, fill_relevance, resolver = self._load_pipeline()

        with self._timer() as sw:
            # Pass 1 — correctness. Dedup identical references so each resolves
            # ONCE, and resolve the uniques concurrently (grounding is HTTP-bound).
            self._correctness_pass(stubs, fill_correctness, resolver, usage, result)

            # Cost ceiling: skip the expensive judge pass if correctness alone
            # already crossed it (rare — bulk usage is a cheap estimate).
            over_ceiling = bool(self.cost_ceiling and usage.cost_usd >= self.cost_ceiling)
            if over_ceiling:
                result.errors.append(
                    f"cost ceiling ${self.cost_ceiling:.2f} reached after correctness; "
                    "relevance pass skipped (records left unverified)"
                )
            else:
                # Pass 2 — relevance over records that resolved (exists != no).
                eligible = [r for r in result.records if _exists(r) is not Exists.NO]
                self._relevance_pass(eligible, fill_relevance, resolver, usage, result)

        # Fold a self-accounting LLM judge's REAL token/cost usage into the run.
        self._fold_judge_usage(usage)
        usage.wall_seconds = sw.seconds

        # Deterministic severity for every record (never judged).
        for rec in result.records:
            rec.severity = derive_severity(_exists(rec), _supports(rec), _priority(rec)).value
        self._stamp_paper_id(result.records, source.paper_id)
        return result

    # ──────────────────────────────────────────────────────────────
    def _correctness_pass(
        self,
        stubs: list[CitationRecord],
        fill_correctness: Any,
        resolver: Any,
        usage: RunUsage,
        result: VerificationResult,
    ) -> None:
        """STEP 1 for all stubs: dedup by reference, resolve uniques concurrently.

        A reference cited in N places resolves ONCE — the heavy part is the
        grounding HTTP — and the verified fields are copied onto its other records.
        Unique references are resolved in parallel (I/O-bound). Records are emitted
        in the original stub order, and bulk-tier usage is recorded per record on
        THIS thread (``record_tier_usage`` is not thread-safe).
        """
        if not stubs:
            return

        # Group records that share a reference; the first is the representative.
        groups: dict[str, list[CitationRecord]] = {}
        for stub in stubs:
            groups.setdefault(_dedup_key(stub), []).append(stub)
        reps = [members[0] for members in groups.values()]

        # Resolve each unique reference concurrently (degrade-not-crash per rep).
        workers = int(
            _setting(self.settings, "resolver_concurrency", _RESOLVER_CONCURRENCY)
            or _RESOLVER_CONCURRENCY
        )
        _parallel_each(
            reps, lambda r: self._resolve_one(r, fill_correctness, resolver, result), workers
        )

        # Copy each representative's verdict onto its siblings (deep copy so each
        # record owns its evidence/resolved — later stages append per record).
        for members in groups.values():
            for sib in members[1:]:
                _apply_correctness(members[0], sib)

        # Record bulk usage + finalize tier, in original stub order, on this thread.
        for stub in stubs:
            self._record_bulk(stub, usage)
            result.records.append(stub)

    def _resolve_one(
        self, rec: CitationRecord, fill_correctness: Any, resolver: Any, result: VerificationResult
    ) -> None:
        """Run STEP 1 grounding for one record (degrade-not-crash).

        Thread-safe to call from a worker: it mutates only ``rec`` and appends to
        ``result.errors`` (GIL-atomic); usage is recorded later on the main thread.
        ``fill_correctness`` mutates ``rec`` in place.
        """
        try:
            fill_correctness(rec, resolver=resolver)
        except Exception as exc:  # noqa: BLE001 — degrade-not-crash boundary
            rec.error = f"correctness stage failed: {exc}"
            result.errors.append(f"{rec.cite_key}: {rec.error}")

    def _record_bulk(self, rec: CitationRecord, usage: RunUsage) -> None:
        """Record the bulk-tier (correctness) usage slice for one record.

        Correctness makes no LLM call, so this is an estimate placeholder kept for
        cross-backend accounting parity (the ``claude_code`` backend reports real
        tokens). Also stamps the bulk tier when nothing else has.
        """
        record_tier_usage(
            usage,
            ModelTier.BULK,
            input_tokens=estimate_tokens(rec.cited_as.raw or rec.cite_key),
            output_tokens=_OUTPUT_TOKEN_FLOOR,
            tool_calls=1,
            model=self.bulk_model,
            pricing=self.pricing,
        )
        if rec.model_tier in (ModelTier.NONE.value, ModelTier.NONE):
            rec.model_tier = ModelTier.BULK.value

    # ──────────────────────────────────────────────────────────────
    def _relevance_pass(
        self,
        eligible: list[CitationRecord],
        fill_relevance: Any,
        resolver: Any,
        usage: RunUsage,
        result: VerificationResult,
    ) -> None:
        """STEP 2 over resolved records — batched when the judge supports it."""
        if not eligible:
            return

        # Batched judge: one query() per chunk amortizes the session overhead.
        if self.judge is not None and hasattr(self.judge, "judge_batch"):
            try:
                from ..stages.relevance import fill_relevance_batch

                fill_relevance_batch(eligible, resolver=resolver, judge_batch=self.judge.judge_batch)
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                result.errors.append(f"relevance batch failed: {exc!r}")
                for rec in eligible:
                    rec.supports_claim = SupportsClaim.UNVERIFIED.value
            for rec in eligible:
                rec.model_tier = ModelTier.JUDGE.value
            return

        # Per-record path: an injected non-batch judge, or deterministic abstain.
        for rec in eligible:
            try:
                fill_relevance(rec, resolver=resolver, judge=self.judge)
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                rec.error = (rec.error + "; " if rec.error else "") + f"relevance stage failed: {exc}"
                result.errors.append(f"{rec.cite_key}: relevance stage failed: {exc}")
            if self.judge is not None:
                rec.model_tier = ModelTier.JUDGE.value
                # Estimate only for judges that do NOT self-account (a real LLM
                # judge folds its true usage in _fold_judge_usage).
                if not hasattr(self.judge, "usage"):
                    record_tier_usage(
                        usage,
                        ModelTier.JUDGE,
                        input_tokens=estimate_tokens(rec.claim.text),
                        output_tokens=_OUTPUT_TOKEN_FLOOR,
                        tool_calls=0,
                        model=self.judge_model,
                        pricing=self.pricing,
                    )

    # ──────────────────────────────────────────────────────────────
    def _fold_judge_usage(self, usage: RunUsage) -> None:
        """Fold a self-accounting LLM judge's REAL usage into the run + JUDGE tier."""
        ju = getattr(self.judge, "usage", None)
        if ju is None:
            return
        usage.add(ju)
        bucket = usage.by_tier.get(ModelTier.JUDGE.value)
        if bucket is None:
            bucket = RunUsage(backend=self.name, model=ju.model)
            usage.by_tier[ModelTier.JUDGE.value] = bucket
        bucket.add(ju)

    # ──────────────────────────────────────────────────────────────
    def _load_pipeline(self) -> tuple[Any, Any, Any]:
        """Lazily import stage fns + resolver from siblings (public API only).

        Imported here (not at module top) so this backend imports even before
        the ``stages`` / ``grounding`` modules exist on a teammate's branch, and
        so neither sibling can drag the SDK into our import path. If a sibling is
        missing we fall back to safe no-op stages / a null resolver, so the
        backend still produces schema-valid ``unverified`` records.
        """
        try:
            from ..stages import fill_correctness, fill_relevance  # type: ignore
        except Exception:  # noqa: BLE001 — sibling not yet on this branch
            fill_correctness, fill_relevance = _noop_stage, _noop_stage
        try:
            from ..grounding import MultiSourceResolver  # type: ignore

            resolver: Any = MultiSourceResolver(settings=self.settings)
        except Exception:  # noqa: BLE001 — sibling not yet on this branch / no ctor kw
            resolver = _try_resolver()
        return fill_correctness, fill_relevance, resolver


# ───────────────────────────────────────────────────────────────
# Fallbacks (keep the spine runnable before siblings land)
# ───────────────────────────────────────────────────────────────
# A small positive output-token floor so usage is non-zero/populated even when
# stages are no-ops; replaced by real counts once stages report them.
_OUTPUT_TOKEN_FLOOR = 16


def _noop_stage(record: CitationRecord, /, **_kwargs: Any) -> CitationRecord:
    """Identity stage used when a real stage module is not importable yet."""
    if record.error is None:
        record.error = "stage module unavailable; record left unverified"
    return record


def _try_resolver() -> Any:
    """Best-effort resolver construction; ``None`` if grounding isn't ready."""
    try:
        from ..grounding import MultiSourceResolver  # type: ignore

        return MultiSourceResolver()
    except Exception:  # noqa: BLE001
        return None


def _setting(settings: Any | None, attr: str, default: Any) -> Any:
    """Read ``attr`` from a Settings object or mapping, else ``default``."""
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)


# Enum coercion helpers (records may store enum *values* due to use_enum_values).
def _exists(rec: CitationRecord) -> Exists:
    return Exists(rec.exists)


def _supports(rec: CitationRecord) -> SupportsClaim:
    return SupportsClaim(rec.supports_claim)


def _priority(rec: CitationRecord) -> Priority:
    return Priority(rec.priority)


# ───────────────────────────────────────────────────────────────
# Dedup + concurrency helpers (STEP 1 fan-out)
# ───────────────────────────────────────────────────────────────
_RESOLVER_CONCURRENCY = 8  # unique references resolved in parallel (HTTP-bound)


def _dedup_key(rec: CitationRecord) -> str:
    """Reference identity for dedup.

    Records that share this key are the SAME cited work (identical ``cited_as``),
    so STEP 1 grounding runs once and the result is copied to the rest. The
    cite_key already names one reference across its claim-sites; fall back to the
    raw reference string, then the (unique) claim_id so dedup is never lossy.
    """
    return rec.cite_key or (rec.cited_as.raw or "").strip().lower() or rec.claim_id


def _apply_correctness(rep: CitationRecord, sib: CitationRecord) -> None:
    """Copy STEP 1 results from a representative onto a sibling sharing its reference.

    Deep-copies the mutable ``evidence`` / ``resolved`` so each record owns its own
    (the relevance stage appends evidence per record); ``metadata_issues`` is a list
    of plain strings, so a shallow copy is enough. ``model_tier`` is left for
    :meth:`AgenticBackend._record_bulk` to stamp uniformly.
    """
    sib.exists = rep.exists
    sib.resolved = rep.resolved.model_copy(deep=True) if rep.resolved is not None else None
    sib.metadata_issues = list(rep.metadata_issues)
    sib.evidence = [e.model_copy(deep=True) for e in rep.evidence]
    if rep.error:
        sib.error = rep.error


def _parallel_each(items: list, fn, max_workers: int) -> None:
    """Run a side-effecting ``fn`` over ``items`` concurrently (I/O-bound).

    Per-item exceptions are swallowed (``fn`` degrades-not-crashes internally).
    Degrades to a serial loop for a single item / single worker.
    """
    if not items:
        return
    workers = max(1, min(int(max_workers), len(items)))
    if workers == 1:
        for it in items:
            fn(it)
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in [ex.submit(fn, it) for it in items]:
            try:
                f.result()
            except Exception:  # noqa: BLE001 — degrade-not-crash per item
                pass


__all__ = ["AgenticBackend"]
