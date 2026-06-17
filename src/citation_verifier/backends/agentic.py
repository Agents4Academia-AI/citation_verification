"""
agentic.py — the "agentic" backend: an explicit, staged pipeline.

This is the deterministic counterpart to the skill-driven ``claude_code``
backend. Instead of handing the whole job to an Agent-SDK loop, it walks each
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
            # Pass 1 — correctness (bulk tier), per record.
            for stub in stubs:
                rec = self._correctness_one(stub, fill_correctness, resolver, usage, result)
                result.records.append(rec)
                if self.cost_ceiling and usage.cost_usd >= self.cost_ceiling:
                    result.errors.append(
                        f"cost ceiling ${self.cost_ceiling:.2f} reached after correctness on "
                        f"{len(result.records)}/{len(stubs)} records; remaining left unverified"
                    )
                    result.records.extend(stubs[len(result.records):])
                    break

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
    def _correctness_one(
        self,
        rec: CitationRecord,
        fill_correctness: Any,
        resolver: Any,
        usage: RunUsage,
        result: VerificationResult,
    ) -> CitationRecord:
        """STEP 1 for one record (degrade-not-crash); records bulk-tier usage."""
        try:
            rec = fill_correctness(rec, resolver=resolver)
        except Exception as exc:  # noqa: BLE001 — degrade-not-crash boundary
            rec.error = f"correctness stage failed: {exc}"
            result.errors.append(f"{rec.cite_key}: {rec.error}")
        finally:
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
        return rec

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


__all__ = ["AgenticBackend"]
