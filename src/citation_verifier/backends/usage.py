"""
usage.py — token / cost accounting for the two backends.

The product compares a "claude_code" backend (skill-driven Agent-SDK loop)
against an "agentic" backend (explicit staged pipeline) on the SAME output
schema, *including token usage and USD cost* (the Notion "FURTHER DIRECTION").
This module is the one place that:

  - maps a Claude Agent-SDK ``ResultMessage`` to the contract's
    :class:`~citation_verifier.interfaces.RunUsage`
    (:func:`usage_from_result_message`);
  - prices a ``RunUsage`` deterministically from a per-model pricing table
    (:func:`estimate_cost`), used by the agentic backend and as a fallback
    when the SDK reports tokens but not cost;
  - estimates token counts when no usage is reported at all
    (:func:`estimate_tokens`), so the agentic pipeline (which calls thin/typed
    judge stubs rather than the SDK) still produces a populated ``RunUsage``.

Pure stdlib + the contract. Import-safe without claude-agent-sdk and without
network. Pricing numbers are *defaults* (USD per million tokens) and are meant
to be overridden from ``config`` / ``.env`` so they can be kept current without
a code change.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import RunUsage
from ..schema import ModelTier

# ───────────────────────────────────────────────────────────────
# Pricing table — USD per 1,000,000 tokens, as (input, output).
# Keys are matched by case-insensitive substring against the model id so that
# dated aliases (e.g. "claude-opus-4-5-20251101") resolve to the family price.
# Override at runtime via `estimate_cost(..., pricing=...)` (Settings injects it).
# These are conservative defaults, NOT a quote; keep them current via config.
# ───────────────────────────────────────────────────────────────
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-opus": (15.0, 75.0),
}

# Fallback price (USD/Mtok) when a model id matches nothing in PRICING.
_DEFAULT_PRICE: tuple[float, float] = (3.0, 15.0)

# Rough chars-per-token used only when NO token count is available at all.
# 4 chars/token is the long-standing English-text rule of thumb; this is a
# floor estimate, explicitly marked, never a substitute for a real count.
_CHARS_PER_TOKEN = 4.0


def price_for(model: str, pricing: dict[str, tuple[float, float]] | None = None) -> tuple[float, float]:
    """Return ``(input_per_mtok, output_per_mtok)`` for ``model``.

    Matching is case-insensitive substring against the table keys (longest key
    first so ``claude-3-5-haiku`` wins over a hypothetical ``claude-3``). Falls
    back to :data:`_DEFAULT_PRICE` for unknown models so cost is never crashy.
    """
    table = pricing or PRICING
    name = (model or "").lower()
    for key in sorted(table, key=len, reverse=True):
        if key.lower() in name:
            return table[key]
    return _DEFAULT_PRICE


def estimate_cost(
    usage: RunUsage,
    model: str | None = None,
    *,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Compute USD cost for ``usage`` from token counts and the pricing table.

    Deterministic and side-effect free: ``tokens -> cost`` always yields the
    same value for the same inputs (the acceptance "round-trips deterministically"
    check). Cache-read tokens are billed at the input rate as a conservative
    approximation; refine in :data:`PRICING` if a cache tier is configured.

    ``model`` defaults to ``usage.model`` when omitted.
    """
    in_rate, out_rate = price_for(model or usage.model, pricing)
    billable_input = usage.input_tokens + usage.cache_read_tokens + usage.cache_creation_tokens
    cost = (billable_input * in_rate + usage.output_tokens * out_rate) / 1_000_000
    return round(cost, 6)


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text`` (~4 chars/token, marked-approximate).

    Used as a fallback ONLY when the SDK / API does not report usage — e.g. the
    agentic pipeline's deterministic stages or a thin judge stub. The estimate
    is intentionally crude and should be replaced by a real count
    (``client.messages.count_tokens``) when a live model is wired in.
    """
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def usage_from_result_message(msg: Any, model: str, backend: str) -> RunUsage:
    """Map an Agent-SDK ``ResultMessage`` to a :class:`RunUsage`.

    The SDK's ``ResultMessage`` exposes (across versions) some subset of:
    ``usage`` (a dict with ``input_tokens`` / ``output_tokens`` and optional
    cache fields), ``total_cost_usd``, ``num_turns``, and ``duration_ms``. We
    read defensively via ``getattr`` / ``dict.get`` so a missing field degrades
    to zero rather than raising, and we fall back to :func:`estimate_cost` when
    the SDK gives tokens but no cost. ``msg`` is typed ``Any`` so this module
    never needs the SDK at import time.
    """
    raw = getattr(msg, "usage", None) or {}
    if not isinstance(raw, dict):
        # Some SDK builds expose a usage object rather than a dict.
        raw = {
            "input_tokens": getattr(raw, "input_tokens", 0),
            "output_tokens": getattr(raw, "output_tokens", 0),
            "cache_read_input_tokens": getattr(raw, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(raw, "cache_creation_input_tokens", 0),
        }

    usage = RunUsage(
        backend=backend,
        model=model,
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
        cache_creation_tokens=int(raw.get("cache_creation_input_tokens", 0) or 0),
        num_turns=int(getattr(msg, "num_turns", 0) or 0),
        wall_seconds=float(getattr(msg, "duration_ms", 0) or 0) / 1000.0,
    )

    reported_cost = getattr(msg, "total_cost_usd", None)
    usage.cost_usd = (
        float(reported_cost)
        if reported_cost is not None
        else estimate_cost(usage, model)
    )
    return usage


def record_tier_usage(
    total: RunUsage,
    tier: ModelTier | str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_calls: int = 0,
    model: str = "",
    pricing: dict[str, tuple[float, float]] | None = None,
) -> RunUsage:
    """Add a per-stage/per-tier slice to ``total`` and return that slice.

    Accumulates into ``total`` (input/output/cost/tool_calls) AND into
    ``total.by_tier[tier]`` so the two-tier routing breakdown (bulk vs judge)
    can be reported per run, per the team decision to log ``model_tier``.
    The returned slice is the freshly-recorded usage for this call.
    """
    # ModelTier is a str-Enum, so isinstance(tier, str) is always True; coerce
    # through ModelTier to guarantee a plain string key (e.g. "bulk"/"judge").
    tier_name = ModelTier(tier).value
    slice_usage = RunUsage(
        backend=total.backend,
        model=model or total.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tool_calls,
    )
    slice_usage.cost_usd = estimate_cost(slice_usage, slice_usage.model, pricing=pricing)

    # Accumulate into the running total.
    total.add(slice_usage)

    # Accumulate into the per-tier bucket.
    bucket = total.by_tier.get(tier_name)
    if bucket is None:
        bucket = RunUsage(backend=total.backend, model=slice_usage.model)
        total.by_tier[tier_name] = bucket
    bucket.add(slice_usage)
    return slice_usage


__all__ = [
    "PRICING",
    "price_for",
    "estimate_cost",
    "estimate_tokens",
    "usage_from_result_message",
    "record_tier_usage",
]
