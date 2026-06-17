"""
config.py — runtime settings for the citation verifier.

Settings are read from the process environment plus an optional ``.env`` file
(parsed with the stdlib — no python-dotenv dependency). NOTHING here is required
to run the keyless floor (schema + render + eval + Crossref/arXiv grounding):
every field has a safe default and loading NEVER raises on a missing key.

Two-tier model routing lives here: ``model_bulk`` (cheap, high-throughput
correctness) and ``model_judge`` (strong relevance/justification). The mapping
``ModelTier -> model id`` is :func:`model_for`. Model ids are config, never
hard-coded into the backends.

Secrets (API keys) only ever come from the environment / a gitignored ``.env``;
they are never written to disk by this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from .schema import ModelTier

__all__ = ["Settings", "load_settings", "model_for", "apply_auth"]

# Default per-tier model ids. Overridable via MODEL_BULK / MODEL_JUDGE so the
# real model strings live in config, not in source.
_DEFAULT_MODEL_BULK = "claude-haiku-4-5-20251001"
_DEFAULT_MODEL_JUDGE = "claude-opus-4-6"


class Settings(BaseModel):
    """Resolved runtime configuration. Construct via :func:`load_settings`.

    All fields are optional with safe defaults; the keyless floor runs with an
    all-defaults instance. Keys default to ``None`` (absent) rather than empty
    strings so callers can branch on "is this source configured?".
    """

    # ── LLM backend (only the 'claude_code' / live backends need a key) ──
    anthropic_api_key: str | None = Field(
        default=None, description="Claude API key; absent => use an authed Claude Code install or skip live runs."
    )
    model_bulk: str = Field(
        default=_DEFAULT_MODEL_BULK, description="Cheap, high-throughput model for the BULK (correctness) tier."
    )
    model_judge: str = Field(
        default=_DEFAULT_MODEL_JUDGE, description="Strong model for the JUDGE (relevance/justification) tier."
    )
    cost_ceiling_usd: float = Field(
        default=0.50, ge=0.0, description="Per-run USD ceiling; 0 disables the cap. Orchestrator aborts gracefully past it."
    )
    enable_relevance_judge: bool = Field(
        default=False,
        description="Turn on the LLM relevance judge (STEP 2) in the 'agentic' backend. "
        "False => agentic stays keyless and abstains on supports_claim.",
    )

    # ── Grounding sources ──
    crossref_mailto: str | None = Field(
        default=None, description="Email for Crossref's polite pool (no key needed; just higher rate limits)."
    )
    s2_api_key: str | None = Field(default=None, description="Semantic Scholar key; absent => S2 source skipped.")
    openalex_api_key: str | None = Field(default=None, description="OpenAlex key; absent => OpenAlex source skipped.")
    enable_web_search: bool = Field(
        default=False, description="Gate for last-resort web search/fetch grounding. False => never touch the web."
    )

    # ── Paths ──
    papers_dir: Path = Field(default=Path("papers"), description="Per-paper artifact dir (source, refs, report.json, run.json).")
    chbench_data_dir: Path = Field(
        default=Path("/scratch/datasets/chbench"),
        description="Full CitationHallucinationBench (off-repo). Smoke gold ships under evals/smoke/.",
    )

    model_config = {"extra": "ignore"}


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` ``.env`` file (stdlib only, fail-soft).

    Supports blank lines, ``#`` comments, ``export KEY=...`` prefixes and
    single/double-quoted values. Never raises: an unreadable file yields ``{}``.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _opt(value: str | None) -> str | None:
    """Treat empty/whitespace env values as absent (None)."""
    if value is None:
        return None
    value = value.strip()
    return value or None


def _as_bool(value: str | None, default: bool) -> bool:
    """Parse a truthy/falsey env string; fall back to ``default`` when unset."""
    if value is None or not value.strip():
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _as_float(value: str | None, default: float) -> float:
    """Parse a float env string; fall back to ``default`` on junk."""
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Build :class:`Settings` from ``os.environ`` plus an optional ``.env``.

    Precedence: the real process environment wins over the ``.env`` file (the
    file only fills gaps). Missing keys are fine — defaults apply. This function
    never imports the SDK and never raises on a missing/blank key.

    Args:
        env_file: Path to a ``.env`` file to layer under the environment, or
            ``None`` to use the process environment only.

    Returns:
        A fully-populated, validated :class:`Settings`.
    """
    file_env: dict[str, str] = {}
    if env_file is not None:
        file_env = _parse_env_file(Path(env_file))

    def get(name: str) -> str | None:
        # os.environ wins; .env fills the gap.
        return os.environ.get(name, file_env.get(name))

    papers = _opt(get("PAPERS_DIR")) or "papers"
    chbench = _opt(get("CHBENCH_DATA_DIR")) or "/scratch/datasets/chbench"

    return Settings(
        anthropic_api_key=_opt(get("ANTHROPIC_API_KEY")),
        model_bulk=_opt(get("MODEL_BULK")) or _DEFAULT_MODEL_BULK,
        model_judge=_opt(get("MODEL_JUDGE")) or _DEFAULT_MODEL_JUDGE,
        cost_ceiling_usd=_as_float(get("COST_CEILING_USD"), 0.50),
        enable_relevance_judge=_as_bool(get("ENABLE_RELEVANCE_JUDGE"), False),
        crossref_mailto=_opt(get("CROSSREF_MAILTO")),
        s2_api_key=_opt(get("S2_API_KEY")),
        openalex_api_key=_opt(get("OPENALEX_API_KEY")),
        enable_web_search=_as_bool(get("ENABLE_WEB_SEARCH"), False),
        papers_dir=Path(papers),
        chbench_data_dir=Path(chbench),
    )


def apply_auth(settings: Settings | None = None) -> str:
    """Select the SDK auth source: the API key if configured, else the subscription.

    The Claude Agent SDK reads ``ANTHROPIC_API_KEY`` from the process environment
    and otherwise falls back to the authenticated Claude Code session (the
    subscription). A key set only in ``.env`` would never reach the SDK, so we
    bridge it into the environment here. When no key is configured we leave the
    environment untouched, so the SDK uses the Claude Code subscription.

    Returns ``"api_key"`` or ``"subscription"`` — which path the SDK will use
    (for logging only; this function makes the choice take effect).
    """
    key = getattr(settings, "anthropic_api_key", None) if settings is not None else None
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
    return "api_key" if (key or os.environ.get("ANTHROPIC_API_KEY")) else "subscription"


def model_for(tier: ModelTier | str, settings: Settings) -> str:
    """Resolve a :class:`ModelTier` to a concrete model id (two-tier routing).

    ``BULK`` -> ``settings.model_bulk``; ``JUDGE`` -> ``settings.model_judge``.
    ``NONE`` (deterministic / no LLM) routes to the bulk model as a harmless
    default so callers always get a usable id.

    Args:
        tier: A :class:`ModelTier` or its string token (``"bulk"``/``"judge"``).
        settings: The resolved settings carrying the model ids.

    Returns:
        The model id string for that tier.
    """
    t = ModelTier(tier)
    if t is ModelTier.JUDGE:
        return settings.model_judge
    return settings.model_bulk
