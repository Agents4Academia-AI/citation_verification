"""
bot/config.py — runtime configuration for the Discord front-end.

Reads the bot's own settings (token, optional guild id, default backend, an
upper bound on how many citations to verify per command) from the process
environment plus the same optional ``.env`` file the rest of the package uses.
Reuses :func:`citation_verifier.config.load_settings` for everything the
verifier itself needs (``papers_dir``, model routing, keys) so the bot never
re-implements pipeline config.

Nothing here imports ``discord`` or touches the network, and loading never
raises on a missing key — an absent ``DISCORD_BOT_TOKEN`` simply yields
``token=None`` so the caller can print one clear, actionable error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, _parse_env_file, load_settings

__all__ = ["BotConfig", "load_bot_config"]

_VALID_BACKENDS = ("agentic", "claude_code")


@dataclass
class BotConfig:
    """Resolved Discord-bot configuration.

    Attributes:
        token: The Discord bot token (``None`` when unset — the bot can't run).
        guild_id: A guild (server) id to register the command in for *instant*
            availability; ``None`` falls back to a (slow, up to ~1h) global sync.
        default_backend: Backend used when ``/check`` is invoked without one.
        max_citations: Hard cap on citations verified per command (0 = no cap);
            keeps a single ``/check`` bounded in time/cost on huge bibliographies.
        settings: The verifier's own resolved :class:`Settings`.
    """

    token: str | None
    guild_id: int | None
    default_backend: str
    max_citations: int
    settings: Settings


def _read(name: str, env_file: dict[str, str]) -> str | None:
    """os.environ wins; the ``.env`` file fills the gap; blank -> ``None``."""
    value = os.environ.get(name, env_file.get(name))
    if value is None:
        return None
    value = value.strip()
    return value or None


def _as_int(value: str | None, default: int) -> int:
    """Parse an int env string; fall back to ``default`` on junk/absent."""
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_bot_config(env_file: str | Path | None = ".env") -> BotConfig:
    """Build :class:`BotConfig` from the environment + an optional ``.env``.

    Args:
        env_file: Path to a ``.env`` file layered *under* the process
            environment, or ``None`` to use the environment only.

    Returns:
        A fully-resolved :class:`BotConfig`. Never raises on missing keys.
    """
    file_env: dict[str, str] = _parse_env_file(Path(env_file)) if env_file is not None else {}

    backend = (_read("BOT_DEFAULT_BACKEND", file_env) or "agentic").lower()
    if backend not in _VALID_BACKENDS:
        backend = "agentic"

    return BotConfig(
        token=_read("DISCORD_BOT_TOKEN", file_env),
        guild_id=_as_int(_read("DISCORD_GUILD_ID", file_env), 0) or None,
        default_backend=backend,
        max_citations=_as_int(_read("BOT_MAX_CITATIONS", file_env), 0),
        settings=load_settings(env_file),
    )
