"""
bot/discord_bot.py — the Discord client and the ``/check`` slash command.

Pipeline, end to end, per ``/check``:

  1. parse the argument with :func:`citation_verifier.ingest.parse_arxiv_id`
     (accepts a bare id ``2505.03335``, an ``/abs/`` URL, or a ``/pdf/`` URL —
     all normalize to the same id);
  2. ``interaction.response.defer()`` (Discord needs an ack within 3s; the real
     work may take minutes on a fresh paper);
  3. run :func:`citation_verifier.orchestrator.run_verification` in a worker
     thread (it is blocking/network-bound) so the gateway heartbeat keeps going;
  4. post a compact embed + the full Markdown report (see :mod:`.report`).

The blocking verifier is the *same* code path the CLI uses — the bot adds no
verification logic. Repeat checks of one paper are instant: ``resume=True`` reads
the cached ``report.json``. Results are cached per (backend, limit, and a
fingerprint of the verdict-affecting settings) so switching backend/model/limit
never returns a stale verdict, and concurrent identical requests are coalesced
onto a single run.

Robustness invariants (every interaction is answered exactly once):
  - all user/exception text is clipped below Discord's 2000-char content limit;
  - the post-verification render+send is guarded, with safe fallbacks;
  - a tree-level ``on_error`` backstop answers anything that still escapes.

The ``discord`` import lives here, not in the package ``__init__``, so importing
the package stays dependency-free; running the bot is what needs discord.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

import discord
from discord import app_commands

from ..ingest import parse_arxiv_id
from ..orchestrator import run_verification
from .config import BotConfig, load_bot_config
from .report import _clip, build_response

__all__ = ["CitationBot", "main"]

log = logging.getLogger("cverify.bot")

# OAuth2 scopes + permissions for the invite link (View Channel, Send Messages,
# Embed Links, Attach Files). 1024 | 2048 | 16384 | 32768.
_INVITE_PERMISSIONS = 52224
_INVITE_SCOPES = "bot+applications.commands"
_CONTENT_LIMIT = 1900  # safely under Discord's 2000-char message-content cap

_BACKEND_CHOICES = [
    app_commands.Choice(name="agentic — fast, free, grounds existence", value="agentic"),
    app_commands.Choice(name="claude_code — deeper LLM check (slower, costs tokens)", value="claude_code"),
]


class _CommandTree(app_commands.CommandTree):
    """A CommandTree that always answers the interaction, even on an unexpected error.

    discord.py's default ``on_error`` only logs; with a deferred interaction that
    would leave the user staring at a perpetual "thinking…" spinner. This backstop
    sends a short message via whichever channel is still open.
    """

    async def on_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        log.exception("unhandled app-command error", exc_info=error)
        msg = "⚠️ Something went wrong handling that command. Please try again."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(msg, ephemeral=True)
            else:
                await interaction.response.send_message(msg, ephemeral=True)
        except discord.HTTPException:
            pass


class CitationBot(discord.Client):
    """A minimal slash-command client exposing ``/check``, ``/help`` and ``/ping``.

    Owns a :class:`_CommandTree`; on startup it syncs the commands to the
    configured guild (instant) or globally (slow) and logs an invite URL so an
    admin can add the bot to a server.
    """

    def __init__(self, cfg: BotConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.tree = _CommandTree(self)
        # Per cache-key in-flight runs, so concurrent identical /checks coalesce
        # onto ONE worker thread (no duplicate work, no report.json write race).
        self._inflight: dict[str, asyncio.Task] = {}
        self._register_commands()

    # ── lifecycle ────────────────────────────────────────────────
    async def setup_hook(self) -> None:
        """Sync slash commands before the gateway connects.

        Guild-scoped sync is *instant* (what you want for a test server); a
        global sync can take up to ~1 hour to propagate. Guild sync fails soft:
        if the bot hasn't been invited to the guild yet, fall back to global.
        """
        if self.cfg.guild_id:
            guild = discord.Object(id=self.cfg.guild_id)
            try:
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                log.info("Synced %d command(s) to guild %s", len(synced), self.cfg.guild_id)
                # Remove any stale GLOBAL registrations so commands don't appear
                # twice — a prior run may have global-synced as a fallback before
                # the bot was invited. The guild copies above are unaffected.
                self.tree.clear_commands(guild=None)
                await self.tree.sync()
                log.info("Cleared global commands (guild sync is authoritative)")
                return
            except discord.HTTPException as exc:
                # Almost always: the bot has not been invited to that guild yet.
                log.warning(
                    "Guild sync to %s failed (%s). Invite the bot to that server first "
                    "(see the Invite URL below), then restart. Falling back to a global sync.",
                    self.cfg.guild_id,
                    exc,
                )
        synced = await self.tree.sync()
        log.info("Synced %d global command(s) (propagation can take ~1h)", len(synced))

    async def on_ready(self) -> None:
        """Log who we are and a ready-to-use invite link."""
        user = self.user
        log.info("Logged in as %s (id=%s)", user, getattr(user, "id", "?"))
        if user is not None:
            log.info(
                "Invite URL: https://discord.com/oauth2/authorize"
                "?client_id=%s&permissions=%d&scope=%s",
                user.id,
                _INVITE_PERMISSIONS,
                _INVITE_SCOPES,
            )
        guilds = ", ".join(f"{g.name}({g.id})" for g in self.guilds) or "(none yet)"
        log.info("In %d guild(s): %s", len(self.guilds), guilds)
        log.info("Ready. Commands: /check, /help, /ping. Try: /check 2505.03335")

    # ── help ─────────────────────────────────────────────────────
    def _help_embed(self) -> discord.Embed:
        """The /help card: what the bot does, how to call it, and the legend."""
        embed = discord.Embed(
            title="📚 Citation Verifier — help",
            description=(
                "I check whether the citations in an arXiv paper are **real** and "
                "**support the claims** that cite them, then flag the hallucinated ones."
            ),
            color=0x4263EB,
        )
        embed.add_field(
            name="/check `paper` [backend] [limit]",
            value=(
                "Verify a paper. `paper` accepts any of:\n"
                "• `2505.03335` (bare id)\n"
                "• `https://arxiv.org/abs/2505.03335`\n"
                "• `https://arxiv.org/pdf/2505.03335`\n"
                f"**backend** — `agentic` (fast, free; default `{self.cfg.default_backend}`) "
                "or `claude_code` (deeper LLM check; slower, costs tokens).\n"
                "**limit** — check only the first N citations (0 = all)."
            ),
            inline=False,
        )
        embed.add_field(name="/help", value="Show this message.", inline=True)
        embed.add_field(name="/ping", value="Health + latency.", inline=True)
        embed.add_field(
            name="How to read a result",
            value=(
                "🔴 **Fabricated** — the cited paper could not be found (likely hallucinated).\n"
                "🟠 **Doesn't support** — the source is real but doesn't back the claim.\n"
                "🟡 **Unverified** — couldn't confirm (no DOI/arXiv match; e.g. blog/system card).\n"
                "🟢 **OK** — exists and supports the claim.\n"
                "A full per-citation report is attached to every `/check`."
            ),
            inline=False,
        )
        embed.set_footer(text="Agents4Academia · citation_verification")
        return embed

    # ── command registration ─────────────────────────────────────
    def _register_commands(self) -> None:
        @self.tree.command(
            name="help",
            description="How to use the citation-verification bot.",
        )
        async def help_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(embed=self._help_embed(), ephemeral=True)

        @self.tree.command(
            name="ping",
            description="Health check — is the bot alive and how fast?",
        )
        async def ping(interaction: discord.Interaction) -> None:
            latency_ms = round(self.latency * 1000)
            await interaction.response.send_message(
                f"🏓 Pong! Gateway latency `{latency_ms} ms` · default backend `{self.cfg.default_backend}`.",
                ephemeral=True,
            )

        @self.tree.command(
            name="check",
            description="Verify the citations in an arXiv paper and report hallucinations.",
        )
        @app_commands.describe(
            paper="arXiv id or URL — e.g. 2505.03335, https://arxiv.org/abs/2505.03335, /pdf/2505.03335",
            backend="Verification backend (default: agentic).",
            limit="Max citations to check (0 = all; useful to keep a run quick/cheap).",
        )
        @app_commands.choices(backend=_BACKEND_CHOICES)
        async def check(
            interaction: discord.Interaction,
            paper: app_commands.Range[str, 1, 256],
            backend: app_commands.Choice[str] | None = None,
            limit: app_commands.Range[int, 0, 500] = 0,
        ) -> None:
            backend_name = backend.value if backend else self.cfg.default_backend
            await self._handle_check(interaction, str(paper), backend_name, int(limit))

    # ── the command body ─────────────────────────────────────────
    async def _handle_check(
        self, interaction: discord.Interaction, paper: str, backend: str, limit: int
    ) -> None:
        arxiv_id = parse_arxiv_id(paper)
        if not arxiv_id:
            await interaction.response.send_message(
                f"❌ Couldn't find an arXiv id in `{_clip(paper, 200)}`.\n"
                "Try a bare id (`2505.03335`) or an arxiv.org `/abs/` or `/pdf/` URL.",
                ephemeral=True,
            )
            return

        # Ack within 3s; the verification may take minutes on a fresh paper.
        await interaction.response.defer(thinking=True)

        try:
            result = await self._run_coalesced(arxiv_id, backend, limit)
        except Exception as exc:  # noqa: BLE001 — never let the command hang
            log.exception("verification failed for %s", arxiv_id)
            await self._safe_followup(
                interaction,
                _clip(
                    f"⚠️ Verification failed for `{arxiv_id}` (`{backend}`): "
                    f"`{type(exc).__name__}: {exc}`",
                    _CONTENT_LIMIT,
                ),
            )
            return

        try:
            embed, files = build_response(result, arxiv_id, backend)
            await interaction.followup.send(embed=embed, files=files)
        except discord.HTTPException:
            log.exception("failed to post report for %s", arxiv_id)
            await self._safe_followup(
                interaction,
                f"⚠️ Verified `{arxiv_id}` but couldn't post the report (too large or a "
                "transient Discord error). Try again with a smaller `limit`.",
            )
        except Exception:  # noqa: BLE001 — rendering must never hang the interaction
            log.exception("failed to render report for %s", arxiv_id)
            await self._safe_followup(
                interaction,
                f"⚠️ Verified `{arxiv_id}` but couldn't render the report. See the bot logs.",
            )

    @staticmethod
    async def _safe_followup(interaction: discord.Interaction, message: str) -> None:
        """Send a followup that itself never raises (last-resort answer)."""
        try:
            await interaction.followup.send(message)
        except discord.HTTPException:
            log.warning("could not deliver followup to interaction")

    # ── run coalescing + cache keying ─────────────────────────────
    async def _run_coalesced(self, arxiv_id: str, backend: str, limit: int):
        """Run (or join) a verification for this cache key on a worker thread.

        Concurrent identical requests await the SAME task instead of spawning a
        second thread that would duplicate the network work and race on
        ``report.json``.
        """
        key, out_dir, eff = self._cache_key(arxiv_id, backend, limit)
        existing = self._inflight.get(key)
        if existing is not None:
            return await existing
        task = asyncio.create_task(
            asyncio.to_thread(self._run_at, arxiv_id, backend, eff, out_dir)
        )
        self._inflight[key] = task
        try:
            return await task
        finally:
            self._inflight.pop(key, None)

    def _cache_key(self, arxiv_id: str, backend: str, limit: int) -> tuple[str, Path, int]:
        """Derive (cache-key, out_dir, effective_limit) for a request.

        The artifact dir — and thus the resume cache — is keyed by backend, the
        effective citation limit, AND a fingerprint of the settings that change a
        verdict (models, web-search gate, source-key presence), so a config
        change never serves a stale cached result.
        """
        cfg = self.cfg
        eff = limit
        if cfg.max_citations:
            eff = min(limit, cfg.max_citations) if limit else cfg.max_citations
        s = cfg.settings
        fp = hashlib.sha1(
            f"{s.model_judge}|{s.model_bulk}|{int(s.enable_web_search)}|"
            f"{bool(s.s2_api_key)}|{bool(s.openalex_api_key)}|"
            f"{bool(s.crossref_mailto)}".encode()
        ).hexdigest()[:8]
        base = backend if not eff else f"{backend}-top{eff}"
        out_dir = Path(s.papers_dir) / arxiv_id / f"{base}-{fp}"
        return str(out_dir), out_dir, eff

    def _run_at(self, arxiv_id: str, backend: str, effective_limit: int, out_dir: Path):
        """Blocking verification (runs in a worker thread)."""
        return run_verification(
            arxiv_id,
            backend=backend,
            settings=self.cfg.settings,
            resume=True,
            out_dir=str(out_dir),
            max_citations=effective_limit,
        )


def _configure_logging() -> None:
    """Send our logs + discord.py's to stderr at a sensible level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("discord").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    """Load config and run the bot. Returns a process exit code.

    Exit codes: ``0`` clean stop, ``2`` missing/invalid ``DISCORD_BOT_TOKEN``.
    """
    _configure_logging()
    cfg = load_bot_config()
    if not cfg.token:
        print(
            "error: DISCORD_BOT_TOKEN is not set. Put it in .env (DISCORD_BOT_TOKEN=...) "
            "or export it, then re-run `cverify-bot`.",
            file=sys.stderr,
        )
        return 2

    # Bridge the Crossref polite-pool email from .env into the process env:
    # grounding.paper_lookup reads CROSSREF_MAILTO from os.environ at import, so
    # without this the .env value is silently ignored and we stay in the slower,
    # easily-throttled anonymous pool.
    if cfg.settings.crossref_mailto:
        os.environ.setdefault("CROSSREF_MAILTO", cfg.settings.crossref_mailto)

    intents = discord.Intents.none()
    intents.guilds = True
    client = CitationBot(
        cfg,
        intents=intents,
        allowed_mentions=discord.AllowedMentions.none(),  # bot messages never ping
    )

    log.info(
        "Starting bot (default backend=%s, guild=%s, max_citations=%s)",
        cfg.default_backend,
        cfg.guild_id or "global",
        cfg.max_citations or "∞",
    )
    try:
        client.run(cfg.token, log_handler=None)
    except discord.LoginFailure:
        print("error: Discord rejected the token (LoginFailure). Check DISCORD_BOT_TOKEN.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
