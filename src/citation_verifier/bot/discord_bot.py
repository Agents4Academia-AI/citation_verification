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
  4. post a compact embed + the full Markdown report (see :mod:`.report`) — via
     the interaction followup, or, when the run outlives Discord's 15-minute
     interaction-token window, a plain channel message that @-mentions the
     caller (so a slow verification still delivers instead of vanishing).

The blocking verifier is the *same* code path the CLI uses — the bot adds no
verification logic. Caching is **off by default** during the testing phase
(``resume=cfg.use_cache``) so every ``/check`` runs fresh; when it is enabled the
artifact dir is keyed per (backend, test-vs-full mode, and a fingerprint of the
verdict-affecting settings) so a test-sample dir and a full-run dir never
collide. Concurrent identical requests are coalesced onto a single run.

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
from .report import _TOO_LARGE, _clip, build_response

__all__ = ["CitationBot", "main"]

log = logging.getLogger("cverify.bot")

# OAuth2 scopes + permissions for the invite link (View Channel, Send Messages,
# Embed Links, Attach Files). 1024 | 2048 | 16384 | 32768.
_INVITE_PERMISSIONS = 52224
_INVITE_SCOPES = "bot+applications.commands"
_CONTENT_LIMIT = 1900  # safely under Discord's 2000-char message-content cap

_BACKEND_CHOICES = [
    app_commands.Choice(name="agentic — fast, free; grounds existence", value="agentic"),
    app_commands.Choice(name="claude_code — deeper LLM check; slower, costs tokens", value="claude_code"),
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
                "**support the claims** that cite them, then flag the hallucinated "
                "ones. 🧪 By default I verify only a small **test sample** (the first "
                f"{self.cfg.test_limit}); pass `full:true` for a complete, whole-paper "
                "verdict."
            ),
            color=0x4263EB,
        )
        embed.add_field(
            name="/check `paper` [backend] [full]",
            value=(
                "Verify a paper. `paper` accepts any of:\n"
                "• `2505.03335` (bare id)\n"
                "• `https://arxiv.org/abs/2505.03335`\n"
                "• `https://arxiv.org/pdf/2505.03335`\n"
                f"🧪 **Default = test sample.** I verify only the first {self.cfg.test_limit} "
                "citations and label the result loudly as PARTIAL — it is **not** a "
                "full-paper verdict.\n"
                "**full** — pass `full:true` to verify **every** citation. That result "
                "is the real verdict (no test banner).\n"
                "**backend** — `agentic` (fast, free) or `claude_code` (deeper LLM "
                "check; slower, costs tokens)."
            ),
            inline=False,
        )
        embed.add_field(name="Default backend", value=f"`{self.cfg.default_backend}`", inline=True)
        embed.add_field(name="/help", value="Show this message.", inline=True)
        embed.add_field(name="/ping", value="Health, latency + current mode.", inline=True)
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
            description="Health check — is the bot alive, and what mode is it in?",
        )
        async def ping(interaction: discord.Interaction) -> None:
            latency_ms = round(self.latency * 1000)
            cache = "on" if self.cfg.use_cache else "off"
            await interaction.response.send_message(
                f"🏓 Pong! Gateway `{latency_ms} ms` · default backend "
                f"`{self.cfg.default_backend}` · 🧪 test sample: first "
                f"{self.cfg.test_limit} (pass full:true for all) · cache {cache}.",
                ephemeral=True,
            )

        # NOTE: Discord caps a command description and each describe() value at
        # 100 chars — keep these short (the /help card carries the long form).
        @self.tree.command(
            name="check",
            description="Verify an arXiv paper's citations (🧪 test sample by default; full:true = whole paper).",
        )
        @app_commands.describe(
            paper="arXiv id or URL — e.g. 2505.03335, https://arxiv.org/abs/2505.03335, /pdf/2505.03335",
            backend="Verification backend. Default: agentic.",
            full=(
                "Verify ALL citations (the whole-paper verdict). "
                f"Off = a 🧪 test sample of the first {self.cfg.test_limit}."
            ),
        )
        @app_commands.choices(backend=_BACKEND_CHOICES)
        async def check(
            interaction: discord.Interaction,
            paper: app_commands.Range[str, 1, 256],
            backend: app_commands.Choice[str] | None = None,
            full: bool = False,
        ) -> None:
            backend_name = backend.value if backend else self.cfg.default_backend
            await self._handle_check(interaction, str(paper), backend_name, full=bool(full))

    # ── the command body ─────────────────────────────────────────
    async def _handle_check(
        self, interaction: discord.Interaction, paper: str, backend: str, *, full: bool
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
            result, cache_hit = await self._run_coalesced(arxiv_id, backend, full=full)
        except Exception as exc:  # noqa: BLE001 — never let the command hang
            log.exception("verification failed for %s", arxiv_id)
            await self._safe_followup(interaction, self._failure_message(arxiv_id, backend, exc))
            return

        try:
            await self._deliver_report(
                interaction, arxiv_id, backend, result, is_test=not full, cached=cache_hit
            )
        except discord.HTTPException:
            log.exception("failed to post report for %s", arxiv_id)
            await self._safe_followup(
                interaction, f"⚠️ Verified `{arxiv_id}`, but couldn't post it. {_TOO_LARGE}"
            )
        except Exception:  # noqa: BLE001 — rendering must never hang the interaction
            log.exception("failed to render report for %s", arxiv_id)
            await self._safe_followup(
                interaction,
                f"⚠️ Verified `{arxiv_id}` but couldn't render the report. See the bot logs.",
            )

    @staticmethod
    def _failure_message(arxiv_id: str, backend: str, exc: Exception) -> str:
        """A friendly, clipped failure line; raw exception detail goes to logs only.

        The common case is a valid-shape-but-nonexistent arXiv id — ``parse``
        accepts the shape, then ingest 404s on download — so map fetch/404
        failures to a "does that paper exist?" hint instead of a stack-ish dump.
        """
        text = str(exc).lower()
        if getattr(exc, "status", 0) == 404 or any(
            k in text for k in ("404", "not found", "download", "fetch")
        ):
            return _clip(
                f"❌ Couldn't fetch `{arxiv_id}` from arXiv — does that paper exist? "
                "Double-check the id.",
                _CONTENT_LIMIT,
            )
        return _clip(
            f"⚠️ Verification failed for `{arxiv_id}` (`{backend}`). The team can see "
            "details in the bot logs. Please try again.",
            _CONTENT_LIMIT,
        )

    # ── result delivery (survives Discord's 15-min interaction-token expiry) ──
    async def _deliver_report(
        self,
        interaction: discord.Interaction,
        arxiv_id: str,
        backend: str,
        result,
        *,
        is_test: bool,
        cached: bool = False,
    ) -> None:
        """Post the finished report, even when the run outlived the interaction.

        Discord invalidates the interaction token ~15 min after ``defer()``, so a
        long verification can no longer answer through ``interaction.followup``.
        The fast path still uses the followup (it threads neatly under the slash
        command); on token expiry we fall back to a plain channel message that
        @-mentions the caller so they still get pinged. ``build_response`` is pure,
        so the embed/file is rebuilt for the fallback — the first attempt consumed
        the attachment's byte buffer.
        """
        embed, files = build_response(result, arxiv_id, backend, is_test=is_test, cached=cached)
        try:
            await interaction.followup.send(embed=embed, files=files)
            return
        except discord.HTTPException as exc:
            if not self._token_expired(exc):
                raise  # genuine size/transient error — handled by the caller
            log.info(
                "interaction token expired for %s (run > 15 min); posting to channel", arxiv_id
            )

        channel = self._channel_of(interaction)
        if channel is None:
            log.warning("no channel available to deliver late report for %s", arxiv_id)
            return
        # rebuild: the first attempt consumed the attachment's byte buffer
        embed, files = build_response(result, arxiv_id, backend, is_test=is_test, cached=cached)
        mention = interaction.user.mention if interaction.user else ""
        suffix = "" if is_test else " full:true"
        await channel.send(
            content=_clip(
                f"{mention} ⏱️ `/check {arxiv_id}{suffix}` took over 15 min — "
                "here's the result:".strip(),
                _CONTENT_LIMIT,
            ),
            embed=embed,
            files=files,
        )

    async def _safe_followup(self, interaction: discord.Interaction, message: str) -> None:
        """Deliver a last-resort answer that never raises — past the token expiry too.

        Tries the interaction followup first; if the token has expired (the run
        took > 15 min) it falls back to a channel message that pings the caller.
        """
        try:
            await interaction.followup.send(message)
            return
        except discord.HTTPException:
            pass
        channel = self._channel_of(interaction)
        if channel is None:
            log.warning("could not deliver followup to interaction")
            return
        try:
            mention = interaction.user.mention if interaction.user else ""
            await channel.send(_clip(f"{mention} {message}".strip(), _CONTENT_LIMIT))
        except discord.HTTPException:
            log.warning("could not deliver followup to channel")

    def _channel_of(self, interaction: discord.Interaction):
        """The channel the command came from, re-resolved from the client if absent."""
        channel = interaction.channel
        if channel is None and interaction.channel_id is not None:
            channel = self.get_channel(interaction.channel_id)
        return channel

    @staticmethod
    def _token_expired(exc: discord.HTTPException) -> bool:
        """True when a followup failed because Discord expired the interaction token.

        ~15 min after ``defer()`` the webhook send returns 401/404 — Discord error
        code 50027 (Invalid Webhook Token) or 10015 (Unknown Webhook).
        """
        return getattr(exc, "code", 0) in (50027, 10015) or getattr(exc, "status", 0) in (401, 404)

    # ── run coalescing + cache keying ─────────────────────────────
    async def _run_coalesced(self, arxiv_id: str, backend: str, *, full: bool):
        """Run (or join) a verification for this cache key on a worker thread.

        Returns ``(result, cache_hit)``. ``cache_hit`` is True only when caching
        is enabled (``BOT_USE_CACHE``) AND a prior ``report.json`` already existed
        for this key, so the footer can be marked ♻️ CACHED. Concurrent identical
        requests await the SAME task instead of spawning a second thread that
        would duplicate the network work and race on ``report.json``.
        """
        key, out_dir, max_cit = self._cache_key(arxiv_id, backend, full)
        # Detect a genuine cache hit BEFORE the run writes a fresh report.json.
        cache_hit = self.cfg.use_cache and (out_dir / "report.json").exists()
        existing = self._inflight.get(key)
        if existing is not None:
            return await existing, cache_hit
        task = asyncio.create_task(
            asyncio.to_thread(self._run_at, arxiv_id, backend, max_cit, out_dir)
        )
        self._inflight[key] = task
        try:
            return await task, cache_hit
        finally:
            self._inflight.pop(key, None)

    def _cache_key(self, arxiv_id: str, backend: str, full: bool) -> tuple[str, Path, int]:
        """Derive (cache-key, out_dir, max_citations) for a request.

        The artifact dir is keyed by backend, the test-vs-full *mode*, AND a
        fingerprint of the settings that change a verdict (models, web-search
        gate, source-key presence). A test-sample dir (``-test5-…``) and a full
        dir (``-full-…``) can never collide, so enabling caching later can never
        serve a 5-pair sample as a whole-paper verdict.
        """
        cfg = self.cfg
        s = cfg.settings
        max_cit = 0 if full else cfg.test_limit
        fp = hashlib.sha1(
            f"{s.model_judge}|{s.model_bulk}|{int(s.enable_web_search)}|"
            f"{bool(s.s2_api_key)}|{bool(s.openalex_api_key)}|"
            f"{bool(s.crossref_mailto)}".encode()
        ).hexdigest()[:8]
        base = f"{backend}-full" if full else f"{backend}-test{cfg.test_limit}"
        out_dir = Path(s.papers_dir) / arxiv_id / f"{base}-{fp}"
        return str(out_dir), out_dir, max_cit

    def _run_at(self, arxiv_id: str, backend: str, max_citations: int, out_dir: Path):
        """Blocking verification (runs in a worker thread).

        ``resume`` is gated on ``BOT_USE_CACHE`` (off by default during the
        testing phase), so every ``/check`` runs fresh unless caching is enabled.
        """
        return run_verification(
            arxiv_id,
            backend=backend,
            settings=self.cfg.settings,
            resume=self.cfg.use_cache,
            out_dir=str(out_dir),
            max_citations=max_citations,
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
        "Starting bot (default backend=%s, guild=%s, test_limit=%d, cache=%s)",
        cfg.default_backend,
        cfg.guild_id or "global",
        cfg.test_limit,
        "on" if cfg.use_cache else "off",
    )
    try:
        client.run(cfg.token, log_handler=None)
    except discord.LoginFailure:
        print("error: Discord rejected the token (LoginFailure). Check DISCORD_BOT_TOKEN.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
