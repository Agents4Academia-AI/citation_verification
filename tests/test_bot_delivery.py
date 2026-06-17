"""
tests/test_bot_delivery.py — the /check result-delivery path.

Focus: a verification that outlives Discord's 15-minute interaction-token window
must still reach the user. The fast path posts via ``interaction.followup``; on
token expiry the bot falls back to a plain channel message that @-mentions the
caller. These tests drive ``CitationBot._deliver_report`` / ``_safe_followup``
with fakes — no gateway, no network — so the branch logic is pinned down.
"""

from __future__ import annotations

import asyncio

import discord
import pytest

import citation_verifier.bot.discord_bot as dbot
from citation_verifier.bot.discord_bot import CitationBot


# ── fakes ─────────────────────────────────────────────────────────
class FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = "fake"


def _http_exc(code: int, status: int) -> discord.HTTPException:
    """A real ``discord.HTTPException`` carrying a Discord error ``code``."""
    return discord.HTTPException(FakeResp(status), {"code": code, "message": "nope"})


class FakeFollowup:
    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.sent: list[tuple] = []

    async def send(self, *args, **kwargs) -> None:
        if self.fail is not None:
            raise self.fail
        self.sent.append((args, kwargs))


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    async def send(self, *args, **kwargs) -> None:
        self.sent.append((args, kwargs))


class FakeUser:
    mention = "<@123>"


class FakeInteraction:
    def __init__(self, followup: FakeFollowup, channel: FakeChannel | None) -> None:
        self.followup = followup
        self.channel = channel
        self.channel_id = 999
        self.user = FakeUser()


def _bot() -> CitationBot:
    # Bypass discord.Client.__init__ (no gateway/loop); the delivery helpers only
    # touch self.<method>, build_response, and the interaction/channel fakes.
    return CitationBot.__new__(CitationBot)


@pytest.fixture(autouse=True)
def _stub_build_response(monkeypatch):
    # Isolate delivery from rendering: a trivial (embed, files) with no buffers.
    monkeypatch.setattr(dbot, "build_response", lambda *a, **k: (discord.Embed(), []))


# ── _token_expired ────────────────────────────────────────────────
def test_token_expired_detects_expiry_codes_and_statuses():
    assert CitationBot._token_expired(_http_exc(50027, 401))  # Invalid Webhook Token
    assert CitationBot._token_expired(_http_exc(10015, 404))  # Unknown Webhook
    assert CitationBot._token_expired(_http_exc(0, 404))      # bare 404
    # A real, non-expiry failure (e.g. payload too large) must NOT look expired.
    assert not CitationBot._token_expired(_http_exc(40005, 413))


# ── _deliver_report ───────────────────────────────────────────────
def test_fast_path_uses_followup_no_channel_fallback():
    fu, ch = FakeFollowup(), FakeChannel()
    inter = FakeInteraction(fu, ch)
    asyncio.run(_bot()._deliver_report(inter, "2505.03335", "agentic", object()))
    assert len(fu.sent) == 1  # delivered via the interaction
    assert ch.sent == []      # no fallback needed


def test_expired_token_falls_back_to_channel_with_mention():
    fu = FakeFollowup(fail=_http_exc(50027, 401))  # token died mid-run (> 15 min)
    ch = FakeChannel()
    inter = FakeInteraction(fu, ch)
    asyncio.run(_bot()._deliver_report(inter, "2505.03335", "agentic", object()))
    assert len(ch.sent) == 1
    content = ch.sent[0][1]["content"]
    assert "<@123>" in content and "2505.03335" in content
    assert ch.sent[0][1]["embed"] is not None  # the report still rides along


def test_non_expiry_httpexception_propagates_and_skips_channel():
    fu = FakeFollowup(fail=_http_exc(40005, 413))  # genuine "too large"
    ch = FakeChannel()
    inter = FakeInteraction(fu, ch)
    with pytest.raises(discord.HTTPException):
        asyncio.run(_bot()._deliver_report(inter, "x", "agentic", object()))
    assert ch.sent == []  # caller's except-block handles it, not the fallback


# ── _safe_followup ────────────────────────────────────────────────
def test_safe_followup_falls_back_to_channel_when_token_expired():
    fu = FakeFollowup(fail=_http_exc(50027, 401))
    ch = FakeChannel()
    inter = FakeInteraction(fu, ch)
    asyncio.run(_bot()._safe_followup(inter, "⚠️ failed"))
    assert len(ch.sent) == 1
    assert "<@123>" in ch.sent[0][0][0]
