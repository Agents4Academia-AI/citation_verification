"""
tests/test_bot_config.py — bot config parsing under the testing-phase knobs.

BOT_TEST_LIMIT (test-sample size) + BOT_USE_CACHE (resume gate, off by default)
replace the retired BOT_MAX_CITATIONS silent cap. Loading is offline and never
raises. Each test points env_file at a tmp .env and clears the relevant process
env so the real repo .env never leaks in.
"""

from __future__ import annotations

import logging

import pytest

from citation_verifier.bot.config import BotConfig, load_bot_config

_BOT_ENV_KEYS = (
    "BOT_TEST_LIMIT",
    "BOT_USE_CACHE",
    "BOT_MAX_CITATIONS",
    "BOT_DEFAULT_BACKEND",
    "DISCORD_BOT_TOKEN",
    "DISCORD_GUILD_ID",
)


@pytest.fixture(autouse=True)
def _clear_bot_env(monkeypatch):
    for k in _BOT_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _env(tmp_path, body: str):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def test_defaults_when_unset(tmp_path):
    cfg = load_bot_config(_env(tmp_path, ""))
    assert cfg.test_limit == 5
    assert cfg.use_cache is False


def test_test_limit_and_use_cache_parse(tmp_path):
    cfg = load_bot_config(_env(tmp_path, "BOT_TEST_LIMIT=12\nBOT_USE_CACHE=1\n"))
    assert cfg.test_limit == 12
    assert cfg.use_cache is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("", False),
        ("garbage", False),
    ],
)
def test_use_cache_truthiness(tmp_path, value, expected):
    cfg = load_bot_config(_env(tmp_path, f"BOT_USE_CACHE={value}\n"))
    assert cfg.use_cache is expected


def test_test_limit_junk_falls_back_to_default(tmp_path):
    cfg = load_bot_config(_env(tmp_path, "BOT_TEST_LIMIT=not-a-number\n"))
    assert cfg.test_limit == 5


def test_bot_max_citations_is_retired_warned_and_ignored(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="cverify.bot"):
        cfg = load_bot_config(_env(tmp_path, "BOT_MAX_CITATIONS=25\n"))
    assert "retired" in caplog.text.lower()
    assert not hasattr(cfg, "max_citations")


def test_botconfig_fields():
    fields = BotConfig.__dataclass_fields__
    assert "max_citations" not in fields  # retired
    assert "test_limit" in fields
    assert "use_cache" in fields
