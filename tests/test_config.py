"""
Offline tests for auth selection: default to the Claude Code subscription, but
use ANTHROPIC_API_KEY when one is configured (env or .env).
"""

from __future__ import annotations

import os

from citation_verifier.config import Settings, apply_auth


def test_apply_auth_subscription_when_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert apply_auth(Settings()) == "subscription"
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_apply_auth_uses_env_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert apply_auth(Settings()) == "api_key"


def test_apply_auth_bridges_dotenv_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        result = apply_auth(Settings(anthropic_api_key="sk-from-dotenv"))
        assert result == "api_key"
        # bridged into the environment so the SDK (which reads os.environ) sees it
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-from-dotenv"
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
