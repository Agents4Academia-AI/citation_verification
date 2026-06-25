"""Offline tests for grounding.url_validate — URL normalization + status classify.
All HTTP is monkeypatched, so these run with no network."""

from __future__ import annotations

import citation_verifier.grounding.url_validate as uv


def test_normalize_url_strips_citation_artifacts():
    assert uv.normalize_url("https://github.com/ganler/code-r1,2025") == "https://github.com/ganler/code-r1"
    assert uv.normalize_url("https: //github.com/Deep-Agent/R1-V") == "https://github.com/Deep-Agent/R1-V"
    assert uv.normalize_url("https://oatllm.notion.site/oat-zero .") == "https://oatllm.notion.site/oat-zero"
    assert uv.normalize_url("https://openai.com/index/o3-system-card/") == "https://openai.com/index/o3-system-card/"


def test_classify_maps_status_codes():
    assert uv._classify(200) == "live"
    assert uv._classify(301) == "live"
    assert uv._classify(403) == "blocked"  # bot-block / gated — NOT proof of absence
    assert uv._classify(429) == "blocked"
    assert uv._classify(404) == "dead"
    assert uv._classify(None) == "error"


def test_github_url_goes_through_the_api(monkeypatch):
    seen: dict[str, str] = {}

    def fake(url, headers, timeout=8.0):
        seen["url"] = url
        return 200

    monkeypatch.setattr(uv, "_http_status", fake)
    chk = uv.validate_citation_url("https://github.com/ganler/code-r1,2025")
    assert chk.status == "live" and chk.method == "github_api"
    assert seen["url"] == "https://api.github.com/repos/ganler/code-r1"  # normalized + API form


def test_generic_url_is_a_browser_fetch(monkeypatch):
    monkeypatch.setattr(uv, "_http_status", lambda url, headers, timeout=8.0: 403)
    chk = uv.validate_citation_url("https://openai.com/index/o3-o4-mini-system-card/")
    assert chk.status == "blocked" and chk.method == "fetch"


def test_non_url_returns_none():
    assert uv.validate_citation_url("Smith et al., 2024") is None
