"""
Offline tests for grounding rate-limit handling: every rate-limited source is
paced (so a single paper's concurrent correctness pass fetches ALL its refs
instead of 429-dropping the tail), and the ``requests``-based S2/OpenAlex path
shares the urllib path's throttle + ``Retry-After`` retry.

No network: a fake ``requests`` module drives :func:`_requests_get_json`, and
``time.sleep`` is neutralized so retries don't actually wait.
"""

from __future__ import annotations

import pytest

from citation_verifier.grounding import paper_lookup as pl


class _FakeResp:
    def __init__(self, status: int, headers: dict | None = None, payload: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeRequests:
    """Stand-in for the ``requests`` module: hands back queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        return self._responses.pop(0)


def test_rate_limited_hosts_are_all_throttled():
    # arXiv was throttled before; DBLP + S2 + OpenAlex are the ones that 429-dropped
    # refs within a single paper. All must now carry a positive per-host interval.
    for host in ("export.arxiv.org", "api.semanticscholar.org", "dblp.org", "api.openalex.org"):
        assert pl._HOST_MIN_INTERVAL.get(host, 0.0) > 0.0


def test_parse_retry_after():
    assert pl._parse_retry_after("5") == 5.0
    assert pl._parse_retry_after("0") == 0.0
    assert pl._parse_retry_after(None) is None
    assert pl._parse_retry_after("Wed, 21 Oct 2025 07:28:00 GMT") is None  # HTTP-date -> None
    assert pl._parse_retry_after("9999") == pl._RETRY_AFTER_CAP  # capped


def test_requests_get_json_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)  # don't actually wait
    fake = _FakeRequests([
        _FakeResp(429, {"Retry-After": "0"}),   # rate-limited once
        _FakeResp(200, {}, {"ok": 1}),          # then OK
    ])
    data = pl._requests_get_json(fake, "https://api.openalex.org/works", params={}, headers={})
    assert data == {"ok": 1}
    assert fake.calls == 2  # retried instead of dropping the ref


def test_requests_get_json_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    fake = _FakeRequests([_FakeResp(429, {"Retry-After": "0"}) for _ in range(5)])
    with pytest.raises(Exception):  # noqa: B017 — caller wraps this and fails soft
        pl._requests_get_json(fake, "https://api.semanticscholar.org/x", retries=2)
    assert fake.calls == 3  # initial + 2 retries, then raise
