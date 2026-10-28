"""
backends/web_search.py — ask a model with web search where a cited work lives.

The model is a finder, nothing more. Given one bibliography reference it searches the web
and names the result URLs that are the cited work's own pages; :mod:`grounding.web_lookup`
then reads those URLs and decides, by the resolver's own rules, whether the work exists.
Two guards keep the model's memory out of that decision:

* only URLs that appeared in the search results are returned — one the model writes from
  memory is dropped; and
* the model never says whether a work exists, only where its pages are.

Two transports, as for the relevance judge:

* **Messages API** (an ``ANTHROPIC_API_KEY`` is configured): Anthropic's server-side
  ``web_search`` tool. Each search is billed ($10 per 1,000) on top of the tokens.
* **Agent SDK** (the Claude Code subscription): a session whose only tool is
  ``WebSearch``, with no project settings and an empty working directory.

Usage is accounted in :attr:`WebSearcher.usage`. An unreachable model raises
:class:`~citation_verifier.errors.ModelUnavailable`, so the caller can stop asking.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Any

from ..config import apply_auth
from ..errors import ModelUnavailable
from ..interfaces import RunUsage
from .usage import usage_from_message, usage_from_result_message

__all__ = ["WebSearcher", "web_search_tool"]

# USD per search on the Messages API (pricing: $10 per 1,000 searches).
_PRICE_PER_SEARCH = 0.01
_MAX_URLS = 5
_FALLBACK_URLS = 8

_SYSTEM = (
    "You locate a cited work on the web. You are given one bibliography reference. Search "
    "the web for it: its title in quotes with the first author's surname first, then other "
    "wordings — the title alone, the report number or venue — if that finds nothing. A "
    "technical report, thesis, dissertation or working paper is issued by an institution: "
    "also search that institution's own website for it — restricted to the institution's "
    "domain — and for the institution's list of its reports. Then answer with JSON only, no "
    'prose: {"urls": ["…"]} — at most five URLs, taken from your search results, of pages '
    "that ARE the cited work: its arXiv, DOI or publisher page, its proceedings or "
    "OpenReview page, a publisher or library page for a book, the issuing institution's page "
    "for a report or thesis (or its list of reports), or, for software, a blog post, a model "
    "card, a dataset, a leaderboard or a report, its official page. Best first. A page that only "
    "mentions, reviews or cites the work is not it. If no result is the cited work, answer "
    '{"urls": []}. Never give a URL that did not appear in your search results, and do not '
    "rely on what you remember about the work."
)

_URL_RE = re.compile(r"https?://[^\s\"'<>\])}]+")


def web_search_tool(model: str, max_uses: int) -> dict:
    """The Messages API web search tool definition for ``model``.

    The dynamic-filtering version (``web_search_20260209``) runs on Opus and Sonnet 4.6 and
    later and on Fable; every other model takes the basic ``web_search_20250305``.
    """
    m = (model or "").lower()
    dynamic = bool(
        re.search(r"claude-(?:opus|sonnet)-(?:4-[6-9]|[5-9])", m) or "claude-fable" in m
    )
    return {
        "type": "web_search_20260209" if dynamic else "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }


class WebSearcher:
    """Callable ``reference -> (urls_chosen, urls_seen)`` over a model with web search.

    ``urls_chosen`` are the model's picks, already restricted to ``urls_seen`` — every URL
    the searches returned.
    """

    def __init__(self, settings: Any = None, *, model: str | None = None, max_uses: int | None = None) -> None:
        self.settings = settings
        self.model = model or _setting(settings, "model_bulk", "claude-haiku-4-5-20251001")
        self.max_uses = int(max_uses or _setting(settings, "web_search_max_uses", 8) or 8)
        self.mode = "messages" if _has_api_key(settings) else "sdk"
        self.usage = RunUsage(backend="agentic", model=self.model)
        self.searches = 0
        self.calls = 0
        self._lock = threading.Lock()
        self._client: Any = None

    def drain_usage(self) -> RunUsage:
        """The usage accumulated since the last drain (and reset it)."""
        with self._lock:
            usage, self.usage = self.usage, RunUsage(backend="agentic", model=self.model)
        return usage

    def __call__(self, reference: str) -> tuple[list[str], list[str]]:
        user = f"Reference:\n{reference}\n\nFind this work on the web."
        if self.mode == "messages":
            text, seen, finished = self._run_messages(user)
        else:
            import claude_agent_sdk as sdk

            apply_auth(self.settings)
            text, seen, finished = asyncio.run(self._run_query(sdk, user))
        chosen = _chosen(text, seen)
        if not chosen and not finished:
            # Out of searches before it answered: what the searches returned is still
            # read — the reading, not the model's pick, decides — best-ranked first.
            chosen = seen[:_FALLBACK_URLS]
        return chosen, seen

    # ── Messages API ───────────────────────────────────────────────
    def _run_messages(self, user: str) -> tuple[str, list[str], bool]:
        client = self._anthropic_client()
        messages: list[dict] = [{"role": "user", "content": user}]
        tools = [web_search_tool(self.model, self.max_uses)]
        text: list[str] = []
        seen: list[str] = []
        finished = False
        for _ in range(3):  # a server-side loop that pauses is resumed, a bounded number of times
            try:
                resp = client.messages.create(
                    model=self.model, max_tokens=1024, system=_SYSTEM, messages=messages, tools=tools
                )
            except Exception as exc:
                if type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                    raise ModelUnavailable(f"{type(exc).__name__}: {exc}") from exc
                raise
            searches = int(
                getattr(getattr(getattr(resp, "usage", None), "server_tool_use", None), "web_search_requests", 0)
                or 0
            )
            usage = usage_from_message(resp, self.model, "agentic")
            usage.cost_usd = round(usage.cost_usd + searches * _PRICE_PER_SEARCH, 6)
            usage.tool_calls = searches
            with self._lock:
                self.usage.add(usage)
                self.searches += searches
                self.calls += 1
            for block in resp.content:
                kind = getattr(block, "type", None)
                if kind == "text":
                    text.append(block.text)
                elif kind == "web_search_tool_result" and isinstance(getattr(block, "content", None), list):
                    seen += [r.url for r in block.content if getattr(r, "url", None)]
            if getattr(resp, "stop_reason", None) != "pause_turn":
                finished = True
                break
            messages = [{"role": "user", "content": user}, {"role": "assistant", "content": resp.content}]
        return "".join(text), list(dict.fromkeys(seen)), finished

    def _anthropic_client(self) -> Any:
        with self._lock:
            if self._client is None:
                import anthropic

                key = _setting(self.settings, "anthropic_api_key", None)
                self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
            return self._client

    # ── Agent SDK (subscription) ───────────────────────────────────
    async def _run_query(self, sdk: Any, user: str) -> tuple[str, list[str], bool]:
        options = _search_options(sdk, self.model, self.max_uses)
        text: list[str] = []
        seen: list[str] = []
        searches = 0
        usage: RunUsage | None = None
        failure = ""
        out_of_turns = False
        try:
            async for message in sdk.query(prompt=user, options=options):
                if isinstance(message, sdk.AssistantMessage):
                    if getattr(message, "error", None):
                        failure = str(message.error)
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            text.append(block.text)
                        elif getattr(block, "name", "") in ("WebSearch", "web_search"):
                            searches += 1
                elif isinstance(message, sdk.UserMessage):
                    seen += _URL_RE.findall(
                        _flatten(message.content) + " " + _flatten(getattr(message, "tool_use_result", None))
                    )
                elif isinstance(message, sdk.ResultMessage):
                    usage = usage_from_result_message(message, self.model, "agentic")
                    if getattr(message, "subtype", "") == "error_max_turns":
                        out_of_turns = True  # it carries no text: the subtype says it
                    elif getattr(message, "is_error", False):
                        failure = failure or str(getattr(message, "result", "") or "error result")
        except Exception as exc:  # noqa: BLE001 — e.g. the session ran out of turns
            if re.search(r"(?i)maximum number of turns", str(exc)):
                out_of_turns = True
            else:
                failure = failure or str(exc)
        if failure and re.search(r"(?i)maximum number of turns", failure):
            failure, out_of_turns = "", True
        # Out of turns (each search is one): the searches ran, the answer did not; what
        # they returned is still read.
        finished = not out_of_turns
        if failure and re.search(r"(?i)auth|401|403|login|credit|billing|quota", failure):
            raise ModelUnavailable(" ".join(failure.split())[:300])
        if failure:
            raise RuntimeError(f"web search call failed: {' '.join(failure.split())[:300]}")
        with self._lock:
            if usage is not None:
                usage.tool_calls = searches
                self.usage.add(usage)
            self.searches += searches
            self.calls += 1
        return "".join(text), list(dict.fromkeys(u.rstrip(".,;") for u in seen)), finished


def _search_options(sdk: Any, model: str, max_uses: int) -> Any:
    """Agent-SDK options for one search session: WebSearch and nothing else.

    Like the relevance judge's sessions, no user or project settings are loaded and the
    session sits in an empty directory, so the repository's instructions never reach it.
    """
    import dataclasses
    import tempfile
    from pathlib import Path

    cwd = Path(tempfile.gettempdir()) / "cverify-websearch"
    try:
        cwd.mkdir(parents=True, exist_ok=True)
    except OSError:
        cwd = Path(tempfile.gettempdir())
    wanted: dict[str, Any] = {
        "system_prompt": _SYSTEM,
        "tools": ["WebSearch"],
        "allowed_tools": ["WebSearch"],
        "setting_sources": [],
        "cwd": str(cwd),
        "model": model,
        "max_turns": max_uses + 1,  # one turn per search, one to answer
    }
    try:
        known = {f.name for f in dataclasses.fields(sdk.ClaudeAgentOptions)}
    except TypeError:
        known = set(wanted)
    return sdk.ClaudeAgentOptions(**{k: v for k, v in wanted.items() if k in known})


def _chosen(text: str, seen: list[str]) -> list[str]:
    """The model's URLs, kept only when a search returned them, in its order."""
    urls: list[str] = []
    for m in re.finditer(r"\{.*?\}", text or "", re.DOTALL):
        try:
            data = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get("urls"), list):
            urls = [u for u in data["urls"] if isinstance(u, str)]
    returned = {_url_key(u): u for u in seen}
    out = [returned[_url_key(u)] for u in urls if _url_key(u) in returned]
    return list(dict.fromkeys(out))[:_MAX_URLS]


def _url_key(url: str) -> str:
    """A URL compared loosely: scheme, ``www.``, a trailing slash and the fragment dropped."""
    u = re.sub(r"^https?://(?:www\.)?", "", (url or "").strip().lower())
    return u.split("#", 1)[0].rstrip("/")


def _flatten(content: Any) -> str:
    """Tool-result content (a string, or nested lists/dicts of blocks) as one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return " ".join(_flatten(v) for v in content.values())
    if isinstance(content, list | tuple):
        return " ".join(_flatten(v) for v in content)
    return _flatten(getattr(content, "__dict__", None)) or str(content)


def _has_api_key(settings: Any) -> bool:
    import os

    return bool(_setting(settings, "anthropic_api_key", None) or os.environ.get("ANTHROPIC_API_KEY"))


def _setting(settings: Any, attr: str, default: Any) -> Any:
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)
