"""
relevance_judge — the LLM relevance judge for STEP 2 (the `agentic` backend seam).

The deterministic `agentic` backend abstains on relevance unless a judge is wired
into `stages.relevance`. This module provides that judge. It decides whether a
cited paper supports the SPECIFIC claim it is attached to, judging ONLY from
retrieved text — never model memory:

  - **L0** — the cited paper's abstract (always available from grounding).
  - **L1** — abstract + the cited paper's Introduction, fetched best-effort when the
    paper is on arXiv (degrades to L0 on any failure).

No L2 (deep full-text retrieval): locating a paraphrased passage needs semantic
retrieval — out of scope. When the abstract(+intro) lacks the specific claimed
evidence, the judge HONESTLY abstains (`partial`/`unverified`) instead of guessing.

**Batching.** The Claude Code subscription is reached via the Agent SDK's
``query()``, which re-pays Claude Code's session overhead (~tens of k tokens) on
every call. Judging one citation per ``query()`` multiplies that overhead by the
citation count. :meth:`LLMRelevanceJudge.judge_batch` therefore judges a whole
chunk of citations in ONE ``query()`` (default ``_BATCH_SIZE`` per call), so the
overhead is paid ``ceil(N / batch)`` times instead of ``N``.

Auth: the Agent SDK uses ``ANTHROPIC_API_KEY`` if one is configured (env or
``.env``), else the authenticated Claude Code SUBSCRIPTION — default (no key) is
the subscription; a key switches to the per-token API
(see :func:`citation_verifier.config.apply_auth`). The SDK is a lazy import: with
it absent, :func:`build_relevance_judge` returns ``None`` and the agentic backend
falls back to its honest deterministic abstain (the keyless floor stays up).
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from typing import Any

from ..config import apply_auth
from ..interfaces import RunUsage
from ..schema import ModelTier, Priority, SupportsClaim
from ..stages.relevance import RelevanceVerdict
from .usage import usage_from_result_message

_HTTP_TIMEOUT = 20
_INTRO_CAP = 2500
_BATCH_SIZE = 40  # citations judged per query() call (amortizes session overhead)


def build_relevance_judge(settings: Any | None = None) -> "LLMRelevanceJudge | None":
    """Return an :class:`LLMRelevanceJudge`, or ``None`` if the SDK is unavailable.

    ``None`` means "no judge" — the caller (agentic) keeps its honest deterministic
    abstain, so the keyless floor still runs with the SDK absent.
    """
    try:
        import claude_agent_sdk  # noqa: F401
    except Exception:
        return None
    return LLMRelevanceJudge(settings=settings)


class LLMRelevanceJudge:
    """Callable relevance judge satisfying ``stages.relevance.RelevanceJudge``.

    Single-call (``__call__``) and batched (``judge_batch``) entry points share one
    LLM round-trip (:meth:`_run`), which accumulates REAL token/cost usage into
    :attr:`usage` (the agentic backend folds this into the run's accounting).
    """

    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings
        self.model = _setting(settings, "model_judge", "claude-opus-4-6")
        self.batch_size = int(_setting(settings, "relevance_batch_size", _BATCH_SIZE) or _BATCH_SIZE)
        self.calls = 0  # number of query() invocations (= chunks)
        self.usage = RunUsage(backend="agentic", model=self.model)

    # ── single-record entry (the per-record seam) ───────────────────
    def __call__(
        self,
        *,
        claim: str,
        abstract: str,
        cited_title: str = "",
        resolved_title: str = "",
        resolved: Any = None,
    ) -> RelevanceVerdict:
        return self.judge_batch([{"claim": claim, "abstract": abstract, "resolved": resolved}])[0]

    # ── batched entry (one query() per chunk) ────────────────────────
    def judge_batch(self, items: list[dict]) -> list[RelevanceVerdict]:
        """Judge many citations, chunked into ``batch_size`` per ``query()`` call.

        ``items``: ``[{"claim": str, "abstract": str, "resolved": <Resolved|None>}]``.
        Returns a list of :class:`RelevanceVerdict` aligned to ``items``. Items with
        no retrievable evidence abstain immediately (no model call); chunk failures
        degrade every item in the chunk to ``unverified``.
        """
        prepared: list[tuple[str, str]] = []
        for it in items:
            evidence, _level = self._assemble_evidence((it.get("abstract") or ""), it.get("resolved"))
            prepared.append(((it.get("claim") or ""), evidence))

        verdicts: list[RelevanceVerdict | None] = [None] * len(items)
        todo = [i for i, (_c, ev) in enumerate(prepared) if ev.strip()]
        for i in range(len(items)):
            if not prepared[i][1].strip():
                verdicts[i] = _abstain("no abstract or introduction to judge from")

        for chunk in _chunks(todo, self.batch_size):
            try:
                got = self._judge_chunk([prepared[i] for i in chunk])
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                got = {}
                _ = exc
            for local_i, gi in enumerate(chunk):
                verdicts[gi] = got.get(local_i) or _abstain("batch judge returned no verdict")

        return [v if v is not None else _abstain("no verdict") for v in verdicts]

    # ── evidence (L0 / L1) ───────────────────────────────────────────
    def _assemble_evidence(self, abstract: str, resolved: Any) -> tuple[str, str]:
        text = (abstract or "").strip()
        intro = _fetch_intro(resolved)
        if intro:
            return ((text + "\n\n[Introduction]\n" + intro).strip(), "L1")
        return (text, "L0")

    # ── one chunk -> {local_index: verdict} ──────────────────────────
    def _judge_chunk(self, chunk: list[tuple[str, str]]) -> dict[int, RelevanceVerdict]:
        import claude_agent_sdk as sdk

        payload = [{"i": i, "claim": c, "evidence": ev} for i, (c, ev) in enumerate(chunk)]
        user = (
            "Judge each item below. For each, use ONLY that item's own evidence "
            "text.\n" + json.dumps(payload, ensure_ascii=False)
        )
        apply_auth(self.settings)  # API key if configured, else the subscription
        text = asyncio.run(self._run(sdk, _BATCH_SYSTEM, user))
        return _parse_batch_verdicts(text)

    async def _run(self, sdk: Any, system: str, user: str) -> str:
        options = sdk.ClaudeAgentOptions(
            system_prompt=system,
            allowed_tools=[],   # no tools: a grounded judgement, not an agent loop
            model=self.model,
            max_turns=2,
        )
        out: list[str] = []
        async for message in sdk.query(prompt=user, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        out.append(block.text)
            elif isinstance(message, sdk.ResultMessage):
                self.usage.add(usage_from_result_message(message, self.model, "agentic"))
        self.calls += 1
        return "".join(out)


# ───────────────────────────────────────────────────────────────
# Prompt
# ───────────────────────────────────────────────────────────────
_BATCH_SYSTEM = (
    "You are a citation RELEVANCE judge. You receive a JSON array of items; each "
    "has an index `i`, a CLAIM from a citing paper, and TEXT from the cited work "
    "(its abstract, sometimes plus its introduction). For EACH item decide, using "
    "ONLY that item's text (never prior knowledge), whether the cited work supports "
    "THAT specific claim. If the specific claimed fact/result/method is not present "
    "in the text, use 'unverified' (or 'partial' if clearly related but not "
    "confirming); do NOT guess 'supports'. Classify priority: 'obligatory' (the "
    "claim depends on this source — a method used/extended, a baseline, a dataset, "
    "a specific result/quote) or 'helpful' (background / see-also). Return ONE JSON "
    "array, exactly one object per input item, echoing its `i`: "
    '[{"i":0,"supports_claim":"supports|partial|does_not|unverified",'
    '"priority":"obligatory|helpful","confidence":0.0-1.0,'
    '"justification":"one short sentence"}]. Output JSON only, no prose.'
)


# ───────────────────────────────────────────────────────────────
# Helpers (pure / network — no SDK), unit-testable offline
# ───────────────────────────────────────────────────────────────
def _setting(settings: Any | None, attr: str, default: Any) -> Any:
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), max(1, n)):
        yield seq[i : i + max(1, n)]


def _abstain(reason: str) -> RelevanceVerdict:
    return RelevanceVerdict(
        supports_claim=SupportsClaim.UNVERIFIED, justification=reason, model_tier=ModelTier.JUDGE
    )


_VERDICT_ALIASES = {
    "does not": "does_not",
    "doesnt": "does_not",
    "n/a": "unverified",
    "na": "unverified",
    "unknown": "unverified",
}


def _verdict_from_row(data: dict) -> RelevanceVerdict:
    sc = _coerce(SupportsClaim, data.get("supports_claim"), SupportsClaim.UNVERIFIED)
    pr = _coerce(Priority, data.get("priority"), None)
    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    return RelevanceVerdict(
        supports_claim=sc,
        priority=pr,
        confidence=conf,
        justification=str(data.get("justification") or "").strip(),
        model_tier=ModelTier.JUDGE,
    )


def _parse_verdict(text: str) -> RelevanceVerdict:
    """Parse a single JSON-object verdict (fail-soft -> abstain)."""
    blob = _first_json(text, "{", "}")
    data: dict = {}
    if blob:
        try:
            parsed = json.loads(blob)
            data = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            data = {}
    return _verdict_from_row(data)


def _parse_batch_verdicts(text: str) -> dict[int, RelevanceVerdict]:
    """Parse a JSON array of verdicts into ``{i: RelevanceVerdict}`` (fail-soft)."""
    blob = _first_json(text, "[", "]")
    out: dict[int, RelevanceVerdict] = {}
    if not blob:
        return out
    try:
        rows = json.loads(blob)
    except json.JSONDecodeError:
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("i"), int):
            out[row["i"]] = _verdict_from_row(row)
    return out


def _coerce(enum_cls: Any, value: Any, default: Any) -> Any:
    if value is None:
        return default
    raw = str(value).strip().lower()
    token = _VERDICT_ALIASES.get(raw, raw.replace(" ", "_").replace("-", "_"))
    try:
        return enum_cls(token)
    except ValueError:
        return default


def _first_json(text: str, opener: str, closer: str) -> str | None:
    """Return the outermost ``opener..closer`` span (prefers a fenced block)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    candidate = fence.group(1) if fence else text
    s, e = candidate.find(opener), candidate.rfind(closer)
    return candidate[s : e + 1] if s != -1 and e > s else None


def _fetch_intro(resolved: Any) -> str:
    """Best-effort: the cited paper's Introduction from arXiv HTML. Fail-soft ''."""
    arxiv_id = getattr(resolved, "arxiv_id", None) if resolved is not None else None
    if not arxiv_id:
        return ""
    stem = str(arxiv_id).split("v")[0]
    try:
        req = urllib.request.Request(
            f"https://arxiv.org/html/{stem}",
            headers={"User-Agent": "citation-verifier/0.1"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", "ignore")
    except Exception:
        return ""
    return _slice_intro(_html_to_text(html))


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


def _slice_intro(text: str) -> str:
    """Slice 'Introduction' .. next-section from plain text; capped. '' if absent."""
    m = re.search(r"introduction", text, flags=re.IGNORECASE)
    if not m:
        return ""
    tail = text[m.end() : m.end() + _INTRO_CAP * 2]
    nxt = re.search(
        r"\b(related work|background|preliminaries|2\s+[A-Z])", tail, flags=re.IGNORECASE
    )
    intro = (tail[: nxt.start()] if nxt else tail).strip()
    return (intro[:_INTRO_CAP] + "…") if len(intro) > _INTRO_CAP else intro


__all__ = ["build_relevance_judge", "LLMRelevanceJudge"]
