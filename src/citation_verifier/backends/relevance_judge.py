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
evidence, the judge HONESTLY abstains (`partial`/`inconclusive`) instead of guessing.

**Two transports (chosen by auth).** The judge needs an LLM but reaches it two
ways:

  - **messages** — the direct Anthropic *Messages API* (`anthropic` SDK). Used
    when an ``ANTHROPIC_API_KEY`` is configured. This is the cheaper per-token
    path: a plain stateless request, so it avoids the Agent SDK's per-call Claude
    Code SESSION overhead (~tens of k tokens), and it lets us prompt-cache the
    static judge instructions (``cache_control`` on the system block).
  - **query** — the Agent SDK ``query()`` loop. The keyless Claude Code
    SUBSCRIPTION path (the raw Messages API has no subscription auth), kept as the
    floor so the judge still runs with no key.

**Batching + concurrency.** Each transport re-pays a fixed per-call overhead, so
:meth:`LLMRelevanceJudge.judge_batch` judges a whole chunk of citations in ONE
call (default ``_BATCH_SIZE``) — overhead paid ``ceil(N / batch)`` times, not
``N``. Chunks and the best-effort arXiv intro fetches both run concurrently
(bounded), and intros are de-duplicated by arXiv id so a paper cited many times
is fetched once.

Auth: the SDK/`anthropic` client use ``ANTHROPIC_API_KEY`` if one is configured
(env or ``.env``), else the authenticated Claude Code SUBSCRIPTION via ``query()``
— default (no key) is the subscription (see
:func:`citation_verifier.config.apply_auth`). Both SDKs are lazy imports: with
neither available, :func:`build_relevance_judge` returns ``None`` and the agentic
backend falls back to its honest deterministic abstain (the keyless floor stays up).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import urllib.request
from typing import Any

from ..config import apply_auth
from ..interfaces import RunUsage
from ..schema import ModelTier, Priority, SupportsClaim
from ..stages.relevance import RelevanceVerdict
from .usage import usage_from_message, usage_from_result_message

_HTTP_TIMEOUT = 20
_INTRO_CAP = 2500
_BATCH_SIZE = 40  # citations judged per LLM call (amortizes per-call overhead)
_CHUNK_CONCURRENCY = 4  # chunks judged in parallel (bounded to respect rate limits)
_INTRO_CONCURRENCY = 8  # arXiv intro fetches in flight at once (pure HTTP)


def build_relevance_judge(settings: Any | None = None) -> LLMRelevanceJudge | None:
    """Return an :class:`LLMRelevanceJudge`, or ``None`` if no LLM transport is available.

    Picks the transport by auth (see module docstring): the direct Messages API
    when an API key is configured and ``anthropic`` is importable, else the Agent
    SDK ``query()`` subscription path. ``None`` means "no judge" — the caller
    (agentic) keeps its honest deterministic abstain, so the keyless floor still
    runs with both SDKs absent.
    """
    mode = _select_mode(settings)
    if mode is None:
        return None
    return LLMRelevanceJudge(settings=settings, mode=mode)


def _select_mode(settings: Any | None) -> str | None:
    """Pick the LLM transport: prefer the cheaper Messages API when a key exists."""
    if _has_api_key(settings) and _module_available("anthropic"):
        return "messages"
    if _module_available("claude_agent_sdk"):
        return "query"
    return None


def _has_api_key(settings: Any | None) -> bool:
    """True when an Anthropic API key is configured (settings or environment)."""
    return bool(_setting(settings, "anthropic_api_key", None) or os.environ.get("ANTHROPIC_API_KEY"))


def _module_available(name: str) -> bool:
    """True if a module can be imported (lazy probe; never raises)."""
    try:
        __import__(name)
        return True
    except Exception:
        return False


class LLMRelevanceJudge:
    """Callable relevance judge satisfying ``stages.relevance.RelevanceJudge``.

    Single-call (``__call__``) and batched (``judge_batch``) entry points share
    one LLM round-trip per chunk (:meth:`_judge_chunk`), which accumulates REAL
    token/cost usage into :attr:`usage` (the agentic backend folds this into the
    run's accounting). ``mode`` selects the transport (``"messages"`` | ``"query"``).
    """

    def __init__(self, *, settings: Any | None = None, mode: str = "query") -> None:
        self.settings = settings
        self.mode = mode
        self.model = _setting(settings, "model_judge", "claude-opus-4-8")
        self.batch_size = int(_setting(settings, "relevance_batch_size", _BATCH_SIZE) or _BATCH_SIZE)
        self.chunk_concurrency = int(
            _setting(settings, "judge_concurrency", _CHUNK_CONCURRENCY) or _CHUNK_CONCURRENCY
        )
        self.calls = 0  # number of LLM invocations (= chunks)
        self.usage = RunUsage(backend="agentic", model=self.model)
        # Guards usage/calls accumulation and the intro cache across chunk threads.
        self._lock = threading.Lock()
        self._intro_cache: dict[str, str] = {}  # arXiv stem -> intro text ("" if none)
        self._client: Any = None  # lazily-built anthropic client (messages mode)

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

    # ── batched entry: GROUPED BY CITATION (evidence sent once per cite) ──
    def judge_batch(self, items: list[dict]) -> list[RelevanceVerdict]:
        """Judge many claim-sites, grouped by citation so evidence is sent once.

        ``items``: ``[{"cite_key", "claim_id", "claim", "abstract", "resolved"}]`` —
        one per claim-site. Relevance is genuinely per-claim, but a citation's
        evidence (abstract+intro) is a property of the cited PAPER. So we group by
        ``cite_key`` and send each citation's evidence to the model ONCE alongside
        all of its claims, instead of re-sending the same abstract per pair (the
        cost/latency lever ported from the claude_code backend). Returns a list of
        :class:`RelevanceVerdict` aligned to ``items``.

        Citations with no retrievable evidence abstain all their claims (no model
        call); a chunk failure degrades every claim in it to ``unresolved/inconclusive``.
        """
        # Warm the intro cache once (dedup'd by arXiv id + fetched in parallel).
        self._prefetch_intros(items)

        # Group claim-sites by citation; assemble each citation's evidence ONCE.
        groups: dict[str, dict] = {}
        order: list[str] = []
        for idx, it in enumerate(items):
            ck = it.get("cite_key") or f"__row{idx}"  # ungrouped fallback (e.g. __call__)
            grp = groups.get(ck)
            if grp is None:
                evidence, _level = self._assemble_evidence(it.get("abstract") or "", it.get("resolved"))
                grp = groups[ck] = {"evidence": evidence, "members": []}
                order.append(ck)
            grp["members"].append((idx, it.get("claim_id") or str(idx), it.get("claim") or ""))

        verdicts: list[RelevanceVerdict | None] = [None] * len(items)

        # Citations with no evidence abstain all their claims (no model call).
        todo: list[str] = []
        for ck in order:
            if groups[ck]["evidence"].strip():
                todo.append(ck)
            else:
                for idx, _cid, _c in groups[ck]["members"]:
                    verdicts[idx] = _abstain("no abstract or introduction to judge from")

        # Chunk by CITATION; one LLM call per chunk, chunks judged concurrently.
        chunk_lists = list(_chunks(todo, self.batch_size))
        results = _parallel_map(
            chunk_lists,
            lambda cks: self._judge_citation_chunk([(ck, groups[ck]) for ck in cks]),
            self.chunk_concurrency,
        )
        for cks, got in zip(chunk_lists, results, strict=False):
            got = got or {}
            for ck in cks:
                per_claim = got.get(ck, {})
                for idx, cid, _c in groups[ck]["members"]:
                    verdicts[idx] = per_claim.get(cid) or _abstain("batch judge returned no verdict")

        return [v if v is not None else _abstain("no verdict") for v in verdicts]

    # ── evidence (L0 / L1) ───────────────────────────────────────────
    def _assemble_evidence(self, abstract: str, resolved: Any) -> tuple[str, str]:
        text = (abstract or "").strip()
        intro = self._intro_for(resolved)
        if intro:
            return ((text + "\n\n[Introduction]\n" + intro).strip(), "L1")
        return (text, "L0")

    def _intro_for(self, resolved: Any) -> str:
        """Cached, dedup'd intro lookup (one fetch per arXiv id across the run)."""
        stem = _arxiv_stem(resolved)
        if not stem:
            return ""
        with self._lock:
            if stem in self._intro_cache:
                return self._intro_cache[stem]
        intro = _fetch_intro_by_stem(stem)  # network, outside the lock
        with self._lock:
            self._intro_cache[stem] = intro
        return intro

    def _prefetch_intros(self, items: list[dict]) -> None:
        """Fetch every distinct, not-yet-cached arXiv intro concurrently (L1 evidence)."""
        stems: list[str] = []
        seen: set[str] = set()
        for it in items:
            stem = _arxiv_stem(it.get("resolved"))
            if not stem or stem in seen:
                continue
            seen.add(stem)
            with self._lock:
                cached = stem in self._intro_cache
            if not cached:
                stems.append(stem)
        if not stems:
            return
        fetched = _parallel_map(stems, _fetch_intro_by_stem, _INTRO_CONCURRENCY)
        with self._lock:
            for stem, intro in zip(stems, fetched, strict=False):
                self._intro_cache[stem] = intro if intro is not None else ""

    # ── one chunk of CITATIONS -> {cite_key: {claim_id: verdict}} ─────
    def _judge_citation_chunk(
        self, chunk: list[tuple[str, dict]]
    ) -> dict[str, dict[str, RelevanceVerdict]]:
        """Judge a chunk of citations in one call: evidence once, a verdict per claim."""
        payload = [
            {
                "cite_key": ck,
                "evidence": grp["evidence"],
                "claims": [{"claim_id": cid, "claim": c} for (_i, cid, c) in grp["members"]],
            }
            for ck, grp in chunk
        ]
        user = (
            "Judge each citation's claims using ONLY that citation's evidence "
            "text.\n" + json.dumps(payload, ensure_ascii=False)
        )
        # Output is one verdict per CLAIM, so size the cap by total claims in the chunk.
        n_claims = sum(len(grp["members"]) for _ck, grp in chunk)
        if self.mode == "messages":
            text = self._run_messages(_GROUP_SYSTEM, user, _max_output_tokens(n_claims))
        else:
            import claude_agent_sdk as sdk

            apply_auth(self.settings)  # API key if configured, else the subscription
            text = asyncio.run(self._run_query(sdk, _GROUP_SYSTEM, user))
        return _parse_group_verdicts(text)

    # ── transport: direct Messages API (cheaper per-token path) ───────
    def _run_messages(self, system: str, user: str, max_tokens: int) -> str:
        """One Messages API round-trip; records real usage. No agent session.

        The static system block carries a ``cache_control`` breakpoint so it is
        prompt-cached across chunks. (Caching only engages above the model's
        minimum cacheable prefix; the dominant saving here is simply NOT paying the
        Agent-SDK session overhead the ``query()`` path incurs.) No thinking/effort
        params: this is a bounded JSON classification, kept lean for cost/latency.
        """
        client = self._anthropic_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        with self._lock:
            self.usage.add(usage_from_message(resp, self.model, "agentic"))
            self.calls += 1
        return "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )

    def _anthropic_client(self) -> Any:
        """Lazily build (once) the anthropic client; reused across chunks/threads."""
        with self._lock:
            if self._client is None:
                import anthropic

                key = _setting(self.settings, "anthropic_api_key", None)
                self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
            return self._client

    # ── transport: Agent SDK query() (subscription floor) ─────────────
    async def _run_query(self, sdk: Any, system: str, user: str) -> str:
        options = sdk.ClaudeAgentOptions(
            system_prompt=system,
            allowed_tools=[],   # no tools: a grounded judgement, not an agent loop
            model=self.model,
            max_turns=2,
        )
        out: list[str] = []
        usage: RunUsage | None = None
        async for message in sdk.query(prompt=user, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        out.append(block.text)
            elif isinstance(message, sdk.ResultMessage):
                usage = usage_from_result_message(message, self.model, "agentic")
        with self._lock:
            if usage is not None:
                self.usage.add(usage)
            self.calls += 1
        return "".join(out)


# ───────────────────────────────────────────────────────────────
# Prompt
# ───────────────────────────────────────────────────────────────
_GROUP_SYSTEM = (
    "You are a citation RELEVANCE judge. You receive a JSON array of CITATIONS; each "
    "has a `cite_key`, the retrieved `evidence` text from the cited work (its "
    "abstract, sometimes plus its introduction or full-text excerpts marked with "
    "[section] tags), and a list of `claims` (each a `claim_id` + the claim text "
    "from the citing paper). For EACH claim decide, using ONLY that citation's "
    "evidence (never prior knowledge), whether the cited work supports THAT specific "
    "claim. Choose the label by this rubric:\n"
    "- 'supports': the evidence states or backs the specific claimed fact, result, "
    "or method (do not infer from prior knowledge).\n"
    "- 'partial': the evidence is on-topic and related but weaker or narrower than "
    "the claim asserts.\n"
    "- 'does_not': you HAVE evidence and it does NOT support THIS specific claim — "
    "it contradicts the claim, OR the cited work's actual topic/scope is materially "
    "different from or narrower than the claim so it cannot back it (e.g. a paper "
    "about a text-generation language model cited for a claim about healthcare or "
    "customer-service deployment). Use 'does_not' whenever the retrieved evidence "
    "fails to support the claim — NOT only on explicit contradiction.\n"
    "- 'inconclusive': ONLY when the retrieved evidence is INSUFFICIENT to judge — "
    "the work is on the SAME topic as the claim but the specific detail is absent "
    "from the retrieved text and could plausibly appear elsewhere in the paper you "
    "could not read. Do NOT use 'inconclusive' as a soft 'does_not' for an off-topic "
    "source.\n"
    "Calibrate to how much you can see: if the evidence is only an abstract and the "
    "paper is plainly on the claim's topic, prefer 'inconclusive'/'partial' over "
    "'does_not' (the body may still support it); if the evidence includes full-text "
    "excerpts, or the source is clearly off the claim's topic, prefer 'does_not'. "
    "Never guess 'supports'. Classify each claim's "
    "priority: 'obligatory' (the claim depends on this source — a method "
    "used/extended, a baseline, a dataset, a specific result/quote) or 'helpful' "
    "(background / see-also). Do NOT judge existence or metadata — only relevance. "
    "Return ONE JSON array, exactly one object per input CITATION, echoing its "
    "`cite_key`, with one verdict per claim_id: "
    '[{"cite_key":"...","claims":[{"claim_id":"...",'
    '"supports_claim":"supports|partial|does_not|inconclusive",'
    '"priority":"obligatory|helpful","confidence":0.0-1.0,'
    '"justification":"one short sentence"}]}]. Output JSON only, no prose.'
)


# ───────────────────────────────────────────────────────────────
# Concurrency (I/O-bound fan-out; stdlib only)
# ───────────────────────────────────────────────────────────────
def _parallel_map(items: list, fn, max_workers: int) -> list:
    """Run ``fn`` over ``items`` concurrently (I/O-bound); failures map to ``None``.

    Order-preserving. Used for the network-bound steps (arXiv intro fetch, judge
    chunks) so a paper with many citations is not paced one round-trip at a time.
    Degrades to a serial loop for a single item / single worker.
    """
    if not items:
        return []
    workers = max(1, min(int(max_workers), len(items)))
    if workers == 1:
        out = []
        for it in items:
            try:
                out.append(fn(it))
            except Exception:  # noqa: BLE001 — degrade-not-crash per item
                out.append(None)
        return out
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception:  # noqa: BLE001 — degrade-not-crash per item
                out.append(None)
        return out


def _max_output_tokens(n_items: int) -> int:
    """Output-token cap for a chunk of ``n_items`` (one small JSON object each)."""
    return min(8192, max(512, 96 * n_items))


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
        supports_claim=SupportsClaim.INCONCLUSIVE, justification=reason, model_tier=ModelTier.JUDGE
    )


_VERDICT_ALIASES = {
    "does not": "does_not",
    "doesnt": "does_not",
    "n/a": "inconclusive",
    "na": "inconclusive",
    "unknown": "inconclusive",
    "unverified": "inconclusive",  # accept the old token; map to the renamed value
}


# A justification that states a clear contradiction (the evidence opposes the
# claim) must not be softened to "partial"/"supports" — that is a "does not"
# support. Conservative: only fires on explicit contradiction language, not on
# mere absence ("does not mention/clarify", which is honest abstention).
_CONTRADICTION_RE = re.compile(
    r"\bcontradict|\bdoes not support\b|\bdo not support\b|\binconsistent with\b"
    r"|\brefut|\bchronolog|\bthe opposite\b",
    re.IGNORECASE,
)


def _verdict_from_row(data: dict) -> RelevanceVerdict:
    sc = _coerce(SupportsClaim, data.get("supports_claim"), SupportsClaim.INCONCLUSIVE)
    pr = _coerce(Priority, data.get("priority"), None)
    conf = data.get("confidence")
    conf = float(conf) if isinstance(conf, int | float) else None
    justification = str(data.get("justification") or "").strip()
    # A clearly-contradictory justification overrides a soft supports/partial label.
    if sc in (SupportsClaim.SUPPORTS, SupportsClaim.PARTIAL) and _CONTRADICTION_RE.search(
        justification
    ):
        sc = SupportsClaim.DOES_NOT
    return RelevanceVerdict(
        supports_claim=sc,
        priority=pr,
        confidence=conf,
        justification=justification,
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


def _parse_group_verdicts(text: str) -> dict[str, dict[str, RelevanceVerdict]]:
    """Parse the grouped output into ``{cite_key: {claim_id: RelevanceVerdict}}``.

    Mirrors the citation-grouped prompt: a JSON array of citation objects, each
    echoing its ``cite_key`` and carrying one verdict per ``claim_id``. Fail-soft:
    a garbled chunk yields ``{}`` (callers then abstain); a citation/claim missing
    its id is skipped (that claim abstains).
    """
    blob = _first_json(text, "[", "]")
    out: dict[str, dict[str, RelevanceVerdict]] = {}
    if not blob:
        return out
    try:
        rows = json.loads(blob)
    except json.JSONDecodeError:
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("cite_key"), str):
            continue
        per_claim: dict[str, RelevanceVerdict] = {}
        for cl in row.get("claims") or []:
            if isinstance(cl, dict) and isinstance(cl.get("claim_id"), str):
                per_claim[cl["claim_id"]] = _verdict_from_row(cl)
        out[row["cite_key"]] = per_claim
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


def _arxiv_stem(resolved: Any) -> str:
    """The version-stripped arXiv id of a resolved record, or '' if none."""
    arxiv_id = getattr(resolved, "arxiv_id", None) if resolved is not None else None
    return str(arxiv_id).split("v")[0] if arxiv_id else ""


def _fetch_intro_by_stem(stem: str) -> str:
    """Best-effort: the cited paper's Introduction from arXiv HTML. Fail-soft ''."""
    if not stem:
        return ""
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
