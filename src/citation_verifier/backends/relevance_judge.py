"""
relevance_judge — the LLM relevance judge for STEP 2 (the `agentic` backend seam).

The deterministic `agentic` backend abstains on relevance unless a judge is wired
into `stages.relevance.fill_relevance`. This module provides that judge. It decides
whether a cited paper supports the SPECIFIC claim it is attached to, judging ONLY
from retrieved text — never model memory:

  - **L0** — the cited paper's abstract (always available from grounding).
  - **L1** — abstract + the cited paper's Introduction, fetched best-effort when the
    paper is on arXiv (degrades to L0 on any failure).

We deliberately do NOT do deep full-text retrieval (L2): locating the supporting
passage under paraphrase needs semantic retrieval, which is the complexity we are
avoiding. When the abstract(+intro) does not contain the specific claimed evidence,
the judge HONESTLY abstains (`partial`/`unverified`) instead of guessing.

Auth: this uses the Claude Agent SDK, i.e. the Claude Code SUBSCRIPTION quota — no
per-token API key needed (leave ANTHROPIC_API_KEY blank and stay logged in to Claude
Code). The SDK is a lazy import: with it absent, :func:`build_relevance_judge`
returns ``None`` and the agentic backend falls back to its honest deterministic
abstain (the keyless floor stays up).
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.request
from typing import Any

from ..schema import ModelTier, Priority, SupportsClaim
from ..stages.relevance import RelevanceVerdict

_HTTP_TIMEOUT = 20
_INTRO_CAP = 2500


def build_relevance_judge(settings: Any | None = None) -> "LLMRelevanceJudge | None":
    """Return an :class:`LLMRelevanceJudge`, or ``None`` if the SDK is unavailable.

    ``None`` means "no judge" — the caller (agentic) then keeps its honest
    deterministic abstain, so the keyless floor still runs with the SDK absent.
    """
    try:
        import claude_agent_sdk  # noqa: F401
    except Exception:
        return None
    return LLMRelevanceJudge(settings=settings)


class LLMRelevanceJudge:
    """Callable relevance judge satisfying ``stages.relevance.RelevanceJudge``."""

    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings
        self.model = _setting(settings, "model_judge", "claude-opus-4-6")

    def __call__(
        self,
        *,
        claim: str,
        abstract: str,
        cited_title: str = "",
        resolved_title: str = "",
        resolved: Any = None,
    ) -> RelevanceVerdict:
        evidence, level = self._assemble_evidence(abstract, resolved)
        if not evidence.strip():
            # No retrieved text to judge from -> abstain honestly (never guess).
            return RelevanceVerdict(
                supports_claim=SupportsClaim.UNVERIFIED,
                justification="no abstract or introduction available to judge from",
                model_tier=ModelTier.JUDGE,
            )
        try:
            return self._judge(claim, evidence, resolved_title or cited_title, level)
        except Exception as exc:  # noqa: BLE001 — degrade-not-crash: abstain on error
            return RelevanceVerdict(
                supports_claim=SupportsClaim.UNVERIFIED,
                justification=f"relevance judge error: {exc!r}",
                model_tier=ModelTier.JUDGE,
            )

    # ── evidence assembly (L0 / L1) ──────────────────────────────────
    def _assemble_evidence(self, abstract: str, resolved: Any) -> tuple[str, str]:
        text = (abstract or "").strip()
        intro = _fetch_intro(resolved)
        if intro:
            return ((text + "\n\n[Introduction]\n" + intro).strip(), "L1")
        return (text, "L0")

    # ── the LLM call (Claude Code subscription) ──────────────────────
    def _judge(self, claim: str, evidence: str, title: str, level: str) -> RelevanceVerdict:
        import claude_agent_sdk as sdk

        also_intro = ", plus its introduction" if level == "L1" else ""
        system = (
            "You are a citation RELEVANCE judge. You are given a CLAIM from a paper "
            f"and TEXT from the cited work (its abstract{also_intro}). Decide whether "
            "the cited work supports THIS SPECIFIC claim, judging ONLY from the "
            "provided text — NEVER from prior knowledge. If the specific claimed "
            "fact/result/method is not present in the text, answer 'unverified' (or "
            "'partial' if clearly related but not confirming the specific point); do "
            "NOT guess 'supports'. Also classify priority: 'obligatory' (the claim "
            "depends on this source — a method used/extended, a baseline, a dataset, "
            "a specific result/quote) or 'helpful' (background / see-also). "
            "Output ONE JSON object only, no prose: "
            '{"supports_claim":"supports|partial|does_not|unverified",'
            '"priority":"obligatory|helpful","confidence":0.0-1.0,'
            '"justification":"one sentence quoting the supporting/contradicting text, '
            'or noting the specific evidence is absent"}'
        )
        user = (
            f"CLAIM (from the citing paper):\n{claim}\n\n"
            f"CITED WORK{(' — ' + title) if title else ''} — provided text:\n{evidence}\n\n"
            "Return the JSON verdict."
        )
        text = asyncio.run(self._run(sdk, system, user))
        return _parse_verdict(text)

    async def _run(self, sdk: Any, system: str, user: str) -> str:
        options = sdk.ClaudeAgentOptions(
            system_prompt=system,
            allowed_tools=[],   # no tools: a single grounded judgement, not an agent loop
            model=self.model,
            max_turns=2,
        )
        out: list[str] = []
        async for message in sdk.query(prompt=user, options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        out.append(block.text)
        return "".join(out)


# ───────────────────────────────────────────────────────────────
# Helpers (pure / network — no SDK), unit-testable offline
# ───────────────────────────────────────────────────────────────
def _setting(settings: Any | None, attr: str, default: Any) -> Any:
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)


_VERDICT_ALIASES = {
    "does not": "does_not",
    "doesnt": "does_not",
    "n/a": "unverified",
    "na": "unverified",
    "unknown": "unverified",
}


def _parse_verdict(text: str) -> RelevanceVerdict:
    """Parse the judge's JSON object into a :class:`RelevanceVerdict` (fail-soft)."""
    blob = _first_json_object(text)
    data: dict = {}
    if blob:
        try:
            parsed = json.loads(blob)
            data = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            data = {}
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


def _coerce(enum_cls: Any, value: Any, default: Any) -> Any:
    if value is None:
        return default
    raw = str(value).strip().lower()
    token = _VERDICT_ALIASES.get(raw, raw.replace(" ", "_").replace("-", "_"))
    try:
        return enum_cls(token)
    except ValueError:
        return default


def _first_json_object(text: str) -> str | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    candidate = fence.group(1) if fence else text
    s, e = candidate.find("{"), candidate.rfind("}")
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
    """Slice 'Introduction' .. next-section from plain text; capped. '' if not found."""
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
