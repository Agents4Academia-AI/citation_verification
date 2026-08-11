"""
tables/llm.py — the two model-backed seams of table verification.

Both are optional and injected; the rest of the subsystem runs without them.

  * :func:`build_glosser` — turns the passages retrieved for a column header into one
    checkable sentence plus the yes/no question each cited paper is asked. It may only
    *compress* supplied text: a column with no supporting passage is left undefined.
  * :func:`build_cell_judge` — for ONE cited paper, reports for each property whether
    the retrieved evidence shows the work ``has`` it, explicitly ``lacks`` it, or is
    ``unclear``. It is never shown the table's ✓/✗, so it cannot rubber-stamp them;
    :mod:`citation_verifier.tables.verify` does the comparison.

Transport is reused from the prose judge (``LLMRelevanceJudge._run_messages`` /
``_run_query``), so auth, batching, model routing, and usage accounting behave exactly
as they already do for relevance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

__all__ = ["build_glosser", "build_cell_judge", "GLOSS_SYSTEM", "CELL_SYSTEM"]


GLOSS_SYSTEM = (
    "You turn a comparison-table column header into ONE checkable property.\n"
    "You are given the header, the table caption, and passages retrieved from the paper "
    "that mention the header.\n\n"
    "RULES\n"
    "1. Ground the meaning ONLY in the supplied passages and caption. Never use outside "
    "knowledge of what the term usually means.\n"
    "2. If the passages do not actually pin the term down, return an EMPTY gloss — do not "
    "guess. An undefined column is a finding, not a gap for you to fill.\n"
    "3. 'gloss' is one sentence stating the property a method must satisfy to earn a ✓.\n"
    "4. 'test_question' is a single yes/no question to ask about ANOTHER paper, phrased so "
    "that 'yes' means the property holds. No hedging.\n\n"
    'Return ONLY a JSON array, one object per column, in order: '
    '[{"i":0,"gloss":"...","test_question":"..."}]'
)

CELL_SYSTEM = (
    "You check whether a cited paper has certain properties, using ONLY the evidence given.\n\n"
    "For each property answer exactly one of:\n"
    "  has     — the evidence shows the work HAS the property. Quote the sentence.\n"
    "  lacks   — the evidence shows the work explicitly does NOT/CANNOT have it. Quote it.\n"
    "  unclear — the evidence does not settle it.\n\n"
    "RULE 1 — absence is not refutation. If the evidence simply never mentions the "
    "property, the answer is 'unclear', NOT 'lacks'. Only answer 'lacks' when the text "
    "positively indicates the limitation (e.g. it states the method is supervised, needs "
    "retraining, is single-hop, or lists it as a limitation).\n\n"
    "RULE 2 — match the definition at its own level of precision; do not tighten or "
    "loosen it.\n"
    "  * FORMAL definition (contains an equation, 'if and only if', or names one specific "
    "mechanism/space the operation must occur in): the work qualifies only if the evidence "
    "shows THAT mechanism. An analogous-sounding mechanism does not qualify — e.g. if the "
    "property requires new task parameters to be a function of existing task parameters, a "
    "method interpolating raw features and labels does NOT satisfy it. If the evidence does "
    "not identify the mechanism, answer 'unclear'.\n"
    "  * QUALITATIVE definition (a capability or outcome in plain words): judge it on "
    "substance, exactly as written. Do NOT demand a formal proof it never asked for — if "
    "the evidence plainly shows the capability, answer 'has'. Being over-cautious here is "
    "as wrong as being loose above: it hides real errors.\n"
    "Decide which kind each definition is before answering.\n"
    "In 'justification', first name the exact part of the definition you are matching, then "
    "the evidence for it.\n\n"
    "RULE 3 — if the evidence is not actually about the named work (wrong paper, unrelated "
    "text), answer 'unclear' for every property and say so. Never judge from your own "
    "knowledge of the work.\n\n"
    "Do not reason about whether some table is right — just report what the evidence shows.\n\n"
    'Return ONLY a JSON array, one object per property, in order: '
    '[{"col_index":1,"answer":"has|lacks|unclear","quote":"...","justification":"...",'
    '"confidence":0.0}]'
)


def _judge_transport(judge: Any) -> Callable[[str, str, int], str]:
    """Adapt an ``LLMRelevanceJudge`` into ``(system, user, max_tokens) -> text``."""

    def run(system: str, user: str, max_tokens: int) -> str:
        if getattr(judge, "mode", "query") == "messages":
            return judge._run_messages(system, user, max_tokens)  # noqa: SLF001 — shared seam
        import asyncio  # noqa: PLC0415 — lazy, mirrors the prose judge

        import claude_agent_sdk as sdk  # noqa: PLC0415 — optional dependency

        from ..config import apply_auth  # noqa: PLC0415

        apply_auth(getattr(judge, "settings", None))  # API key if configured, else subscription
        return asyncio.run(judge._run_query(sdk, system, user))  # noqa: SLF001 — shared seam

    return run


def _parse_json_array(text: str) -> list[dict]:
    """Best-effort parse of a JSON array from a model response (fenced or bare)."""
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    # Try EVERY '[' rather than only the first: a preamble like "answers for [col 1] and
    # [col 2]:" brace-matches to a non-JSON span, and giving up there silently discards a
    # whole row's judgement (every property then defaults to unclear).
    start = body.find("[")
    while start >= 0:
        depth, end = 0, -1
        for i in range(start, len(body)):
            if body[i] == "[":
                depth += 1
            elif body[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > 0:
            try:
                data = json.loads(body[start:end])
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(data, list):
                    rows = [d for d in data if isinstance(d, dict)]
                    if rows:
                        return rows
        start = body.find("[", start + 1)
    return []


def build_glosser(judge: Any) -> Callable[[list[dict]], list[dict]]:
    """A ``glosser`` for :func:`~citation_verifier.tables.dimensions.resolve_dimensions`."""
    run = _judge_transport(judge)

    def glosser(columns: list[dict]) -> list[dict]:
        lines = []
        for i, col in enumerate(columns):
            snips = "\n".join(
                f"   - [{s.get('source')}] {s.get('quote')}" for s in col.get("snippets", [])
            ) or "   (no passage found in the paper)"
            lines.append(
                f"[{i}] header: {col.get('header')!r}\n"
                f"   caption: {col.get('caption') or '(none)'}\n"
                f"   passages:\n{snips}"
            )
        user = "Columns to define:\n\n" + "\n\n".join(lines)
        out = _parse_json_array(run(GLOSS_SYSTEM, user, 160 * max(1, len(columns)) + 200))
        by_i = {int(d["i"]): d for d in out if str(d.get("i", "")).isdigit()}
        return [by_i.get(i, {}) for i in range(len(columns))]

    return glosser


def build_cell_judge(judge: Any) -> Callable[[dict], list[dict]]:
    """A ``judge`` for :func:`~citation_verifier.tables.verify.verify_table`."""
    run = _judge_transport(judge)

    def cell_judge(payload: dict) -> list[dict]:
        props = payload.get("properties", []) or []
        listing = "\n".join(
            f"[col {p.get('col_index')}] {p.get('name')}\n"
            f"    property: {p.get('definition')}\n"
            f"    question: {p.get('question')}"
            for p in props
        )
        user = (
            f"CITED WORK: {payload.get('row_label')}\n\n"
            f"EVIDENCE FROM THAT WORK:\n{payload.get('evidence')}\n\n"
            f"PROPERTIES TO CHECK:\n{listing}"
        )
        return _parse_json_array(run(CELL_SYSTEM, user, 220 * max(1, len(props)) + 200))

    return cell_judge
