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

__all__ = ["build_glosser", "build_cell_judge", "build_table_ocr",
           "GLOSS_SYSTEM", "CELL_SYSTEM", "OCR_SYSTEM"]


GLOSS_SYSTEM = (
    "You turn a comparison-table column header into ONE checkable property.\n"
    "You are given the header, the table caption, and passages retrieved from the paper "
    "that mention the header.\n\n"
    "RULES\n"
    "1. Ground the meaning ONLY in the supplied passages and caption. Never use outside "
    "knowledge of what the term usually means.\n"
    "2. If the passages do not actually pin the term down, return an EMPTY gloss — do not "
    "guess. An undefined column is a finding, not a gap for you to fill. In particular a "
    "passage that merely USES the term is not a definition: an ablation result ('without "
    "X the score drops to 0'), a description of the authors' own pipeline, or a sentence "
    "naming the term in passing all leave the column undefined.\n"
    "3. A worked example IS a definition. Papers routinely introduce a capability by "
    "showing it — a sample dialogue, an input/output pair, a short listing. State the "
    "property the example demonstrates; do not return an empty gloss merely because the "
    "passage shows rather than states.\n"
    "4. State the PROPERTY, never the authors' preferred means of achieving it. "
    "\"Repeatable under self-occlusion\" is the property; \"…by using point clouds rather "
    "than multi-view images\" is what these authors chose to do about it, and writing it "
    "into the gloss convicts every method that made the other choice.\n"
    "5. The gloss must SEPARATE this column from the other columns listed. If what you "
    "would write applies equally to a neighbouring column, you have not defined this one "
    "— return an empty gloss instead.\n"
    "6. 'gloss' is one sentence stating the property a method must satisfy to earn a ✓.\n"
    "7. 'test_question' is a single yes/no question to ask about ANOTHER paper, phrased so "
    "that 'yes' means the property holds. No hedging.\n\n"
    'Return ONLY a JSON array, one object per column, in order: '
    '[{"i":0,"gloss":"...","test_question":"..."}]'
)

CELL_SYSTEM = (
    "You check whether a cited paper has certain properties, using ONLY the evidence given.\n\n"
    "The evidence opens with a COVERAGE line saying whether you are looking at the whole "
    "paper or only its title and abstract. It decides which answers are available to "
    "you.\n\n"
    "For each property answer exactly one of:\n"
    "  has     — the evidence shows the work HAS the property. Quote the sentence.\n"
    "  lacks   — the evidence shows the work explicitly does NOT/CANNOT have it. Quote it.\n"
    "  absent  — COVERAGE is full text, you read it, and the work never claims this "
    "property anywhere. Available ONLY when coverage is full text.\n"
    "  unclear — you do not have enough of the paper to say. The only answer available "
    "when coverage is title and abstract only.\n"
    "  wrong_paper — the evidence is about some other work entirely (see RULE 3).\n\n"
    "RULE 1 — absence is not refutation, but it is not nothing either. If the evidence "
    "positively indicates the limitation (it says the method is supervised, needs "
    "retraining, is single-hop, or lists it as a limitation), answer 'lacks'. If you have "
    "the FULL TEXT and the property is simply never claimed, answer 'absent' — a paper "
    "advertises what its method can do, so silence across the whole paper is informative "
    "even though it refutes nothing. With only a title and abstract, silence means "
    "nothing: answer 'unclear'.\n\n"
    "RULE 2 — match the definition at its own level of precision; do not tighten or "
    "loosen it. Decide which kind it is before answering.\n"
    "  * FORMAL — it contains an equation, an 'if and only if', or names one specific "
    "mechanism, representation or space the operation must occur in. Then the work "
    "qualifies ONLY if the evidence shows that same mechanism. A different mechanism that "
    "achieves a similar end does not qualify, however close it sounds; if the evidence "
    "does not identify the mechanism at all, answer 'unclear'.\n"
    "  * QUALITATIVE — it states a capability or outcome in plain words. Judge it on "
    "substance, exactly as written. Do NOT demand a formal proof it never asked for: if "
    "the evidence plainly shows the capability, answer 'has'. Over-caution here is as "
    "wrong as looseness above — it hides real errors.\n"
    "In 'justification', first name the exact part of the definition you are matching, then "
    "the evidence for it.\n\n"
    "RULE 3 — if the evidence is not actually about the named work (wrong paper, unrelated "
    "text), answer 'wrong_paper' for every property. Never judge from your own knowledge "
    "of the work. This is different from 'unclear': 'unclear' means you read the right "
    "paper and it did not settle the question, while 'wrong_paper' means the cell was "
    "never checked at all.\n\n"
    "RULE 4 — a definition that lists several requirements needs evidence for EVERY one. "
    "'insertion and extraction are efficient' is not established by evidence that "
    "extraction is efficient; that is 'unclear'. Say which conjunct is unevidenced.\n\n"
    "RULE 5 — a property stated as a MATTER OF DEGREE (efficient, negligible, "
    "substantial, minimal, scalable, robust, not compromised) is not established by:\n"
    "  * the work's own favourable adjective about itself — an abstract calling a method "
    "efficient or its impact negligible is a claim, not a demonstration; and a comparison "
    "table's mark is a RELATIVE ranking the authors are entitled to make; or\n"
    "  * a stated aim or motivation — 'our goal is to increase diversity' says what the "
    "work set out to do, not what it achieved.\n"
    "Answer 'has' for such a property only on a concrete mechanism or a measurement. "
    "Otherwise 'unclear'.\n\n"
    "RULE 6 — respect the definition's SCOPE. When it quantifies over a reference set "
    "('a substantial portion of the TRUE distribution', 'across all instances', 'every "
    "computation'), evidence that the work does the thing over SOME set says nothing "
    "about that reference set. A mechanism that produces a distribution of tasks does not "
    "establish that the distribution covers the true one. Answer 'unclear' unless the "
    "evidence reaches the scope the definition names.\n\n"
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
            cand = [s for s in col.get("snippets", []) if s.get("source") != "self-context"]
            ctx = [s for s in col.get("snippets", []) if s.get("source") == "self-context"]
            snips = "\n".join(
                f"   - [{s.get('source')}] {s.get('quote')}" for s in cand
            ) or "   (no passage found in the paper)"
            if ctx:
                snips += "\n   background (about the CITING paper's own system — read it "
                snips += "for what the column is about, never adopt it as the definition):\n"
                snips += "\n".join(f"   - {s.get('quote')}" for s in ctx)
            sibs = ", ".join(map(repr, col.get("siblings") or [])) or "(none)"
            legend = " | ".join(col.get("legend") or []) or "(none)"
            lines.append(
                f"[{i}] header: {col.get('header')!r}\n"
                f"   other columns of the same table: {sibs}\n"
                f"   caption: {col.get('caption') or '(none)'}\n"
                f"   legend under the table: {legend}\n"
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
            f"    property: {p.get('definition')}"
            + (f"\n    question: {p['question']}" if p.get("question") else "")
            for p in props
        )
        user = (
            f"CITED WORK: {payload.get('row_label')}\n\n"
            f"EVIDENCE FROM THAT WORK:\n{payload.get('evidence')}\n\n"
            f"PROPERTIES TO CHECK:\n{listing}"
        )
        return _parse_json_array(run(CELL_SYSTEM, user, 220 * max(1, len(props)) + 200))

    return cell_judge


OCR_SYSTEM = (
    "You read one comparison table from a page image and return it as a grid.\n\n"
    "RULES\n"
    "1. Read ONLY the capability/comparison table — the one whose rows are methods and "
    "whose columns are properties. Ignore results tables, figures and body text.\n"
    "2. Reproduce the grid EXACTLY as printed. Do not fix, infer or complete anything. An "
    "empty cell stays empty.\n"
    "3. Normalise each cell to one of: 'yes' (✓ ✔ full circle where the legend says so), "
    "'no' (✗ ✘ ×), 'partial' (▲ ◐ half circle, a middle grade), '' (blank/–/N/A), or the "
    "literal text when the cell is a word or number ('Human', 'Input', '~2K', 'O(1)').\n"
    "4. USE THE PAPER'S OWN LEGEND. The caption often states what each symbol means "
    "(\"▲: medium\", \"○ denotes white-box\"). Report it in 'legend' as symbol -> meaning, "
    "and grade the cells by it — never by your own assumption about a glyph.\n"
    "5. Keep each row's citation exactly as printed, in whatever style the page uses "
    "(a bracketed number, or an author-year in parentheses) — it is how the row is "
    "matched to a reference.\n"
    "6. If the table is TRANSPOSED (methods across the top, properties down the side), "
    "return it with methods as ROWS anyway, and set 'transposed': true.\n\n"
    'Return ONLY JSON: {"header": ["Method", "<col>", …], '
    '"rows": [["<row label>", "<cell>", …], …], '
    '"legend": {"▲": "medium"}, "transposed": false}'
)


def build_table_ocr(judge: Any) -> Callable[[bytes, str], list[list[str]] | None]:
    """A vision reader for the ``ocr`` hook of :func:`~citation_verifier.tables.pdf_grid.tables_from_pdf`.

    Text extraction loses exactly what this subsystem depends on: symbol-font glyphs, the
    column a mark belongs to, and which text is a row label rather than the abstract
    printed beside the table. Reading the rendered page instead sidesteps all three, and
    a vision model can also report the paper's own symbol legend — which is what makes a
    ▲ interpretable at all.

    Requires the **Messages API** (an ``ANTHROPIC_API_KEY``): images cannot be sent
    through the Agent-SDK ``query()`` transport the subscription path uses, so without a
    key this raises and :func:`tables_from_pdf` keeps its geometric parse.

    Returns a callable ``(png_bytes, caption) -> grid | None`` where ``grid`` is the
    header row followed by the body rows. ``None`` on any failure, so the caller keeps
    its geometric parse.
    """
    import base64

    def ocr(png: bytes, caption: str) -> list[list[str]] | None:
        client = judge._anthropic_client()  # noqa: SLF001 — shared transport
        resp = client.messages.create(
            model=getattr(judge, "model", "claude-opus-4-8"),
            max_tokens=3000,
            system=[{"type": "text", "text": OCR_SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                 "data": base64.b64encode(png).decode()}},
                    {"type": "text", "text": f"Table caption (may state the legend): {caption or '(none)'}"},
                ],
            }],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )
        data = _parse_json_object(text)
        if not data:
            return None
        header = [str(h or "") for h in (data.get("header") or [])]
        rows = [
            [str(c or "") for c in row]
            for row in (data.get("rows") or [])
            if isinstance(row, list)
        ]
        if not header or len(rows) < 2:
            return None
        # The legend rides along on the grid so the caller can attach it to the table.
        ocr.last_legend = {str(k): str(v) for k, v in (data.get("legend") or {}).items()}
        return [header, *rows]

    ocr.last_legend = {}
    return ocr


def _parse_json_object(text: str) -> dict:
    """Best-effort parse of a single JSON object from a model response."""
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text
    start = body.find("{")
    while start >= 0:
        depth, end = 0, -1
        for i in range(start, len(body)):
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
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
                if isinstance(data, dict):
                    return data
        start = body.find("{", start + 1)
    return {}
