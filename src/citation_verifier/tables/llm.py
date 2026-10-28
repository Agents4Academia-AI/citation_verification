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
    "You are given the header, the table caption, passages retrieved from the paper "
    "that mention the header, and — when the paper has one — the passage in which it "
    "introduces the table. That introducing passage is the most direct statement of what "
    "the columns mean, usually in words the header does not contain: the shortcoming it "
    "attributes to the compared methods ('refine only a single rule', 'need retraining') "
    "names exactly what the column checks, and you may define the column from it.\n\n"
    "RULES\n"
    "1. Ground the meaning ONLY in the supplied passages, the introducing passage and the "
    "caption. Never use outside knowledge of what the term usually means.\n"
    "2. If the passages do not actually pin the term down, return an EMPTY gloss — do not "
    "guess. An undefined column is a finding, not a gap for you to fill. In particular a "
    "passage that merely USES the term is not a definition: an ablation result ('without "
    "X the score drops to 0'), a description of the authors' own pipeline, or a sentence "
    "naming the term in passing all leave the column undefined.\n"
    "3. A worked example IS a definition. Papers routinely introduce a capability by "
    "showing it — a sample dialogue, an input/output pair, a short listing. State the "
    "property the example demonstrates; do not return an empty gloss merely because the "
    "passage shows rather than states.\n"
    "4. State the PROPERTY, never the authors' preferred means of achieving it, and never "
    "the way they MEASURED it. A desideratum is often stated and then followed by what "
    "these authors did about it (\"…therefore we use X rather than Y\"): the first half is "
    "the criterion, the second their design choice, and writing it into the gloss "
    "convicts every method that made the other choice. Likewise an experimental protocol "
    "(\"measured by …\", \"evaluated with a classifier\") is how these authors tested the "
    "property, not what the property is; folding it in makes the criterion stricter than "
    "the paper's own definition.\n"
    "5. The gloss must SEPARATE this column from the other columns listed. If what you "
    "would write applies equally to a neighbouring column, you have not defined this one "
    "— return an empty gloss instead.\n"
    "6. 'gloss' is one sentence stating the property a method must satisfy to earn a ✓.\n"
    "7. 'test_question' is a single yes/no question to ask about ANOTHER paper, phrased so "
    "that 'yes' means the property holds. No hedging.\n"
    "8. 'search_terms' is how OTHER papers word this property: 6-12 words or short "
    "phrases — synonyms and near-synonyms, the standard names of the mechanisms that "
    "provide it, and the usual phrasing of its absence. The column's name is this "
    "paper's coinage; a cited paper that has the property will call it something else "
    "(a detector 'proven to be rotation invariant' never says 'equivariant'). Here, and "
    "only here, your knowledge of the field's vocabulary is wanted: the terms widen a "
    "search, they assert nothing. Give them even when the gloss is empty.\n\n"
    'Return ONLY a JSON array, one object per column, in order: '
    '[{"i":0,"gloss":"...","test_question":"...","search_terms":["...","..."]}]'
)

CELL_SYSTEM = (
    "You check whether a cited paper has certain properties, using ONLY the evidence given.\n\n"
    "The evidence opens with a COVERAGE block saying what you are holding. Usually it is "
    "the COMPLETE text of the cited paper: then nothing was selected or left out, what "
    "the paper does not say there it does not say anywhere, and you settle every question "
    "by reading. When the paper was too long or could not be fetched you get excerpts "
    "instead, and the block says how many, out of a paper of what length, and for each "
    "property whether the paper mentions it anywhere — by the column's name or by any of "
    "the other wordings the code searched for, which are listed.\n\n"
    "For each property answer exactly one of:\n"
    "  has     — the work has the property: it says so, or what it describes building "
    "gives the property. Quote the sentence.\n"
    "  lacks   — the work does not have the property: it says so, or what it describes "
    "building cannot have it. Quote the sentence.\n"
    "  absent  — the work never claims this property AND its construction does not settle "
    "it either way. With the COMPLETE text you may answer this whenever you have read the "
    "paper through and found no such claim (a reference title, an acknowledgement, a "
    "definition of a term, an unrelated use of the word are not claims). With excerpts it "
    "is available ONLY when the COVERAGE block marks the property 'mentioned in paper: "
    "no', or marks it 'all shown' and none of the shown mentions has THIS work claiming "
    "the property; if only some mentions are shown, you have not read all the paper says: "
    "'unclear'.\n"
    "  unclear — you do not have enough of the paper to say. The only answer available "
    "when coverage is title and abstract only.\n"
    "  wrong_paper — the evidence is about some other work entirely (see RULE 3).\n\n"
    "RULE 1 — the paper's vocabulary is not the question; its CONSTRUCTION is. A work "
    "that has a property usually never uses the column's word for it, and a work that "
    "lacks one almost never says so. Decide from what the paper describes ITSELF as "
    "building, training on and computing:\n"
    "  * the construction gives the property -> 'has'. A detector trained to agree with "
    "itself under sampled transformations is robust to them whether or not it says so.\n"
    "  * the construction cannot have the property -> 'lacks'. A method trained on "
    "hand-annotated labels is not unsupervised though it never says 'supervised'; a "
    "pipeline assembled entirely from components carrying no such guarantee does not have "
    "a guarantee the definition demands. Name the described component that decides it and "
    "quote that description.\n"
    "  * the construction does not settle it -> 'absent' or 'unclear' by the rules above; "
    "if the property is simply not addressed in what you were shown, 'unclear' — unless "
    "the COVERAGE block says the paper mentions it nowhere or that every mention is "
    "shown, in which case 'absent'.\n"
    "This is deduction from what the paper says it built, and it is the ordinary way "
    "these cells are settled. It is NOT predicting how well the method would PERFORM "
    "('it would probably degrade under X', 'this may not generalise'): a prediction "
    "about behaviour is your opinion, and there the answer is 'unclear'. Absence of the "
    "word is never itself the reason for either answer, and never infer a limitation from "
    "the excerpts merely being silent.\n"
    "A difficulty the paper raises in order to address it ('X may fail when …; to handle "
    "this we …', 'a naive approach breaks under …, so we …') is not a limitation of the "
    "method: read on to the remedy before answering 'lacks', and quote the remedy if it "
    "establishes the property.\n\n"
    "RULE 2 — match the definition at its own level of precision; do not tighten or "
    "loosen it. Decide which kind it is before answering.\n"
    "  * FORMAL — it contains an equation, an 'if and only if', or names one specific "
    "mechanism, representation or space the operation must occur in. Then the work "
    "qualifies ONLY if the evidence shows that same mechanism. A different mechanism that "
    "achieves a similar end does not qualify, however close it sounds: when the evidence "
    "identifies the work's mechanism and it is a different one, answer 'lacks' and quote "
    "the mechanism; if the evidence does not identify the mechanism at all, answer "
    "'unclear'. A formal definition is still read as the citing paper reads it (RULE 7): "
    "where its criterion says a part of the formula is trivially met, handled separately, "
    "or not what the column is about, that part needs no evidence — check the rest.\n"
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
    "A definition of the form 'A and B' is not established by evidence for A alone; that "
    "is 'unclear'. Say which conjunct is unevidenced. EXCEPT where the citing paper's own "
    "criterion (RULE 7) sets a conjunct aside — saying it is trivially satisfied, dealt "
    "with by other means, or not where the difficulty lies. The table's marks were made "
    "under that reading, so a work meeting the conjuncts that remain earns 'has'; name the "
    "conjunct you set aside and the criterion that licensed it.\n\n"
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
    "RULE 7 — a property may come with 'how the citing paper applies it': sentences from "
    "the paper that made the table, saying what qualifies, what disqualifies, which "
    "notions it treats as interchangeable, or which part of a requirement it considers "
    "trivial or already handled. That criterion governs how strictly the definition is "
    "read — a mechanism it admits qualifies even if the definition, read literally, would "
    "not; a requirement it sets aside is not checked; and one it rules out does not "
    "qualify however the definition reads. It is NOT evidence about the cited work: whether this work meets the "
    "criterion is decided from the excerpts alone, and your quote must come from them. "
    "If the criterion presupposes something about the work that the excerpts contradict, "
    "the excerpts win.\n\n"
    "RULE 8 — answer the property you were ASKED, not the one next to it. The request "
    "names the table's other columns; they are deliberately near-neighbours of this one, "
    "and evidence that establishes one of THEM establishes nothing here. If the passage "
    "you would quote is really about a neighbouring column, this property is not "
    "evidenced by it.\n\n"
    "Do not reason about whether some table is right — just report what the evidence shows.\n\n"
    'Return ONLY a JSON array, one object per property, in order: '
    '[{"col_index":1,"answer":"has|lacks|unclear","quote":"...","justification":"...",'
    '"confidence":0.0}]'
)


# A model call that fails is usually the transport having a bad minute — an overloaded
# API, a rate limit, a dropped connection — not a request that can never succeed. One
# uncaught failure costs a whole ROW: every cell of that cited paper is reported as
# "never judged". Measured across runs: five cells of one table in one run, six in
# another, always whole rows and never the same row twice.
_JUDGE_ATTEMPTS = 4
_JUDGE_BACKOFF = 6.0  # seconds, doubling


def _with_retry(call: Callable[[], str]) -> str:
    """``call()``, retried with exponential backoff. Raises the last error if all fail."""
    import time  # noqa: PLC0415 — lazy, keeps import cost off the offline path

    from ..errors import ModelUnavailable  # noqa: PLC0415

    last: Exception | None = None
    for attempt in range(_JUDGE_ATTEMPTS):
        try:
            return call()
        except ModelUnavailable:
            raise  # an expired login is not cured by asking again
        except Exception as exc:  # noqa: BLE001 — any transport failure is worth a retry
            last = exc
            if attempt + 1 < _JUDGE_ATTEMPTS:
                time.sleep(_JUDGE_BACKOFF * (2**attempt))
    raise last if last else RuntimeError("judge call failed")


def _judge_transport(judge: Any) -> Callable[..., str]:
    """Adapt an ``LLMRelevanceJudge`` into ``(system, user, max_tokens, prefix) -> text``.

    ``prefix`` is long material that stays constant while the question varies — a whole
    cited paper asked about one column after another.

    Where it travels depends on the transport, and this is not cosmetic: the prefix is
    most of what a call costs, and it is paid in full unless the transport can recognise
    it as the same prefix next time.

    * The Messages API takes it as a separately cached system block.
    * The Agent SDK caches the SYSTEM PROMPT, so the paper belongs there — but a system
      prompt given as a string rides on the CLI command line, which a whole paper
      overruns (128k and 123k characters failed with ``CLIConnectionError`` on every
      attempt). Given as a FILE it has no such limit. Put in the user message instead,
      as it was, the paper sat in one block with the question, no two calls shared a
      prefix, and every cell of a row paid to send the paper again: measured, 25k tokens
      of cache writes per cell and none read back.

    Retries a failed call: see :data:`_JUDGE_ATTEMPTS`.
    """

    def once(system: str, user: str, max_tokens: int, prefix: str = "") -> str:
        if getattr(judge, "mode", "query") == "messages":
            return judge._run_messages(system, user, max_tokens, prefix)  # noqa: SLF001 — shared seam
        import asyncio  # noqa: PLC0415 — lazy, mirrors the prose judge

        import claude_agent_sdk as sdk  # noqa: PLC0415 — optional dependency

        from ..config import apply_auth  # noqa: PLC0415

        apply_auth(getattr(judge, "settings", None))  # API key if configured, else subscription
        prompt: Any = _system_prompt_file(system, prefix) if prefix else system
        if prefix and not isinstance(prompt, dict):
            user = f"{prefix}\n\n{user}"  # no file could be written: send it the old way
        return asyncio.run(judge._run_query(sdk, prompt, user))  # noqa: SLF001 — shared seam

    def run(system: str, user: str, max_tokens: int, prefix: str = "") -> str:
        return _with_retry(lambda: once(system, user, max_tokens, prefix))

    return run


def _system_prompt_file(system: str, prefix: str) -> Any:
    """``system`` + ``prefix`` as a system-prompt FILE for the Agent SDK, or ``system``.

    One file per distinct content, named by its hash: the cells of a row ask about the
    same paper, so they name the same file, send the same system prompt, and all but the
    first read it from the prompt cache. Fail-soft — if the file cannot be written the
    plain string is returned and the caller sends the prefix in the user message.
    """
    import hashlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    text = f"{system}\n\n{prefix}"
    name = "system-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24] + ".txt"
    try:
        folder = Path(tempfile.gettempdir()) / "cverify-judge"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        if not path.exists() or path.stat().st_size != len(text.encode("utf-8")):
            tmp = path.with_suffix(f".{id(text)}.tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
    except OSError:
        return system
    return {"type": "file", "path": str(path)}


def _parse_json_array(text: str) -> list[dict]:
    """Best-effort parse of a JSON array of objects from a model response.

    The array is found by DECODING at each ``[``, not by counting brackets: a model
    prefaces its answer with prose ("answers for [col 1] and [col 2]:"), and the quotes
    inside the answer carry brackets of their own — one cited paper writes its actions as
    "[1" with no closing bracket, the count never returned to zero, and two cells of that
    row came back with no answer and no error.
    """
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    decoder = json.JSONDecoder()
    for body in ([fence.group(1)] if fence else []) + [text]:
        start = body.find("[")
        while start >= 0:
            try:
                data, _ = decoder.raw_decode(body, start)
            except json.JSONDecodeError:
                data = None
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
            intro = " ".join(col.get("table_context") or []).strip()
            lines.append(
                f"[{i}] header: {col.get('header')!r}\n"
                f"   other columns of the same table: {sibs}\n"
                f"   caption: {col.get('caption') or '(none)'}\n"
                f"   legend under the table: {legend}\n"
                f"   where the paper introduces the table: {intro[:1500] or '(none)'}\n"
                f"   passages:\n{snips}"
            )
        user = "Columns to define:\n\n" + "\n\n".join(lines)
        out = _parse_json_array(run(GLOSS_SYSTEM, user, 300 * max(1, len(columns)) + 200))
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
            + (f"\n    how the citing paper applies it: {p['criterion']}" if p.get("criterion") else "")
            + (f"\n    question: {p['question']}" if p.get("question") else "")
            for p in props
        )
        others = ", ".join(map(repr, payload.get("other_properties") or []))
        # The cited work is the same for every property of this row and, on a re-run or a
        # flip pass, for the same row again — so it travels as a cacheable prefix, apart
        # from the question.
        prefix = (
            f"CITED WORK: {payload.get('row_label')}\n\n"
            f"EVIDENCE FROM THAT WORK:\n{payload.get('evidence')}"
        )
        user = f"PROPERTY TO CHECK:\n{listing}"
        if others:
            user += (
                f"\n\nThe same table has separate columns for {others}. Those are other "
                f"questions; do not answer them here, and do not let evidence that belongs "
                f"to one of them settle this one."
            )
        text = run(CELL_SYSTEM, user, 220 * max(1, len(props)) + 200, prefix)
        rows = _parse_json_array(text)
        if not rows and (text or "").strip():
            # An answer that cannot be read is a failed call, not an "unclear" cell: left
            # as an empty list it became a verdict with no justification and no error.
            raise ValueError(f"the judge's answer was not a JSON array: {' '.join(text.split())[:120]!r}")
        return rows

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
