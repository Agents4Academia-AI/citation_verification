"""
closed_book.py — the no-retrieval baseline for table-cell verification.

Refari answers "does cited work R have property D?" from text it *retrieved* from R.
This baseline answers the same question from the model's own parametric knowledge of R:
no retrieval, no tools, no evidence. Everything else is held fixed —

  * the same cells (``out/cells.json``, recovered from the archived Refari run),
  * the same grounded property definition for each column,
  * the same blinding (the judge never sees the table's ✓/✗),
  * the same mark-vs-answer decision table (imported from ``tables.verify``),
  * the same judge model and transport.

So the only variable is where the evidence comes from. That makes this both the
headline baseline (what you get by just asking a model) and the retrieval ablation.

Two independent passes are run over identical inputs. Because the judge is blinded to
the mark, a ✓↔✗ flip cannot change its answer — so re-running the *same* input is
exactly the counterfactual-consistency probe the paper reports for Refari.

Usage::

    python scripts/baselines/closed_book.py [--passes 2] [--concurrency 4] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from citation_verifier.backends.relevance_judge import build_relevance_judge
from citation_verifier.config import load_settings
from citation_verifier.tables.llm import _judge_transport, _parse_json_array

# The mark-vs-answer table Refari itself uses. Imported (not copied) so the baseline
# can never silently drift from the system it is being compared against.
from citation_verifier.tables.verify import _DECISION  # noqa: PLC2701 — deliberate shared seam

HERE = Path(__file__).resolve().parent
CELLS = HERE / "out" / "cells.json"
OUT = HERE / "out" / "closed_book.json"

# Cells Refari never sent to a judge: a self/uncited row, or a column the citing paper
# never defines. Both are properties of the CITING paper, not of the verification
# method, so the baseline inherits them unchanged and the denominators stay identical.
STRUCTURAL = {"skipped", "undefined"}

CLOSED_BOOK_SYSTEM = (
    "You check whether a cited paper has certain properties, using ONLY your own "
    "knowledge of that paper. No evidence is supplied and you have no tools.\n\n"
    "For each property answer exactly one of:\n"
    "  has     — the work HAS the property.\n"
    "  lacks   — the work does NOT/CANNOT have it.\n"
    "  unclear — you cannot settle it.\n\n"
    "RULE 1 — answer from what you actually know about the named work. If you do not "
    "know the work, or are not confident enough to commit, answer 'unclear'.\n\n"
    "RULE 2 — match the definition at its own level of precision; do not tighten or "
    "loosen it.\n"
    "  * FORMAL definition (contains an equation, 'if and only if', or names one specific "
    "mechanism/space the operation must occur in): the work qualifies only if it uses "
    "THAT mechanism. An analogous-sounding mechanism does not qualify. If you cannot "
    "identify the mechanism, answer 'unclear'.\n"
    "  * QUALITATIVE definition (a capability or outcome in plain words): judge it on "
    "substance, exactly as written. Do NOT demand a formal proof it never asked for.\n"
    "Decide which kind each definition is before answering.\n"
    "In 'justification', first name the exact part of the definition you are matching, "
    "then what you are relying on.\n\n"
    "Do not reason about whether some table is right — just report what you know.\n\n"
    'Return ONLY a JSON array, one object per property, in order: '
    '[{"col_index":1,"answer":"has|lacks|unclear","quote":"","justification":"...",'
    '"confidence":0.0}]'
)


def group_rows(cells: list[dict]) -> list[dict]:
    """One judge call per cited work, covering all of its columns (Refari's cost shape)."""
    rows: dict[tuple[str, str], dict] = {}
    for c in cells:
        if c["refari_verdict"] in STRUCTURAL:
            continue
        key = (c["paper"], c["row_label"])
        row = rows.setdefault(
            key,
            {
                "paper": c["paper"],
                "row_label": c["row_label"],
                "cited_title": c["cited_title"],
                "cited_meta": c["cited_meta"],
                "cite_key": c["cite_key"],
                "properties": [],
            },
        )
        row["properties"].append(
            {
                "col_index": len(row["properties"]),
                "name": c["dimension"],
                "definition": c["gloss"],
                "cell_uid": c["cell_uid"],
            }
        )
    return list(rows.values())


def build_user(row: dict) -> str:
    """The request for one cited work: its identity, then the properties to check."""
    listing = "\n".join(
        f"[col {p['col_index']}] {p['name']}\n    property: {p['definition']}"
        for p in row["properties"]
    )
    ref = row["cited_title"] + (f"\n{row['cited_meta']}" if row["cited_meta"] else "")
    return (
        f"CITED WORK: {row['row_label']}\n\n"
        f"REFERENCE AS PRINTED BY THE CITING PAPER:\n{ref}\n\n"
        f"NO EVIDENCE IS PROVIDED. Answer from your own knowledge of this work.\n\n"
        f"PROPERTIES TO CHECK:\n{listing}"
    )


def judge_row(run, row: dict) -> dict[str, dict]:
    """``cell_uid -> answer dict``. A failed call degrades to 'unclear', never raises."""
    try:
        raw = run(CLOSED_BOOK_SYSTEM, build_user(row), 220 * len(row["properties"]) + 200)
        got = _parse_json_array(raw)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash, same as verify.py
        got = []
        print(f"    ! {row['paper']}/{row['row_label']}: {exc!r}")
    by_col = {
        int(d["col_index"]): d
        for d in got
        if isinstance(d, dict) and str(d.get("col_index", "")).lstrip("-").isdigit()
    }
    out = {}
    for p in row["properties"]:
        d = by_col.get(p["col_index"]) or {}
        ans = str(d.get("answer", "unclear")).strip().lower()
        out[p["cell_uid"]] = {
            "answer": ans if ans in ("has", "lacks", "unclear") else "unclear",
            "justification": str(d.get("justification", "") or "")[:500],
            "confidence": d.get("confidence"),
        }
    return out


def run_pass(run, rows: list[dict], concurrency: int, tag: str) -> dict[str, dict]:
    """Judge every row once, `concurrency` calls in flight."""
    answers: dict[str, dict] = {}
    lock, done = threading.Lock(), [0]
    sem = threading.Semaphore(concurrency)

    def work(row: dict) -> None:
        with sem:
            got = judge_row(run, row)
        with lock:
            answers.update(got)
            done[0] += 1
            print(f"  [{tag}] {done[0]}/{len(rows)} {row['paper']}/{row['row_label']}")

    threads = [threading.Thread(target=work, args=(r,)) for r in rows]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return answers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    rows = group_rows(cells)
    if args.limit:
        rows = rows[: args.limit]
    judged = {p["cell_uid"] for r in rows for p in r["properties"]}
    print(f"{len(rows)} cited works · {len(judged)} cells judged "
          f"· {len(cells) - len(judged)} structural (skipped/undefined)")

    settings = load_settings()
    judge = build_relevance_judge(settings)
    if judge is None:
        print("no LLM transport available (no API key, no claude-agent-sdk)")
        return 1
    run = _judge_transport(judge)
    print(f"judge: {judge.model} via {judge.mode}")

    t0 = time.time()
    passes = [run_pass(run, rows, args.concurrency, f"pass{i + 1}") for i in range(args.passes)]
    elapsed = time.time() - t0

    records = []
    for c in cells:
        rec = dict(c)
        if c["cell_uid"] in judged:
            per = [p.get(c["cell_uid"], {}) for p in passes]
            rec["baseline_answers"] = [a.get("answer", "unclear") for a in per]
            rec["baseline_justifications"] = [a.get("justification", "") for a in per]
            rec["baseline_confidences"] = [a.get("confidence") for a in per]
            rec["baseline_verdicts"] = [
                _DECISION.get((c["claimed"], a.get("answer", "unclear")), "unverifiable")
                for a in per
            ]
            rec["baseline_verdict"] = rec["baseline_verdicts"][0]
            rec["baseline_self_consistent"] = len(set(rec["baseline_answers"])) == 1
        else:
            rec["baseline_verdict"] = c["refari_verdict"]  # inherited structural exclusion
            rec["baseline_self_consistent"] = True
        records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "meta": {
                    "model": judge.model,
                    "transport": judge.mode,
                    "passes": args.passes,
                    "llm_calls": judge.calls,
                    "usage": judge.usage.__dict__ if hasattr(judge.usage, "__dict__") else {},
                    "elapsed_s": round(elapsed, 1),
                    "cells_judged": len(judged),
                    "cited_works": len(rows),
                },
                "cells": records,
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n{len(records)} cells -> {args.out}  ({elapsed:.0f}s, {judge.calls} LLM calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
