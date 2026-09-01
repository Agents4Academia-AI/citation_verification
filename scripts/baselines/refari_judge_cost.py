#!/usr/bin/env python3
"""
refari_judge_cost.py — price Refari's cell-judging stage on frozen inputs.

The archived audit report records verdicts but no token or cost accounting, so the
system we propose had no cost figure to sit beside the two baselines. Re-running Refari
end to end would produce *different* verdicts and desynchronise the paper, so instead we
replay only the stage the baselines also perform — deciding a cell given whatever
evidence the system holds — over the inputs the report already froze:

  * Refari's own judge, ``tables.llm.build_cell_judge`` with the real ``CELL_SYSTEM``
    prompt. Imported, not reimplemented.
  * One call per cited work, covering all its columns — Refari's own batching shape.
  * The column definitions and retrieved evidence parsed out of the 0829 report.

This yields a cost per judged cell that is directly comparable to ``closed_book.py`` and
``scitance.py``, all three measuring the same stage with the same model and transport.

WHAT IT DOES NOT MEASURE
------------------------
Refari's retrieval and column-grounding stages, which the baselines do not perform and
which the archived run did not instrument. Refari's true end-to-end cost is therefore
strictly HIGHER than the number this prints, and the paper says so.

ONE KNOWN DEVIATION
-------------------
``build_cell_judge`` accepts a per-column ``question`` (the yes/no test the glosser
emits). The report does not record it, so only the ``definition`` is supplied. Verdict
agreement with the archived run is printed as a check on how much that costs us.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

from citation_verifier.backends.relevance_judge import build_relevance_judge
from citation_verifier.config import load_settings
from citation_verifier.tables.llm import build_cell_judge
from citation_verifier.tables.verify import _DECISION  # noqa: PLC2701 — deliberate seam

HERE = Path(__file__).resolve().parent
CELLS = HERE / "out" / "cells.json"
OUT = HERE / "out" / "refari_judge_cost.json"
STRUCTURAL = {"skipped", "undefined"}


def group_rows(cells: list[dict]) -> list[dict]:
    """One judge call per cited work, as Refari does. Evidence is the work's quotes."""
    rows: dict[tuple[str, str], dict] = {}
    for c in cells:
        if c["refari_verdict"] in STRUCTURAL:
            continue
        key = (c["paper"], c["row_label"])
        row = rows.setdefault(key, {"row_label": c["row_label"], "paper": c["paper"],
                                    "quotes": [], "properties": []})
        q = (c.get("evidence") or "").strip()
        if q and q not in row["quotes"]:
            row["quotes"].append(q)
        row["properties"].append({
            "col_index": len(row["properties"]),
            "name": c["dimension"],
            "definition": c["gloss"],
            "question": "",
            "cell_uid": c["cell_uid"],
            "claimed": c["claimed"],
            "archived": c["refari_verdict"],
        })
    return list(rows.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    rows = group_rows(cells)
    if args.limit:
        rows = rows[: args.limit]
    n_cells = sum(len(r["properties"]) for r in rows)
    print(f"{len(rows)} cited works · {n_cells} cells · replaying Refari's judge only")

    settings = load_settings()
    judge = build_relevance_judge(settings)
    if judge is None:
        print("no LLM transport available")
        return 1
    cell_judge = build_cell_judge(judge)
    print(f"judge: {judge.model} via {judge.mode}")

    results: dict[str, str] = {}
    lock, done = threading.Lock(), [0]
    sem = threading.Semaphore(args.concurrency)

    def work(row: dict) -> None:
        payload = {
            "row_label": row["row_label"],
            "evidence": "\n\n".join(row["quotes"]) or "(no evidence retrieved)",
            "properties": row["properties"],
        }
        with sem:
            try:
                got = cell_judge(payload)
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                print(f"    ! {row['paper']}/{row['row_label']}: {exc!r}")
                got = []
        by_col = {int(d["col_index"]): d for d in got
                  if isinstance(d, dict) and str(d.get("col_index", "")).lstrip("-").isdigit()}
        with lock:
            for p in row["properties"]:
                ans = str((by_col.get(p["col_index"]) or {}).get("answer", "unclear")).lower()
                ans = ans if ans in ("has", "lacks", "unclear") else "unclear"
                results[p["cell_uid"]] = _DECISION.get((p["claimed"], ans), "unverifiable")
            done[0] += 1
            print(f"  {done[0]}/{len(rows)} {row['paper']}/{row['row_label']}")

    t0 = time.time()
    threads = [threading.Thread(target=work, args=(r,)) for r in rows]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    archived = {p["cell_uid"]: p["archived"] for r in rows for p in r["properties"]}
    agree = sum(1 for u, v in results.items() if v == archived[u])
    usage = judge.usage.__dict__ if hasattr(judge.usage, "__dict__") else {}
    cost = float(usage.get("cost_usd", 0.0))
    print(f"\ncost ${cost:.2f} over {n_cells} cells = ${cost / n_cells:.4f}/cell "
          f"· {judge.calls} calls · {elapsed:.1f}s")
    print(f"verdict agreement with the archived run: {agree}/{len(results)} "
          f"({100 * agree / len(results):.1f}%)")

    OUT.write_text(json.dumps({
        "meta": {
            "what": "Refari cell-judging stage replayed on frozen glosses + evidence",
            "excludes": "retrieval and column grounding (not instrumented in the archive)",
            "model": judge.model, "transport": judge.mode, "llm_calls": judge.calls,
            "usage": usage, "elapsed_s": round(elapsed, 1),
            "cells_judged": n_cells, "cited_works": len(rows),
            "agreement_with_archive": f"{agree}/{len(results)}",
        },
        "verdicts": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
