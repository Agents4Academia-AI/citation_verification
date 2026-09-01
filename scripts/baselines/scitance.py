#!/usr/bin/env python3
"""
scitance.py — a literature baseline: SCitance-style zero-shot claim verification.

Alvarez, Bennett and Wang, *Zero-shot Scientific Claim Verification Using LLMs and
Citation Text* (SDP 2024, ``scitance2024``). Their finding is that prompting an LLM
directly with a citation sentence ("citance") and the cited work's evidence matches
fine-tuned scientific claim verifiers. It is the closest published method to our task
that is a *prompting procedure* rather than fine-tuned weights, so it can be run
faithfully with a different model. (We checked the alternatives: ``zhang2025atomic``
needs 350 fine-tuning examples, and ``minicheck2024`` is a fine-tuned small model.
Running either "with Claude" would measure our port, not their method.)

Why this baseline earns its place: the introduction claims table-cell verification is
"not simply reference matching or sentence-level entailment". SCitance *is* sentence-level
entailment over a citation claim. Running it turns that assertion into a measurement.

WHAT IS COPIED VERBATIM
-----------------------
* ``SCITANCE_PROMPT`` — their Appendix-A prompt, the ``NEI=Yes / citance-abstract pairs /
  zero-shot`` row of Table 4, which is their best GPT-4 configuration (Micro-F1 80.1,
  above their few-shot 75.4). Not one word is changed; only the two ``{}`` slots are filled.
* Their three-label set, SUPPORTS / CONTRADICTS / NOT_ENOUGH_INFO.

WHAT IS ADAPTED, AND WHY
------------------------
* *The claim.* They verify a citance — one sentence written by the citing authors. Our
  unit is a table cell, so the cell is verbalised mechanically (no LLM in the loop, so
  the step adds no capability): "<row> has the property "<column>"." for ✓, "does not
  have" for ✗. That is literally what the cell asserts.
* *The evidence.* They pass the cited work's abstract. We pass the sentence Refari
  retrieved from the cited work, so this row isolates the *decision procedure* rather
  than re-testing retrieval. Where Refari retrieved nothing, the model is told so.
* *Temperature.* They use 0.2. Our shared transport does not expose temperature, so this
  runs at the judge's default. Recorded here because it is a real deviation.

WHAT THIS EXPOSES (the differences that are the point)
------------------------------------------------------
1. **Not blinded.** SCitance is shown the claim *including its polarity*, so it can
   rubber-stamp the table. Refari never sees the ✓/✗ and compares afterwards.
2. **No grounded column meaning.** SCitance gets the column header, not the citing
   paper's definition of it. ``--gloss`` re-runs with the definition appended, as a
   charitable variant, so the comparison cannot be accused of crippling the baseline.
3. **Three labels, not four.** There is no "may not support"; that verdict needs full
   text plus the observation that a property is never claimed.

Usage::

    python scripts/baselines/scitance.py                 # as published
    python scripts/baselines/scitance.py --gloss         # charitable: + column definition
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path

from citation_verifier.backends.relevance_judge import build_relevance_judge
from citation_verifier.config import load_settings
from citation_verifier.tables.llm import _judge_transport

HERE = Path(__file__).resolve().parent
CELLS = HERE / "out" / "cells.json"

# Same structural exclusions as Refari and the closed-book baseline, so all three
# report over one denominator. These are properties of the CITING paper.
STRUCTURAL = {"skipped", "undefined"}

# Verbatim, from Table 4 of the paper (NEI=Yes, citance-abstract pairs, zero-shot).
SCITANCE_PROMPT = (
    "Please obey the following: With a specific abstract, please make an estimation "
    "whether the abstract SUPPORTS, CONTRADICTS, or if there is NOT_ENOUGH_INFO to "
    "determine. You must choose SUPPORTS or CONTRADICTS or NOT_ENOUGH_INFO. Please "
    "return your answer as only the capitalized token, as well as an explanation or "
    "rationale for the answer. Abstract: {} Claim: {}"
)

# Their label set -> ours. Direct, with no mark-vs-answer step: because SCitance sees the
# claim's polarity, "SUPPORTS" already means "the table's mark is right".
LABEL = {
    "SUPPORTS": "supported",
    "CONTRADICTS": "contradicted",
    "NOT_ENOUGH_INFO": "unverifiable",
}
_TOKEN = re.compile(r"\b(SUPPORTS|CONTRADICTS|NOT_ENOUGH_INFO)\b")

NO_EVIDENCE = (
    "No supporting sentence could be retrieved from the cited work for this property."
)


def verbalise(cell: dict, *, with_gloss: bool) -> str:
    """The table cell as a citance. Mechanical: no model, so it grants no capability."""
    verb = "has" if cell["claimed"] == "yes" else "does not have"
    claim = f'{cell["row_label"]} {verb} the property "{cell["dimension"]}".'
    if with_gloss and cell.get("gloss"):
        claim += f' Here, "{cell["dimension"]}" means: {cell["gloss"]}'
    return claim


def judge_cell(run, cell: dict, *, with_gloss: bool) -> dict:
    """One cell, one call. A failed call degrades to NOT_ENOUGH_INFO, never raises."""
    evidence = cell.get("evidence") or NO_EVIDENCE
    user = SCITANCE_PROMPT.format(evidence, verbalise(cell, with_gloss=with_gloss))
    try:
        raw = run("", user, 400)
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash, as elsewhere in the repo
        print(f"    ! {cell['cell_uid']}: {exc!r}")
        raw = ""
    m = _TOKEN.search(raw or "")
    token = m.group(1) if m else "NOT_ENOUGH_INFO"
    return {
        "scitance_label": token,
        "scitance_verdict": LABEL[token],
        "scitance_rationale": (raw or "").strip()[:500],
        "scitance_parsed": bool(m),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gloss", action="store_true",
                    help="charitable variant: append the grounded column definition")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or HERE / "out" / (
        "scitance_gloss.json" if args.gloss else "scitance.json"
    )

    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    todo = [c for c in cells if c["refari_verdict"] not in STRUCTURAL]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} cells to judge · {len(cells) - len(todo)} structural "
          f"· gloss={'on' if args.gloss else 'off'}")
    print(f"  cells with retrieved evidence: {sum(1 for c in todo if c.get('evidence'))}"
          f"/{len(todo)}")

    settings = load_settings()
    judge = build_relevance_judge(settings)
    if judge is None:
        print("no LLM transport available (no API key, no claude-agent-sdk)")
        return 1
    run = _judge_transport(judge)
    print(f"judge: {judge.model} via {judge.mode}")

    answers: dict[str, dict] = {}
    lock, done = threading.Lock(), [0]
    sem = threading.Semaphore(args.concurrency)

    def work(cell: dict) -> None:
        with sem:
            got = judge_cell(run, cell, with_gloss=args.gloss)
        with lock:
            answers[cell["cell_uid"]] = got
            done[0] += 1
            print(f"  {done[0]}/{len(todo)} {cell['cell_uid']} -> {got['scitance_label']}")

    t0 = time.time()
    threads = [threading.Thread(target=work, args=(c,)) for c in todo]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    records = []
    for c in cells:
        rec = dict(c)
        if c["cell_uid"] in answers:
            rec.update(answers[c["cell_uid"]])
        else:
            rec["scitance_verdict"] = c["refari_verdict"]  # inherited structural exclusion
            rec["scitance_label"] = "STRUCTURAL"
            rec["scitance_parsed"] = True
        records.append(rec)

    unparsed = [r["cell_uid"] for r in records if not r.get("scitance_parsed", True)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {
            "method": "scitance2024 zero-shot, NEI, citance-abstract prompt (verbatim)",
            "gloss_variant": args.gloss,
            "model": judge.model,
            "transport": judge.mode,
            "llm_calls": judge.calls,
            "usage": judge.usage.__dict__ if hasattr(judge.usage, "__dict__") else {},
            "elapsed_s": round(elapsed, 1),
            "cells_judged": len(answers),
            "unparsed": unparsed,
        },
        "cells": records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(answers)} cells · {round(elapsed, 1)}s · {len(unparsed)} unparsed "
          f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
