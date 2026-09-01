#!/usr/bin/env python3
"""
score_human.py — score every system against the human cell labels.

Reads the frozen annotation snapshot (``$CV_OUT/human_labels_v1.json``, pulled from the
three-author sheet) and joins it on ``cell_uid`` to whichever system outputs are present:

    Refari       out/cells.json            refari_verdict
    Closed-book  out/closed_book.json      baseline_verdict
    SCitance     out/scitance.json         scitance_verdict
    SCitance+def out/scitance_gloss.json   scitance_verdict   (charitable variant)

Three things this script is deliberately careful about, because each one would silently
inflate a headline number:

1. **Unlabelled rows are excluded.** A row whose ``checker`` is blank carries the sheet's
   default ``Supported``, which is not a human judgement. As of the 2026-09-01 pull that
   is all 20 MARRS cells — the fifth paper is still being annotated.
2. **``skipped`` cells are excluded**, matching the 99-cell denominator the report uses.
3. **Hybrid labels stay hybrid.** Five cells carry two verdicts. They are kept out of the
   strict number and scored generously (correct if the prediction matches either branch)
   elsewhere, rather than collapsed to whichever branch flatters a system.

The headline is NOT accuracy. On this label distribution a constant "Supported"
classifier outscores every system, so that constant is printed as its own row and the
script leads with the error budget instead.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("CV_OUT", "/scratch/datasets/citation_verification_outputs"))

# name -> (file under out/, field holding that system's verdict)
SYSTEMS = [
    ("Refari", "cells.json", "refari_verdict"),
    ("Closed-book", "closed_book.json", "baseline_verdict"),
    ("SCitance", "scitance.json", "scitance_verdict"),
    ("SCitance+def", "scitance_gloss.json", "scitance_verdict"),
]

GOLD = {
    "Supported": {"supported"},
    "Contradicted": {"contradicted"},
    "May Not Support": {"may_not_support"},
    "May not.": {"may_not_support"},
    "Undetermined": {"undefined"},
    "Contradicted/May Not": {"contradicted", "may_not_support"},
    "Supported / May Not Support": {"supported", "may_not_support"},
}
ADVERSE = {"contradicted", "may_not_support"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ci(k: int, n: int) -> str:
    if n == 0:
        return "     n/a       "
    lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} {100 * k / n:5.1f}% [{100 * lo:.0f},{100 * hi:.0f}]"


def load() -> tuple[dict, list[dict]]:
    labels_path = OUT_DIR / "human_labels_v1.json"
    if not labels_path.exists():
        print(f"missing {labels_path}", file=sys.stderr)
        raise SystemExit(1)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    preds: dict[str, dict[str, str]] = {}
    for name, fname, field in SYSTEMS:
        path = HERE / "out" / fname
        if not path.exists():
            print(f"  (skipping {name}: no {fname})", file=sys.stderr)
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        cells = blob["cells"] if isinstance(blob, dict) else blob
        preds[name] = {c["cell_uid"]: c[field] for c in cells if field in c}

    rows = []
    for r in labels["rows"]:
        got = {n: p[r["cell_uid"]] for n, p in preds.items() if r["cell_uid"] in p}
        if "Refari" not in got:
            continue
        rows.append({
            "uid": r["cell_uid"], "paper": r["paper"],
            "labelled": bool(r["checker"]), "gold": GOLD[r["verdict"]],
            "note": r["note"], "pred": got,
        })
    return labels, rows


def main() -> int:
    labels, rows = load()
    names = [n for n, _, _ in SYSTEMS if any(n in r["pred"] for r in rows)]
    scored = [r for r in rows if r["labelled"] and r["pred"]["Refari"] != "skipped"]
    strict = [r for r in scored if len(r["gold"]) == 1]
    unlabelled = sorted({r["paper"] for r in rows if not r["labelled"]})

    print(f"labels pulled {labels['pulled']} · {len(rows)} rows joined · systems: {names}")
    print(f"excluded: unlabelled {unlabelled or '(none)'}, plus 'skipped' cells")
    print(f"SCORED n={len(scored)}  {dict(Counter(r['paper'] for r in scored))}")
    print(f"  gold: {dict(Counter(sorted(r['gold'])[0] if len(r['gold']) == 1 else 'hybrid' for r in scored))}")

    print("\n=== what each system emits (99 eligible cells) ===")
    elig = [r for r in rows if r["pred"]["Refari"] != "skipped"]
    print(f"  {'':<14} " + " ".join(f"{v:>14}" for v in
          ("supported", "contradicted", "may_not", "undet.")))
    for n in names:
        c = Counter(r["pred"].get(n) for r in elig if n in r["pred"])
        undet = c["unverifiable"] + c["undefined"]
        print(f"  {n:<14} " + " ".join(f"{x:>14}" for x in
              (c["supported"], c["contradicted"], c["may_not_support"], undet)))

    print(f"\n=== error budget against {len(scored)} human labels ===")
    for n in names:
        S = [r for r in scored if n in r["pred"]]
        green = [r for r in S if r["pred"][n] == "supported"]
        ok = [r for r in green if "supported" in r["gold"]]
        acc = [r for r in S if r["pred"][n] == "contradicted"]
        conf = [r for r in acc if "contradicted" in r["gold"]]
        over = [r for r in acc if "contradicted" not in r["gold"] and "may_not_support" in r["gold"]]
        false = [r for r in acc if not (r["gold"] & ADVERSE)]
        idle = [r for r in S if r["pred"][n] == "unverifiable" and r["gold"] == {"supported"}]
        st = [r for r in S if len(r["gold"]) == 1]
        exact = sum(1 for r in st if r["pred"][n] in r["gold"])
        print(f"\n  {n}")
        print(f"    Supported issued / confirmed  {ci(len(ok), len(green))}   wrong green lights: {len(green) - len(ok)}")
        print(f"    Contradicted issued {len(acc):>2}       confirmed {len(conf)}, overstated {len(over)}, cleared {len(false)}")
        if false:
            print(f"      cleared: {', '.join(r['uid'] for r in false)}")
        print(f"    Abstained where label Supported  {len(idle)}")
        print(f"    Exact match                   {ci(exact, len(st))}")
    const = sum(1 for r in strict if "supported" in r["gold"])
    print(f"\n  {'constant Supported':<14} exact match {ci(const, len(strict))}  <- the constant")

    print("\n=== pairwise agreement on the scored set ===")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [r for r in scored if a in r["pred"] and b in r["pred"]]
            same = sum(1 for r in both if r["pred"][a] == r["pred"][b])
            print(f"  {a:<14} vs {b:<14} {same}/{len(both)} ({100 * same / len(both):.1f}%)")

    print("\n=== per paper, exact match ===")
    print(f"  {'paper':<12} " + " ".join(f"{n:>13}" for n in names))
    for pap in dict.fromkeys(r["paper"] for r in strict):
        s = [r for r in strict if r["paper"] == pap]
        cells = []
        for n in names:
            k = [r for r in s if n in r["pred"]]
            hit = sum(1 for r in k if r["pred"][n] in r["gold"])
            cells.append(f"{hit:>2}/{len(k):<2} {100 * hit / len(k):4.0f}%" if k else "     -")
        print(f"  {pap:<12} " + " ".join(f"{c:>13}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
