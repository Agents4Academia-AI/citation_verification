#!/usr/bin/env python3
"""
score_human.py — score Refari and the no-retrieval baseline against the human labels.

Reads the frozen annotation snapshot (``$CV_OUT/human_labels_v1.json``, pulled from the
three-author Google Sheet) and ``out/closed_book.json``, joins them on ``cell_uid``, and
reports the numbers the paper quotes.

Three things this script is deliberately careful about, because each one would silently
inflate a headline number:

1. **Unlabelled rows are excluded.** A row whose ``checker`` is blank carries the sheet's
   default ``Supported``, which is not a human judgement. As of the 2026-09-01 pull that
   is all 20 MARRS cells — the fifth paper is still being annotated.
2. **``skipped`` cells are excluded**, matching the 99-cell denominator the report uses.
3. **Hybrid labels stay hybrid.** Five cells carry two verdicts ("Contradicted/May Not",
   "Supported / May Not Support"). They are kept out of the strict number and scored
   generously (correct if the prediction matches either branch) in the lenient one, rather
   than being collapsed to whichever branch flatters a system.

The headline is NOT accuracy. On this label distribution a constant "Supported" classifier
outscores both systems, so the script prints that constant as a third row and leads with
the error budget instead: wrong green lights, false accusations, needless abstentions.
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

# The human vocabulary is free text typed by three people; map it onto the frozen enum.
# A hybrid label maps to a SET — the annotator genuinely did not pick one.
GOLD = {
    "Supported": {"supported"},
    "Contradicted": {"contradicted"},
    "May Not Support": {"may_not_support"},
    "May not.": {"may_not_support"},
    "Undetermined": {"undefined"},
    "Contradicted/May Not": {"contradicted", "may_not_support"},
    "Supported / May Not Support": {"supported", "may_not_support"},
}
ASSERT_BAD = {"contradicted", "may_not_support"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k:>3}/{n:<3} = {100 * k / n:5.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]"


def main() -> int:
    labels_path = OUT_DIR / "human_labels_v1.json"
    preds_path = HERE / "out" / "closed_book.json"
    for p in (labels_path, preds_path):
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 1

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    preds = {c["cell_uid"]: c for c in json.loads(preds_path.read_text())["cells"]}

    rows = []
    for r in labels["rows"]:
        c = preds.get(r["cell_uid"])
        if c is None:
            print(f"WARN no prediction for {r['cell_uid']}", file=sys.stderr)
            continue
        rows.append(
            {
                "uid": r["cell_uid"],
                "paper": r["paper"],
                "labelled": bool(r["checker"]),
                "gold": GOLD[r["verdict"]],
                "note": r["note"],
                "refari": c["refari_verdict"],
                "baseline": c["baseline_verdict"],
                "consistent": c.get("baseline_self_consistent"),
            }
        )

    scored = [r for r in rows if r["labelled"] and r["refari"] != "skipped"]
    strict = [r for r in scored if len(r["gold"]) == 1]
    unlabelled = sorted({r["paper"] for r in rows if not r["labelled"]})

    print(f"pulled {labels['pulled']}  |  {len(rows)} rows joined")
    print(f"excluded: unlabelled papers {unlabelled or '(none)'}, plus 'skipped' cells")
    print(f"SCORED  n={len(scored)}   ({dict(Counter(r['paper'] for r in scored))})")
    print(f"  gold: {dict(Counter(sorted(r['gold'])[0] if len(r['gold']) == 1 else 'hybrid' for r in scored))}")

    print("\n--- accuracy, and why the paper does not lead with it ---")
    for name in ("refari", "baseline"):
        ok = sum(1 for r in strict if r[name] in r["gold"])
        print(f"  {name:<22} strict 4-way {pct(ok, len(strict))}")
    const = sum(1 for r in strict if "supported" in r["gold"])
    print(f"  {'always-say-supported':<22} strict 4-way {pct(const, len(strict))}  <- the constant wins")

    print("\n--- error budget (the numbers that matter) ---")
    for name in ("refari", "baseline"):
        green = [r for r in scored if r[name] == "supported"]
        good_green = [r for r in green if "supported" in r["gold"]]
        acc = [r for r in scored if r[name] == "contradicted"]
        false_acc = [r for r in acc if not (r["gold"] & ASSERT_BAD)]
        overstated = [r for r in acc if "contradicted" not in r["gold"] and "may_not_support" in r["gold"]]
        idle = [r for r in scored if r[name] == "unverifiable" and r["gold"] == {"supported"}]
        print(f"  {name}")
        print(f"    green-light precision   {pct(len(good_green), len(green))}   (coverage {len(green)}/{len(scored)})")
        print(f"    wrong green lights      {len(green) - len(good_green)}")
        print(f"    contradictions issued   {len(acc)}  -> false {len(false_acc)}, overstated {len(overstated)}")
        if false_acc:
            print(f"      false: {', '.join(r['uid'] for r in false_acc)}")
        print(f"    needless abstentions    {len(idle)}  (said unverifiable, human says supported)")

    cons = [r for r in scored if r["consistent"] is not None]
    ok = sum(1 for r in cons if r["consistent"])
    print(f"\n  baseline self-consistency  {pct(ok, len(cons))}")

    print("\n--- per paper, strict 4-way ---")
    for pap in dict.fromkeys(r["paper"] for r in strict):
        s = [r for r in strict if r["paper"] == pap]
        rk = sum(1 for r in s if r["refari"] in r["gold"])
        bk = sum(1 for r in s if r["baseline"] in r["gold"])
        print(f"  {pap:<12} n={len(s):<3} refari {100 * rk / len(s):5.1f}%   baseline {100 * bk / len(s):5.1f}%")

    flagged = [r for r in scored if r["note"] and "main paper" in r["note"]]
    if flagged:
        print(f"\n  NOTE — {len(flagged)} cell(s) where an annotator says Refari's evidence came from the")
        print("  citing paper rather than the cited work: " + ", ".join(r["uid"] for r in flagged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
