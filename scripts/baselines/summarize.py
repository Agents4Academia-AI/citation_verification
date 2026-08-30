"""
summarize.py — compare the no-retrieval baseline against Refari on identical cells.

Human cell labels are not finished, so nothing here claims who is *right*. What it
does measure, and what needs no gold at all:

  * **Determinacy** — how often each system commits to Supported/Contradicted rather
    than abstaining. A system that always commits looks better on coverage and worse
    on honesty; which it is cannot be read off this number alone.
  * **Behaviour on cells with no retrievable evidence** — Refari abstains there by
    construction. Whatever the baseline returns on those cells is produced from
    memory alone, which is the failure mode the task exists to catch.
  * **Agreement** — a confusion matrix over the cells both systems committed on.
  * **Self-consistency** — two passes over byte-identical input, comparable to the
    counterfactual-consistency figure already reported for Refari.

Usage::

    python scripts/baselines/summarize.py [--json] [--latex]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "out" / "closed_book.json"

PAPERS = ["LLM-FE", "ATU", "USEEK", "REMARK-LLM", "MARRS"]
DETERMINATE = {"supported", "contradicted", "may_not_support"}
ABSTAIN = {"unverifiable", "undefined"}


def bucket(v: str) -> str:
    """Collapse to the four buckets the paper's output-distribution table uses."""
    return v if v in ("supported", "contradicted", "may_not_support") else "undetermined"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.run.read_text(encoding="utf-8"))
    cells = data["cells"]
    eligible = [c for c in cells if c["refari_verdict"] != "skipped"]
    judged = [c for c in eligible if "baseline_answers" in c]

    # ── per-paper distribution, both systems ──────────────────────────
    dist: dict[str, dict[str, Counter]] = defaultdict(lambda: {"refari": Counter(), "baseline": Counter()})
    for c in eligible:
        dist[c["paper"]]["refari"][bucket(c["refari_verdict"])] += 1
        dist[c["paper"]]["baseline"][bucket(c["baseline_verdict"])] += 1

    # ── headline counts ───────────────────────────────────────────────
    r_det = sum(c["refari_verdict"] in DETERMINATE for c in eligible)
    b_det = sum(c["baseline_verdict"] in DETERMINATE for c in eligible)

    # Cells Refari could not settle from retrieved evidence. Anything the baseline
    # commits to here came from memory alone.
    r_abstained = [c for c in judged if c["refari_verdict"] in ABSTAIN]
    b_commits_on_abstain = [c for c in r_abstained if c["baseline_verdict"] in DETERMINATE]

    # Agreement over the cells BOTH committed on.
    both = [c for c in judged if c["refari_verdict"] in DETERMINATE and c["baseline_verdict"] in DETERMINATE]
    agree = sum(bucket(c["refari_verdict"]) == bucket(c["baseline_verdict"]) for c in both)
    confusion = Counter((bucket(c["refari_verdict"]), bucket(c["baseline_verdict"])) for c in both)

    # Self-consistency across two passes on identical input.
    inconsistent = [c for c in judged if not c["baseline_self_consistent"]]

    # Contradictions: the finding that actually costs a paper something.
    r_contra = [c for c in eligible if c["refari_verdict"] == "contradicted"]
    b_contra = [c for c in eligible if c["baseline_verdict"] == "contradicted"]

    n = len(eligible)
    print(f"cells: {n} eligible · {len(judged)} judged by both · "
          f"{n - len(judged)} undefined-column (inherited)\n")
    print(f"{'paper':<12} {'Refari (S/C/M/U)':>20}   {'Closed-book (S/C/M/U)':>22}")
    for p in PAPERS:
        r, b = dist[p]["refari"], dist[p]["baseline"]
        f = lambda x: f"{x['supported']:>3} {x['contradicted']:>3} {x['may_not_support']:>3} {x['undetermined']:>3}"  # noqa: E731
        print(f"{p:<12} {f(r):>20}   {f(b):>22}")
    tr = Counter(bucket(c["refari_verdict"]) for c in eligible)
    tb = Counter(bucket(c["baseline_verdict"]) for c in eligible)
    f = lambda x: f"{x['supported']:>3} {x['contradicted']:>3} {x['may_not_support']:>3} {x['undetermined']:>3}"  # noqa: E731
    print(f"{'TOTAL':<12} {f(tr):>20}   {f(tb):>22}")

    print(f"\ndeterminate:  Refari {r_det}/{n} ({r_det / n:.1%})   "
          f"closed-book {b_det}/{n} ({b_det / n:.1%})")
    print(f"contradicted: Refari {len(r_contra)}   closed-book {len(b_contra)}")

    if r_abstained:
        k, m = len(b_commits_on_abstain), len(r_abstained)
        print(f"\non the {m} cells Refari could not settle from retrieved evidence,")
        print(f"  the closed-book judge still commits on {k}/{m} ({k / m:.1%}) — from memory alone")
        print("  of those:", dict(Counter(c["baseline_verdict"] for c in b_commits_on_abstain)))

    if both:
        print(f"\nagreement where both commit: {agree}/{len(both)} ({agree / len(both):.1%})")
        for (r, b), k in sorted(confusion.items(), key=lambda kv: -kv[1]):
            flag = "" if r == b else "   <- disagreement"
            print(f"  Refari {r:<16} / baseline {b:<16} {k:>3}{flag}")

    ni = len(inconsistent)
    print(f"\nself-consistency (2 passes, identical input): "
          f"{len(judged) - ni}/{len(judged)} ({1 - ni / len(judged):.1%}) on judged cells; "
          f"{n - ni}/{n} ({1 - ni / n:.1%}) over all eligible")
    for c in inconsistent:
        print(f"  flip: {c['cell_uid']:<14} {c['row_label'][:22]:<24} {c['dimension'][:26]:<28} "
              f"{c['baseline_answers']}")

    meta = data["meta"]
    print(f"\nmodel {meta['model']} via {meta['transport']} · {meta['llm_calls']} calls "
          f"· {meta['elapsed_s']}s · est ${meta.get('usage', {}).get('cost_usd', 0):.2f}")

    if args.latex:
        print("\n% ── LaTeX: baseline vs Refari, per paper ──")
        for p in PAPERS:
            r, b = dist[p]["refari"], dist[p]["baseline"]
            print(f"{p} & {b['supported']} & {b['contradicted']} & {b['undetermined']} "
                  f"& {r['supported']} & {r['contradicted']} & {r['may_not_support']} "
                  f"& {r['undetermined']} \\\\")
        print(f"Total & {tb['supported']} & {tb['contradicted']} & {tb['undetermined']} "
              f"& {tr['supported']} & {tr['contradicted']} & {tr['may_not_support']} "
              f"& {tr['undetermined']} \\\\")

    if args.json:
        print("\n" + json.dumps(
            {
                "eligible": n,
                "judged": len(judged),
                "determinate": {"refari": r_det, "baseline": b_det},
                "contradicted": {"refari": len(r_contra), "baseline": len(b_contra)},
                "refari_abstentions": len(r_abstained),
                "baseline_commits_on_refari_abstentions": len(b_commits_on_abstain),
                "agreement": {"both_commit": len(both), "agree": agree},
                "self_consistent_judged": len(judged) - ni,
                "self_consistent_eligible": n - ni,
                "totals": {"refari": dict(tr), "baseline": dict(tb)},
            },
            indent=1,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
