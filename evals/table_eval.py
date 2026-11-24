"""
table_eval.py — score comparison-table findings against human cell labels.

    python -m evals.table_eval --labels labels.csv papers/*/tables.json [--json]

On the scoring side of the boundary, like :mod:`evals.run_eval`: it reads the agent's
output as JSON on disk and imports nothing from the verifier. The report the agent writes
never sees the labels, so the product cannot be shaped by the evaluation set.

What is scored
--------------
The unit is one table cell — "cited method R has / lacks property D" — and the question
is the reviewer's: *should a person look at this cell?* So the two verdicts that ask for
a look, ``contradicted`` and ``may_not_support``, are scored together as **flagged**, and
a human label of "contradicted", "may not support" or the undecided "supported / may not
support" is a flagged cell too (a cell an experienced reader could not sign off on is one
worth raising).

    precision = flagged cells a person also flags / all cells we flag
    recall    = flagged cells a person also flags / all cells a person flags

A cell we leave open (``unverifiable`` / ``undefined``) is a miss when the person flags
it, and otherwise is counted apart: it is neither a correct clearance nor a false alarm.
Cells the person marks "undetermined" (the paper never defines the column) are reported
separately and enter neither rate. Skipped cells (the authors' own row) are ignored.

Inputs
------
``--labels`` is a CSV with one row per cell. Columns are matched by name, loosely:
a table id (``Table``), the row (``Cited method (row)``), the column (``Property
column``) and the human verdict (any header starting ``HUMAN VERDICT``).

Each findings file is either the ``tables.json`` a run writes
(``{"tables": [TableReport, …]}``), one ``TableReport`` per line, or the flat audit
format of earlier experiments (``{name: {"table_id": …, "rows": [{row, col, verdict}]}}``).
Cells are joined on ``(table id, row label, column header)``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

__all__ = ["load_labels", "load_findings", "score", "main"]

FLAGGED = ("contradicted", "may_not_support")
OPEN = ("unverifiable", "undefined")

Key = tuple[str, str, str]  # (table id, row label, column header)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def human_class(raw: str) -> str | None:
    """``flag`` / ``clear`` / ``undetermined`` from an annotator's wording, or ``None``."""
    h = _norm(raw)
    if not h:
        return None
    if h.startswith(("contradicted", "may not", "supported /", "incorrect")):
        return "flag"
    if h.startswith(("supported", "correct")):
        return "clear"
    if h.startswith(("undetermined", "indeterminate")):
        return "undetermined"
    return None


def _column(fieldnames: list[str], *needles: str) -> str | None:
    for name in fieldnames:
        low = name.lower()
        if any(low.startswith(n) or n in low for n in needles):
            return name
    return None


def load_labels(path: str | Path) -> dict[Key, str]:
    """``{(table, row, column): human class}`` from a labels CSV."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        verdict = _column(fields, "human verdict")
        table = next((f for f in fields if f.strip().lower() == "table"), None) or _column(fields, "table id")
        row = _column(fields, "cited method", "row")
        col = _column(fields, "property column", "column")
        missing = [n for n, c in (("verdict", verdict), ("table", table), ("row", row), ("column", col)) if not c]
        if missing:
            raise ValueError(f"labels CSV lacks column(s) for: {', '.join(missing)} (have {fields})")
        out: dict[Key, str] = {}
        for rec in reader:
            cls = human_class(rec.get(verdict, ""))
            if cls is not None:
                out[(_norm(rec[table]), _norm(rec[row]), _norm(rec[col]))] = cls
    return out


def load_findings(paths: list[str | Path]) -> dict[Key, str]:
    """``{(table, row, column): verdict}`` from one or more findings files."""
    out: dict[Key, str] = {}

    def from_report(rep: dict) -> None:
        table_id = (rep.get("table") or {}).get("table_id", "")
        for f in rep.get("findings") or []:
            out[(_norm(table_id), _norm(f.get("row_label")), _norm(f.get("dimension")))] = f.get("verdict", "")

    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:  # one TableReport per line
            for line in text.splitlines():
                if line.strip():
                    from_report(json.loads(line))
            continue
        if isinstance(data, dict) and isinstance(data.get("tables"), list):
            for rep in data["tables"]:
                from_report(rep)
        elif isinstance(data, dict) and "findings" in data:
            from_report(data)
        elif isinstance(data, dict):  # flat audit format of earlier experiments
            for entry in data.values():
                if not isinstance(entry, dict):
                    continue
                for r in entry.get("rows") or []:
                    out[(_norm(entry.get("table_id")), _norm(r.get("row")), _norm(r.get("col")))] = r.get("verdict", "")
        elif isinstance(data, list):
            for rep in data:
                from_report(rep)
    return out


def score(findings: dict[Key, str], labels: dict[Key, str]) -> dict:
    """Confusion counts and rates for ``flagged``, plus what was left out and why."""
    tp = fp = fn = tn = 0
    left_open, on_undetermined = Counter(), Counter()
    detail: dict[str, list[Key]] = {"tp": [], "fp": [], "fn": [], "missed_open": []}
    unjoined = [k for k in labels if k not in findings]
    for key, human in labels.items():
        verdict = findings.get(key)
        if verdict is None or verdict == "skipped":
            continue
        ours = "flag" if verdict in FLAGGED else "open" if verdict in OPEN else "clear"
        if human == "undetermined":
            on_undetermined[ours] += 1
            continue
        if ours == "open":
            left_open[human] += 1
            if human == "flag":
                fn += 1
                detail["missed_open"].append(key)
            continue
        if human == "flag" and ours == "flag":
            tp += 1
            detail["tp"].append(key)
        elif human == "clear" and ours == "flag":
            fp += 1
            detail["fp"].append(key)
        elif human == "flag" and ours == "clear":
            fn += 1
            detail["fn"].append(key)
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cells_labelled": len(labels),
        "cells_joined": len(labels) - len(unjoined),
        "unjoined": [list(k) for k in unjoined],
        "verdicts": dict(Counter(findings[k] for k in labels if k in findings)),
        "flagged": {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        "left_open": dict(left_open),
        "on_human_undetermined": dict(on_undetermined),
        "detail": {k: [list(x) for x in v] for k, v in detail.items()},
    }


def _print(result: dict) -> None:
    f = result["flagged"]
    print("comparison-table eval")
    print(f"  labelled={result['cells_labelled']}  joined={result['cells_joined']}  verdicts={result['verdicts']}")
    print("\n  FLAGGED (contradicted + may_not_support)")
    print(f"    TP {f['tp']}  FP {f['fp']}  FN {f['fn']}  TN {f['tn']}")
    print(f"    precision {f['precision']:.1%}   recall {f['recall']:.1%}   F1 {f['f1']:.1%}")
    print(f"  left open by us (unverifiable/undefined), by human label: {result['left_open']}")
    print(f"  cells the human marked undetermined, by our call: {result['on_human_undetermined']}")
    for name, title in (("fp", "false alarms"), ("fn", "missed (we cleared)"),
                        ("missed_open", "missed (we left open)")):
        if result["detail"][name]:
            print(f"\n  {title}:")
            for table, row, col in result["detail"][name]:
                print(f"    {table} · {row} × {col}")
    if result["unjoined"]:
        print(f"\n  {len(result['unjoined'])} labelled cell(s) had no finding to join, e.g. {result['unjoined'][:3]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="table_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("findings", nargs="+", help="tables.json file(s) / audit JSON")
    parser.add_argument("--labels", required=True, help="CSV of human cell labels")
    parser.add_argument("--json", action="store_true", help="print the metrics as JSON")
    args = parser.parse_args(argv)
    result = score(load_findings(args.findings), load_labels(args.labels))
    if args.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        _print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
