"""
build_manifest.py — recover the exact cell set the Refari run reported.

The Refari table run is archived as a rendered per-cell audit
(``table_audit_percell.md`` in the shared outputs dir). This script parses it back
into a machine-readable manifest so a BASELINE can be run on *identical* cells:
same tables, same rows, same column glosses, same ✓/✗ marks.

Parsing a rendered report rather than re-running extraction is deliberate. It
guarantees the baseline and Refari share a denominator — otherwise a re-extraction
that recovers 94 cells instead of 99 makes the comparison meaningless.

Usage::

    python scripts/baselines/build_manifest.py [SRC_MD] [OUT_JSON]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC = Path(
    "/scratch/datasets/citation_verification_outputs/table_audit_percell.md"
)
OUT = Path(__file__).resolve().parent / "out" / "cells.json"

# Rendered verdict glyph -> machine token. Refari's own five CellVerdict values plus
# the "may_not_support" refinement the report adds when full text WAS retrieved and
# the property is still never claimed (verify.py folds this into `unverifiable`).
VERDICT = {
    "✅": "supported",
    "❌": "contradicted",
    "❗": "may_not_support",
    "❓": "unverifiable",
    "⬜": "undefined",
    "➖": "skipped",
}

# The citing paper's assertion for the cell.
MARK = {"具备": "yes", "不具备": "no"}

# Column-definition provenance -> tables.model.GlossSource token.
GLOSS_SOURCE = {
    "正文定义": "body",
    "表题": "caption",
    "图例": "legend",
    "仅提及": "mention",
    "论文未定义": "header_only",
    "模型还原": "mention",
}

_PAPER = re.compile(r"^# (?!合计)(.+)$")
_CITING = re.compile(r"^引用方论文：\[(.+?)\]\((.+?)\)\s*·\s*表\s*`(.+?)`")
_ROW = re.compile(r"^### 被引方法：(.+)$")
_REF = re.compile(r"^- \*\*被引论文\*\*：(.*?)\s*$")
_KEY = re.compile(r"^- \*\*引用键\*\*：`(.+?)`\s*$")
_CELL = re.compile(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*\*\*(.+?)\*\*\s*\|\s*(.)")
_GLOSS = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*$")


def parse(md: str) -> list[dict]:
    """Walk the report top-down, carrying paper / column / row state into each cell."""
    cells: list[dict] = []
    paper = citing_url = table_id = ""
    glosses: dict[str, dict] = {}
    row_label = ref_title = ref_meta = cite_key = ""
    section = ""

    lines = md.splitlines()
    for i, line in enumerate(lines):
        if m := _PAPER.match(line):
            paper, glosses, section = m.group(1).strip(), {}, ""
            continue
        if m := _CITING.match(line):
            citing_url, table_id = m.group(2), m.group(3)
            continue
        if line.startswith("## 这张表的列定义"):
            section = "gloss"
            continue
        if line.startswith("## 逐格"):
            section = "cells"
            continue
        if m := _ROW.match(line):
            row_label, ref_title, ref_meta, cite_key = m.group(1).strip(), "", "", ""
            continue
        if m := _REF.match(line):
            ref_title = m.group(1).strip()
            # The authors · year · venue line is wrapped onto the next line.
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "·" in nxt and not nxt.startswith("- "):
                ref_meta = nxt.strip()
            continue
        if m := _KEY.match(line):
            cite_key = m.group(1).strip()
            continue

        if section == "gloss" and (m := _GLOSS.match(line)):
            header, src, quote = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if header == "属性列":
                continue
            src = src.replace("⚠", "").strip()
            glosses[header] = {
                "gloss": "" if quote.startswith("（") else quote,
                "gloss_source": GLOSS_SOURCE.get(src, "none"),
                "gloss_source_raw": src,
            }
            continue

        if section == "cells" and (m := _CELL.match(line)):
            idx, header, claimed, glyph = (g.strip() for g in m.groups())
            g = glosses.get(header, {"gloss": "", "gloss_source": "none", "gloss_source_raw": ""})
            cells.append(
                {
                    "cell_uid": f"{paper}:{idx}",
                    "paper": paper,
                    "citing_url": citing_url,
                    "table_id": table_id,
                    "index": int(idx),
                    "row_label": row_label,
                    "cited_title": ref_title,
                    "cited_meta": ref_meta,
                    "cite_key": cite_key,
                    "dimension": header,
                    "claimed": MARK.get(claimed, "unknown"),
                    "refari_verdict": VERDICT.get(glyph, "unknown"),
                    **g,
                }
            )

    return cells


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    cells = parse(src.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter

    print(f"{len(cells)} cells -> {out}")
    print("  by verdict:", dict(Counter(c["refari_verdict"] for c in cells)))
    print("  by paper:  ", dict(Counter(c["paper"] for c in cells)))
    print("  unknown marks:", sum(c["claimed"] == "unknown" for c in cells))
    print("  missing gloss:", sum(not c["gloss"] for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
