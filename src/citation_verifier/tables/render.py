"""
tables/render.py — deterministic Markdown for table-level findings.

Mirrors :mod:`citation_verifier.render`: the output is built in Python from the models,
never written by a model, so the same report is produced for the same findings.

Three blocks per table:

  1. **The grid as the paper printed it** — so a reader can see the claim being audited.
  2. **The columns** — what each header was taken to mean, and the quote it came from
     (an undefined column is called out, since its marks are not checkable).
  3. **The findings** — one row per checked cell, worst first.
"""

from __future__ import annotations

from .model import (
    MARK_STR,
    VERDICT_STR,
    CellVerdict,
    GlossSource,
    TableReport,
)
from .verify import asymmetry_summary

__all__ = ["render_table_report", "render_table_reports"]

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "ok": 3}
_GLOSS_LABEL = {
    GlossSource.CAPTION.value: "caption",
    GlossSource.LEGEND.value: "legend",
    GlossSource.BODY.value: "body text",
    GlossSource.HEADER_ONLY.value: "**never defined**",
    GlossSource.NONE.value: "—",
}


def _cell(text: str) -> str:
    """Make a string safe for a Markdown table cell."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def _grid_block(report: TableReport) -> list[str]:
    """The table as printed: rows × columns with the original marks."""
    t = report.table
    heads = [d.header or f"c{d.col_index}" for d in t.dimensions]
    out = [
        "| Method | " + " | ".join(_cell(h) for h in heads) + " | Cited as |",
        "|---|" + "---|" * len(heads) + "---|",
    ]
    for row in t.rows:
        marks = []
        for d in t.dimensions:
            c = t.cell(row.row_index, d.col_index)
            marks.append(MARK_STR.get(c.mark, "?") if c else "")
        label = _cell(row.label) + (" *(ours)*" if row.is_self else "")
        cited = ", ".join(f"`{k}`" for k in row.cite_keys) or "—"
        out.append(f"| {label} | " + " | ".join(marks) + f" | {cited} |")
    return out


def _columns_block(report: TableReport) -> list[str]:
    """What each column was taken to mean, and where that meaning came from."""
    out = [
        "| Column | Meaning used for checking | Source |",
        "|---|---|---|",
    ]
    for d in report.table.dimensions:
        gloss = _cell(d.gloss) or "*(not defined anywhere in the paper)*"
        out.append(
            f"| {_cell(d.header)} | {gloss[:220]} | {_GLOSS_LABEL.get(d.gloss_source, d.gloss_source)} |"
        )
    return out


def _findings_block(report: TableReport) -> list[str]:
    """Checked cells, worst first. Purely-structural skips are omitted as noise."""
    rows = [
        f
        for f in report.findings
        if f.verdict != CellVerdict.SKIPPED.value
    ]
    if not rows:
        return ["*No checkable cells in this table.*"]
    rows.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.row_label, f.dimension))
    out = [
        "| # | Cited work | Column | Claimed | Verdict | Why |",
        "|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(rows, 1):
        why = _cell(f.justification)
        if f.evidence_quote:
            why = f"{why} — “{_cell(f.evidence_quote)[:160]}”"
        flag = " ⚑" if f.understates_prior_work else ""
        out.append(
            f"| {i} | {_cell(f.row_label)}"
            f"{f' (`{f.cite_key}`)' if f.cite_key else ''} | {_cell(f.dimension)} "
            f"| {MARK_STR.get(f.claimed, '?')} | {VERDICT_STR.get(f.verdict, f.verdict)}{flag} "
            f"| {why[:260]} |"
        )
    return out


def render_table_report(report: TableReport) -> str:
    """Render one :class:`TableReport` as a Markdown section."""
    t = report.table
    s = asymmetry_summary(report)
    head = f"### Table `{t.table_id}`" + (f" — {t.caption}" if t.caption else "")

    parts = [head, "", *_grid_block(report), "", "**Columns**", "", *_columns_block(report), ""]

    verdicts = report.counts()
    bits = [
        f"{VERDICT_STR.get(k, k)} {v}"
        for k, v in sorted(verdicts.items(), key=lambda kv: _order_verdict(kv[0]))
        if k != CellVerdict.SKIPPED.value
    ]
    parts += ["**Findings** — " + (", ".join(bits) if bits else "none"), "", *_findings_block(report), ""]

    flags: list[str] = []
    if s["understated_prior_work"]:
        cells = ", ".join(
            f"{c['row']} / {c['column']}" for c in s["understated_cells"][:6]
        )
        flags.append(
            f"⚑ **{s['understated_prior_work']} ✗ mark(s) refuted by the cited work itself** "
            f"({cells}) — prior work is understated, so the novelty this table claims is overstated."
        )
    if s["undefined_columns"]:
        flags.append(
            "⚠ Column(s) never defined in the paper: "
            + ", ".join(f"**{c}**" for c in s["undefined_columns"])
            + " — the ✓/✗ in them cannot be checked, or reproduced by a reader."
        )
    if s["self_all_yes"]:
        flags.append(
            "ℹ The authors' own row is ✓ on every column — the asymmetry this table rests on."
        )
    if t.warnings:
        flags.append(
            "⚠ The grid was read imperfectly from the "
            + ("PDF" if t.source.startswith("pdf") else "source")
            + " — " + "; ".join(_cell(w) for w in t.warnings)
            + ". Check these rows against the paper before acting on them."
        )
    if flags:
        parts += flags + [""]
    if report.notes:
        parts += ["*Notes: " + "; ".join(_cell(n) for n in report.notes) + "*", ""]
    return "\n".join(parts).rstrip() + "\n"


def _order_verdict(token: str) -> int:
    order = [
        CellVerdict.CONTRADICTED.value,
        CellVerdict.MAY_NOT_SUPPORT.value,
        CellVerdict.UNDEFINED.value,
        CellVerdict.UNVERIFIABLE.value,
        CellVerdict.SUPPORTED.value,
        CellVerdict.SKIPPED.value,
    ]
    return order.index(token) if token in order else len(order)


def render_table_reports(reports: list[TableReport]) -> str:
    """Render every table report as one '## Comparison tables' section.

    Returns ``""`` when there are no tables, so callers can append unconditionally.
    """
    if not reports:
        return ""
    total = sum(
        1
        for r in reports
        for f in r.findings
        if f.verdict != CellVerdict.SKIPPED.value
    )
    understated = sum(1 for r in reports for f in r.findings if f.understates_prior_work)
    head = [
        "## Comparison tables",
        "",
        f"{len(reports)} comparison table(s); {total} cell claim(s) checked"
        + (f"; **{understated} refuted ✗ mark(s)**" if understated else "")
        + ".",
        "",
        "Each cell asserts *“this cited method has this property”*. A ✗ refuted by the "
        "cited paper itself understates prior work and inflates the novelty claimed here.",
        "",
    ]
    return "\n".join(head + [render_table_report(r) for r in reports]).rstrip() + "\n"
