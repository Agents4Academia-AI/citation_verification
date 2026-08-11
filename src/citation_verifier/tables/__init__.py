"""
tables — TABLE-LEVEL citation verification.

The prose pipeline answers *"does this citation support this sentence?"*. This
subsystem answers a question prose never asks: **is the ✓/✗ this paper put next to a
competitor in its comparison table actually true?**

Those cells are where novelty is positioned, they are rarely restated in the text, and
a wrong ✗ ("prior work can't do this") silently inflates the paper's contribution — so
they are both high-value and, until now, unchecked (table citation sites are ``skipped``
by the prose pipeline).

Four stages, each usable on its own::

    grid        latex_grid.tables_from_latex / pdf_grid.tables_from_pdf
                  -> ComparisonTable (rows × dimensions × marks, rows bound to cite keys)
    meaning     dimensions.resolve_dimensions
                  -> what each terse header actually asserts, quoted from the paper
    check       verify.verify_table
                  -> one verdict per cell, from evidence in the CITED paper
    summarize   verify.asymmetry_summary
                  -> understated prior work, undefined columns, all-✓ own row

Typical use::

    from citation_verifier.tables import extract_tables, verify_tables

    tables = extract_tables(source)                     # PaperSource -> [ComparisonTable]
    reports = verify_tables(tables, body_text=body,     # -> [TableReport]
                            evidence_for=my_fetcher, judge=my_judge)

Offline and dependency-light: extraction is pure text work (PDF path needs the ``pdf``
extra), and both model seams are injected.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dimensions import find_definition_snippets, resolve_dimensions
from .latex_grid import (
    looks_like_comparison_table,
    normalize_mark,
    parse_tabular,
    strip_tex,
    tables_from_latex,
)
from .model import (
    MARK_STR,
    VERDICT_STR,
    CellFinding,
    CellMark,
    CellVerdict,
    ComparisonTable,
    Dimension,
    DimensionKind,
    GlossSource,
    TableCell,
    TableReport,
    TableRow,
    derive_cell_severity,
)
from .pdf_grid import tables_from_pdf
from .verify import asymmetry_summary, verify_table

__all__ = [
    # model
    "CellMark", "CellVerdict", "DimensionKind", "GlossSource",
    "Dimension", "TableRow", "TableCell", "ComparisonTable", "CellFinding", "TableReport",
    "derive_cell_severity", "MARK_STR", "VERDICT_STR",
    # stages
    "tables_from_latex", "tables_from_pdf", "parse_tabular", "normalize_mark", "strip_tex",
    "looks_like_comparison_table", "find_definition_snippets", "resolve_dimensions",
    "verify_table", "asymmetry_summary",
    # convenience
    "extract_tables", "read_body_text", "verify_tables",
]


def read_body_text(source: Any) -> str:
    """All of the citing paper's text, for recovering column meanings.

    Prefers the LaTeX sources (definitions often live in a ``\\begin{definition}`` block
    in a different file from the table); falls back to the extracted PDF text.

    Args:
        source: a :class:`citation_verifier.interfaces.PaperSource`.

    Returns:
        Concatenated text, or ``""`` when nothing is readable.
    """
    work_dir = getattr(source, "work_dir", None)
    if work_dir:
        tex_dir = Path(work_dir) / "tex"
        if tex_dir.is_dir():
            parts = []
            for p in sorted(tex_dir.rglob("*.tex")):
                try:
                    parts.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    continue
            if parts:
                return "\n".join(parts)
    pdf_path = getattr(source, "pdf_path", None)
    if pdf_path:
        try:
            from ..extract.pdf import extract_pdf_text  # noqa: PLC0415 — lazy

            return extract_pdf_text(pdf_path)
        except Exception:  # noqa: BLE001 — no body text is survivable
            return ""
    return ""


def _dedupe(tables: list[ComparisonTable]) -> list[ComparisonTable]:
    """Drop tables that are the same grid seen twice.

    arXiv sources routinely ship a file in two places (``table/x.tex`` and a stray author
    copy under another directory), and both get read. Without this, every finding — and
    every "understates prior work" count and judge call — is doubled.
    """
    seen: set[tuple] = set()
    out: list[ComparisonTable] = []
    for t in tables:
        sig = (
            tuple(d.header for d in t.dimensions),
            tuple((r.label, tuple(r.cite_keys)) for r in t.rows),
            tuple(c.mark for c in t.cells),
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(t)
    return out


def extract_tables(
    source: Any,
    *,
    method_names: set[str] | None = None,
    require_comparison: bool = True,
) -> list[ComparisonTable]:
    """Find the comparison tables in an ingested paper.

    Uses the LaTeX sources when available (exact marks and real ``\\cite`` keys) and
    otherwise falls back to the PDF.

    Args:
        source: a :class:`citation_verifier.interfaces.PaperSource`.
        method_names: the paper's own method name(s), used to flag the "ours" row.
        require_comparison: keep only capability matrices, dropping results tables.

    Returns:
        The tables found, possibly empty. Never raises.
    """
    paper_id = getattr(source, "paper_id", "") or ""
    work_dir = getattr(source, "work_dir", None)
    out: list[ComparisonTable] = []

    if work_dir:
        tex_dir = Path(work_dir) / "tex"
        if tex_dir.is_dir():
            for p in sorted(tex_dir.rglob("*.tex")):
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                out.extend(
                    tables_from_latex(
                        text,
                        paper_id=paper_id,
                        section=p.stem,
                        method_names=method_names,
                        require_comparison=require_comparison,
                    )
                )
    out = _dedupe(out)
    if out:
        return out

    pdf_path = getattr(source, "pdf_path", None)
    if pdf_path:
        out.extend(
            tables_from_pdf(
                pdf_path,
                paper_id=paper_id,
                method_names=method_names,
                require_comparison=require_comparison,
            )
        )
    return out


def verify_tables(
    tables: list[ComparisonTable],
    *,
    body_text: str,
    evidence_for: Callable[[str | None, str], tuple[str, str]],
    judge: Callable[[dict], list[dict]] | None = None,
    glosser: Callable[[list[dict]], list[dict]] | None = None,
    verify_self_rows: bool = False,
) -> list[TableReport]:
    """Resolve column meanings, then verify every checkable cell.

    Args:
        tables: tables from :func:`extract_tables`.
        body_text: the citing paper's text (see :func:`read_body_text`), searched for
            the definitions of the column headers.
        evidence_for: ``(cite_key, row_label) -> (evidence_text, source)``.
        judge: the cell judge (see :func:`citation_verifier.tables.llm.build_cell_judge`).
        glosser: optional column glosser; without it the retrieved passage is the gloss.
        verify_self_rows: also check the authors' own row.

    Returns:
        One :class:`TableReport` per table, in the same order.
    """
    reports: list[TableReport] = []
    for table in tables:
        resolve_dimensions(
            table.dimensions,
            body_text,
            caption=table.caption,
            legend=table.legend,
            glosser=glosser,
        )
        reports.append(
            verify_table(
                table,
                evidence_for=evidence_for,
                judge=judge,
                verify_self_rows=verify_self_rows,
            )
        )
    return reports
