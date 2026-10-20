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

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dimensions import find_definition_snippets, resolve_dimensions
from .evidence import build_evidence_provider, compose_evidence
from .latex_grid import (
    collect_macro_names,
    included_sources,
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
    "collect_macro_names",
    "verify_table", "asymmetry_summary",
    "build_evidence_provider", "compose_evidence",
    # convenience
    "extract_tables", "read_body_text", "verify_tables",
    "choose_extraction", "extraction_quality",
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


_OURS_MARKER_RE = re.compile(
    r"\s*(?:\((?:[^()]*)\)|\[[^\]]*\])\s*$|\s*[-—–]\s*(?:ours?|this\s+(?:work|paper))\s*$",
    re.IGNORECASE,
)


def _own_name(label: str) -> str:
    """The bare method name inside an "ours" row label.

    ``"CaT (this work)"`` -> ``"CaT"``. The parenthetical has to go: the name is later
    matched word-by-word against prose to spot sentences that describe the citing paper's
    own system, and the marker never appears there.
    """
    name = (label or "").strip()
    for _ in range(2):  # "X (ours) [1]" carries two trailing markers
        name = _OURS_MARKER_RE.sub("", name).strip()
    return "" if len(name) <= 2 or _OURS_ONLY_RE.fullmatch(name) else name


_OURS_ONLY_RE = re.compile(r"(?:ours?|our\s+method|this\s+(?:work|paper)|proposed)", re.I)


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
            sources = []
            # Only the files the document actually pulls in: an arXiv tree ships drafts
            # and previous submissions beside the live ones, and one such draft carried an
            # outdated version of a table under the same \label as the published one.
            for p in included_sources(sorted(tex_dir.rglob("*.tex"))):
                try:
                    sources.append((p, p.read_text(encoding="utf-8", errors="ignore")))
                except OSError:
                    continue
            # Aliases are declared in the preamble but used in an \input-ed table file,
            # so they must be collected across the whole paper before parsing any table.
            macros = collect_macro_names("\n".join(text for _p, text in sources))
            for p, text in sources:
                out.extend(
                    tables_from_latex(
                        text,
                        paper_id=paper_id,
                        section=p.stem,
                        method_names=method_names,
                        require_comparison=require_comparison,
                        macros=macros,
                    )
                )
    latex_tables = _dedupe(out)

    pdf_tables: list[ComparisonTable] = []
    pdf_path = getattr(source, "pdf_path", None)
    if pdf_path:
        try:
            pdf_tables = tables_from_pdf(
                pdf_path,
                paper_id=paper_id,
                method_names=method_names,
                require_comparison=require_comparison,
            )
        except Exception:  # noqa: BLE001 — the PDF is the second opinion, never fatal
            pdf_tables = []

    return choose_extraction(latex_tables, pdf_tables)


def extraction_quality(table: ComparisonTable) -> float:
    """How usable an extraction of one table is, in ``[0, 1]``.

    Neither source wins outright and the failures are not symmetric. LaTeX gives exact
    ``\\cite`` keys but the tree can hold a stale draft of the same table — one paper's
    source shipped a five-column version the document never ``\\input``s while the PDF
    carried the published six-column one. The PDF is always the published artifact but its
    headers get mangled by line-breaking and its row labels can be prose from the
    surrounding column.

    Scored on what verification actually needs, in order: a row is useless without a
    citation to resolve, a column is useless without a name, and a cell is useless without
    a mark.
    """
    if not table.rows or not table.dimensions:
        return 0.0
    cited = sum(1 for r in table.rows if r.cite_keys or r.is_self) / len(table.rows)
    named = sum(1 for d in table.dimensions if d.header.strip()) / len(table.dimensions)
    labelled = sum(1 for r in table.rows if _looks_like_a_method_name(r.label)) / len(table.rows)
    total = len(table.rows) * len(table.dimensions)
    marked = sum(
        1 for c in table.cells
        if c.mark in (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)
    ) / max(1, total)
    return 0.40 * cited + 0.25 * named + 0.20 * labelled + 0.15 * marked


_PROSE_LABEL_RE = re.compile(
    r"\b(?:the|and|of|for|with|that|this|which|while|from|are|is|was|were|has|have)\b",
    re.IGNORECASE,
)


def _looks_like_a_method_name(label: str) -> bool:
    """True when a row label names a method rather than being a scrap of prose.

    The PDF path takes row labels from the band beside the marks, and a table set next to
    body text picks up sentences ("adaptability and deeper contextual understanding").
    Function words in quantity are what separate those from "Skeleton Merger [29]".
    """
    text = (label or "").strip()
    if not text or len(text) > 60:
        return False
    return len(_PROSE_LABEL_RE.findall(text)) <= 1


def _same_table(a: ComparisonTable, b: ComparisonTable) -> bool:
    """True when two extractions are of the same printed table.

    Matched on the mark grid rather than on labels or headers, because those are exactly
    what the two paths disagree about. Marks are what both read reliably.
    """
    ga = [c.mark for c in sorted(a.cells, key=lambda c: (c.row_index, c.col_index))]
    gb = [c.mark for c in sorted(b.cells, key=lambda c: (c.row_index, c.col_index))]
    real = {CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value}
    sa = [m for m in ga if m in real]
    sb = [m for m in gb if m in real]
    if len(sa) < 4 or len(sb) < 4:
        return False
    overlap = sum(1 for x, y in zip(sa, sb, strict=False) if x == y)
    return overlap / max(len(sa), len(sb)) >= 0.7


def choose_extraction(
    latex_tables: list[ComparisonTable], pdf_tables: list[ComparisonTable]
) -> list[ComparisonTable]:
    """Keep the better extraction of each table, and record what the other one said.

    A table seen by both paths gets whichever extraction scores higher, with the loser's
    mark grid recorded in ``warnings`` when the two disagree — two independent readings of
    the same printed table are the cheapest cross-check available, and a disagreement is
    exactly the cell a human should look at. A table only one path found is kept as-is.
    """
    if not pdf_tables:
        return latex_tables
    if not latex_tables:
        return pdf_tables

    chosen: list[ComparisonTable] = []
    unmatched_pdf = list(pdf_tables)
    for lt in latex_tables:
        peer = next((pt for pt in unmatched_pdf if _same_table(lt, pt)), None)
        if peer is None:
            chosen.append(lt)
            continue
        unmatched_pdf.remove(peer)
        ql, qp = extraction_quality(lt), extraction_quality(peer)
        win, lose, win_src, lose_src = (
            (lt, peer, "latex", "pdf") if ql >= qp else (peer, lt, "pdf", "latex")
        )
        win.warnings.append(
            f"cross-checked against the {lose_src} extraction "
            f"(quality {max(ql, qp):.2f} vs {min(ql, qp):.2f}); kept {win_src}"
        )
        for note in _mark_disagreements(win, lose):
            win.warnings.append(note)
        chosen.append(win)
    chosen.extend(unmatched_pdf)
    return chosen


def _mark_disagreements(win: ComparisonTable, lose: ComparisonTable) -> list[str]:
    """Cells where the two extractions read a different mark."""
    other = {(c.row_index, c.col_index): c.mark for c in lose.cells}
    out = []
    for c in win.cells:
        got = other.get((c.row_index, c.col_index))
        if got is not None and got != c.mark:
            row = next((r.label for r in win.rows if r.row_index == c.row_index), "?")
            col = next((d.header for d in win.dimensions if d.col_index == c.col_index), "?")
            out.append(f"extractions disagree on '{row}' × '{col}': {c.mark} vs {got}")
    return out[:20]


def verify_tables(
    tables: list[ComparisonTable],
    *,
    body_text: str,
    evidence_for: Callable[[str | None, str], tuple[str, str]],
    judge: Callable[[dict], list[dict]] | None = None,
    glosser: Callable[[list[dict]], list[dict]] | None = None,
    verify_self_rows: bool = False,
    method_names: set[str] | None = None,
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
        method_names: the citing paper's own method name(s), so a sentence describing
            that method is not adopted as a column definition.

    Returns:
        One :class:`TableReport` per table, in the same order.
    """
    reports: list[TableReport] = []
    for table in tables:
        # The "ours" row label IS the paper's own method name, spelled the way the paper
        # spells it. Harvesting it means a caption or body sentence boasting about that
        # method is recognised as self-description even when the caller passed no
        # `method_names` — which is the common case, since nothing upstream knows it.
        own = set(method_names or set()) | {
            n for r in table.rows if r.is_self for n in [_own_name(r.label)] if n
        }
        resolve_dimensions(
            table.dimensions,
            body_text,
            caption=table.caption,
            legend=table.legend,
            glosser=glosser,
            own_names=own,
            paper_id=table.paper_id,
            table_id=table.table_id,
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
