"""
tables/pdf_grid.py — recover comparison-table grids from a PDF.

The fallback path, used when no LaTeX e-print exists (most ACL/NAACL camera-ready PDFs).

**Why generic table detection is not enough.** Comparison tables are typeset booktabs
style: horizontal rules only, no vertical lines. PyMuPDF's default ``find_tables()``
segments columns from ruling lines, so on these tables it collapses the entire mark
matrix into one cell (measured on ACL PDFs: a 7x5 matrix came back as ``2x2`` with the
cell text ``"✗ ✓ ✓ ✓ ✓\\n✓ ✗ ✗ ✓ ✗\\n…"``). ``strategy="text"`` recovers the rows but
still merges adjacent mark columns and swallows the caption and body text around it.

**What this module does instead: anchor the grid on the marks themselves.** In a
capability matrix the ✓/✗ glyphs are strongly aligned — one tight x-cluster per column,
one y-cluster per row — which makes them a far better skeleton than any ruling line:

  1. find the mark glyphs on the page (they extract as real Unicode ✓ U+2713 / ✗ U+2717
     in practice; the symbol-font artifacts are mapped too),
  2. cluster their x-centres into column anchors and y-centres into row anchors,
  3. read each row's label from the text band beside the mark block (left, or right when
     the table puts the marks first), and each column's header from the text above it.

That also bounds the table to the mark block, so surrounding prose cannot leak in.
``find_tables(strategy="text")`` is kept as a fallback for tables whose cells are words
or numbers rather than symbols.

One thing the PDF path cannot do: a PDF row says "MetaAug [21]" or "MetaAug (Rajendran
et al., 2020)", not ``\\cite{rajendran2020meta}``. This module extracts the *marker*;
binding it to a reference is the caller's job (the extract layer already builds that map).

Requires PyMuPDF (the ``pdf`` extra). Import-safe without it: the import is lazy and
:func:`tables_from_pdf` returns ``[]`` when it is unavailable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .latex_grid import _is_self_row
from .model import CellMark, ComparisonTable, Dimension, TableCell, TableRow

__all__ = ["tables_from_pdf", "mark_from_pdf_cell", "citation_markers"]

# Glyph artifacts seen when ✓/✗ come from symbol fonts and lose their unicode mapping.
_PDF_YES = {"", "", "", "✓", "✔", "√"}
_PDF_NO = {"", "", "", "✗", "✘", "✕", "×"}
_PDF_PARTIAL = {"▲", "◐", "◑", "半"}
_ALL_MARK_CHARS = _PDF_YES | _PDF_NO | _PDF_PARTIAL

# In-text citation markers inside a row label.
# A caption line must never be mistaken for column headers.
_CAPTION_RE = re.compile(r"\s*(?:Table|TABLE|Tab\.)\s*[IVXLC0-9]+\s*[.:]", re.IGNORECASE)

_NUM_MARKER_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_AY_MARKER_RE = re.compile(
    r"\(?\b([A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-z'’-]+)?)"
    r",?\s*(?:et\s+al\.?,?\s*)?\(?((?:19|20)\d{2}[a-z]?)\)?"
)


def mark_from_pdf_cell(raw: str) -> str:
    """Classify a PDF cell string into a :class:`CellMark` value.

    Handles the symbol-font artifacts above before falling back to words/values.
    """
    s = (raw or "").strip()
    if not s:
        return CellMark.EMPTY.value
    chars = set(s)
    if chars & _PDF_YES:
        return CellMark.YES.value
    if chars & _PDF_NO:
        return CellMark.NO.value
    if chars & _PDF_PARTIAL:
        return CellMark.PARTIAL.value
    low = re.sub(r"\s+", " ", s).strip().lower()
    # "N/A" is an abstention, not the paper asserting the method lacks the property.
    if low in {"-", "–", "—", "", "n/a", "na", "n.a.", "--", "?"}:
        return CellMark.EMPTY.value
    if low in {"yes", "y", "true", "full"}:
        return CellMark.YES.value
    if low in {"no", "n", "false", "none"}:
        return CellMark.NO.value
    if low in {"partial", "partially", "limited", "partly", "medium"}:
        return CellMark.PARTIAL.value
    return CellMark.VALUE.value


def citation_markers(label: str) -> list[str]:
    """Citation markers inside a PDF row label.

    ``"MetaAug [21]"`` -> ``["21"]``; ``"MetaAug (Rajendran et al., 2020)"`` ->
    ``["Rajendran 2020"]``. These are *markers*, not resolved keys — the caller maps
    them onto the paper's reference list.
    """
    out: list[str] = []
    for m in _NUM_MARKER_RE.finditer(label or ""):
        for part in re.split(r"[,–-]", m.group(1)):
            if part.strip().isdigit():
                out.append(part.strip())
    if not out:
        for m in _AY_MARKER_RE.finditer(label or ""):
            out.append(f"{m.group(1)} {m.group(2)}")
    return out


def _is_mark_token(text: str) -> bool:
    """True when a whole word token is nothing but mark glyphs (``"✓"``, ``"✗✗"``)."""
    t = (text or "").strip()
    return bool(t) and all(ch in _ALL_MARK_CHARS for ch in t)


def _cluster_1d(values: list[float], gap: float) -> list[float]:
    """Group sorted 1-D values whose neighbours are within ``gap``; return the centres."""
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for v in ordered[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _densest_block(marks: list[tuple], page_width: float) -> list[tuple]:
    """The largest spatially-contiguous group of marks on the page.

    Marks are grouped by vertical gaps (separate tables) and then by horizontal gaps
    wider than a page column (side-by-side tables in a two-column layout); the biggest
    resulting group wins. Prevents two unrelated tables from being spliced into one grid.
    """
    if not marks:
        return marks

    def split(items: list[tuple], key, gap: float) -> list[list[tuple]]:
        ordered = sorted(items, key=key)
        groups: list[list[tuple]] = [[ordered[0]]]
        for w in ordered[1:]:
            if key(w) - key(groups[-1][-1]) > gap:
                groups.append([])
            groups[-1].append(w)
        return groups

    mid = page_width / 2
    best: list[tuple] = []
    # Vertical gaps first: a gap much larger than a text line means a different table.
    for band in split(marks, lambda w: (w[1] + w[3]) / 2, 60.0):
        # Then horizontal — but ONLY at the page gutter. A wide `table*` spans the page
        # and its own inter-column pitch can exceed any fixed width threshold, so
        # splitting on gap size alone shreds it into single columns and the table is lost.
        # A genuine two-column layout separates at the middle of the page.
        blocks = [band]
        ordered = sorted(band, key=lambda w: (w[0] + w[2]) / 2)
        centres = [(w[0] + w[2]) / 2 for w in ordered]
        gaps = [b - a for a, b in zip(centres, centres[1:], strict=False) if b - a > 1]
        # The gutter must be far wider than this table's OWN column pitch. A wide table's
        # internal gap can straddle the page centre too, so "crosses the middle" alone
        # shreds it; comparing against the median pitch tells the two cases apart.
        pitch = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
        threshold = max(page_width * 0.18, pitch * 2.5)
        for a, b in zip(ordered, ordered[1:], strict=False):
            ax, bx = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
            if bx - ax > threshold and ax < mid < bx:
                blocks = [
                    [w for w in band if (w[0] + w[2]) / 2 <= ax],
                    [w for w in band if (w[0] + w[2]) / 2 >= bx],
                ]
                break
        for block in blocks:
            if len(block) > len(best):
                best = block
    return best if len(best) >= 4 else marks


def _grid_from_marks(page, *, x_tol: float = 12.0, y_tol: float = 6.0) -> list[list[str]] | None:
    """Reconstruct a capability matrix using the ✓/✗ glyphs as the grid skeleton.

    Args:
        page: a PyMuPDF ``Page``.
        x_tol: half-width of a column: how far a glyph may sit from its anchor.
        y_tol: half-height of a row.

    Returns:
        ``[[label, mark, mark, …], …]`` with a header row first, or ``None`` when the
        page has too few marks to be a comparison table.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word_no)
    marks = [w for w in words if _is_mark_token(w[4])]
    if len(marks) < 4:
        return None

    # A two-column page (every ACL/NAACL paper) can hold two unrelated tables that share
    # y-bands. Clustering all marks together would splice them into one grid and attribute
    # the right-hand table's cells to the left-hand table's cited rows — inventing
    # high-severity accusations the paper never made. Isolate the densest mark block and
    # parse only that: one capability matrix per page is the realistic case.
    marks = _densest_block(marks, page.rect.width)
    if len(marks) < 4:
        return None

    col_x = _cluster_1d([(w[0] + w[2]) / 2 for w in marks], x_tol)
    row_y = _cluster_1d([(w[1] + w[3]) / 2 for w in marks], y_tol)
    if len(col_x) < 2 or len(row_y) < 2:
        return None
    # Require a real matrix, without punishing sparsity: blank cells ("not applicable")
    # are normal, so demand that at least two ROWS carry two or more marks rather than a
    # total count — the old `len(marks) < len(col_x)+len(row_y)` dropped a valid 3x3 with
    # five marks while still admitting a stray "3 × 3" from prose.
    dense_rows = sum(
        1
        for ry in row_y
        if sum(1 for w in marks if abs((w[1] + w[3]) / 2 - ry) <= y_tol) >= 2
    )
    if dense_rows < 2:
        return None

    mark_left = min(w[0] for w in marks)
    mark_right = max(w[2] for w in marks)
    top_y = min(w[1] for w in marks)

    # Row labels sit beside the mark block — usually to its left, but some tables put the
    # marks first and the method names to the right. Pick whichever side carries words.
    def band(ry: float, *, left: bool) -> str:
        picked = [
            w for w in words
            if abs((w[1] + w[3]) / 2 - ry) <= y_tol
            and (w[2] < mark_left - 3 if left else w[0] > mark_right + 3)
            and not _is_mark_token(w[4])
        ]
        picked.sort(key=lambda w: w[0])
        return " ".join(w[4] for w in picked).strip()

    left_labels = [band(ry, left=True) for ry in row_y]
    right_labels = [band(ry, left=False) for ry in row_y]
    use_left = sum(bool(re.search(r"[A-Za-z]", s)) for s in left_labels) >= sum(
        bool(re.search(r"[A-Za-z]", s)) for s in right_labels
    )
    labels = left_labels if use_left else right_labels

    # Headers: the text directly above the mark block. Each word goes to its NEAREST
    # column anchor rather than a fixed window — header phrases are wider than the glyph
    # they label ("Threshold Free" over a single ✓), so a fixed window clips them.
    # Only the text LINES immediately above the marks, and never a caption line: a wide
    # window pulls "Table 1: Comparison of …" into the headers, which then match nothing
    # in the body and get reported as columns the paper never defined.
    band = [
        w for w in words
        if w[3] <= top_y - 1 and top_y - w[3] <= 60 and not _is_mark_token(w[4])
    ]
    # Group into lines FIRST and test each unclipped — "Table 1:" sits left of the mark
    # block, so clipping before the caption test removes the very words that identify it
    # and the caption prose is then bucketed into the column headers (IEEE/ACM/ACL put
    # captions above tables, so this is the common case, not the exotic one).
    lines: dict[int, list[tuple]] = {}
    for w in band:
        lines.setdefault(round(w[1] / 3), []).append(w)
    header_words = []
    for _key, line in sorted(lines.items(), reverse=True)[:3]:
        joined = " ".join(w[4] for w in sorted(line, key=lambda w: w[0]))
        if _CAPTION_RE.match(joined) or _CAPTION_RE.search(joined):
            continue
        header_words.extend(
            w for w in line if mark_left - 90 <= (w[0] + w[2]) / 2 <= mark_right + 90
        )
        if len(header_words) >= len(col_x):
            break
    buckets: list[list[tuple]] = [[] for _ in col_x]
    for w in header_words:
        wx = (w[0] + w[2]) / 2
        nearest = min(range(len(col_x)), key=lambda i: abs(col_x[i] - wx))
        if abs(col_x[nearest] - wx) <= max(x_tol * 3.5, _median_gap(col_x) / 2):
            buckets[nearest].append(w)
    headers = []
    for bucket in buckets:
        bucket.sort(key=lambda w: (round(w[1]), w[0]))
        headers.append(" ".join(w[4] for w in bucket).strip())

    grid: list[list[str]] = [["Method", *headers]]
    for ry, label in zip(row_y, labels, strict=False):
        row = [label]
        for cx in col_x:
            hit = [
                w[4] for w in marks
                if abs((w[1] + w[3]) / 2 - ry) <= y_tol and abs((w[0] + w[2]) / 2 - cx) <= x_tol
            ]
            row.append(hit[0] if hit else "")
        grid.append(row)
    # Drop rows that ended up with no label AND no marks.
    body = [r for r in grid[1:] if r[0].strip() or any(c.strip() for c in r[1:])]
    return [grid[0], *body] if body else None


def _median_gap(centres: list[float]) -> float:
    """Median spacing between adjacent column anchors (0 for fewer than two)."""
    if len(centres) < 2:
        return 0.0
    gaps = sorted(b - a for a, b in zip(centres, centres[1:], strict=False))
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def grid_issues(grid: list[list[str]]) -> list[str]:
    """Signs that a geometric parse is untrustworthy — the trigger for OCR escalation.

    Text extraction degrades in ways that are visible in the parsed grid itself: headers
    that came out blank, row labels that merged several methods into one line, or rows
    whose mark count does not match the column count. Rendering the region and reading it
    with a vision model recovers exactly these cases, so they are reported rather than
    silently returned as if they were clean.

    Args:
        grid: ``[[label, mark, …], …]`` with the header row first.

    Returns:
        Human-readable issue strings; empty when the grid looks sound.
    """
    if not grid or len(grid) < 2:
        return ["no rows parsed"]
    issues: list[str] = []
    header, body = grid[0], grid[1:]
    ncols = len(header) - 1

    blank_headers = sum(1 for h in header[1:] if not h.strip())
    if blank_headers:
        issues.append(f"{blank_headers}/{ncols} column header(s) came out empty")

    blank_labels = sum(1 for r in body if not r[0].strip())
    if blank_labels:
        issues.append(f"{blank_labels}/{len(body)} row label(s) came out empty")

    # A label naming several methods usually means two printed rows were merged.
    crowded = sum(1 for r in body if len(re.findall(r"[A-Za-z]{3,}\s*\(", r[0])) >= 2)
    if crowded:
        issues.append(f"{crowded} row label(s) look like several merged rows")

    ragged = sum(1 for r in body if sum(1 for c in r[1:] if c.strip()) not in (0, ncols))
    if ragged:
        issues.append(f"{ragged}/{len(body)} row(s) have a partial mark count")
    return issues


def _clean_ocr_grid(grid: object) -> list[list[str]] | None:
    """Normalize whatever an OCR/vision callback returned into a rectangular grid.

    Vision models emit ragged rows, ``None`` cells and trailing empty rows; the geometric
    path never does, so the rest of the module assumes rectangularity.
    """
    if not isinstance(grid, list) or len(grid) < 2:
        return None
    rows: list[list[str]] = []
    for row in grid:
        if not isinstance(row, list | tuple):
            continue
        cells = [("" if c is None else str(c)).strip() for c in row]
        if any(cells):
            rows.append(cells)
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


def _render_region(page, *, dpi: int = 220) -> bytes:
    """Render the page to PNG bytes for a vision/OCR pass."""
    import fitz  # noqa: PLC0415 — lazy, optional dependency

    zoom = dpi / 72.0
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")


def _caption_near(page, bbox, *, above_pt: float = 60.0, below_pt: float = 40.0) -> str:
    """The 'Table N: …' caption immediately above or below a table's bbox."""
    import fitz  # noqa: PLC0415 — lazy, optional dependency

    x0, y0, x1, y1 = bbox
    for rect in (
        fitz.Rect(x0 - 20, max(0, y0 - above_pt), x1 + 20, y0),
        fitz.Rect(x0 - 20, y1, x1 + 20, y1 + below_pt),
    ):
        text = " ".join((page.get_textbox(rect) or "").split())
        m = re.search(r"((?:Table|TABLE|Tab\.)\s*[IVXLC0-9]+[.:]?\s.*)", text)
        if m:
            return m.group(1).strip()
    return ""


def tables_from_pdf(
    pdf_path: str | Path,
    *,
    paper_id: str = "",
    method_names: set[str] | None = None,
    require_comparison: bool = True,
    max_pages: int = 12,
    ocr: Callable[[bytes, str], list[list[str]] | None] | None = None,
) -> list[ComparisonTable]:
    """Extract comparison tables from a PDF.

    Order of attack: mark-anchored reconstruction (the reliable path for borderless
    capability matrices — see the module docstring), then PyMuPDF's text strategy for
    word/number matrices, and finally — when a parse looks damaged — an optional OCR /
    vision pass over the rendered page.

    Args:
        pdf_path: the PDF to read.
        paper_id: stamped onto each table.
        method_names: the citing paper's own method name(s), to flag the "ours" row.
        require_comparison: keep only capability matrices (drop results tables).
        max_pages: only scan the first N pages — comparison tables sit early, and this
            bounds the cost on long appendices.
        ocr: ``(png_bytes, caption) -> grid | None`` escalation used when
            :func:`grid_issues` reports problems (clipped headers, merged rows, ragged
            mark counts) or when no grid could be parsed at all. Its grid wins when it
            is cleaner than the geometric one.

    Returns:
        The tables found. ``[]`` when PyMuPDF is unavailable or nothing matches; never
        raises on a malformed PDF.
    """
    method_names = method_names or set()
    try:
        import fitz  # noqa: PLC0415 — lazy, optional dependency
    except ImportError:
        return []

    out: list[ComparisonTable] = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:  # noqa: BLE001 — unreadable PDF: no tables, no crash
        return []

    with doc:
        for pno, page in enumerate(doc):
            if pno >= max_pages:
                break

            # 1) Mark-anchored: the ✓/✗ glyphs ARE the grid.
            try:
                grid = _grid_from_marks(page)
            except Exception:  # noqa: BLE001 — geometry failure is not fatal
                grid = None
            if grid and (not require_comparison or _looks_like_comparison_pdf(grid)):
                caption = _caption_on_page(page)
                issues = grid_issues(grid)
                source = "pdf"
                if issues and ocr is not None:
                    # Text extraction looks damaged — read the rendered page instead.
                    # A vision model happily returns ragged/None rows, so scoring the
                    # replacement must be inside the guard too.
                    try:
                        better = _clean_ocr_grid(ocr(_render_region(page), caption))
                        if better and len(grid_issues(better)) < len(issues):
                            grid, issues, source = better, grid_issues(better), "pdf+ocr"
                    except Exception:  # noqa: BLE001 — OCR is best-effort
                        pass
                table = _build_table(
                    grid,
                    table_id=f"{paper_id or 'paper'}-p{pno + 1}-marks",
                    paper_id=paper_id,
                    caption=caption,
                    method_names=method_names,
                    source=source,
                    issues=issues,
                )
                if table is not None:
                    out.append(table)
                    continue  # one capability matrix per page is the realistic case

            # 2) Fallback: PyMuPDF's text strategy, for word/number matrices.
            try:
                found = page.find_tables(strategy="text")
            except Exception:  # noqa: BLE001 — finder failure on one page is not fatal
                continue
            for tno, tbl in enumerate(getattr(found, "tables", []) or []):
                try:
                    grid = [[(c or "").strip() for c in row] for row in tbl.extract()]
                except Exception:  # noqa: BLE001
                    continue
                grid = [r for r in grid if any(c for c in r)]
                if len(grid) < 2 or (require_comparison and not _looks_like_comparison_pdf(grid)):
                    continue
                table = _build_table(
                    grid,
                    table_id=f"{paper_id or 'paper'}-p{pno + 1}-table-{tno + 1}",
                    paper_id=paper_id,
                    caption=_caption_near(page, tbl.bbox),
                    method_names=method_names,
                )
                if table is not None:
                    out.append(table)
    return out


def _build_table(
    grid: list[list[str]],
    *,
    table_id: str,
    paper_id: str,
    caption: str,
    method_names: set[str],
    source: str = "pdf",
    issues: list[str] | None = None,
) -> ComparisonTable | None:
    """Turn a parsed grid (header row first) into a :class:`ComparisonTable`.

    ``issues`` (from :func:`grid_issues`) are carried on the table as legend notes so the
    report can say the grid was read imperfectly instead of implying it is exact.
    """
    if len(grid) < 2:
        return None
    header = grid[0]
    ncols = max(len(r) for r in grid)
    dims = [
        Dimension(
            col_index=c,
            header=(header[c] if c < len(header) else "").strip(),
            kind=_dimension_kind_pdf(grid, c),
        )
        for c in range(1, ncols)
    ]
    rows: list[TableRow] = []
    cells: list[TableCell] = []
    for r, row in enumerate(grid[1:]):
        label = (row[0] if row else "").strip()
        if not label:
            continue
        markers = citation_markers(label)
        rows.append(
            TableRow(
                row_index=r,
                label=label,
                cite_keys=markers,
                # strict: in a PDF a missing marker means "we failed to parse it",
                # not "this is the authors' own method".
                is_self=_is_self_row(label, markers, method_names=method_names, strict=True),
            )
        )
        for c in range(1, ncols):
            raw = row[c] if c < len(row) else ""
            cells.append(
                TableCell(
                    cell_id=f"{table_id}:r{r}:c{c}",
                    row_index=r,
                    col_index=c,
                    raw=raw,
                    mark=mark_from_pdf_cell(raw),
                )
            )
    if not rows:
        return None
    return ComparisonTable(
        table_id=table_id,
        paper_id=paper_id,
        caption=caption,
        source=source,
        warnings=list(issues or []),
        dimensions=dims,
        rows=rows,
        cells=cells,
    )


def _caption_on_page(page) -> str:
    """The first 'Table N: …' caption on the page (the mark block has no bbox of its own)."""
    text = " ".join((page.get_text() or "").split())
    m = re.search(r"((?:Table|TABLE|Tab\.)\s*[IVXLC0-9]+\s*[.:]\s+.{0,300})", text)
    return m.group(1).strip() if m else ""


def _looks_like_comparison_pdf(grid: list[list[str]], *, min_rows: int = 2) -> bool:
    """Same intent as the LaTeX heuristic, using the PDF mark vocabulary."""
    if len(grid) < min_rows + 1:
        return False
    marks = total = 0
    for row in grid[1:]:
        for cell in row[1:]:
            m = mark_from_pdf_cell(cell)
            if m == CellMark.EMPTY.value:
                continue
            total += 1
            if m in (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value):
                marks += 1
    return total >= 3 and marks / total >= 0.6


def _dimension_kind_pdf(grid: list[list[str]], col: int) -> str:
    """Column value space, using PDF mark classification."""
    seen = {mark_from_pdf_cell(row[col]) for row in grid[1:] if col < len(row)}
    seen.discard(CellMark.EMPTY.value)
    if not seen or seen <= {CellMark.YES.value, CellMark.NO.value}:
        return "binary"
    if seen <= {CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value}:
        return "graded"
    if CellMark.VALUE.value in seen:
        nums = sum(
            1
            for row in grid[1:]
            if col < len(row) and re.fullmatch(r"[\d.,]+\s*[KMBG%]?", (row[col] or "x").strip())
        )
        return "numeric" if nums >= 2 else "categorical"
    return "categorical"
