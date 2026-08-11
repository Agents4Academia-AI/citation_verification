"""
tables/latex_grid.py — recover comparison-table GRIDS from LaTeX source.

LaTeX is the high-fidelity path: the ✓/✗ marks are macros (``\\ding{51}``,
``\\cmark``, ``$\\times$``) and each row names its reference with a real ``\\cite`` key,
so a row binds to a bibliography entry exactly rather than by fuzzy text matching.

What makes this non-trivial in practice (all seen in the wild)::

    \\begin{wraptable}[6]{r}{8.5cm}                 % not always `table`
      \\caption{\\small Summary of existing ...}     % nested braces / font macros
      \\label{tab:related_work}
      \\resizebox{74mm}{11.2mm}{                    % arbitrary wrappers around tabular
        \\begin{tabular}{c|c|c|c}
          \\toprule
          \\multicolumn{1}{l}{\\multirow{2}{*}{\\textbf{Method}}} & ...  \\\\
          \\multicolumn{1}{l}{\\multirow{2}{*}{}} & ...                \\\\  % spacer row
          \\midrule
          \\multicolumn{1}{l}{\\multirow{1}{*}{MetaAug~\\cite{rajendran2020meta}}} &
          \\multicolumn{1}{c}{\\multirow{1}{*}{\\ding{51}}} & ...       \\\\
        \\end{tabular}}
    \\end{wraptable}

so every cell must be unwrapped through ``\\multicolumn``/``\\multirow``/``\\textbf``
before its mark can be read, ``\\multicolumn{n}{..}{..}`` with ``n > 1`` must expand so
column indices stay aligned, and all-empty spacer rows must be dropped.

Pure stdlib and offline: this module only parses text.
"""

from __future__ import annotations

import re

from .model import CellMark, ComparisonTable, Dimension, DimensionKind, TableCell, TableRow

__all__ = [
    "tables_from_latex",
    "parse_tabular",
    "normalize_mark",
    "strip_tex",
    "looks_like_comparison_table",
]

# Float environments a comparison table can live in (mirrors extract/latex.py).
_FLOAT_NAMES = (
    "table", "table*", "wraptable", "sidewaystable", "threeparttable", "floatrow",
)
# The inner grid environments.
_GRID_NAMES = ("tabular", "tabular*", "tabularx", "tabulary", "longtable", "array")

# Rule/spacing commands that carry no cell content.
_RULE_RE = re.compile(
    r"\\(?:toprule|midrule|bottomrule|hline|cline|cmidrule|specialrule|addlinespace|morecmidrules)"
    r"(?:\s*\[[^\]]*\])?(?:\s*\([^)]*\))?(?:\s*\{[^}]*\})?"
)
# Row separator: \\ (optionally starred / with spacing) or \tabularnewline.
_ROWSEP_RE = re.compile(r"\\\\(?:\s*\*)?(?:\s*\[[^\]]*\])?|\\tabularnewline")
# Every \cite-family spelling: natbib (starred forms included), biblatex and apacite.
# Missing one makes a row look uncited, which the "ours" heuristic then reads as the
# authors' own method — silently excusing a competitor's row from verification.
_CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|citeyearpar|Citep|Citet"
    r"|parencite|Parencite|autocite|Autocite|textcite|Textcite|footcite|smartcite"
    r"|shortcite|shortcites|citeA|citeNP|fullcite)\*?\s*"
    r"(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}"
)

# Marks. Order matters: check the explicit macros before falling back to words.
# pifont: 51/52 are the check marks (✓✔); 53-56 are the crosses (✕✖✗✘).
# SYMBOLS ONLY. Word forms ("Yes"/"No"/"Full") are handled by the exact-match branch in
# normalize_mark, because a substring search on a text cell misreads ordinary content:
# "No-Reference" -> ✗, "Full fine-tuning" -> ✓, "limited-memory BFGS" -> ▲. Those turn a
# categorical column into fabricated ✓/✗ claims and then into false accusations.
_YES_PAT = re.compile(
    r"\\ding\s*\{\s*5[12]\s*\}|\\(?:checkmark|cmark|greencheck|CheckmarkBold|Checkmark)\b"
    r"|\\usym\s*\{\s*2713\s*\}|[✓✔🗸]",
    re.IGNORECASE,
)
_NO_PAT = re.compile(
    r"\\ding\s*\{\s*5[3-6]\s*\}|\\(?:xmark|redcross|XSolidBrush|ding55)\b"
    r"|\\usym\s*\{\s*271[78]\s*\}|\$?\\times\$?|[✗✘×✕✖]",
    re.IGNORECASE,
)
_PARTIAL_PAT = re.compile(
    r"\\(?:halfcirc|LEFTcircle|RIGHTcircle|halfstar|Circle)\b|[▲◐◑⯨◒◓]",
    re.IGNORECASE,
)


# ───────────────────────────────────────────────────────────────
# Small TeX utilities
# ───────────────────────────────────────────────────────────────
def _strip_comments(tex: str) -> str:
    """Drop unescaped ``%`` line comments (a commented-out row must not become data)."""
    return re.sub(r"(?<!\\)%[^\n]*", "", tex)


def _balanced(text: str, open_idx: int) -> tuple[str, int]:
    """Content of the brace group starting at ``open_idx``, and the index past its ``}``.

    Args:
        text: the source.
        open_idx: index of the opening ``{``.

    Returns:
        ``(content, end)``; ``("", open_idx + 1)`` when the group never closes.
    """
    if open_idx >= len(text) or text[open_idx] != "{":
        return "", open_idx + 1
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i + 1
    return "", open_idx + 1


def _macro_arg(tex: str, macro: str, *, occurrence: int = 0) -> str:
    """The braced argument of ``\\macro{...}`` (balanced), or ``""``.

    Skips optional ``[...]`` arguments so ``\\caption[short]{long}`` yields the long form.
    """
    seen = -1
    for m in re.finditer(rf"\\{macro}\s*", tex):
        i = m.end()
        while i < len(tex) and tex[i] == "[":  # optional args
            j = tex.find("]", i)
            if j < 0:
                break
            i = j + 1
            while i < len(tex) and tex[i].isspace():
                i += 1
        if i < len(tex) and tex[i] == "{":
            seen += 1
            if seen == occurrence:
                return _balanced(tex, i)[0]
    return ""


def strip_tex(s: str) -> str:
    """Reduce a TeX fragment to readable plain text (for headers, labels, captions)."""
    s = _strip_comments(s)
    s = _CITE_RE.sub(" ", s)
    s = re.sub(r"\\(?:label|ref|cref|vspace|hspace|resizebox|footnote)\s*\{[^}]*\}", " ", s)
    # Unwrap common one-argument formatting macros, innermost first.
    for _ in range(6):
        new = re.sub(
            r"\\(?:textbf|textit|texttt|textsc|emph|mathrm|mathbf|text|bm|underline|"
            r"scalebox|makecell|thead|tabincell)\s*(?:\{[^{}]*\})?\s*\{([^{}]*)\}",
            r"\1",
            s,
        )
        if new == s:
            break
        s = new
    s = re.sub(r"\\(?:small|footnotesize|scriptsize|tiny|large|Large|normalsize|centering|bf|it)\b", " ", s)
    s = re.sub(r"\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?", " ", s)  # any remaining macro
    s = s.replace("~", " ").replace("\\&", "&")
    s = re.sub(r"[{}$]", "", s)
    return re.sub(r"\s+", " ", s).strip(" .,;:")


def normalize_mark(raw: str) -> str:
    """Classify a raw cell into a :class:`CellMark` value.

    Macros win over words: ``\\ding{55}`` is ✗ even though the stripped text is empty.
    A cell that carries real content (``"gradient-based"``, ``"3.2M"``) is ``value``.
    """
    cell = _strip_comments(raw or "").strip()
    if not cell:
        return CellMark.EMPTY.value
    # Symbol/macro evidence, before text stripping loses it.
    if _PARTIAL_PAT.search(cell):
        return CellMark.PARTIAL.value
    yes, no = bool(_YES_PAT.search(cell)), bool(_NO_PAT.search(cell))
    if yes and not no:
        return CellMark.YES.value
    if no and not yes:
        return CellMark.NO.value

    text = strip_tex(cell)
    if not text:
        return CellMark.EMPTY.value
    low = text.lower()
    # "N/A" is an abstention ("not applicable / not reported"), NOT the paper asserting
    # the method lacks the property — treating it as ✗ manufactures a false accusation.
    if low in {"-", "–", "—", "n/a", "na", "n.a.", "--", "?"}:
        return CellMark.EMPTY.value
    # Whole-cell word forms only — never a substring of longer content.
    if re.fullmatch(r"(partial(ly)?|limited|partly|some|medium|mid)", low):
        return CellMark.PARTIAL.value
    if re.fullmatch(r"(yes|y|true|full|high|supported)", low):
        return CellMark.YES.value
    if re.fullmatch(r"(no|n|false|low|none|unsupported)", low):
        return CellMark.NO.value
    if "▲" in text or "◐" in text:
        return CellMark.PARTIAL.value
    return CellMark.VALUE.value


# ───────────────────────────────────────────────────────────────
# Grid parsing
# ───────────────────────────────────────────────────────────────
def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` only at brace depth 0 and outside math mode."""
    out, buf, depth, math = [], [], 0, False
    i, n, slen = 0, len(text), len(sep)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:  # escaped char: copy verbatim
            buf.append(text[i : i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "$":
            math = not math
        if depth == 0 and not math and text.startswith(sep, i):
            out.append("".join(buf))
            buf = []
            i += slen
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return out


def _split_rows(body: str) -> list[str]:
    """Split a tabular body into row sources on top-level ``\\\\`` / ``\\tabularnewline``."""
    marks = [(m.start(), m.end()) for m in _ROWSEP_RE.finditer(body)]
    if not marks:
        return [body]
    # Track brace depth / math mode so a `\\` nested in a group (e.g. makecell) never splits.
    depth_at: list[tuple[int, bool]] = []
    d, mm = 0, False
    for i, ch in enumerate(body):
        if ch == "{" and (i == 0 or body[i - 1] != "\\"):
            d += 1
        elif ch == "}" and (i == 0 or body[i - 1] != "\\"):
            d = max(0, d - 1)
        elif ch == "$" and (i == 0 or body[i - 1] != "\\"):
            mm = not mm
        depth_at.append((d, mm))
    rows, prev = [], 0
    for s, e in marks:
        depth, math = depth_at[s] if s < len(depth_at) else (0, False)
        if depth == 0 and not math:
            rows.append(body[prev:s])
            prev = e
    rows.append(body[prev:])
    return rows


def _unwrap_cell(cell: str) -> tuple[str, int]:
    """Unwrap a cell to its content and the number of columns it spans.

    ``\\multicolumn{3}{l|}{X}`` -> ``("X", 3)``; ``\\multirow{2}{*}{X}`` -> ``("X", 1)``.
    Applied repeatedly because these nest (``\\multicolumn{1}{c}{\\multirow{1}{*}{X}}``).
    """
    span = 1
    text = cell.strip()
    for _ in range(6):
        m = re.match(r"^\s*\\multicolumn\s*\{\s*(\d+)\s*\}", text)
        if m:
            span = max(span, int(m.group(1)))
            i = text.find("{", m.end())          # the column-spec arg
            if i < 0:
                break
            _, j = _balanced(text, i)
            k = text.find("{", j)                 # the content arg
            if k < 0:
                break
            text = _balanced(text, k)[0]
            continue
        m = re.match(r"^\s*\\multirow\s*\{[^}]*\}\s*\{[^}]*\}", text)
        if m:
            i = text.find("{", m.end())
            if i < 0:
                break
            text = _balanced(text, i)[0]
            continue
        break
    return text.strip(), span


def _parse_row_cells(row_src: str) -> list[str]:
    """Cells of one row, with ``\\multicolumn`` spans expanded so indices stay aligned."""
    raw_cells = _split_top_level(_RULE_RE.sub(" ", row_src), "&")
    out: list[str] = []
    for c in raw_cells:
        content, span = _unwrap_cell(c)
        out.append(content)
        out.extend([""] * (span - 1))
    return out


def _strip_colspec(body: str) -> str:
    """Drop the column specification that follows ``\\begin{tabular}``.

    The environment body starts with ``{l c c c c}`` (preceded by ``[t]`` and, for
    ``tabular*``/``tabularx``, a width argument). Left in place it becomes the first cell
    of the first row — usually harmless because it merges into the header, but when the
    table opens with a blank spacer row the spec becomes the entire header row and every
    column then looks like one the paper never defined.
    """
    i, n = 0, len(body)
    while i < n and body[i].isspace():
        i += 1
    if i < n and body[i] == "[":  # positional arg
        j = body.find("]", i)
        if j >= 0:
            i = j + 1
            while i < n and body[i].isspace():
                i += 1
    seen = 0
    while i < n and body[i] == "{" and seen < 2:
        content, end = _balanced(body, i)
        # A column spec is only alignment/rule characters (plus p{}/m{} widths).
        if seen == 0 and not re.fullmatch(r"[\s|lcrXYZ@p{}.\d>*<!:;()cm-]*", content or ""):
            break
        i = end
        seen += 1
        while i < n and body[i].isspace():
            i += 1
        if seen == 1 and re.fullmatch(r"[\s|lcrXYZ@p{}.\d>*<!:;()cm-]*", content or ""):
            break  # that was the spec; a second group would be content
    return body[i:]


def parse_tabular(body: str) -> list[list[str]]:
    """Parse a tabular BODY (no ``\\begin``/``\\end``) into a rectangular-ish grid.

    Returns rows of raw (unwrapped, still TeX) cell strings. The leading column spec,
    rule-only rows and all-empty spacer rows are dropped — none carry assertions.
    """
    grid: list[list[str]] = []
    for row_src in _split_rows(_strip_comments(_strip_colspec(body))):
        if not row_src.strip() or not _RULE_RE.sub("", row_src).strip():
            continue
        cells = _parse_row_cells(row_src)
        if not any(strip_tex(c) or _YES_PAT.search(c) or _NO_PAT.search(c) for c in cells):
            continue  # spacer row: every cell empty
        grid.append(cells)
    return grid


def _find_envs(text: str, names: tuple[str, ...]) -> list[tuple[int, int, str]]:
    """Spans ``(start, end, body)`` of ``\\begin{name}...\\end{name}``, honoring nesting."""
    alt = "|".join(re.escape(n) for n in names)
    open_re = re.compile(rf"\\begin\{{({alt})\}}")
    out: list[tuple[int, int, str]] = []
    for m in open_re.finditer(text):
        name = m.group(1)
        close_re = re.compile(rf"\\end\{{{re.escape(name)}\}}")
        open_again = re.compile(rf"\\begin\{{{re.escape(name)}\}}")
        depth, i = 1, m.end()
        while i < len(text) and depth:
            c, o = close_re.search(text, i), open_again.search(text, i)
            if not c:
                break
            if o and o.start() < c.start():
                depth += 1
                i = o.end()
                continue
            depth -= 1
            i = c.end()
            if depth == 0:
                out.append((m.start(), i, text[m.end() : c.start()]))
    return out


def _cite_keys(cell: str) -> list[str]:
    """All ``\\cite``-family keys inside a cell, in order."""
    keys: list[str] = []
    for m in _CITE_RE.finditer(cell):
        keys.extend(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def looks_like_comparison_table(grid: list[list[str]], *, min_rows: int = 2, min_cols: int = 2) -> bool:
    """Heuristic: is this a capability matrix rather than a results table?

    Counts columns that are *predominantly* ✓/✗/▲ rather than the share of marks over the
    whole table. Real comparison tables routinely mix binary properties with descriptive
    ones — Wanda's Table 1 is ``Weight Update ✗/✓ | Calibration Data ✗/✓ | Pruning Metric
    <formula> | Complexity O(1)`` — and a whole-table ratio drops those at 50%, discarding
    exactly the tables this subsystem exists to audit. Results tables still fail, because
    every one of their columns is numeric.
    """
    if len(grid) < min_rows + 1:
        return False
    ncols = max((len(r) for r in grid), default=0)
    marky_cols = 0
    for c in range(1, ncols):
        vals = [normalize_mark(r[c]) for r in grid[1:] if c < len(r)]
        vals = [v for v in vals if v != CellMark.EMPTY.value]
        if len(vals) < 2:
            continue
        marks = [
            v for v in vals
            if v in (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)
        ]
        if len(marks) / len(vals) >= 0.6:
            marky_cols += 1
    return marky_cols >= min_cols


def _dimension_kind(grid: list[list[str]], col: int, *, body_from: int = 1) -> str:
    """Infer a column's value space from the marks actually used in it.

    ``body_from`` is where the DATA rows start; with a two-level header it is 2. Passing
    1 there lets the second header row's words count as values, and since
    ``strip_tex(r"\\ding{51}")`` is ``"51"`` the numeric test then matches every mark —
    the whole table is classified numeric and silently dropped as "descriptive".
    """
    seen = {normalize_mark(row[col]) for row in grid[body_from:] if col < len(row)}
    seen.discard(CellMark.EMPTY.value)
    if not seen:
        return DimensionKind.BINARY.value
    if seen <= {CellMark.YES.value, CellMark.NO.value}:
        return DimensionKind.BINARY.value
    if seen <= {CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value}:
        return DimensionKind.GRADED.value
    if CellMark.VALUE.value in seen:
        nums = sum(
            1
            for row in grid[body_from:]
            if col < len(row) and re.fullmatch(r"[\d.,]+\s*[KMBG%]?", strip_tex(row[col]) or "x")
        )
        return DimensionKind.NUMERIC.value if nums >= 2 else DimensionKind.CATEGORICAL.value
    return DimensionKind.CATEGORICAL.value


def _first_mark_col(grid: list[list[str]]) -> int:
    """Index of the first column that holds ✓/✗ marks (everything left of it labels rows).

    Comparison tables are not always ``method | mark | mark``: a family column added with
    ``\\multirow`` pushes the method names into column 1, and assuming column 0 is the
    label then drops every continuation row (their column 0 is empty).
    """
    ncols = max((len(r) for r in grid), default=0)
    body = grid[1:]
    for c in range(ncols):
        vals = [normalize_mark(r[c]) for r in body if c < len(r)]
        marky = [v for v in vals if v in (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)]
        if vals and len(marky) >= max(1, len([v for v in vals if v != CellMark.EMPTY.value]) * 0.6):
            return max(1, c)
    return 1


def _row_label(cells: list[str], label_cols: int) -> tuple[str, list[str]]:
    """Pick a row's label and cite keys from its (possibly several) label columns.

    Prefers the cell that actually carries a citation — with a ``\\multirow`` family
    column the citation sits in column 1 while column 0 holds the family name (and is
    empty on continuation rows).
    """
    span = cells[: max(1, label_cols)]
    for cell in span:
        keys = _cite_keys(cell)
        if keys:
            return strip_tex(cell) or _macro_label(cell), keys
    for cell in span:
        if strip_tex(cell) or _macro_label(cell):
            return strip_tex(cell) or _macro_label(cell), []
    return "", []


def _macro_label(cell: str) -> str:
    """Fallback label for a row named only by a macro.

    Papers alias their own method (``\\newcommand{\\method}{Wanda}``) and write
    ``\\method`` in the table; stripping TeX leaves nothing and the row would be dropped —
    losing precisely the "ours" row the asymmetry check needs.
    """
    m = re.search(r"\\([A-Za-z@]{3,})\b", cell or "")
    if not m:
        return ""
    name = m.group(1)
    if name in {"textbf", "textit", "emph", "texttt", "multirow", "multicolumn", "gr", "wc"}:
        return ""
    return name


def _merge_header_rows(grid: list[list[str]], label_cols: int) -> tuple[list[str], int]:
    """Combine stacked header rows into one, returning ``(header, rows_consumed)``.

    Two-level headers (``\\multicolumn{2}{c}{Efficiency}`` over ``Fast & Cheap``) put the
    real column names on the SECOND row; taking only the first loses them and leaves
    blank headers that later read as "the paper never defined this column".
    """
    if len(grid) < 3:
        return grid[0], 1
    first, second = grid[0], grid[1]
    # Only merge when the FIRST row actually spans columns: `\multicolumn{2}{c}{Efficiency}`
    # expands to a filled cell followed by empty placeholders, and that gap is what the
    # second row fills in. Without this check an uncited textual data row ("Human
    # annotator | manual | offline | expert") satisfies the shape tests below and is
    # silently absorbed into the header — losing a row the table actually asserts about.
    spans_columns = any(not strip_tex(c) for c in first[label_cols:]) and any(
        strip_tex(c) for c in first[label_cols:]
    )
    if not spans_columns:
        return first, 1
    # The second row is still header material only if it names columns and asserts nothing.
    body_marks = [
        normalize_mark(c)
        for c in second[label_cols:]
        if normalize_mark(c) in (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)
    ]
    named = sum(1 for c in second[label_cols:] if strip_tex(c))
    if body_marks or _cite_keys(" ".join(second)) or named < 2:
        return first, 1
    ncols = max(len(first), len(second))
    merged = []
    for i in range(ncols):
        top = strip_tex(first[i]) if i < len(first) else ""
        sub = strip_tex(second[i]) if i < len(second) else ""
        merged.append(f"{top} — {sub}" if top and sub else (sub or top))
    return merged, 2


def _is_self_row(
    label: str, cite_keys: list[str], *, method_names: set[str], strict: bool = False
) -> bool:
    """True when a row is the authors' own method (the all-✓ row).

    Signals, in order: an explicit "(ours)" marker, or a label matching a known method
    name of the citing paper. "Carries no citation" is a third signal but only outside
    ``strict`` mode: it is reliable for LaTeX (every competitor row has a real ``\\cite``)
    and NOT for PDFs, where marker extraction routinely fails — treating an unparsed row
    as "ours" would silently drop real prior work from verification.

    Args:
        label: the row label as printed.
        cite_keys: citation keys/markers found in that label.
        method_names: the citing paper's own method name(s).
        strict: require positive evidence (used by the PDF path).
    """
    low = strip_tex(label).lower()
    if re.search(r"\b(ours?|our method|this (?:work|paper)|proposed)\b", low):
        return True
    # Whole-token match: a bare substring makes method_names={"SAM"} claim
    # "SAMPLE-Net~\cite{…}" as the authors' own work and skip a real competitor. Hyphenated
    # names ("Fast-DetectGPT", "GPT-4o") must still match as one unit, so test the whole
    # name against the label with word boundaries rather than only its split tokens.
    tokens = set(re.split(r"[^a-z0-9+.]+", low)) - {""}
    for m in method_names:
        if not m:
            continue
        name = m.lower()
        if name in tokens:
            return True
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", low):
            return True
    if strict:
        return False
    return not cite_keys and bool(low)


def tables_from_latex(
    text: str,
    *,
    paper_id: str = "",
    section: str | None = None,
    method_names: set[str] | None = None,
    require_comparison: bool = True,
) -> list[ComparisonTable]:
    """Extract comparison tables from one LaTeX source string.

    Args:
        text: LaTeX source (one ``.tex`` file's contents).
        paper_id: the citing paper's id, stamped onto each table.
        section: section name to record on the tables found here, if known.
        method_names: names of the citing paper's own method(s), used to mark the
            "ours" row when it is not labelled ``(ours)``.
        require_comparison: when True (default) keep only capability matrices —
            results/number tables are dropped.

    Returns:
        The tables found, in document order. Never raises on malformed TeX: a table
        that cannot be parsed is skipped.
    """
    method_names = method_names or set()
    text = _strip_comments(text)
    out: list[ComparisonTable] = []

    # One region per float. A float nested inside another (threeparttable inside table)
    # is skipped, or the same grid would be extracted — and every finding counted — twice.
    floats = _find_envs(text, _FLOAT_NAMES)
    outer = [
        (s, e, b)
        for s, e, b in floats
        if not any(fs < s and e <= fe for fs, fe, _ in floats)
    ]
    covered = [(s, e) for s, e, _ in outer]
    regions: list[tuple[str, bool]] = [(b, True) for _, _, b in outer]
    # A bare tabular outside any float: take only the tabular itself. Widening the window
    # to look for a caption would pick up the PRECEDING figure's caption and label, and a
    # figure caption becomes the gloss used to judge every cited paper.
    for s, e, _body in _find_envs(text, _GRID_NAMES):
        if not any(fs <= s and e <= fe for fs, fe in covered):
            regions.append((text[s:e], False))

    for idx, (region, has_caption) in enumerate(regions):
        grids = _find_envs(region, _GRID_NAMES)
        if not grids:
            continue
        # The innermost/first grid holds the data.
        body = grids[0][2]
        try:
            grid = parse_tabular(body)
        except Exception:  # noqa: BLE001 — malformed TeX: skip this table, keep going
            continue
        if len(grid) < 2:
            continue
        if require_comparison and not looks_like_comparison_table(grid):
            continue

        caption = strip_tex(_macro_arg(region, "caption")) if has_caption else ""
        label = (_macro_arg(region, "label").strip() or None) if has_caption else None
        table_id = label or f"{paper_id or 'paper'}-table-{idx + 1}"

        label_cols = _first_mark_col(grid)
        header, consumed = _merge_header_rows(grid, label_cols)
        body_rows = grid[consumed:]
        ncols = max(len(r) for r in grid)
        dims: list[Dimension] = []
        for c in range(label_cols, ncols):
            head = strip_tex(header[c]) if c < len(header) else ""
            dims.append(
                Dimension(
                    col_index=c,
                    header=head,
                    kind=_dimension_kind(grid, c, body_from=consumed),
                )
            )

        # "This row has no \cite, so it is the authors' own" only holds when the OTHER
        # rows do cite. Some tables cite nobody at all (Wanda's Table 1 names Magnitude
        # and SparseGPT with no keys) — there the signal would brand every competitor as
        # "ours" and skip the whole table.
        # Valid only when essentially every OTHER row cites: with a mixed table (say 2 of
        # 6 rows carrying keys) the four unparsed rows are competitors we failed to bind,
        # not four "ours" rows.
        parsed = [_row_label(row, label_cols) for row in body_rows]
        cites_somewhere = sum(1 for _lbl, keys in parsed if keys)
        strict_self = cites_somewhere < max(1, len(parsed) - 1)

        rows: list[TableRow] = []
        cells: list[TableCell] = []
        for r, (row, (rlabel, keys)) in enumerate(zip(body_rows, parsed, strict=False)):
            if not rlabel and not keys:
                continue
            rows.append(
                TableRow(
                    row_index=r,
                    label=rlabel,
                    cite_keys=keys,
                    is_self=_is_self_row(
                        rlabel, keys, method_names=method_names, strict=strict_self
                    ),
                )
            )
            for c in range(label_cols, ncols):
                raw = row[c] if c < len(row) else ""
                cells.append(
                    TableCell(
                        cell_id=f"{table_id}:r{r}:c{c}",
                        row_index=r,
                        col_index=c,
                        raw=strip_tex(raw) or raw.strip(),
                        mark=normalize_mark(raw),
                    )
                )
        if not rows:
            continue
        out.append(
            ComparisonTable(
                table_id=table_id,
                paper_id=paper_id,
                caption=caption,
                label=label,
                source="latex",
                section=section,
                dimensions=dims,
                rows=rows,
                cells=cells,
            )
        )
    return out
