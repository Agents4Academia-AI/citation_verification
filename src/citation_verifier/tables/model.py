"""
tables/model.py — the data contract for TABLE-LEVEL citation verification.

The prose pipeline verifies ``(claim, citation)`` pairs. This subsystem verifies a
different, finer grain: the **cells of a comparison table**. A "related work" feature
matrix asserts one checkable proposition per cell —

    "cited method R has property D"        (marked ✓ / ✗ / ▲ / a value)

and those propositions are rarely restated in prose, so the prose pipeline never sees
them. They are also where novelty is positioned (and inflated): a ✗ that should be a ✓
understates prior work, which is exactly the failure a reviewer would catch.

Grain and ids::

    ComparisonTable          one table float (caption + label + grid)
      ├── Dimension          one column  (header + the gloss that says what it MEANS)
      ├── TableRow           one row     (method label + cite_key + is_self)
      └── TableCell          one (row, dimension) assertion → cell_id "<table>:r<i>:c<j>"
            └── CellFinding  the verdict for that assertion + evidence + severity

Deliberately separate from :mod:`citation_verifier.schema`: ``CitationRecord`` is the
frozen ``(claim, citation)`` contract and a table cell is not that shape. Nothing here
imports a backend, the SDK, or the network.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CellMark",
    "CellVerdict",
    "DimensionKind",
    "GlossSource",
    "Dimension",
    "TableRow",
    "TableCell",
    "ComparisonTable",
    "CellFinding",
    "TableReport",
    "derive_cell_severity",
    "MARK_STR",
    "VERDICT_STR",
]


# ───────────────────────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────────────────────
class CellMark(str, Enum):
    """What the table *asserts* in a cell (normalized from ✓/✗/▲/text)."""

    YES = "yes"          # ✓ ✔ \cmark \checkmark \ding{51} "Yes"
    NO = "no"            # ✗ ✘ \xmark $\times$ \ding{55} "No"
    PARTIAL = "partial"  # ▲ ◐ half-filled circle, "limited", "partially"
    VALUE = "value"      # categorical/numeric cell ("gradient-based", "3.2M", "O(n)")
    EMPTY = "empty"      # blank cell
    UNKNOWN = "unknown"  # present but not interpretable


class CellVerdict(str, Enum):
    """Did the cell's assertion hold up against the cited paper?"""

    SUPPORTED = "supported"        # retrieved evidence confirms the mark
    CONTRADICTED = "contradicted"  # retrieved evidence refutes the mark — the ✓/✗ is wrong
    # The cited paper's FULL TEXT never claims the property, and the table marks ✓. Weaker
    # than CONTRADICTED — nothing shows the method cannot do it — but it is a real finding:
    # the citing paper credited prior work with a capability its own paper never asserts.
    # Distinct from UNVERIFIABLE, which means we never got enough of the paper to say.
    MAY_NOT_SUPPORT = "may_not_support"
    UNVERIFIABLE = "unverifiable"  # not enough evidence retrieved (abstention, NOT a refutation)
    UNDEFINED = "undefined"        # the dimension is never defined → uncheckable by construction
    SKIPPED = "skipped"            # self row / uncited row / non-binary value cell


class DimensionKind(str, Enum):
    """The value space of a column — decides whether a cell is checkable as a proposition."""

    BINARY = "binary"            # ✓/✗ capability claim
    GRADED = "graded"            # ✓/▲/✗ — a subjective strength ranking
    CATEGORICAL = "categorical"  # e.g. "White-box"/"Black-box", "Injection Method"
    NUMERIC = "numeric"          # counts, sizes, complexity
    LABEL = "label"              # the row-label column itself (method names)


class GlossSource(str, Enum):
    """Where the meaning of a column header was recovered from (never invented).

    Ordered by strength. ``RECOVERED`` sits between them: the paper does carry the meaning,
    in a caption, a symbol legend or an abbreviation expansion the keyword search cannot
    reach, and a model read it off that material. Usable for checking a cell, never strong
    enough to accuse the authors — so it is grouped with ``MENTION`` by the verdict gate
    but must not be reported to a reader as "the paper merely mentions this column".

    The other distinction that matters is ``MENTION`` vs ``HEADER_ONLY``:
    a term the paper discusses but never crisply defines is still checkable (weakly),
    whereas a term that appears *nowhere outside the table* cannot be checked at all —
    and only that second case is reported as an undefined column, because calling a
    column undefined is an accusation against the paper.
    """

    CAPTION = "caption"          # the table caption defined it
    LEGEND = "legend"            # a table footnote / legend line
    BODY = "body"                # a definition in the body (definition env, "X is …", "X: …")
    MENTION = "mention"          # discussed in the body, but never actually defined
    RECOVERED = "recovered"      # a model read the meaning off the caption/legend/context
    HEADER_ONLY = "header_only"  # never occurs outside the table — genuinely undefined
    NONE = "none"


# Rendered (human) strings. Machine tokens above stay stable; these are display only.
MARK_STR: dict[str, str] = {
    CellMark.YES.value: "✓",
    CellMark.NO.value: "✗",
    CellMark.PARTIAL.value: "▲",
    CellMark.VALUE.value: "value",
    CellMark.EMPTY.value: "—",
    CellMark.UNKNOWN.value: "?",
}

VERDICT_STR: dict[str, str] = {
    CellVerdict.SUPPORTED.value: "Supported",
    CellVerdict.CONTRADICTED.value: "Contradicted",
    CellVerdict.MAY_NOT_SUPPORT.value: "May not support",
    CellVerdict.UNVERIFIABLE.value: "Unverifiable",
    CellVerdict.UNDEFINED.value: "Undefined dimension",
    CellVerdict.SKIPPED.value: "Skipped",
}


# ───────────────────────────────────────────────────────────────
# Sub-models
# ───────────────────────────────────────────────────────────────
class _Model(BaseModel):
    """Base config: forbid unknown keys so contract drift fails loudly."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Dimension(_Model):
    """One column: the terse header plus what it actually MEANS.

    Table headers are compressed to fit ("Task-aware", "TDA Free"), so the checkable
    proposition lives in the caption, a legend, or a definition in the body. ``gloss``
    is that recovered meaning and ``gloss_quote`` is the text it came from — if nothing
    was found, ``gloss_source`` is ``header_only`` and cells in this column are reported
    as ``undefined`` rather than guessed at.
    """

    col_index: int
    header: str = Field(default="", description="The header cell as printed, TeX stripped.")
    kind: str = Field(default=DimensionKind.BINARY.value)
    gloss: str = Field(default="", description="What the column asserts, in one checkable sentence.")
    gloss_source: str = Field(default=GlossSource.NONE.value)
    gloss_quote: str = Field(default="", description="Verbatim text the gloss was derived from.")
    test_question: str = Field(
        default="", description="The yes/no question asked of each cited paper for this column."
    )


class TableRow(_Model):
    """One row: the method it names and the reference that backs it."""

    row_index: int
    label: str = Field(default="", description="Row label as printed (method name), TeX stripped.")
    cite_keys: list[str] = Field(
        default_factory=list, description="\\cite keys / markers found in the row label."
    )
    is_self: bool = Field(
        default=False,
        description="True for the authors' own method ('ours' / no citation) — not verified "
        "against an external paper, but counted for the all-✓ asymmetry check.",
    )


class TableCell(_Model):
    """One (row, dimension) assertion — the unit this subsystem verifies."""

    cell_id: str = Field(..., description="Stable id '<table_id>:r<row>:c<col>'.")
    row_index: int
    col_index: int
    raw: str = Field(default="", description="The cell exactly as written (TeX stripped).")
    mark: str = Field(default=CellMark.UNKNOWN.value)


class ComparisonTable(_Model):
    """A capability/feature matrix: rows are cited methods, columns are claimed properties."""

    table_id: str = Field(..., description="Stable id, e.g. 'tab:related_work' or 'table-1'.")
    paper_id: str = Field(default="")
    caption: str = Field(default="")
    label: str | None = None
    source: str = Field(default="latex", description="'latex' | 'pdf'.")
    section: str | None = Field(default=None, description="Section the table float sits in.")
    legend: list[str] = Field(
        default_factory=list,
        description="Footnote/legend lines under the table (symbol keys). Paper text — it "
        "is mined for column definitions, so only real page content belongs here.",
    )
    symbol_legend: dict[str, str] = Field(
        default_factory=dict,
        description="Symbol -> the meaning THIS paper gives it, read from the caption or "
        "legend (e.g. {'\\tmark': 'medium'}). A ▲ means whatever the paper says it means.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Extraction diagnostics (clipped headers, merged rows). Kept apart "
        "from `legend` so tool output is never mined as if the paper had written it.",
    )
    dimensions: list[Dimension] = Field(default_factory=list)
    rows: list[TableRow] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)

    def cell(self, row_index: int, col_index: int) -> TableCell | None:
        """The cell at ``(row_index, col_index)``, or ``None`` if absent."""
        for c in self.cells:
            if c.row_index == row_index and c.col_index == col_index:
                return c
        return None


class CellFinding(_Model):
    """The verification outcome for one table cell."""

    cell_id: str
    table_id: str = Field(default="")
    row_label: str = Field(default="")
    cite_key: str | None = None
    dimension: str = Field(default="", description="The column header this cell is under.")
    claimed: str = Field(default=CellMark.UNKNOWN.value, description="What the table asserts.")
    verdict: str = Field(default=CellVerdict.UNVERIFIABLE.value)
    severity: str = Field(default="ok", description="'high' | 'medium' | 'low' | 'ok'.")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = Field(default="")
    evidence_quote: str = Field(default="", description="The decisive sentence from the cited paper.")
    evidence_source: str = Field(default="", description="URL / arXiv id / API the quote came from.")
    understates_prior_work: bool = Field(
        default=False,
        description="A ✗ that the evidence refutes: the cited work DOES have the property. "
        "The highest-value finding — it inflates the citing paper's novelty.",
    )


class TableReport(_Model):
    """Everything found for one table: the grid, the resolved columns, and the verdicts."""

    table: ComparisonTable
    findings: list[CellFinding] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Verdict tally over the findings (machine tokens as keys)."""
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.verdict] = out.get(f.verdict, 0) + 1
        return out


# ───────────────────────────────────────────────────────────────
# Derived severity (deterministic — never model-guessed)
# ───────────────────────────────────────────────────────────────
def derive_cell_severity(mark: str, verdict: str, *, is_self: bool = False) -> str:
    """Severity for one cell, from ``(mark, verdict, is_self)``.

    The ranking encodes what actually damages a paper:

    * ``high``   — a refuted ✗ on a *cited competitor*: the prior work does have the
      property, so the table understates it and overstates this paper's novelty.
      Also a refuted ✓ on the authors' own row (an unearned capability claim).
    * ``medium`` — a refuted ✓ on a competitor (miscredited), or a cell in a column
      whose meaning is never defined anywhere (uncheckable by construction).
    * ``low``    — could not be verified (an abstention, not a refutation).
    * ``ok``     — supported, or skipped.

    Args:
        mark: a :class:`CellMark` value — what the table asserts.
        verdict: a :class:`CellVerdict` value — what verification concluded.
        is_self: True when the row is the authors' own method.

    Returns:
        ``"high"`` | ``"medium"`` | ``"low"`` | ``"ok"``.
    """
    mark = getattr(mark, "value", mark)
    verdict = getattr(verdict, "value", verdict)

    if verdict == CellVerdict.CONTRADICTED.value:
        if mark == CellMark.NO.value and not is_self:
            return "high"  # understated prior work → inflated novelty
        if mark == CellMark.YES.value and is_self:
            return "high"  # an unearned claim about the authors' own method
        return "medium"
    if verdict == CellVerdict.MAY_NOT_SUPPORT.value:
        # A ✓ the cited work never claims: the citing paper's own assertion, unbacked.
        return "low" if is_self else "medium"
    if verdict == CellVerdict.UNDEFINED.value:
        return "medium"
    if verdict == CellVerdict.UNVERIFIABLE.value:
        return "low"
    return "ok"
