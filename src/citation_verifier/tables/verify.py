"""
tables/verify.py — check each table cell against the paper it cites.

One cell asserts one proposition: *"cited method R has property D"*. This module
retrieves evidence from R's own paper and decides whether the ✓/✗ holds.

Two rules carried over from the prose pipeline, because they are what keep the output
trustworthy:

  * **Never judge from memory.** A verdict is only ever derived from retrieved text
    (abstract, then full-text excerpts selected by the column's gloss).
  * **Absence is not refutation.** "The abstract doesn't mention it" does NOT prove a
    method lacks a property. The judge answers ``has`` / ``lacks`` / ``unclear``, and
    only an explicit ``lacks`` can confirm a ✗. Everything else is ``unverifiable``.

The asymmetry that matters::

    claimed ✗  +  evidence says the work HAS it   ->  CONTRADICTED, severity HIGH
                                                      (prior work understated =
                                                       this paper's novelty inflated)

Cost shape: evidence is fetched and judged **once per row** (one cited paper) covering
all of its columns, not once per cell — so a 5x4 table costs 5 calls, not 20.

Both seams (``evidence_for``, ``judge``) are injected, so the whole module runs offline
in tests with no network and no SDK.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .dimensions import DEFAULT_TEST_QUESTION, dimension_is_checkable
from .model import (
    CellFinding,
    CellMark,
    CellVerdict,
    ComparisonTable,
    Dimension,
    DimensionKind,
    GlossSource,
    TableReport,
    derive_cell_severity,
)

__all__ = ["verify_table", "asymmetry_summary", "build_row_payload"]

# Cross-references survive into a gloss quoted from the source ("(task F in
# Figure~\ref{fig:intro}c)") and tell the judge nothing about the property, while eating
# context and inviting it to reason about a figure it cannot see.
_XREF_RE = re.compile(
    # A whole parenthetical that points at a float goes first — otherwise only the macro
    # inside it is removed and a stub survives ("(tasks C, D and E in b)").
    r"\([^()]{0,90}(?:Figure|Fig\.|Table|Tab\.|Section|Sec\.|Eq\.|Equation|Appendix)[^()]{0,90}\)"
    r"|\\(?:ref|cref|Cref|autoref|eqref|label|cite\w*)\s*\{[^}]*\}"
    r"|\b(?:Figure|Fig\.|Table|Tab\.|Section|Sec\.|Appendix)~?\s*\\?\w*\{?[\w:.-]*\}?",
)


def clean_property_text(text: str) -> str:
    """Tidy a gloss for the judge without changing what it asserts.

    Only cross-references and TeX bookkeeping are removed — the wording, including any
    equation, is left exactly as the paper wrote it, because the judge is required to
    match the definition at its own level of precision.
    """
    out = _XREF_RE.sub(" ", text or "")
    out = re.sub(r"\s*\(\s*\)", "", out)          # brackets emptied by the above
    out = re.sub(r"\s+", " ", out).strip(" .,;:")
    # Removing a reference can leave the clause that introduced it dangling
    # ("…, as reported in"). Drop back to the last complete clause.
    out = re.sub(
        r"[,;]?\s*(?:as\s+)?(?:reported|shown|listed|described|summari[sz]ed|illustrated|given)?"
        r"\s*(?:in|on|by|from|of|see)\s*$",
        "", out, flags=re.IGNORECASE,
    )
    return out.strip(" .,;:")

# What the judge may answer about one (paper, property) pair.
_HAS, _LACKS, _UNCLEAR = "has", "lacks", "unclear"

# (claimed mark, judged answer) -> verdict
# "The whole paper never claims this property." Silence read across the full text, which
# is informative — a paper advertises what its method does — without refuting anything.
_ABSENT = "absent"
# "The evidence is about some other work." A pipeline failure, not a finding about the
# paper: the cell was never checked. Mixed into the ordinary "not enough evidence" bucket
# it reads as though the cited work had been consulted and come up short — measured, a
# cell whose retrieval returned an unrelated paper was reported exactly that way.
_WRONG_PAPER = "wrong_paper"

_DECISION: dict[tuple[str, str], str] = {
    (CellMark.YES.value, _HAS): CellVerdict.SUPPORTED.value,
    (CellMark.YES.value, _LACKS): CellVerdict.CONTRADICTED.value,
    # A ✓ the cited paper never claims anywhere: not refuted, but not backed either.
    (CellMark.YES.value, _ABSENT): CellVerdict.MAY_NOT_SUPPORT.value,
    (CellMark.YES.value, _UNCLEAR): CellVerdict.UNVERIFIABLE.value,
    (CellMark.NO.value, _HAS): CellVerdict.CONTRADICTED.value,   # understates prior work
    (CellMark.NO.value, _LACKS): CellVerdict.SUPPORTED.value,
    # A ✗ the cited paper never contradicts: silence and the mark agree.
    (CellMark.NO.value, _ABSENT): CellVerdict.SUPPORTED.value,
    (CellMark.NO.value, _UNCLEAR): CellVerdict.UNVERIFIABLE.value,
    (CellMark.PARTIAL.value, _HAS): CellVerdict.SUPPORTED.value,
    (CellMark.PARTIAL.value, _LACKS): CellVerdict.CONTRADICTED.value,
    (CellMark.PARTIAL.value, _ABSENT): CellVerdict.MAY_NOT_SUPPORT.value,
    (CellMark.PARTIAL.value, _UNCLEAR): CellVerdict.UNVERIFIABLE.value,
}

_VERIFIABLE_MARKS = (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)


def build_row_payload(
    table: ComparisonTable, row_index: int, dims: list[Dimension], evidence: str
) -> dict:
    """The judge request for one row: the cited work, its evidence, and the properties.

    The claimed marks are deliberately **omitted** — the judge reports what the evidence
    shows about each property and this module compares that to the table afterwards, so
    the model cannot simply agree with the table.
    """
    row = table.rows[row_index]
    return {
        "row_label": row.label,
        "cite_key": row.cite_keys[0] if row.cite_keys else None,
        "evidence": evidence,
        "properties": [
            _property_entry(d) for d in dims
        ],
    }


def _property_entry(dim: Dimension) -> dict:
    """One property for the judge: its name and what the paper says it means.

    ``test_question`` is included only when it carries information. Without a glosser it
    is a template ("Does the cited work satisfy '<header>'?") that merely restates the
    header, so sending it adds a line of noise to every property.
    """
    entry = {
        "col_index": dim.col_index,
        "name": dim.header,
        "definition": clean_property_text(dim.gloss) or dim.header,
    }
    templated = f"Does the cited work satisfy '{dim.header}'?"
    if dim.test_question and dim.test_question not in (templated, DEFAULT_TEST_QUESTION):
        entry["question"] = dim.test_question
    return entry


def _skip(table: ComparisonTable, row, dim: Dimension, cell, reason: str, verdict: str) -> CellFinding:
    """A finding that records why a cell was not verified."""
    return CellFinding(
        cell_id=cell.cell_id,
        table_id=table.table_id,
        row_label=row.label,
        cite_key=row.cite_keys[0] if row.cite_keys else None,
        dimension=dim.header,
        claimed=cell.mark,
        verdict=verdict,
        severity=derive_cell_severity(cell.mark, verdict, is_self=row.is_self),
        justification=reason,
    )


# Gloss grades too weak to license a CONTRADICTED verdict. A contradiction says "the
# paper's ✓/✗ is wrong", which presupposes knowing what the column asserts; these two
# grades mean precisely that the paper never said.
# The paper says nothing at all about this column outside the table. A contradiction here
# would be an accusation over a criterion that does not exist. (A HEADER_ONLY column never
# reaches the judge at all — `dimension_is_checkable` drops it — so only NONE arrives here,
# from a table built without the gloss stage.)
_NO_GLOSS = frozenset({GlossSource.HEADER_ONLY.value, GlossSource.NONE.value})
# The paper does say what the column means, but loosely — a passing mention, or a meaning
# a model recovered from the caption. Weak grounds for an accusation, but reporting that
# the paper "never states" anything would be false, and hiding the contradiction under
# "undefined" loses a real signal.
_LOOSE_GLOSS = frozenset({GlossSource.MENTION.value, GlossSource.RECOVERED.value})
_WEAK_GLOSS = _NO_GLOSS | _LOOSE_GLOSS


def verify_table(
    table: ComparisonTable,
    *,
    evidence_for: Callable[[str | None, str], tuple[str, str]],
    judge: Callable[[dict], list[dict]] | None = None,
    verify_self_rows: bool = False,
) -> TableReport:
    """Verify every checkable cell of ``table``.

    Args:
        table: a table whose dimensions have already been resolved
            (:func:`citation_verifier.tables.dimensions.resolve_dimensions`).
        evidence_for: ``(cite_key, row_label) -> (evidence_text, source)``. Returns
            ``("", "")`` when nothing could be retrieved (then cells are unverifiable).
        judge: ``payload -> [{col_index, answer, quote, justification, confidence}, …]``
            where ``answer`` is ``has`` | ``lacks`` | ``unclear``. ``None`` disables
            judging: every otherwise-checkable cell becomes ``unverifiable``.
        verify_self_rows: also verify the authors' own row. Off by default — its
            evidence is the paper under review, not an external source.

    Returns:
        A :class:`TableReport`. Never raises: a failing row degrades to unverifiable
        cells with the error recorded in ``notes``.
    """
    findings: list[CellFinding] = []
    notes: list[str] = []

    for d in [d for d in table.dimensions if d.gloss_source == GlossSource.HEADER_ONLY.value]:
        notes.append(
            f"column {d.header!r} is never defined in the caption, legend, or body — "
            f"its marks are not checkable as written"
        )

    for row in table.rows:
        row_dims: list[Dimension] = []
        for dim in table.dimensions:
            cell = table.cell(row.row_index, dim.col_index)
            if cell is None:
                continue
            # Row-level reasons come first: a self/uncited row has nothing external to
            # check against, so reporting its cells as "undefined column" would be noise
            # (the column defect is still recorded once, in notes + asymmetry_summary).
            if row.is_self and not verify_self_rows:
                findings.append(
                    _skip(table, row, dim, cell, "authors' own method — no external source",
                          CellVerdict.SKIPPED.value)
                )
                continue
            if not row.cite_keys:
                findings.append(
                    _skip(table, row, dim, cell, "row carries no citation to verify against",
                          CellVerdict.SKIPPED.value)
                )
                continue
            if cell.mark not in _VERIFIABLE_MARKS:
                findings.append(
                    _skip(table, row, dim, cell, f"cell is {cell.mark}, not a ✓/✗ claim",
                          CellVerdict.SKIPPED.value)
                )
                continue
            if dim.gloss_source == GlossSource.HEADER_ONLY.value:
                findings.append(
                    _skip(table, row, dim, cell, "column never defined in the paper",
                          CellVerdict.UNDEFINED.value)
                )
                continue
            if not dimension_is_checkable(dim):
                findings.append(
                    _skip(table, row, dim, cell, f"column kind {dim.kind} is descriptive",
                          CellVerdict.SKIPPED.value)
                )
                continue
            row_dims.append(dim)

        if not row_dims:
            continue

        # One retrieval + one judge call per cited paper, covering all its columns.
        # Without a judge there is nothing to do with the evidence, so don't fetch it.
        evidence, source = "", ""
        if judge is not None:
            try:
                got_ev = evidence_for(row.cite_keys[0] if row.cite_keys else None, row.label)
                # Coerce: a retriever returning None/partial must degrade, not crash.
                # A bare str is indexable, so without the shape check `got_ev[0]` would
                # silently become its FIRST CHARACTER and be judged as if it were evidence.
                if isinstance(got_ev, str):
                    evidence, source = got_ev, ""
                elif isinstance(got_ev, list | tuple):
                    evidence = str((got_ev[0] if got_ev else "") or "")
                    source = str((got_ev[1] if len(got_ev) > 1 else "") or "")
                else:
                    evidence, source = "", ""
                    notes.append(f"{row.label}: evidence_for returned {type(got_ev).__name__}")
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                evidence, source = "", ""
                notes.append(f"{row.label}: evidence retrieval failed: {exc!r}")

        answers: dict[int, dict] = {}
        row_failure = ""
        if evidence.strip() and judge is not None:
            try:
                got = judge(build_row_payload(table, row.row_index, row_dims, evidence))
                for item in got or []:
                    if isinstance(item, dict) and item.get("col_index") is not None:
                        answers[int(item["col_index"])] = item
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                notes.append(f"{row.label}: judge failed: {exc!r}")
                row_failure = f"the judge call for this row failed ({type(exc).__name__})"

        for dim in row_dims:
            cell = table.cell(row.row_index, dim.col_index)
            if cell is None:
                continue
            got = answers.get(dim.col_index) or {}
            answer = str(got.get("answer", _UNCLEAR)).strip().lower()
            if answer not in (_HAS, _LACKS, _ABSENT, _UNCLEAR, _WRONG_PAPER):
                answer = _UNCLEAR
            verdict = _DECISION.get((cell.mark, answer), CellVerdict.UNVERIFIABLE.value)
            if not evidence.strip():
                verdict = CellVerdict.UNVERIFIABLE.value
                got = {"justification": "no evidence could be retrieved for the cited work"}

            note = str(got.get("justification", "") or "")[:500]
            severity_floor = ""
            if row_failure and not note:
                # A crashed judge call is a pipeline failure, and every cell in the row
                # inherits it. Left blank it renders as an ordinary "could not verify" with
                # no reason given, indistinguishable from the cited paper simply not
                # settling the question — measured: five cells of one table read that way.
                note = f"[pipeline failure: {row_failure}, so this cell was never judged]"
            if answer == _WRONG_PAPER:
                note = (
                    "[retrieval failure: the retrieved text is about a different work, so "
                    "this cell was never checked against the cited paper] "
                ) + note
            # An accusation is only as good as the column definition it rests on. When the
            # paper merely MENTIONS a column and never defines it, the judge was comparing
            # the cited work against a sentence that happens to use the term — and a
            # measured 12-of-15 false-positive rate traced back to exactly that. Downgrade
            # the contradiction to "unverifiable" and say which half failed: the claim is
            # not refuted, it is uncheckable because the column has no stated criterion.
            # In a CATEGORICAL column most cells name a technique ("ILP", "SMT",
            # "Heuristics") and a ✗ means "no technique reported for this phase" — a
            # weaker claim than "cannot do this". The judge was asked the binary question
            # ("does the work allocate resources at all?") and answered it, refuting
            # something the table never asserted. Measured: three high-severity
            # accusations against compilers whose ✗ only meant the paper names no
            # allocation technique.
            if (
                verdict == CellVerdict.CONTRADICTED.value
                and cell.mark == CellMark.NO.value
                and dim.kind == DimensionKind.CATEGORICAL.value
            ):
                # Also a statement about the citing paper's table rather than about our
                # reach: the ✗ in a column of technique names is not a capability claim.
                verdict = CellVerdict.UNDEFINED.value
                note = (
                    "[in this column a ✗ means the citing paper reports no technique for "
                    "that phase, not that the cited work cannot do it] "
                ) + note
            if verdict == CellVerdict.CONTRADICTED.value and dim.gloss_source in _NO_GLOSS:
                # UNDEFINED, not UNVERIFIABLE: this is a finding about the CITING paper —
                # it drew a column and never said what earns a mark in it — not a report
                # that we failed to check something. Filed under "we could not verify" it
                # read as our shortcoming and the reader had no way to tell the two apart.
                verdict = CellVerdict.UNDEFINED.value
                note = (
                    "[the citing paper never states what earns a mark in this column, so "
                    "the cell asserts nothing checkable] "
                ) + note
            elif verdict == CellVerdict.CONTRADICTED.value and dim.gloss_source in _LOOSE_GLOSS:
                # The paper DOES say what this column means, just not crisply. Reporting
                # that it "never states" anything would be false, and burying the
                # contradiction under "undefined" hides a real signal — a reviewer reading
                # an earlier run flagged exactly that. Kept as a contradiction, but low
                # severity and labelled: a lead to check, not a finding to headline.
                severity_floor = "low"
                note = (
                    "[weakly grounded: the paper discusses this column but never defines "
                    "it crisply, so treat this as a lead rather than a finding] "
                ) + note
            severity = derive_cell_severity(cell.mark, verdict, is_self=row.is_self)
            if severity_floor:
                severity = severity_floor

            findings.append(
                CellFinding(
                    cell_id=cell.cell_id,
                    table_id=table.table_id,
                    row_label=row.label,
                    cite_key=row.cite_keys[0] if row.cite_keys else None,
                    dimension=dim.header,
                    claimed=cell.mark,
                    verdict=verdict,
                    severity=severity,
                    confidence=_as_confidence(got.get("confidence")),
                    justification=note,
                    evidence_quote=str(got.get("quote", "") or "")[:500],
                    evidence_source=source,
                    understates_prior_work=(
                        verdict == CellVerdict.CONTRADICTED.value
                        and cell.mark == CellMark.NO.value
                        and not row.is_self
                    ),
                )
            )

    return TableReport(table=table, findings=findings, notes=notes)


def _as_confidence(value: Any) -> float | None:
    """Coerce a model-supplied confidence into ``[0, 1]``, or ``None``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, f))


def asymmetry_summary(report: TableReport) -> dict[str, Any]:
    """Table-level signal: is the novelty positioning supported by the evidence?

    A comparison table is persuasive because the authors' row is fuller than everyone
    else's. This quantifies that shape and how much of it survived checking:

    * ``self_all_yes``           — the authors' row is ✓ on every column.
    * ``understated_prior_work`` — refuted ✗ marks on cited competitors (the headline).
    * ``unbacked_ticks``         — ✓ marks the cited paper never claims anywhere in its
      full text: weaker than a refutation, but the citing paper credited prior work with
      something its own paper does not assert.
    * ``undefined_columns``      — columns marked ✓/✗ that the paper never defines.
    """
    table = report.table
    self_rows = [r for r in table.rows if r.is_self]
    self_all_yes = False
    for r in self_rows:
        marks = [c.mark for c in table.cells if c.row_index == r.row_index]
        if marks and all(m == CellMark.YES.value for m in marks):
            self_all_yes = True

    understated = [f for f in report.findings if f.understates_prior_work]
    contradicted = [f for f in report.findings if f.verdict == CellVerdict.CONTRADICTED.value]
    unbacked = [f for f in report.findings if f.verdict == CellVerdict.MAY_NOT_SUPPORT.value]
    undefined_cols = sorted(
        {d.header for d in table.dimensions if d.gloss_source == GlossSource.HEADER_ONLY.value}
    )
    return {
        "table_id": table.table_id,
        "rows": len(table.rows),
        "columns": len(table.dimensions),
        "self_all_yes": self_all_yes,
        "contradicted": len(contradicted),
        "unbacked_ticks": len(unbacked),
        "unbacked_cells": [
            {"row": f.row_label, "column": f.dimension, "cite_key": f.cite_key}
            for f in unbacked
        ],
        "understated_prior_work": len(understated),
        "understated_cells": [
            {"row": f.row_label, "column": f.dimension, "cite_key": f.cite_key}
            for f in understated
        ],
        "undefined_columns": undefined_cols,
        "verdicts": report.counts(),
    }
