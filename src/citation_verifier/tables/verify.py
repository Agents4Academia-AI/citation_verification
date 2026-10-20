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

from ..errors import ModelUnavailable
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

__all__ = [
    "verify_table", "asymmetry_summary", "build_row_payload", "redact_methods",
    "combine_segment_answers",
]

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
    # A ✗ the cited paper never contradicts. Silence is CONSISTENT with the mark, but it
    # does not establish it: "no evidence the method has X" is not "evidence it lacks X".
    # This was SUPPORTED for one run — and confirmed a ✗ against a paper whose excerpts
    # happened not to include its four mentions of exactly that capability.
    (CellMark.NO.value, _ABSENT): CellVerdict.UNVERIFIABLE.value,
    (CellMark.NO.value, _UNCLEAR): CellVerdict.UNVERIFIABLE.value,
    (CellMark.PARTIAL.value, _HAS): CellVerdict.SUPPORTED.value,
    (CellMark.PARTIAL.value, _LACKS): CellVerdict.CONTRADICTED.value,
    (CellMark.PARTIAL.value, _ABSENT): CellVerdict.MAY_NOT_SUPPORT.value,
    (CellMark.PARTIAL.value, _UNCLEAR): CellVerdict.UNVERIFIABLE.value,
}

_VERIFIABLE_MARKS = (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)


def combine_segment_answers(items: list[dict]) -> dict:
    """One answer from the same question asked of each part of a split paper.

    A part is not the paper, so what each part licenses is limited:

    * a part that SHOWS the property, or shows the work cannot have it, has found
      something — one such finding decides the paper, wherever it sits;
    * ``absent`` from a part means only "not in this part", so the paper counts as never
      claiming the property when EVERY part says so;
    * anything else leaves the question open.

    When one part shows the property and another shows its absence, the parts disagree
    about the same paper and this is not for the code to settle: the answer is
    ``unclear``, carrying both parts' reasons, and the cell is reported as unverified
    rather than as a finding built on half a paper.
    """
    seen = [i for i in items if isinstance(i, dict)]
    if not seen:
        return {}
    answers = [str(i.get("answer", "")).strip().lower() for i in seen]
    if all(a == _WRONG_PAPER for a in answers):
        return seen[0]
    live = [(a, i) for a, i in zip(answers, seen, strict=True) if a != _WRONG_PAPER]
    has = [i for a, i in live if a == _HAS]
    lacks = [i for a, i in live if a == _LACKS]
    if has and lacks:
        return {
            "answer": _UNCLEAR,
            "quote": has[0].get("quote") or lacks[0].get("quote") or "",
            "justification": (
                "the parts of this paper point different ways — "
                f"one part: {str(has[0].get('justification', ''))[:220]} | "
                f"another part: {str(lacks[0].get('justification', ''))[:220]}"
            ),
        }
    if has:
        return has[0]
    if lacks:
        return lacks[0]
    if live and all(a == _ABSENT for a, _i in live):
        return dict(live[0][1], answer=_ABSENT)
    return next((i for a, i in live if a == _UNCLEAR), live[0][1] if live else seen[0])


def build_row_payload(
    table: ComparisonTable, row_index: int, dims: list[Dimension], evidence: str
) -> dict:
    """The judge request for one cited work: the work, its evidence, and the properties.

    The claimed marks are deliberately **omitted** — the judge reports what the evidence
    shows about each property and this module compares that to the table afterwards, so
    the model cannot simply agree with the table.

    ``dims`` is normally a SINGLE property (:func:`verify_table` asks one question per
    cell). The names of the table's other columns ride along in ``other_properties``: a
    comparison table's columns are deliberately near-neighbours, and a judge shown one of
    them alone will otherwise answer from evidence that belongs to the column next door.
    """
    # By row_index, not by position: a caller holding a subset of the rows (re-running the
    # ones a failed run lost) still addresses them correctly, and a grid whose spacer rows
    # were dropped does not silently shift every payload onto the wrong cited paper.
    row = next((r for r in table.rows if r.row_index == row_index), None)
    if row is None:
        raise IndexError(f"no row with row_index {row_index} in {table.table_id}")
    names = [r.label for r in table.rows]
    asked = {d.col_index for d in dims}
    return {
        "row_label": row.label,
        "cite_key": row.cite_keys[0] if row.cite_keys else None,
        "evidence": evidence,
        "properties": [
            _property_entry(d, method_names=names) for d in dims
        ],
        "other_properties": [
            d.header for d in table.dimensions if d.col_index not in asked and d.header
        ],
    }


# Trailing markers and qualifiers a row label carries that its name in prose does not:
# "CaT (this work)", "kPAM [12] and its variants".
_LABEL_TAIL_RE = re.compile(r"\s*(?:[\(\[].*$|\band\s+(?:its|their)\s+variants?\b.*$)", re.IGNORECASE)
_REDACTED = "a compared method"


def _looks_like_a_name(label: str) -> bool:
    """A proper name ("MetaMix", "S3K", "Skeleton Merger"), not a category ("Rule-based").

    Rows are sometimes families rather than methods — "Rule-based", "Inference-time" —
    and the citing paper's criterion is often stated in exactly those words ("rule-based
    watermarking is efficient because …"). Redacting them would delete the criterion, not
    the mark. A name shows an internal capital or a digit; a category is ordinary words.
    """
    return bool(re.search(r"(?<=.)[A-Z]|\d", label))


def redact_methods(text: str, method_names: list[str]) -> str:
    """Replace the table's row names in ``text`` with a neutral placeholder.

    The citing paper's criterion notes can say how a property was applied to a named
    method ("X and Y approach this via cross-task mixup, at the expense of erroneous
    tasks"). Read with the names in, that is the table's mark in prose, and the judge would
    be answering the table rather than the cited work's text. With the names out, the
    criterion survives ("cross-task mixup … erroneous tasks") and whether THIS work is
    such a method still has to be read off its own evidence.
    """
    out = text or ""
    for raw in method_names or []:
        name = _LABEL_TAIL_RE.sub("", (raw or "").strip()).strip(" .,;:")
        if len(name) < 3 or not _looks_like_a_name(name):
            continue
        words = [re.escape(w) for w in re.split(r"[\s\-]+", name) if w]
        if not words:
            continue
        pat = re.compile(r"(?<![\w-])" + r"[\s\-]+".join(words) + r"(?![\w-])", re.IGNORECASE)
        out = pat.sub(_REDACTED, out)
    return out


def _property_entry(dim: Dimension, *, method_names: list[str] | None = None) -> dict:
    """One property for the judge: its name and what the paper says it means.

    ``test_question`` is included only when it carries information. Without a glosser it
    is a template ("Does the cited work satisfy '<header>'?") that merely restates the
    header, so sending it adds a line of noise to every property.

    ``criterion`` carries the citing paper's own notes on how it applied the column (see
    :attr:`Dimension.criterion_notes`), with the table's row names redacted.
    """
    entry = {
        "col_index": dim.col_index,
        "name": dim.header,
        "definition": clean_property_text(dim.gloss) or dim.header,
    }
    templated = f"Does the cited work satisfy '{dim.header}'?"
    if dim.test_question and dim.test_question not in (templated, DEFAULT_TEST_QUESTION):
        entry["question"] = dim.test_question
    notes = [
        redact_methods(clean_property_text(n), list(method_names or []))
        for n in (getattr(dim, "criterion_notes", None) or [])
    ]
    notes = [n for n in notes if n]
    if notes:
        entry["criterion"] = " | ".join(notes)
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
    # Why the model cannot be reached, once a call has said so. From then on no evidence
    # is fetched and no question asked: the remaining cells are marked as never judged.
    model_down = ""

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
        named_in_paper: dict[int, bool] = {}
        search_report: dict[int, dict] = {}
        complete_text = False
        segments: list[str] = []
        if judge is not None and not model_down:
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
                    meta = got_ev[2] if len(got_ev) > 2 and isinstance(got_ev[2], dict) else {}
                    named_in_paper = dict(meta.get("named") or {})
                    search_report = dict(meta.get("search") or {})
                    complete_text = bool(meta.get("complete"))
                    segments = [str(x) for x in (meta.get("segments") or []) if str(x)]
                else:
                    evidence, source = "", ""
                    notes.append(f"{row.label}: evidence_for returned {type(got_ev).__name__}")
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                evidence, source = "", ""
                notes.append(f"{row.label}: evidence retrieval failed: {exc!r}")

        # ONE QUESTION PER CELL. Asked together, the properties of a comparison table
        # compete for one bounded answer and for the model's attention, and a failure took
        # every cell of the row with it. Asked separately, each cell gets the whole answer
        # and a failure costs only that cell — and the cited work, which is the long part
        # of the request, is a cacheable prefix that does not change between them.
        answers: dict[int, dict] = {}
        cell_failure: dict[int, str] = {}
        if evidence.strip() and judge is not None:
            # A paper too long for one request arrives already cut into parts; the same
            # question is put to each and the answers combined.
            parts = segments or [evidence]
            for dim in row_dims:
                if model_down:
                    break
                try:
                    per_part = []
                    for part in parts:
                        got = judge(build_row_payload(table, row.row_index, [dim], part))
                        found = next((x for x in (got or []) if isinstance(x, dict)), None)
                        if found is not None:
                            per_part.append(found)
                    item = combine_segment_answers(per_part) if per_part else None
                    if item:
                        # The answer is to the one question asked, whatever index the
                        # model echoed back.
                        answers[dim.col_index] = dict(item, col_index=dim.col_index)
                except ModelUnavailable as exc:
                    model_down = str(exc) or "no reason given"
                    notes.append(f"the model could not be reached ({model_down}); no cell judged from here on")
                except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                    notes.append(f"{row.label} / {dim.header}: judge failed: {exc!r}")
                    cell_failure[dim.col_index] = (
                        f"the judge call for this cell failed ({type(exc).__name__})"
                    )
        if model_down:
            for dim in row_dims:
                if dim.col_index not in answers:
                    cell_failure[dim.col_index] = f"the model could not be reached ({model_down})"

        for dim in row_dims:
            cell = table.cell(row.row_index, dim.col_index)
            if cell is None:
                continue
            got = answers.get(dim.col_index) or {}
            answer = str(got.get("answer", _UNCLEAR)).strip().lower()
            if answer not in (_HAS, _LACKS, _ABSENT, _UNCLEAR, _WRONG_PAPER):
                answer = _UNCLEAR
            absent_note = ""
            rec = search_report.get(dim.col_index) or {}
            # Without a search record nothing is known about what was shown: not all.
            # With the COMPLETE paper there is nothing the judge was not shown, so its
            # reading of the whole text stands as it is.
            all_shown = complete_text or (
                bool(rec) and int(rec.get("shown") or 0) >= int(rec.get("mentions") or 0)
            )
            if answer == _ABSENT and named_in_paper.get(dim.col_index, False) and not all_shown:
                # The judge saw excerpts; the paper mentions this property in passages it
                # was not shown. Paper-level absence is not the judge's to assert then —
                # only the code knows what the whole paper contains. (When every mention
                # WAS shown, the judge has read all the paper says under those wordings,
                # and its 'absent' stands.)
                answer = _UNCLEAR
                n, shown = int(rec.get("mentions") or 0), int(rec.get("shown") or 0)
                where = (
                    f"in {n} passages of which only {shown} were shown"
                    if rec else "by name elsewhere"
                )
                absent_note = (
                    f"[the judge found no claim in the excerpts, but the paper does mention "
                    f"this property {where}, so absence cannot be concluded] "
                )
            verdict = _DECISION.get((cell.mark, answer), CellVerdict.UNVERIFIABLE.value)
            if judge is None:
                # Nothing was looked up and nothing was asked. Saying "no evidence could
                # be retrieved" here would report a retrieval failure that never happened.
                verdict = CellVerdict.UNVERIFIABLE.value
                got = {"justification": "not judged — this run had no model to judge table cells with"}
            elif model_down and dim.col_index in cell_failure:
                # Not a retrieval failure and not an open question: nobody was asked.
                verdict = CellVerdict.UNVERIFIABLE.value
                got = {}
            elif not evidence.strip():
                verdict = CellVerdict.UNVERIFIABLE.value
                got = {"justification": "no evidence could be retrieved for the cited work"}
            elif answer == _ABSENT:
                # Say what the search covered: absence is only as good as the wordings
                # tried, and a reader auditing the cell needs to see them.
                terms = ", ".join(str(t) for t in (rec.get("terms") or [])[:12])
                where = f" under any of: {terms}" if terms else ""
                n = int(rec.get("mentions") or 0)
                if complete_text:
                    read = "[the complete text of the paper was read and it never claims this property"
                elif n:
                    read = (
                        f"[every passage mentioning this property ({n}) was read and none "
                        f"claims it for this work"
                    )
                else:
                    read = f"[the paper never mentions this property{where}"
                lead = (
                    f"{read}; silence is consistent with the ✗ but does not establish it] "
                    if cell.mark == CellMark.NO.value
                    else f"{read}] "
                )
                got = dict(got, justification=lead + str(got.get("justification", "") or ""))

            note = absent_note + str(got.get("justification", "") or "")[:500]
            severity_floor = ""
            row_failure = cell_failure.get(dim.col_index, "")
            if row_failure and not note:
                # A crashed judge call is a pipeline failure. Left blank it renders as an
                # ordinary "could not verify" with no reason given, indistinguishable from
                # the cited paper simply not settling the question — measured: five cells
                # of one table read that way.
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
