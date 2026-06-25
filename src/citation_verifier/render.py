"""
render.py — deterministic rendering of records to the SKILL.md output.

This module is pure (no LLM, no network). It turns a list of
:class:`~citation_verifier.schema.CitationRecord` into:

  - :func:`render_table`   — the EXACT 7-column SKILL.md Markdown table + scope line,
  - :func:`render_summary` — counts + a "Fix before submission" high-severity list,
  - :func:`render_report`  — scope + table + summary + a usage footer,
  - :func:`to_json` / :func:`from_json` — a JSONL round-trip for records.

The SKILL.md table is the FROZEN contract. Every column maps to a schema field
and NO column exists that is not a schema field. Enum machine tokens are mapped
to the human strings the table uses (notably ``does_not`` -> ``does not``) via
the ``*_STR`` maps below.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .interfaces import VerificationResult
from .schema import (
    CitationRecord,
    Exists,
    Priority,
    Severity,
    SupportsClaim,
)

__all__ = [
    "EXISTS_STR",
    "SUPPORTS_STR",
    "SEVERITY_STR",
    "PRIORITY_STR",
    "TABLE_HEADER",
    "render_table",
    "render_summary",
    "render_report",
    "to_json",
    "from_json",
]

# ── enum token -> SKILL.md table string ──────────────────────────────────────
EXISTS_STR: dict[str, str] = {
    Exists.YES.value: "yes",
    Exists.NO.value: "no",
    Exists.UNRESOLVED.value: "unresolved",
}

SUPPORTS_STR: dict[str, str] = {
    SupportsClaim.SUPPORTS.value: "supports",
    SupportsClaim.PARTIAL.value: "partial",
    SupportsClaim.DOES_NOT.value: "does not",  # token does_not -> table string
    SupportsClaim.INCONCLUSIVE.value: "inconclusive",
}

PRIORITY_STR: dict[str, str] = {
    Priority.OBLIGATORY.value: "obligatory",
    Priority.HELPFUL.value: "helpful",
}

SEVERITY_STR: dict[str, str] = {
    Severity.HIGH.value: "high",
    Severity.MEDIUM.value: "medium",
    Severity.LOW.value: "low",
    Severity.OK.value: "ok",
}

# The frozen SKILL.md column header (load-bearing — tests pin this exactly).
_COLUMNS = [
    "#",
    "Citation (authors, title, year)",
    "Cited where (the claim)",
    "Exists?",
    "Match notes",
    "Supports claim?",
    "Explanation",
]
TABLE_HEADER = "| " + " | ".join(_COLUMNS) + " |"
_TABLE_DIVIDER = "| " + " | ".join(["---"] * len(_COLUMNS)) + " |"


def _enum_value(value: object) -> str:
    """Return the ``.value`` of an enum, or the string itself.

    Records may carry either Enum members or their string values (the schema
    uses ``use_enum_values=True``), so rendering must accept both.
    """
    return getattr(value, "value", value) if value is not None else ""


def _cell(text: str | None) -> str:
    """Sanitize a value for a Markdown table cell (escape pipes, collapse newlines)."""
    if not text:
        return ""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _citation_cell(record: CitationRecord) -> str:
    """Render col 2: 'authors, title, year' from ``cited_as`` (full title, untruncated)."""
    cited = record.cited_as
    authors = cited.authors or []
    if len(authors) > 2:
        who = f"{authors[0]} et al."
    else:
        who = ", ".join(authors)
    title = cited.title or ""
    year = str(cited.year) if cited.year else ""
    parts = [p for p in (who, title, year) if p]
    rendered = ", ".join(parts)
    return rendered or _cell(cited.raw) or record.cite_key


def _cite_marker(cite_key: str) -> str:
    """A short per-row citation marker: ``[6]`` for ``ref-6``, else ``[<key>]``."""
    m = re.match(r"ref-(\d+)$", cite_key or "")
    return f"[{m.group(1)}]" if m else f"[{cite_key}]"


def _claim_cell(record: CitationRecord, *, marker: str | None = None) -> str:
    """Render col 3: the claim text (optional cite marker + section prefix).

    When a claim is cited by more than one reference (e.g. ``… and more [6, 7]``)
    every such row is prefixed with its own marker (``[6]`` / ``[7]``) so the
    (claim, citation) pair the row refers to is unambiguous.
    """
    claim = record.claim
    text = claim.text or ""
    if claim.section:
        text = f"[{claim.section}] {text}" if text else f"[{claim.section}]"
    if marker:
        text = f"{marker} {text}".strip()
    return text


def _metadata_cell(record: CitationRecord) -> str:
    """Render col 5 (Match notes): the ``metadata_issues``, semicolon-joined.

    Holds match/metadata discrepancies when ``exists = yes`` and, when
    ``exists = unresolved``, which sources were searched without a confident match.
    """
    return "; ".join(record.metadata_issues)


def _explanation_cell(record: CitationRecord) -> str:
    """Render the final ``Explanation`` column: the note/justification + any links.

    No severity word (``ok`` / ``low`` / …) — severity now lives only in the
    Summary. A clean row with nothing to explain renders empty.
    """
    return record.notes or record.error or ""


def _severity_cell(record: CitationRecord) -> str:
    """Severity + a short note, for the Summary's 'Fix before submission' list."""
    sev = SEVERITY_STR.get(_enum_value(record.severity), "ok")
    note = record.notes or record.error or ""
    if note:
        return f"{sev} — {_cell(note)}"
    return sev


def render_table(records: list[CitationRecord]) -> str:
    """Render records as the EXACT SKILL.md 7-column table + a scope line.

    Args:
        records: The verified records (one row each).

    Returns:
        A Markdown string: a scope line, a blank line, then the table.
    """
    scope = _scope_line(records)
    lines = [scope, "", TABLE_HEADER, _TABLE_DIVIDER]
    # A claim cited by >1 reference shares a char_span across its rows; mark each
    # such row with its own cite marker so the (claim, citation) pair is clear.
    shared_spans = {
        span
        for span, count in Counter(
            rec.claim.char_span for rec in records if rec.claim and rec.claim.char_span
        ).items()
        if count > 1
    }
    # Group rows by reference: references ordered by first appearance, each
    # reference's claims kept in document order. exists/metadata/title/URL are
    # reference-level facts, so clustering a paper's rows keeps them scannable —
    # one paper's uses sit together ("cited 6×: 4 supports, 2 partial") instead of
    # scattering the same reference down a long table.
    first_seen: dict[str, int] = {}
    for i, rec in enumerate(records):
        first_seen.setdefault(rec.cite_key, i)
    order = sorted(range(len(records)), key=lambda i: (first_seen[records[i].cite_key], i))

    for n, idx in enumerate(order, start=1):
        rec = records[idx]
        marker = (
            _cite_marker(rec.cite_key)
            if rec.claim and rec.claim.char_span in shared_spans
            else None
        )
        row = [
            str(n),
            _cell(_citation_cell(rec)),
            _cell(_claim_cell(rec, marker=marker)),
            EXISTS_STR.get(_enum_value(rec.exists), "unresolved"),
            _cell(_metadata_cell(rec)),
            SUPPORTS_STR.get(_enum_value(rec.supports_claim), "inconclusive"),
            _cell(_explanation_cell(rec)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _scope_line(records: list[CitationRecord]) -> str:
    """A one-line scope statement: paper id and number of (claim, citation) pairs."""
    paper_id = ""
    for rec in records:
        if rec.paper and rec.paper.paper_id:
            paper_id = rec.paper.paper_id
            break
        if rec.paper_id:
            paper_id = rec.paper_id
            break
    n = len(records)
    pairs = "pair" if n == 1 else "pairs"
    where = f" for `{paper_id}`" if paper_id else ""
    return f"**Scope:** {n} (claim, citation) {pairs}{where}."


def _inconclusive_breakdown(records: list[CitationRecord]) -> list[tuple[str, int]]:
    """Bucket the ``inconclusive`` rows by WHY, so the count isn't misread as
    "bad citations". Inconclusive is an abstention, not a refutation — and most are
    benign (a citation sitting in a table/figure, which is not assessed for relevance
    by design). Returns ``[(reason, count), …]`` in a fixed order, omitting empties.
    """
    labels = {
        "table": "cited in a table/figure — relevance not assessed by design",
        "unresolved": "reference unresolved — no verified paper to judge against",
        "fulltext": "full text checked — the specific claim still wasn't confirmed",
        "abstract": "abstract is on-topic, but the specific claim isn't stated in it",
    }
    counts: Counter = Counter()
    for r in records:
        if _enum_value(r.supports_claim) != SupportsClaim.INCONCLUSIVE.value:
            continue
        if r.in_table:
            counts["table"] += 1
        elif _enum_value(r.exists) != Exists.YES.value:
            counts["unresolved"] += 1
        elif "full text" in (r.notes or ""):
            counts["fulltext"] += 1
        else:
            counts["abstract"] += 1
    return [(labels[k], counts[k]) for k in ("table", "unresolved", "fulltext", "abstract") if counts[k]]


def render_summary(records: list[CitationRecord]) -> str:
    """Render counts + a 'Fix before submission' list of high-severity rows.

    Args:
        records: The verified records.

    Returns:
        A Markdown summary block.
    """
    n = len(records)
    # exists + metadata are REFERENCE-level facts (identical across a reference's
    # rows) — tally them once per unique cite_key, not per (claim, citation) pair, so
    # a paper cited 6× doesn't count its existence 6 times. supports + severity are
    # per-pair (a reference can support one claim and not another), so stay row-level.
    exists_by_ref: dict[str, str] = {}
    meta_refs: set[str] = set()
    for r in records:
        exists_by_ref.setdefault(r.cite_key, _enum_value(r.exists))
        if r.metadata_issues:
            meta_refs.add(r.cite_key)
    refs = len(exists_by_ref)
    ex = Counter(exists_by_ref.values())
    sc = Counter(_enum_value(r.supports_claim) for r in records)
    meta = len(meta_refs)
    high = [r for r in records if _enum_value(r.severity) == Severity.HIGH.value]

    # Side-by-side stats: reference existence (left) and claim relevance (right).
    lines = [
        "## Summary",
        "",
        f"**{n}** (claim, citation) pairs over **{refs}** unique references · "
        f"**{meta}** with metadata issues · **{len(high)}** high-severity.",
        "",
        "| References — Exists? | n |  | Claims — Supports? | n |",
        "|:---|---:|:-:|:---|---:|",
        f"| yes | {ex.get(Exists.YES.value, 0)} |  | supports | {sc.get(SupportsClaim.SUPPORTS.value, 0)} |",
        f"| unresolved | {ex.get(Exists.UNRESOLVED.value, 0)} |  | partial | {sc.get(SupportsClaim.PARTIAL.value, 0)} |",
        f"| no (fabricated) | {ex.get(Exists.NO.value, 0)} |  | does not | {sc.get(SupportsClaim.DOES_NOT.value, 0)} |",
        f"| **total refs** | **{refs}** |  | inconclusive | {sc.get(SupportsClaim.INCONCLUSIVE.value, 0)} |",
    ]
    breakdown = _inconclusive_breakdown(records)
    if breakdown:
        lines += [
            "",
            '*"inconclusive" = relevance could not be determined (an abstention, not a '
            "refutation) — by cause:*",
            *[f"- {label}: **{cnt}**" for label, cnt in breakdown],
        ]
    if high:
        lines += ["", "### Fix before submission"]
        for r in high:
            lines.append(f"- `{r.cite_key}` — {_cell(_citation_cell(r))}: {_severity_cell(r)}")
    return "\n".join(lines)


def _usage_footer(result: VerificationResult) -> str:
    """Render the per-run token/cost accounting footer."""
    u = result.usage
    lines = [
        "## Run",
        "",
        f"- Backend: `{result.backend}`"
        + (f" · model `{u.model}`" if u.model else ""),
        f"- Tokens: in {u.input_tokens:,} / out {u.output_tokens:,} "
        f"(total {u.total_tokens:,})",
        f"- Cost: ${u.cost_usd:.4f} · turns {u.num_turns} · tool calls {u.tool_calls}",
    ]
    if u.wall_seconds:
        lines.append(f"- Wall time: {u.wall_seconds:.1f}s")
    if result.errors:
        lines += ["", "### Degraded pairs", *[f"- {_cell(e)}" for e in result.errors]]
    return "\n".join(lines)


def render_report(result: VerificationResult) -> str:
    """Render a full Markdown report: scope + table + summary + usage footer.

    Args:
        result: The :class:`VerificationResult` from the orchestrator.

    Returns:
        The complete Markdown report string.
    """
    records = list(result.records)
    return "\n\n".join(
        [
            f"# Citation verification — `{result.paper_id}`",
            render_table(records),
            render_summary(records),
            _usage_footer(result),
        ]
    )


def to_json(records: list[CitationRecord], path: str | Path | None = None) -> str:
    """Serialize records to JSONL (one record per line).

    Args:
        records: The records to serialize.
        path: Optional destination; when given, the JSONL is written there
            (parent dirs created) in addition to being returned.

    Returns:
        The JSONL string.
    """
    lines = [
        json.dumps(r.model_dump(mode="json"), ensure_ascii=False) for r in records
    ]
    text = "\n".join(lines) + ("\n" if lines else "")
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return text


def from_json(path: str | Path) -> list[CitationRecord]:
    """Load records from a JSONL file produced by :func:`to_json`.

    Blank lines are skipped. Each non-blank line must be a valid record object.

    Args:
        path: Path to the JSONL file.

    Returns:
        The parsed list of :class:`CitationRecord`.
    """
    out: list[CitationRecord] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(CitationRecord.model_validate_json(line))
    return out
