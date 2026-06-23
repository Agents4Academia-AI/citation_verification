"""
render.py — deterministic rendering of records to the SKILL.md output.

This module is pure (no LLM, no network). It turns a list of
:class:`~citation_verifier.schema.CitationRecord` into:

  - :func:`render_table`   — the EXACT 8-column SKILL.md Markdown table + scope line,
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
    "Citation (authors, short title, year)",
    "Cited where (the claim)",
    "Exists?",
    "Metadata issues",
    "Supports claim?",
    "Priority",
    "Issue / Severity",
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
    """Render col 2: 'authors, short title, year' from ``cited_as``."""
    cited = record.cited_as
    authors = cited.authors or []
    if len(authors) > 2:
        who = f"{authors[0]} et al."
    else:
        who = ", ".join(authors)
    title = cited.title or ""
    short_title = title if len(title) <= 60 else title[:57].rstrip() + "…"
    year = str(cited.year) if cited.year else ""
    parts = [p for p in (who, short_title, year) if p]
    rendered = ", ".join(parts)
    return rendered or _cell(cited.raw) or record.cite_key


def _claim_cell(record: CitationRecord) -> str:
    """Render col 3: the claim text (with section prefix when known)."""
    claim = record.claim
    text = claim.text or ""
    if claim.section:
        text = f"[{claim.section}] {text}" if text else f"[{claim.section}]"
    return text


def _metadata_cell(record: CitationRecord) -> str:
    """Render col 5: the metadata issues, semicolon-joined."""
    return "; ".join(record.metadata_issues)


def _severity_cell(record: CitationRecord) -> str:
    """Render col 8: severity, optionally prefixed by a short issue note."""
    sev = SEVERITY_STR.get(_enum_value(record.severity), "ok")
    note = record.notes or record.error or ""
    if note:
        return f"{sev} — {_cell(note)}"
    return sev


def render_table(records: list[CitationRecord]) -> str:
    """Render records as the EXACT SKILL.md 8-column table + a scope line.

    Args:
        records: The verified records (one row each).

    Returns:
        A Markdown string: a scope line, a blank line, then the table.
    """
    scope = _scope_line(records)
    lines = [scope, "", TABLE_HEADER, _TABLE_DIVIDER]
    for i, rec in enumerate(records, start=1):
        row = [
            str(i),
            _cell(_citation_cell(rec)),
            _cell(_claim_cell(rec)),
            EXISTS_STR.get(_enum_value(rec.exists), "unresolved"),
            _cell(_metadata_cell(rec)),
            SUPPORTS_STR.get(_enum_value(rec.supports_claim), "inconclusive"),
            PRIORITY_STR.get(_enum_value(rec.priority), "helpful"),
            _severity_cell(rec),
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


def render_summary(records: list[CitationRecord]) -> str:
    """Render counts + a 'Fix before submission' list of high-severity rows.

    Args:
        records: The verified records.

    Returns:
        A Markdown summary block.
    """
    n = len(records)
    exists_no = sum(1 for r in records if _enum_value(r.exists) == Exists.NO.value)
    unresolved = sum(1 for r in records if _enum_value(r.exists) == Exists.UNRESOLVED.value)
    does_not = sum(
        1 for r in records if _enum_value(r.supports_claim) == SupportsClaim.DOES_NOT.value
    )
    high = [r for r in records if _enum_value(r.severity) == Severity.HIGH.value]

    lines = [
        "## Summary",
        "",
        f"- Pairs checked: **{n}**",
        f"- Fabricated / not found (`exists = no`): **{exists_no}**",
        f"- Unresolved: **{unresolved}**",
        f"- Does not support the claim: **{does_not}**",
        f"- High-severity issues: **{len(high)}**",
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
