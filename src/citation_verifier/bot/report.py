"""
bot/report.py — turn a :class:`VerificationResult` into a Discord message.

A full SKILL.md table (often 100+ rows) does not fit a Discord message, so the
bot posts a compact **embed** (headline + counts + the flagged citations) and
attaches the complete Markdown report as a file. All rendering here is pure (no
network, no LLM); it reuses :mod:`citation_verifier.render` for the attached
report and the enum->human-string maps so the bot never re-spells a verdict.

Discord hard limits respected: embed description <= 4096, each field value
<= 1024, footer <= 2048. Everything is clipped defensively.
"""

from __future__ import annotations

import io

import discord

from ..interfaces import VerificationResult
from ..render import (
    EXISTS_STR,
    SEVERITY_STR,
    SUPPORTS_STR,
    render_report,
)
from ..schema import CitationRecord, Exists, Severity, SupportsClaim

__all__ = ["build_response"]

# Embed colors (severity-ranked) and the per-record bullet emoji.
_RED = 0xE03131
_AMBER = 0xF59F00
_GREEN = 0x2F9E44
_SEV_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡", "ok": "🟢"}
_MAX_FLAGGED = 8
# Stay safely under Discord's non-boosted 10 MB upload limit.
_MAX_ATTACH_BYTES = 8 * 1024 * 1024


def _ev(value: object) -> str:
    """Return an enum's ``.value`` (records may hold enum members or strings)."""
    return getattr(value, "value", value) if value is not None else ""


def _clip(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars with an ellipsis when truncated."""
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _short_citation(rec: CitationRecord) -> str:
    """A compact 'authors, short title (year)' label for one record."""
    cited = rec.cited_as
    authors = cited.authors or []
    if len(authors) > 2:
        who = f"{authors[0]} et al."
    elif authors:
        who = ", ".join(authors)
    else:
        who = ""
    title = cited.title or ""
    short_title = title if len(title) <= 50 else title[:47].rstrip() + "…"
    year = f"({cited.year})" if cited.year else ""
    parts = [p for p in (who, short_title, year) if p]
    return ", ".join(parts[:2]) + ((" " + parts[2]) if len(parts) > 2 else "") or (
        cited.raw[:60] if cited.raw else rec.cite_key
    )


def _flag_line(rec: CitationRecord) -> str:
    """One bullet for the 'Flagged citations' field."""
    emoji = _SEV_EMOJI.get(SEVERITY_STR.get(_ev(rec.severity), "ok"), "🟡")
    verdicts = []
    exists = EXISTS_STR.get(_ev(rec.exists), "unverified")
    if exists != "yes":
        verdicts.append(f"exists: {exists}")
    supports = SUPPORTS_STR.get(_ev(rec.supports_claim), "unverified")
    if supports not in ("supports", "unverified"):
        verdicts.append(f"claim: {supports}")
    issue = "; ".join(rec.metadata_issues) or rec.notes or rec.error or ""
    tail = f" — {issue}" if issue else ""
    verdict_str = f" [{', '.join(verdicts)}]" if verdicts else ""
    return _clip(f"{emoji} `{rec.cite_key}` — {_short_citation(rec)}{verdict_str}{tail}", 300)


def _flagged_records(records: list[CitationRecord]) -> list[CitationRecord]:
    """Records worth surfacing, most-severe first, de-duplicated by join key.

    Order: fabricated (exists=no) -> high severity -> doesn't-support -> other
    unverified. The same record never appears twice.
    """
    def bucket(r: CitationRecord) -> int:
        if _ev(r.exists) == Exists.NO.value:
            return 0
        if _ev(r.severity) == Severity.HIGH.value:
            return 1
        if _ev(r.supports_claim) == SupportsClaim.DOES_NOT.value:
            return 2
        if _ev(r.exists) == Exists.UNVERIFIED.value:
            return 3
        return 99

    flagged = [r for r in records if bucket(r) < 99]
    flagged.sort(key=bucket)
    seen: set[tuple[str, str]] = set()
    out: list[CitationRecord] = []
    for r in flagged:
        join = (r.claim_id, r.cite_key)
        if join in seen:
            continue
        seen.add(join)
        out.append(r)
    return out


def build_response(
    result: VerificationResult, arxiv_id: str, backend: str
) -> tuple[discord.Embed, list[discord.File]]:
    """Build the ``(embed, files)`` to post for a finished verification.

    Args:
        result: The orchestrator's :class:`VerificationResult`.
        arxiv_id: The normalized arXiv id that was checked.
        backend: The backend that ran (shown in the embed/footer).

    Returns:
        A two-tuple of the summary :class:`discord.Embed` and a one-element list
        with the full Markdown report attached as a file.
    """
    records = list(result.records)
    n = len(records)
    fab = [r for r in records if _ev(r.exists) == Exists.NO.value]
    unver = [r for r in records if _ev(r.exists) == Exists.UNVERIFIED.value]
    no_support = [r for r in records if _ev(r.supports_claim) == SupportsClaim.DOES_NOT.value]
    high = [r for r in records if _ev(r.severity) == Severity.HIGH.value]

    if fab or high:
        color, headline = _RED, (
            f"🚨 {len(fab)} fabricated / not-found citation(s)"
            if fab
            else f"⚠️ {len(high)} high-severity issue(s)"
        )
    elif no_support or unver:
        color = _AMBER
        bits = []
        if no_support:
            bits.append(f"{len(no_support)} not supporting their claim")
        if unver:
            bits.append(f"{len(unver)} unverifiable")
        headline = "🔍 " + ", ".join(bits)
    elif n == 0:
        color, headline = _AMBER, "No citations were extracted from this paper."
    else:
        color, headline = _GREEN, "✅ No citation hallucinations detected"

    embed = discord.Embed(
        title=f"Citation check — {arxiv_id}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        description=headline,
        color=color,
    )
    embed.add_field(name="Citations checked", value=str(n))
    embed.add_field(name="Fabricated", value=str(len(fab)))
    embed.add_field(name="Unverified", value=str(len(unver)))
    embed.add_field(name="Doesn't support", value=str(len(no_support)))
    embed.add_field(name="High severity", value=str(len(high)))
    embed.add_field(name="Backend", value=f"`{backend}`")

    flagged = _flagged_records(records)
    if flagged:
        lines = [_flag_line(r) for r in flagged[:_MAX_FLAGGED]]
        more = len(flagged) - _MAX_FLAGGED
        if more > 0:
            lines.append(f"…and {more} more — see the attached report.")
        embed.add_field(
            name="Flagged citations", value=_clip("\n".join(lines), 1024), inline=False
        )

    # An explicit, always-shown scope note when a citation cap was applied
    # (so the headline counts aren't misread as a full-paper verdict).
    cap_note = next((e for e in result.errors if e.startswith("limited to first")), None)
    if cap_note:
        embed.add_field(name="Scope", value=_clip(f"⚠️ {cap_note}", 1024), inline=False)

    other_errors = [e for e in result.errors if e is not cap_note]
    if other_errors:
        embed.add_field(
            name="Degraded",
            value=_clip(f"{len(other_errors)} pair(s)/notes — see report tail.", 1024),
            inline=False,
        )

    u = result.usage
    footer = f"{backend} · {u.total_tokens:,} tok · ${u.cost_usd:.4f}"
    if u.wall_seconds:
        footer += f" · {u.wall_seconds:.1f}s"
    embed.set_footer(text=_clip(footer, 2048))

    # Attach the full report, unless it would breach Discord's upload limit.
    report_bytes = render_report(result).encode("utf-8")
    if len(report_bytes) > _MAX_ATTACH_BYTES:
        embed.add_field(
            name="Full report",
            value=_clip(
                f"Too large to attach ({len(report_bytes) // (1024 * 1024)} MB). "
                "Re-run with a smaller `limit`.",
                1024,
            ),
            inline=False,
        )
        return embed, []
    report_file = discord.File(
        io.BytesIO(report_bytes),
        filename=f"citation-report-{arxiv_id}.md",
    )
    return embed, [report_file]
