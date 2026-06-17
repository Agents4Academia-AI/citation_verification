"""
bot/report.py — turn a :class:`VerificationResult` into a Discord message.

A full SKILL.md table (often 100+ rows) does not fit a Discord message, so the
bot posts a compact **embed** (headline + counts + the flagged citations) and
attaches the complete Markdown report as a file. All rendering here is pure (no
network, no LLM); it reuses :mod:`citation_verifier.render` for the attached
report and the enum->human-string maps so the bot never re-spells a verdict.

Scope is the one fact that can mislead a reader, so it is signalled in depth and
is **never** inferred from ``result.errors``. The bot passes an explicit
``is_test`` flag (a bare ``/check`` is a 🧪 *test sample* of the first N
citations; ``full:true`` is the real, whole-paper verdict). On a test run the
embed is forced amber (green is reserved for a clean *whole* paper), a 🧪 banner
always leads the description, counts carry a denominator, and the attached
report gets a 🧪 header + a ``-sample`` filename. The orchestrator's
"limited to first N of M" note is parsed only to recover the display
denominator M — never to decide whether to label.

Discord hard limits respected: embed description <= 4096, each field value
<= 1024, footer <= 2048. Everything is clipped defensively.
"""

from __future__ import annotations

import io
import re

import discord

from ..interfaces import VerificationResult
from ..render import SEVERITY_STR, render_report
from ..schema import CitationRecord, Exists, Severity, SupportsClaim

__all__ = ["build_response"]

# Embed colors (severity-ranked) and the per-record bullet emoji.
_RED = 0xE03131
_AMBER = 0xF59F00
_GREEN = 0x2F9E44
_SEV_EMOJI = {"high": "🔴", "medium": "🟠", "low": "🟡", "ok": "🟢"}
_MAX_FLAGGED = 6
# Stay safely under Discord's non-boosted 10 MB upload limit.
_MAX_ATTACH_BYTES = 8 * 1024 * 1024

# Shared copy: what to do when a report can't be posted (almost always a full run).
_TOO_LARGE = (
    "The full report was too large to post. Re-run **without** `full:true` to get "
    "the quick 🧪 test sample, or fetch the report from the bot logs / CLI."
)


def _ev(value: object) -> str:
    """Return an enum's ``.value`` (records may hold enum members or strings)."""
    return getattr(value, "value", value) if value is not None else ""


def _clip(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars with an ellipsis when truncated."""
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _who_year(rec: CitationRecord) -> str:
    """A compact 'authors (year)' label for one record (no comma-soup title)."""
    cited = rec.cited_as
    authors = cited.authors or []
    if len(authors) > 2:
        who = f"{authors[0]} et al."
    elif authors:
        who = ", ".join(authors)
    else:
        who = cited.title or (cited.raw[:40] if cited.raw else rec.cite_key)
    year = f" ({cited.year})" if cited.year else ""
    return _clip(f"{who}{year}", 80)


def _flag_line(rec: CitationRecord) -> str:
    """One bullet for the 'Flagged citations' field: emoji + plain verdict + cite.

    The full per-citation detail (metadata issues, raw errors) lives in the
    attached ``.md``; the embed bullet stays scannable.
    """
    emoji = _SEV_EMOJI.get(SEVERITY_STR.get(_ev(rec.severity), "ok"), "🟡")
    if _ev(rec.exists) == Exists.NO.value:
        word = "Fabricated"
    elif _ev(rec.supports_claim) == SupportsClaim.DOES_NOT.value:
        word = "Doesn't support"
    elif _ev(rec.exists) == Exists.UNVERIFIED.value:
        word = "Unverified"
    else:
        word = "Flagged"
    return _clip(f"{emoji} `{rec.cite_key}` — {word} — {_who_year(rec)}", 300)


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


def _is_outage(records: list[CitationRecord], errors: list[str]) -> bool:
    """True when the whole run degraded (extraction/backend down), not real findings.

    Primary, string-drift-proof signal: the orchestrator returns exactly one
    ``_degraded_stub`` record on every whole-run degrade path — recognised by its
    sentinel (claim_id == cite_key == "run", exists=unverified). A belt-and-
    suspenders error-string scan catches a future stub-shape change too.
    """
    if (
        len(records) == 1
        and _ev(records[0].exists) == Exists.UNVERIFIED.value
        and records[0].cite_key == "run"
        and records[0].claim_id == "run"
    ):
        return True
    for e in errors:
        el = e.lower()
        if "extract layer unavailable" in el or "extraction failed" in el:
            return True
        if "backend" in el and ("unavailable" in el or "failed" in el):
            return True
    return False


def _denominator(errors: list[str]) -> int | None:
    """Recover M (the paper's total extracted pairs) from the cap note, for display.

    Parsed ONLY to show "n of M"; never used to decide whether to label a run.
    Returns ``None`` when there is no cap note (the small-bibliography case).
    """
    cap_note = next((e for e in errors if e.startswith("limited to first")), None)
    if not cap_note:
        return None
    m = re.search(r"of (\d+) citation pairs", cap_note)
    return int(m.group(1)) if m else None


def build_response(
    result: VerificationResult,
    arxiv_id: str,
    backend: str,
    *,
    is_test: bool = True,
    cached: bool = False,
) -> tuple[discord.Embed, list[discord.File]]:
    """Build the ``(embed, files)`` to post for a finished verification.

    Args:
        result: The orchestrator's :class:`VerificationResult`.
        arxiv_id: The normalized arXiv id that was checked.
        backend: The backend that ran (shown in the footer).
        is_test: Whether this was a 🧪 test sample (a bare ``/check`` — the
            default) rather than a ``full:true`` whole-paper verdict. Drives the
            banner, the forced-amber color, the denominatored counts, and the
            attached report's header/filename.
        cached: Whether the result was served from a prior ``report.json``
            (only possible when ``BOT_USE_CACHE`` is on); marks the footer ♻️.

    Returns:
        A two-tuple of the summary :class:`discord.Embed` and a (0- or 1-element)
        list with the full Markdown report attached as a file.
    """
    records = list(result.records)
    n = len(records)
    outage = _is_outage(records, result.errors)
    M = _denominator(result.errors)

    fab = [r for r in records if _ev(r.exists) == Exists.NO.value]
    unver = [r for r in records if _ev(r.exists) == Exists.UNVERIFIED.value]
    no_support = [r for r in records if _ev(r.supports_claim) == SupportsClaim.DOES_NOT.value]
    high = [r for r in records if _ev(r.severity) == Severity.HIGH.value]

    # ── verdict headline + the severity color a FULL run would use ──
    if outage:
        verdict = (
            "⚠️ Couldn't verify — the pipeline degraded (extraction/backend "
            "unavailable). Nothing was checked; please try again later."
        )
        severity_color = _RED
    elif n == 0:
        verdict = (
            f"📭 No citations were extracted from `{arxiv_id}` — the paper may "
            "have no reference list, or extraction found none."
        )
        severity_color = _AMBER
    elif fab or high:
        severity_color = _RED
        if fab:
            verdict = (
                f"🚨 {len(fab)} of {n} sampled citation(s) fabricated / not found"
                if is_test
                else f"🚨 {len(fab)} fabricated / not-found citation(s)"
            )
        else:
            verdict = (
                f"⚠️ {len(high)} of {n} sampled with high-severity issues"
                if is_test
                else f"⚠️ {len(high)} high-severity issue(s)"
            )
    elif no_support or unver:
        severity_color = _AMBER
        bits = []
        if no_support:
            bits.append(f"{len(no_support)} not supporting their claim")
        if unver:
            bits.append(f"{len(unver)} unverifiable")
        verdict = "🔍 " + ", ".join(bits)
    else:
        severity_color = _GREEN
        verdict = (
            "✅ No hallucinations in this sample"
            if is_test
            else "✅ No citation hallucinations detected"
        )

    # Green is reserved for a clean WHOLE paper: a test run is always amber
    # (so it can never be mistaken for a full verdict), an outage is always red.
    color = _RED if outage else (_AMBER if is_test else severity_color)

    # ── description: 🧪 banner ALWAYS leads a test run; verdict on its own line ──
    if is_test:
        if outage or n == 0:
            banner = (
                "🧪 TEST SAMPLE — partial test run; NOT a full-paper verdict. "
                "Run `full:true` for the whole paper."
            )
        elif M is not None:
            banner = (
                f"🧪 TEST SAMPLE — verified the first {n} of {M} citations. NOT a "
                "full-paper verdict. Run `full:true` for the real verdict."
            )
        else:
            banner = (
                f"🧪 TEST SAMPLE — verified {n} citation(s). NOT a full-paper "
                "verdict; run `full:true` for the whole paper."
            )
        description = f"{banner}\n{verdict}"
    else:
        description = verdict

    embed = discord.Embed(
        title=f"Citation check — {arxiv_id}",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        description=_clip(description, 4096),
        color=color,
    )

    # ── count grid (only when there is something real to count) ──
    if not outage and n > 0:
        if is_test:
            checked = f"{n} of {M} (sample)" if M is not None else f"{n} (sample)"
        else:
            checked = str(n)
        embed.add_field(name="Citations checked", value=checked)
        embed.add_field(name="Fabricated", value=str(len(fab)))
        embed.add_field(name="Doesn't support", value=str(len(no_support)))
        embed.add_field(name="Unverified", value=str(len(unver)))
        embed.add_field(name="High severity", value=str(len(high)))

        flagged = _flagged_records(records)
        if flagged:
            lines = [_flag_line(r) for r in flagged[:_MAX_FLAGGED]]
            more = len(flagged) - _MAX_FLAGGED
            if more > 0:
                lines.append(f"…and {more} more — see the attached report.")
            embed.add_field(
                name="Flagged citations", value=_clip("\n".join(lines), 1024), inline=False
            )

    # ── the attached full report (scope header + -sample/-full filename on a test run) ──
    report_text = render_report(result)
    if is_test:
        if M is not None:
            header = (
                f"> 🧪 **TEST SAMPLE** — this report covers only the first {n} of "
                f"{M} citations in `{arxiv_id}`. It is **not** a full-paper "
                f"verdict. Re-run `/check {arxiv_id} full:true` for the complete "
                "report.\n\n"
            )
        else:
            header = (
                f"> 🧪 **TEST SAMPLE** — this report covers only the first {n} "
                f"citation(s) in `{arxiv_id}`. It is **not** a full-paper verdict. "
                f"Re-run `/check {arxiv_id} full:true` for the complete report.\n\n"
            )
        report_text = header + report_text
    report_bytes = report_text.encode("utf-8")
    attached = len(report_bytes) <= _MAX_ATTACH_BYTES
    if not attached:
        embed.add_field(name="Full report", value=_clip(_TOO_LARGE, 1024), inline=False)

    # ── genuine degradation (per-pair grounding failures), excluding the cap note ──
    other_errors = [e for e in result.errors if not e.startswith("limited to first")]
    if not outage and other_errors:
        k = len(other_errors)
        if attached:
            degraded = f"⚠️ {k} citation(s) couldn't be grounded — see the report tail."
        else:
            degraded = (
                f"⚠️ {k} citation(s) couldn't be grounded (e.g. {other_errors[0]}). "
                "Full detail is in the logs."
            )
        embed.add_field(name="Degraded", value=_clip(degraded, 1024), inline=False)

    # ── single footer: run-economics + scope (♻️ on the rare cache hit) ──
    u = result.usage
    scope = "test sample" if is_test else "full paper"
    if cached:
        footer = (
            f"♻️ CACHED (not re-verified) · {backend} · {scope} · "
            f"{u.total_tokens:,} tok · ${u.cost_usd:.4f} (prior run)"
        )
    else:
        cost_str = f"${u.cost_usd:.4f}" + (" (no LLM)" if u.cost_usd == 0 else "")
        footer = f"{backend} · {scope} · {u.total_tokens:,} tok · {cost_str}"
        if u.wall_seconds:
            footer += f" · {u.wall_seconds:.1f}s"
    embed.set_footer(text=_clip(footer, 2048))

    if not attached:
        return embed, []
    report_file = discord.File(
        io.BytesIO(report_bytes),
        filename=f"citation-report-{arxiv_id}-{'sample' if is_test else 'full'}.md",
    )
    return embed, [report_file]
