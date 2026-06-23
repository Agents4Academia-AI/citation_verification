"""
tests/test_bot_report.py — the /check embed under the test-sample vs full model.

Focus: a bare /check is a 🧪 *test sample* and must be impossible to mistake for
a full-paper verdict (loud banner, forced-amber color, denominatored counts, a
``-sample`` report filename + .md header). A ``full:true`` run is the bare,
real verdict. A whole-run degrade renders as an outage, never as a phantom
"1 unverified". All pure — no gateway, no network, no SDK.
"""

from __future__ import annotations

import pytest

from citation_verifier.bot.config import load_bot_config
from citation_verifier.bot.discord_bot import CitationBot
from citation_verifier.bot.report import _AMBER, _GREEN, _RED, build_response
from citation_verifier.interfaces import RunUsage, VerificationResult
from citation_verifier.schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Exists,
    Severity,
    SupportsClaim,
)

_PID = "2505.03335"


def _rec(
    cite_key: str,
    *,
    exists: Exists = Exists.YES,
    supports: SupportsClaim = SupportsClaim.SUPPORTS,
    severity: Severity = Severity.OK,
) -> CitationRecord:
    return CitationRecord(
        paper_id=_PID,
        claim_id=f"c-{cite_key}",
        cite_key=cite_key,
        claim=Claim(claim_id=f"c-{cite_key}", text="a claim"),
        cited_as=CitedAs(authors=["Smith", "Jones"], title="A Title", year=2021),
        exists=exists,
        supports_claim=supports,
        severity=severity,
    )


def _degraded_stub(error: str) -> CitationRecord:
    """The exact shape orchestrator._degraded_stub returns on a whole-run degrade."""
    return CitationRecord(
        paper_id=_PID,
        claim_id="run",
        cite_key="run",
        claim=Claim(claim_id="run", text=""),
        cited_as=CitedAs(),
        exists=Exists.UNRESOLVED,
        supports_claim=SupportsClaim.INCONCLUSIVE,
        error=error,
    )


def _result(records, *, errors=None, backend="agentic") -> VerificationResult:
    u = RunUsage(backend=backend, input_tokens=600, output_tokens=400, wall_seconds=2.0)
    return VerificationResult(
        paper_id=_PID, backend=backend, records=list(records), usage=u, errors=list(errors or [])
    )


def _field(embed, name):
    return next((f for f in embed.fields if f.name == name), None)


def _attached_text(files) -> str:
    return files[0].fp.getvalue().decode("utf-8")


# ── default = a loudly-labeled test sample ─────────────────────────
def test_default_is_labeled_test_sample():
    embed, files = build_response(_result([_rec("a"), _rec("b"), _rec("c")]), _PID, "agentic")
    assert embed.color.value == _AMBER  # a clean SAMPLE is amber, never green
    lines = embed.description.split("\n")
    assert len(lines) >= 2 and lines[0].startswith("🧪 TEST SAMPLE")
    assert _field(embed, "Citations checked").value.endswith("(sample)")
    assert files[0].filename.endswith("-sample.md")
    assert _attached_text(files).startswith("> 🧪 **TEST SAMPLE**")


def test_test_banner_fires_when_paper_smaller_than_N():
    # 2 records, NO "limited to first" cap note (the <=N blind spot in the old code).
    embed, files = build_response(_result([_rec("a"), _rec("b")]), _PID, "agentic")
    assert embed.description.split("\n")[0].startswith("🧪 TEST SAMPLE")
    assert "2 citation" in embed.description
    assert embed.color.value == _AMBER
    assert files[0].filename.endswith("-sample.md")


def test_test_sample_shows_denominator_from_cap_note():
    note = "limited to first 5 of 142 citation pairs (max_citations); the rest were not verified"
    embed, _ = build_response(_result([_rec(str(i)) for i in range(5)], errors=[note]), _PID, "agentic")
    assert "5 of 142" in embed.description
    assert _field(embed, "Citations checked").value == "5 of 142 (sample)"


def test_fabricated_in_sample_stays_amber_but_screams():
    rec = _rec("ghost", exists=Exists.NO, severity=Severity.HIGH)
    embed, _ = build_response(_result([rec, _rec("ok")]), _PID, "agentic")
    assert embed.color.value == _AMBER  # color is the scope channel on a sample
    assert "🚨" in embed.description  # severity still screams via the headline
    assert _field(embed, "Fabricated").value == "1"


# ── full:true = the real, unbanned verdict ─────────────────────────
def test_full_run_is_unbanned_and_uncapped():
    embed, files = build_response(
        _result([_rec("a"), _rec("b"), _rec("c")]), _PID, "agentic", is_test=False
    )
    assert embed.color.value == _GREEN  # clean WHOLE paper -> green is allowed
    assert "🧪" not in embed.description
    assert _field(embed, "Citations checked").value == "3"
    assert files[0].filename.endswith("-full.md")
    assert not _attached_text(files).startswith("> 🧪")
    assert "full paper" in embed.footer.text


def test_full_fabricated_is_red():
    rec = _rec("ghost", exists=Exists.NO, severity=Severity.HIGH)
    embed, _ = build_response(_result([rec, _rec("ok")]), _PID, "agentic", is_test=False)
    assert embed.color.value == _RED
    assert embed.description.startswith("🚨")


# ── whole-run degrade is an outage, not a phantom "1 unverified" ───
@pytest.mark.parametrize(
    "error",
    [
        "backend 'claude_code' failed: RuntimeError('boom')",
        "backend 'agentic' unavailable (sibling 'backends' package not installed)",
        "extract layer unavailable (sibling 'extract' package not installed)",
        "extraction failed: ValueError('bad tex')",
    ],
)
def test_backend_degrade_renders_outage_not_phantom_unverified(error):
    embed, _ = build_response(_result([_degraded_stub(error)], errors=[error]), _PID, "agentic", is_test=False)
    assert embed.color.value == _RED
    assert embed.description.startswith("⚠️ Couldn't verify")
    assert "🔍" not in embed.description  # not the phantom "unverifiable" headline
    assert _field(embed, "Citations checked") is None  # no count grid on an outage
    assert _field(embed, "Flagged citations") is None  # the lone stub is never flagged


def test_empty_extraction_is_inbox_not_outage_and_still_banners():
    embed, _ = build_response(_result([]), _PID, "agentic")  # n==0, no errors
    lines = embed.description.split("\n")
    assert lines[0].startswith("🧪 TEST SAMPLE")  # scope is unconditional
    assert "📭 No citations were extracted" in embed.description
    assert embed.color.value == _AMBER


# ── cache footer ───────────────────────────────────────────────────
def test_cached_result_marks_footer_and_drops_time():
    embed, _ = build_response(_result([_rec("a")]), _PID, "agentic", is_test=False, cached=True)
    assert embed.footer.text.startswith("♻️ CACHED")
    assert "prior run" in embed.footer.text
    assert "2.0s" not in embed.footer.text  # the resume wall-time is meaningless


# ── the cache dir can never serve a sample as a full verdict ───────
def test_cache_key_test_vs_full_never_collide():
    bot = CitationBot.__new__(CitationBot)
    bot.cfg = load_bot_config(None)
    _key_t, dir_t, max_t = bot._cache_key(_PID, "agentic", False)
    _key_f, dir_f, max_f = bot._cache_key(_PID, "agentic", True)
    assert max_t == bot.cfg.test_limit and max_f == 0
    assert dir_t.name != dir_f.name
    assert f"-test{bot.cfg.test_limit}-" in dir_t.name
    assert "-full-" in dir_f.name
