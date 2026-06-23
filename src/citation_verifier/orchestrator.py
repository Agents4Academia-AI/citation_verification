"""
orchestrator.py — THE runtime spine: ``run_verification``.

The single public entry point (exported lazily as
``citation_verifier.run_verification``). End to end:

  1. ingest    : normalize the input -> :class:`PaperSource` (downloads under papers/<id>/).
  2. extract   : pick the LaTeX or PDF extractor -> record *stubs* (key + claim + cited_as).
  3. backend   : run the chosen :class:`VerificationBackend` ('agentic' or 'claude_code')
                 over the stubs -> filled records + per-run :class:`RunUsage`.
  4. finalize  : where the backend did not set a model-judged severity, derive it
                 deterministically via :func:`schema.derive_severity`.
  5. persist   : write ``papers/<id>/report.json`` and ``papers/<id>/run.json``.

Design rules honored from docs/decisions-phy.md:
  - degrade-not-crash: a per-pair failure becomes one ``unresolved/inconclusive`` row with
    ``record.error`` set and is appended to ``result.errors``; the run continues.
  - resumable: when ``resume`` is True and a prior ``report.json`` exists, its
    records are reused instead of re-running the backend.
  - import-safe: the SDK and the sibling extract/backends packages are imported
    lazily inside the function, so ``import citation_verifier`` (and this module)
    work without the SDK, without network, and even before those siblings exist.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Settings, load_settings
from .interfaces import (
    PaperSource,
    RunUsage,
    VerificationResult,
)
from .schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Exists,
    ModelTier,
    Severity,
    SupportsClaim,
    derive_severity,
)

__all__ = ["run_verification", "finalize_severity"]


def finalize_severity(records: list[CitationRecord]) -> None:
    """Set a deterministic severity on every record that lacks a model-judged one.

    Per docs/DECISIONS.md, severity is derived from
    ``(exists, supports_claim, priority)`` unless the backend explicitly judged
    it. A record is treated as "not model-judged" when its severity is still the
    schema default (``ok``) and its model tier is ``none`` (i.e. produced
    deterministically / by extraction rather than by a relevance judge). This is
    in-place and idempotent.

    Args:
        records: The records to finalize (mutated in place).
    """
    for rec in records:
        sev = rec.severity
        sev_val = getattr(sev, "value", sev)
        tier = getattr(rec.model_tier, "value", rec.model_tier)
        model_set = sev_val != Severity.OK.value and tier == ModelTier.JUDGE.value
        if not model_set:
            rec.severity = derive_severity(rec.exists, rec.supports_claim, rec.priority)


def _append_source_links(records: list[CitationRecord]) -> None:
    """Append the resolved source link to each existing citation's explanation.

    When a reference resolved (``exists == yes``), show *where* it was found in the
    rendered notes so the verdict is auditable. In-place, idempotent, and
    single-line (safe inside the markdown table cell).
    """
    for rec in records:
        if getattr(rec.exists, "value", rec.exists) != Exists.YES.value or rec.resolved is None:
            continue
        r = rec.resolved
        url = (r.url or "").strip()
        if not url and r.doi:
            url = f"https://doi.org/{r.doi}"
        if not url and r.arxiv_id:
            url = f"https://arxiv.org/abs/{r.arxiv_id}"
        if not url or url in (rec.notes or ""):
            continue
        base = (rec.notes or "").rstrip()
        rec.notes = f"{base} · source: {url}" if base else f"source: {url}"


def _degraded_stub(source: PaperSource, error: str) -> CitationRecord:
    """Build a single ``unresolved/inconclusive`` placeholder record when a whole run degrades.

    Used when ingestion succeeds but extraction yields nothing or the backend is
    unavailable — the run still returns a valid (empty-but-honest) record set.
    """
    return CitationRecord(
        paper_id=source.paper_id,
        claim_id="run",
        cite_key="run",
        claim=Claim(claim_id="run", text=""),
        cited_as=CitedAs(),
        exists=Exists.UNRESOLVED,
        supports_claim=SupportsClaim.INCONCLUSIVE,
        error=error,
    )


def _load_resume(work_dir: Path) -> list[CitationRecord] | None:
    """Load previously-written records from ``report.json`` for resumption.

    Returns ``None`` when there is nothing to resume (no file / unreadable /
    empty). The on-disk format is the JSON array written by :func:`_write_report`.
    """
    report = work_dir / "report.json"
    if not report.exists():
        return None
    try:
        raw = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    items = raw.get("records") if isinstance(raw, dict) else raw
    if not items:
        return None
    try:
        return [CitationRecord.model_validate(item) for item in items]
    except Exception:
        return None


def _load_run(work_dir: Path) -> RunUsage | None:
    """Reload the persisted :class:`RunUsage` from ``run.json`` (for resume).

    Returns ``None`` when there is nothing to reload (no file / unreadable /
    malformed). Restoring the real token/cost accounting on a cache hit keeps
    the resumed result honest instead of reporting a zeroed usage — and lets the
    resume path rewrite ``run.json`` with the genuine numbers rather than zeros.
    """
    run = work_dir / "run.json"
    if not run.exists():
        return None
    try:
        raw = json.loads(run.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    u = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(u, dict):
        return None
    try:
        return RunUsage(
            backend=str(u.get("backend", "")),
            model=str(u.get("model", "")),
            input_tokens=int(u.get("input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cache_read_tokens=int(u.get("cache_read_tokens", 0) or 0),
            cache_creation_tokens=int(u.get("cache_creation_tokens", 0) or 0),
            cost_usd=float(u.get("cost_usd", 0.0) or 0.0),
            num_turns=int(u.get("num_turns", 0) or 0),
            tool_calls=int(u.get("tool_calls", 0) or 0),
            wall_seconds=float(u.get("wall_seconds", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        return None


def _write_report(work_dir: Path, result: VerificationResult) -> None:
    """Persist ``report.json`` (records) under the per-paper work dir."""
    payload = {
        "paper_id": result.paper_id,
        "backend": result.backend,
        "records": [r.model_dump(mode="json") for r in result.records],
        "errors": result.errors,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_run(work_dir: Path, result: VerificationResult) -> None:
    """Persist ``run.json`` (RunUsage + provenance) under the per-paper work dir."""
    u = result.usage
    payload = {
        "paper_id": result.paper_id,
        "backend": result.backend,
        "usage": {
            "backend": u.backend,
            "model": u.model,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
            "cache_read_tokens": u.cache_read_tokens,
            "cache_creation_tokens": u.cache_creation_tokens,
            "cost_usd": u.cost_usd,
            "num_turns": u.num_turns,
            "tool_calls": u.tool_calls,
            "wall_seconds": u.wall_seconds,
            "by_tier": {
                tier: {
                    "input_tokens": tu.input_tokens,
                    "output_tokens": tu.output_tokens,
                    "cost_usd": tu.cost_usd,
                }
                for tier, tu in u.by_tier.items()
            },
        },
        "num_records": len(result.records),
        "num_errors": len(result.errors),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _get_extractor(source: PaperSource):
    """Lazily fetch the extractor for a source (sibling 'extract' package).

    Import-safe: if the extract package is not yet present (a teammate's module
    not merged), returns ``None`` so the caller degrades instead of crashing.
    """
    try:
        from .extract import get_extractor  # type: ignore
    except Exception:
        return None
    try:
        return get_extractor(source)
    except Exception:
        return None


def _get_backend(name: str, settings: Settings | None = None):
    """Lazily fetch a backend from the registry (sibling 'backends' package).

    Forwards ``settings`` so model routing / the relevance-judge toggle from
    ``.env`` actually reach the backend. Import-safe: returns ``None`` if the
    backends package (or the named backend) is unavailable, so the orchestrator
    degrades to an honest, empty result.
    """
    try:
        from .backends import get_backend  # type: ignore
    except Exception:
        return None
    try:
        return get_backend(name, settings=settings)
    except Exception:
        return None


def run_verification(
    source: str,
    *,
    backend: str = "agentic",
    settings: Settings | None = None,
    resume: bool = True,
    out_dir: str | Path | None = None,
    max_citations: int = 0,
) -> VerificationResult:
    """Verify the citations in a paper draft and return a :class:`VerificationResult`.

    Args:
        source: arXiv id, arXiv URL, or a local PDF path.
        backend: Which backend to run — ``"agentic"`` (staged pipeline) or
            ``"claude_code"`` (skill-driven, grounded, concurrent judge).
        settings: Resolved settings; loaded from env/.env when omitted.
        resume: Reuse a prior ``papers/<id>/report.json`` if present.
        out_dir: Override for the per-paper artifact dir (defaults to the
            ingested ``work_dir`` under ``settings.papers_dir``).
        max_citations: When > 0, verify only the first N extracted
            ``(claim, citation)`` stubs (keeps a run bounded in time/cost on a
            large bibliography); a note is appended to ``.errors``. 0 = all.

    Returns:
        A :class:`VerificationResult` with filled records, aggregated
        :class:`RunUsage`, and any degradation messages in ``.errors``.
    """
    settings = settings or load_settings()
    started = time.monotonic()

    # 1) Ingest. A genuine bad-input error is allowed to raise; everything past
    #    here is degrade-not-crash.
    from . import ingest as _ingest

    paper_source = _ingest.ingest(source, settings=settings)
    work_dir = Path(out_dir) if out_dir is not None else Path(
        paper_source.work_dir or (Path(settings.papers_dir) / paper_source.paper_id)
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    result = VerificationResult(
        paper_id=paper_source.paper_id,
        backend=backend,
        source=paper_source,
        usage=RunUsage(backend=backend),
    )

    # Resume from a prior report if asked and available.
    if resume:
        prior = _load_resume(work_dir)
        if prior is not None:
            result.records = prior
            finalize_severity(result.records)
            _append_source_links(result.records)
            # Restore the real token/cost accounting so a cached hit reports the
            # genuine usage (and does not overwrite run.json with zeros).
            prior_usage = _load_run(work_dir)
            if prior_usage is not None:
                prior_usage.backend = backend
                result.usage = prior_usage
            result.usage.wall_seconds = time.monotonic() - started
            _write_report(work_dir, result)
            _write_run(work_dir, result)
            return result

    # 2) Extract record stubs.
    stubs: list[CitationRecord] = []
    extractor = _get_extractor(paper_source)
    if extractor is None:
        result.errors.append(
            "extract layer unavailable (sibling 'extract' package not installed); "
            "no record stubs produced"
        )
    else:
        try:
            stubs = list(extractor.extract(paper_source))
        except Exception as exc:  # degrade-not-crash at the extraction boundary
            result.errors.append(f"extraction failed: {exc!r}")
            stubs = []

    # Optional bound: verify only the first N (claim, citation) stubs.
    if max_citations and len(stubs) > max_citations:
        result.errors.append(
            f"limited to first {max_citations} of {len(stubs)} citation pairs "
            "(max_citations); the rest were not verified"
        )
        stubs = stubs[:max_citations]

    # 3) Run the chosen backend over the stubs.
    backend_impl = _get_backend(backend, settings)
    if backend_impl is None:
        result.errors.append(
            f"backend {backend!r} unavailable (sibling 'backends' package not "
            "installed); returning unresolved/inconclusive stubs"
        )
        # Without a backend we still return whatever stubs we have, unresolved/inconclusive.
        result.records = stubs or [
            _degraded_stub(paper_source, "no extractor and no backend available")
        ]
    else:
        try:
            backend_result = backend_impl.verify(paper_source, stubs)
            result.records = list(backend_result.records)
            result.usage.add(backend_result.usage)
            result.usage.model = backend_result.usage.model or result.usage.model
            for tier, tu in backend_result.usage.by_tier.items():
                bucket = result.usage.by_tier.setdefault(tier, RunUsage(backend=backend))
                bucket.add(tu)
            result.errors.extend(backend_result.errors)
        except Exception as exc:  # whole-backend failure degrades, run completes
            result.errors.append(f"backend {backend!r} failed: {exc!r}")
            result.records = stubs or [_degraded_stub(paper_source, repr(exc))]

    # 4) Deterministic severity for any record the backend did not judge.
    finalize_severity(result.records)
    _append_source_links(result.records)

    # 5) Persist artifacts.
    result.usage.wall_seconds = time.monotonic() - started
    _write_report(work_dir, result)
    _write_run(work_dir, result)
    return result
