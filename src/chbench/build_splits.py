"""
build_splits.py — assemble gold records into committed JSONL splits.

Writes two splits (one record per line, schema-mirrored):

  * ``smoke.jsonl`` — a small (~18-pair) curated subset for CI / fast contract
    regression. Deliberately balanced to include positives the agent must catch:
    fabricated (``exists='no'``) and metadata-error records are prioritized into
    the smoke split so a green smoke run means "schema-valid + non-trivial
    correctness signal" (decisions-phy.md). This file is intended to seed
    ``evals/smoke/gold.jsonl``.
  * ``full.jsonl`` — every gold record (the full benchmark; lives off-repo on
    /scratch per the 30-day-scratch / git-split policy).

JSONL (not a single JSON array) so splits are append-friendly and streamable.
Pure stdlib + the contract schema. Offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from citation_verifier.schema import CitationRecord, Exists


def _is_priority_positive(rec: CitationRecord) -> bool:
    """True for records the smoke split should preferentially include.

    Positives = fabricated (exists=no) or any record carrying metadata issues —
    these exercise the agent's hardest correctness decisions.
    """
    return rec.exists == Exists.NO or bool(rec.metadata_issues)


def _select_smoke(records: list[CitationRecord], smoke_n: int) -> list[CitationRecord]:
    """Pick up to ``smoke_n`` records, front-loading positives for balance.

    Deterministic: positives first (in input order), then negatives, truncated.
    Guarantees at least the available positives are represented when room allows.
    """
    positives = [r for r in records if _is_priority_positive(r)]
    negatives = [r for r in records if not _is_priority_positive(r)]
    ordered = positives + negatives
    return ordered[:smoke_n]


def write_jsonl(records: list[CitationRecord], path: str | Path) -> Path:
    """Write ``records`` as one JSON object per line. Creates parent dirs.

    Returns the written :class:`pathlib.Path`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return out


def read_jsonl(path: str | Path) -> list[CitationRecord]:
    """Read a gold JSONL file back into validated :class:`CitationRecord` objects.

    Raises pydantic ``ValidationError`` on a malformed/out-of-contract line, so
    this doubles as a load-time schema check.
    """
    records: list[CitationRecord] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(CitationRecord.model_validate_json(line))
    return records


def build_splits(
    records: list[CitationRecord],
    out_dir: str | Path,
    smoke_n: int = 18,
) -> dict[str, Path]:
    """Write ``smoke.jsonl`` and ``full.jsonl`` from gold ``records``.

    Args:
        records: gold :class:`CitationRecord` objects (labels populated).
        out_dir: directory to write the splits into (created if missing).
        smoke_n: target size of the smoke split (positives front-loaded).

    Returns:
        Mapping ``{"smoke": Path, "full": Path}`` of the written files. The smoke
        file is suitable to copy to ``evals/smoke/gold.jsonl`` for CI.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke = _select_smoke(records, smoke_n)
    return {
        "smoke": write_jsonl(smoke, out / "smoke.jsonl"),
        "full": write_jsonl(records, out / "full.jsonl"),
    }
