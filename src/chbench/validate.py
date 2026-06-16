"""
validate.py — validate a gold JSONL split against the frozen spec.

Runs JSON Schema validation of every record in a ``.jsonl`` (or ``.json`` array)
file against ``spec/v0.1/record.schema.json`` (the committed, language-agnostic
contract exported from :mod:`citation_verifier.schema`). This is the dataset's
gate: a gold file that does not validate is not allowed into the benchmark.

Beyond pure schema validity it also asserts the dataset-specific invariant that
gold records carry labels (``labels`` present and ``labels.exists`` set), because
a "gold" record with no label is useless for scoring.

``jsonschema`` is a core dependency; falls back to pydantic validation (via the
contract model) if the spec file is missing, so validation still runs. Offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Locate the repo root (…/src/chbench/validate.py -> repo root)."""
    return Path(__file__).resolve().parents[2]


def default_schema_path() -> Path:
    """Path to the committed JSON Schema spec for a CitationRecord."""
    return _repo_root() / "spec" / "v0.1" / "record.schema.json"


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file or a JSON array file. Raises on parse error."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        return list(data)
    records: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {i}: invalid JSON: {exc}") from exc
    return records


def _validate_with_jsonschema(
    records: list[dict[str, Any]], schema_path: Path
) -> list[str] | None:
    """Validate via jsonschema against ``schema_path``. None if jsonschema absent."""
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors: list[str] = []
    for idx, rec in enumerate(records):
        for err in validator.iter_errors(rec):
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            errors.append(f"record {idx} ({loc}): {err.message}")
    return errors


def _validate_with_pydantic(records: list[dict[str, Any]]) -> list[str]:
    """Fallback validation using the contract model directly."""
    from citation_verifier.schema import CitationRecord

    errors: list[str] = []
    for idx, rec in enumerate(records):
        try:
            CitationRecord.model_validate(rec)
        except Exception as exc:  # pydantic ValidationError or similar
            errors.append(f"record {idx}: {exc}")
    return errors


def _validate_labels_present(records: list[dict[str, Any]]) -> list[str]:
    """Gold-specific invariant: every record must carry a populated label."""
    errors: list[str] = []
    for idx, rec in enumerate(records):
        labels = rec.get("labels")
        if not labels:
            errors.append(f"record {idx}: gold record missing 'labels'")
        elif labels.get("exists") is None:
            errors.append(f"record {idx}: gold 'labels.exists' not set")
    return errors


def validate_dataset(
    path: str | Path,
    *,
    schema_path: str | Path | None = None,
    require_labels: bool = True,
) -> list[str]:
    """Validate a gold dataset file; return a list of human-readable errors.

    Args:
        path: a ``.jsonl`` (one record/line) or ``.json`` (array) gold file.
        schema_path: override for the JSON Schema; defaults to
            ``spec/v0.1/record.schema.json``.
        require_labels: also enforce the gold invariant that every record carries
            a populated ``labels`` block (set False to validate plain agent
            output against the same schema).

    Returns:
        An empty list if the file is fully valid (and, when requested, every
        record is labelled); otherwise one message per problem found. Never
        raises for *validation* failures — only for unreadable/parse-broken files.
    """
    p = Path(path)
    if not p.exists():
        return [f"dataset file not found: {p}"]

    try:
        records = _load_records(p)
    except (ValueError, json.JSONDecodeError) as exc:
        return [str(exc)]

    if not records:
        return [f"{p}: contains no records"]

    spec = Path(schema_path) if schema_path else default_schema_path()
    errors: list[str]
    if spec.exists():
        js_errors = _validate_with_jsonschema(records, spec)
        errors = js_errors if js_errors is not None else _validate_with_pydantic(records)
    else:
        errors = [f"spec not found at {spec}; using pydantic fallback", *_validate_with_pydantic(records)]
        # Drop the informational note if pydantic found nothing else.
        if len(errors) == 1:
            errors = _validate_with_pydantic(records)

    if require_labels:
        errors = [*errors, *_validate_labels_present(records)]
    return errors
