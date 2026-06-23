"""
test_schema.py — guard the FROZEN contract (schema.py + spec/record.schema.json).

All offline. These tests fail loudly if anyone drifts the enums, the primary
key, the deterministic severity map, or the committed JSON Schema export.
"""

from __future__ import annotations

import json

import jsonschema
import pytest
from pydantic import ValidationError

from citation_verifier import schema as S
from citation_verifier.schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Exists,
    Priority,
    Severity,
    SupportsClaim,
    derive_severity,
    json_schema,
)

# The enum vocabularies exactly as the SKILL.md table promises (machine tokens).
EXPECTED_ENUMS = {
    "exists": ["yes", "no", "unresolved"],
    "supports_claim": ["supports", "partial", "does_not", "inconclusive"],
    "priority": ["obligatory", "helpful"],
    "severity": ["high", "medium", "low", "ok"],
}


# ── enums mirror SKILL.md ───────────────────────────────────────────────
def test_exists_enum_values() -> None:
    assert [e.value for e in Exists] == EXPECTED_ENUMS["exists"]


def test_supports_claim_enum_values() -> None:
    assert [e.value for e in SupportsClaim] == EXPECTED_ENUMS["supports_claim"]


def test_priority_enum_values() -> None:
    assert [e.value for e in Priority] == EXPECTED_ENUMS["priority"]


def test_severity_enum_values() -> None:
    assert [e.value for e in Severity] == EXPECTED_ENUMS["severity"]


def test_does_not_token_is_underscored() -> None:
    # The token is a valid identifier; the renderer prints the table string "does not".
    assert SupportsClaim.DOES_NOT.value == "does_not"


# ── primary key ─────────────────────────────────────────────────────────
def test_record_key_is_paper_claim_cite() -> None:
    r = CitationRecord(
        paper_id="p1", claim_id="c1", cite_key="k1",
        claim=Claim(claim_id="c1"), cited_as=CitedAs(),
    )
    assert r.key == ("p1", "c1", "k1")


# ── deterministic severity derivation ───────────────────────────────────
@pytest.mark.parametrize(
    ("exists", "supports", "priority", "expected"),
    [
        (Exists.NO, SupportsClaim.INCONCLUSIVE, Priority.HELPFUL, Severity.HIGH),
        (Exists.NO, SupportsClaim.SUPPORTS, Priority.OBLIGATORY, Severity.HIGH),
        (Exists.YES, SupportsClaim.DOES_NOT, Priority.OBLIGATORY, Severity.HIGH),
        (Exists.YES, SupportsClaim.PARTIAL, Priority.OBLIGATORY, Severity.MEDIUM),
        (Exists.YES, SupportsClaim.DOES_NOT, Priority.HELPFUL, Severity.LOW),
        (Exists.UNRESOLVED, SupportsClaim.INCONCLUSIVE, Priority.HELPFUL, Severity.LOW),
        (Exists.YES, SupportsClaim.PARTIAL, Priority.HELPFUL, Severity.LOW),
        (Exists.YES, SupportsClaim.SUPPORTS, Priority.OBLIGATORY, Severity.OK),
        (Exists.YES, SupportsClaim.SUPPORTS, Priority.HELPFUL, Severity.OK),
    ],
)
def test_derive_severity(exists, supports, priority, expected) -> None:
    assert derive_severity(exists, supports, priority) is expected


def test_derive_severity_accepts_raw_strings() -> None:
    assert derive_severity("no", "inconclusive", "obligatory") is Severity.HIGH


# ── round-trip ──────────────────────────────────────────────────────────
def test_record_json_round_trip(sample_records) -> None:
    assert sample_records, "fixture must not be empty"
    for rec in sample_records:
        again = CitationRecord.model_validate_json(rec.model_dump_json())
        assert again.key == rec.key
        assert again.model_dump() == rec.model_dump()


def test_fixture_covers_every_enum_value(sample_records) -> None:
    dumps = [r.model_dump() for r in sample_records]
    for field, values in EXPECTED_ENUMS.items():
        seen = {d[field] for d in dumps}
        assert seen == set(values), f"{field}: fixture covers {seen}, expected {set(values)}"


# ── extra keys are forbidden (contract drift fails loudly) ──────────────
def test_extra_keys_rejected() -> None:
    payload = {
        "paper_id": "p", "claim_id": "c", "cite_key": "k",
        "claim": {"claim_id": "c"}, "cited_as": {}, "bogus_field": 1,
    }
    with pytest.raises(ValidationError):
        CitationRecord.model_validate(payload)


# ── committed spec is byte-identical to the live model export ───────────
def test_committed_spec_matches_model(spec_path) -> None:
    committed = spec_path.read_text(encoding="utf-8")
    generated = json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"
    assert committed == generated, (
        "spec/v0.1/record.schema.json drifted from schema.py — "
        "run `make schema` (python -m citation_verifier.schema) and commit."
    )


def test_spec_version_matches_module() -> None:
    spec = json_schema()
    assert spec["x-schema-version"] == S.SCHEMA_VERSION


# ── every fixture/gold record validates against the JSON Schema ─────────
def test_records_validate_against_json_schema(sample_records, spec_path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(spec)
    for rec in sample_records:
        payload = json.loads(rec.model_dump_json())
        errors = sorted(validator.iter_errors(payload), key=str)
        assert not errors, f"{rec.key}: {[e.message for e in errors]}"
