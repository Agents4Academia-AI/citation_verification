"""
schema.py — THE CONTRACT.  [FROZEN — changes require team sign-off]

The single source of truth for one row of the citation-verification table.

`CitationRecord` is keyed by ``(paper_id, claim_id, cite_key)`` and is 1:1 with
the SKILL.md output table — every table column maps to a field here and nothing
in the table exists that is not a field here. The agent emits these records; the
deterministic renderer (`render.py`) turns them into the SKILL.md Markdown table;
and CitationHallucinationBench (`src/chbench/`) mirrors this exact schema as its
gold-label format, so agent output and gold labels agree by construction.

Closed enums mirror SKILL.md EXACTLY (machine tokens; the renderer maps them to
the human strings used in the table):

    exists          : yes | no | unresolved
    supports_claim  : supports | partial | does_not | inconclusive
    priority        : obligatory | helpful
    severity        : high | medium | low | ok

This module is pure pydantic v2 + stdlib. It MUST import and run without
claude-agent-sdk and without any network access.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Bump on any breaking change to this file; stamped onto every record and onto
# the exported JSON Schema under spec/<SCHEMA_VERSION>/.
SCHEMA_VERSION = "0.1"


# ───────────────────────────────────────────────────────────────
# Enums — the frozen value vocabularies (mirror SKILL.md)
# ───────────────────────────────────────────────────────────────
class Exists(str, Enum):
    """Does the cited work exist / could it be resolved? (SKILL.md 'Exists?')"""

    YES = "yes"
    NO = "no"
    UNRESOLVED = "unresolved"


class SupportsClaim(str, Enum):
    """Does the cited work support the specific claim? (SKILL.md 'Supports claim?')

    NOTE the token is ``does_not`` (a valid identifier); the renderer prints the
    SKILL.md table string ``does not``.
    """

    SUPPORTS = "supports"
    PARTIAL = "partial"
    DOES_NOT = "does_not"
    INCONCLUSIVE = "inconclusive"


class Priority(str, Enum):
    """Is the citation obligatory or merely helpful? (SKILL.md 'Priority')"""

    OBLIGATORY = "obligatory"
    HELPFUL = "helpful"


class Severity(str, Enum):
    """Issue severity for the row. (SKILL.md 'Issue / Severity')

    May be judged by the model OR derived deterministically from
    ``(exists, supports_claim, priority)`` via :func:`derive_severity`
    (see docs/DECISIONS.md). The schema permits either; the field records
    whichever was used.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OK = "ok"


class ModelTier(str, Enum):
    """Which routed model produced this record (two-tier routing, logged per record)."""

    BULK = "bulk"  # cheap, high-throughput (correctness) — e.g. Haiku-class
    JUDGE = "judge"  # strong (relevance/justification) — e.g. Opus/Sonnet-class
    NONE = "none"  # produced without an LLM (e.g. extraction stub, deterministic)


class MatchMethod(str, Enum):
    """How the resolver matched the cited reference to a canonical record."""

    DOI = "doi"
    ARXIV = "arxiv"
    FUZZY_TITLE = "fuzzy_title"
    NONE = "none"  # no confident match (drives exists=no / unresolved upstream)


# ───────────────────────────────────────────────────────────────
# Sub-models
# ───────────────────────────────────────────────────────────────
class _Model(BaseModel):
    """Base config: forbid unknown keys so contract drift fails loudly."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class Paper(_Model):
    """The draft under verification (the citing paper)."""

    paper_id: str = Field(..., description="Stable id, e.g. arXiv id '2310.06825' or a content hash.")
    title: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    source_kind: str | None = Field(
        default=None, description="How it was ingested: 'arxiv_latex' | 'arxiv_pdf' | 'pdf'."
    )
    tex_available: bool = Field(
        default=False, description="True if LaTeX e-print (.bbl/.bib + \\cite sites) was used."
    )


class Claim(_Model):
    """A single in-text claim site that carries the citation."""

    claim_id: str = Field(..., description="Deterministic id for the claim site (stable across runs).")
    text: str = Field(default="", description="The sentence/claim the citation is attached to.")
    section: str | None = Field(default=None, description="Section/heading the claim appears in.")
    char_span: tuple[int, int] | None = Field(
        default=None, description="(start, end) offsets in the extracted source text, if known."
    )


class CitedAs(_Model):
    """The reference exactly as written in the draft's bibliography (claimed metadata)."""

    raw: str = Field(default="", description="The full reference string as written.")
    authors: list[str] = Field(default_factory=list)
    title: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, v: Any) -> Any:
        """Accept '2017' / '2017.' / None; reject junk softly by returning None."""
        if v in (None, ""):
            return None
        try:
            return int(str(v)[:4])
        except (TypeError, ValueError):
            return None


class Resolved(_Model):
    """The canonical record the reference resolved to (ground for correctness)."""

    source: str | None = Field(default=None, description="'crossref' | 'arxiv' | 'dblp' | 's2' | 'openalex'.")
    match_method: MatchMethod = MatchMethod.NONE
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    url_valid: bool | None = Field(default=None, description="Result of validating the cited/resolved URL.")
    abstract: str | None = Field(default=None, description="Used as relevance evidence when available.")


class Evidence(_Model):
    """A retrieved snippet/record backing a verdict. NEVER from model memory."""

    kind: str = Field(..., description="'metadata' | 'abstract' | 'web' | 'snippet'.")
    source: str = Field(..., description="Provenance: api name, URL, or DOI/arXiv id.")
    quote: str = Field(default="", description="The exact retrieved text supporting the verdict.")
    url: str | None = None


class Labels(_Model):
    """Ground-truth labels for CitationHallucinationBench. None on agent output.

    Mirrors the judged axes of `CitationRecord` so a gold record is just a
    `CitationRecord` whose verdicts ARE the truth; this nested form lets a single
    record additionally carry gold for join-and-score without a parallel schema.
    """

    exists: Exists | None = None
    supports_claim: SupportsClaim | None = None
    priority: Priority | None = None
    severity: Severity | None = None
    is_hallucinated: bool | None = Field(
        default=None, description="Positive class for correctness P/R/F1 (fabricated/wrong-metadata)."
    )
    provenance: str | None = Field(
        default=None, description="How gold was made (human / different-resolver); for anti-circularity audit."
    )


# ───────────────────────────────────────────────────────────────
# The record
# ───────────────────────────────────────────────────────────────
class CitationRecord(_Model):
    """One verified (claim, citation) pair — one row of the SKILL.md table.

    Primary key: ``(paper_id, claim_id, cite_key)``. This is also the eval/gold
    join key. One pair per record (a reference cited in N places => N records).
    """

    schema_version: str = Field(default=SCHEMA_VERSION)

    # ── key ──
    paper_id: str
    claim_id: str
    cite_key: str = Field(..., description="BibTeX key / \\cite key, e.g. 'vaswani2017attention'.")

    # ── context (table cols 2 & 3) ──
    paper: Paper | None = None
    claim: Claim
    cited_as: CitedAs

    # ── correctness (table cols 4 & 5) ──
    exists: Exists = Exists.UNRESOLVED
    resolved: Resolved | None = None
    metadata_issues: list[str] = Field(default_factory=list)

    # ── relevance & priority (table cols 6 & 7) ──
    supports_claim: SupportsClaim = SupportsClaim.INCONCLUSIVE
    priority: Priority = Priority.HELPFUL

    # ── severity (table col 8) ──
    severity: Severity = Severity.OK

    # ── comparison objectiveness (STEP 3 — reserved seam; n/a for MVP) ──
    compared_against: bool | None = Field(
        default=None,
        description="For obligatory baselines: was it actually compared against? None=n/a/unchecked.",
    )

    # ── provenance / accounting ──
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_tier: ModelTier = ModelTier.NONE
    notes: str | None = None
    error: str | None = Field(default=None, description="Set when a pair degraded instead of crashing.")

    # ── gold (None on agent output; populated only in chbench/eval) ──
    labels: Labels | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """The join key shared with the eval harness and gold labels."""
        return (self.paper_id, self.claim_id, self.cite_key)


# ───────────────────────────────────────────────────────────────
# Deterministic severity derivation (see docs/DECISIONS.md)
# ───────────────────────────────────────────────────────────────
def derive_severity(
    exists: Exists | str,
    supports_claim: SupportsClaim | str,
    priority: Priority | str,
) -> Severity:
    """Map the three judged axes to a severity (reproducible, agent/gold-agreeing).

    Rules (highest wins):
      - fabricated (exists=no)                                  -> high
      - obligatory cite that does_not support                  -> high
      - obligatory cite with metadata/partial trouble          -> medium
      - helpful cite that is wrong/partial, or any unresolved/inconclusive  -> low
      - everything clean                                       -> ok
    """
    ex = Exists(exists)
    sc = SupportsClaim(supports_claim)
    pr = Priority(priority)

    if ex is Exists.NO:
        return Severity.HIGH
    if pr is Priority.OBLIGATORY and sc is SupportsClaim.DOES_NOT:
        return Severity.HIGH
    if pr is Priority.OBLIGATORY and sc is SupportsClaim.PARTIAL:
        return Severity.MEDIUM
    if sc is SupportsClaim.DOES_NOT:  # helpful cite that does not support
        return Severity.LOW
    if ex is Exists.UNRESOLVED or sc is SupportsClaim.INCONCLUSIVE:
        return Severity.LOW
    if sc is SupportsClaim.PARTIAL:  # helpful + partial
        return Severity.LOW
    return Severity.OK


# ───────────────────────────────────────────────────────────────
# JSON Schema export (committed to spec/<version>/record.schema.json)
# ───────────────────────────────────────────────────────────────
def json_schema() -> dict[str, Any]:
    """Return the language-agnostic JSON Schema for a single CitationRecord."""
    schema = CitationRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://agents4academia/citation_verifier/spec/{SCHEMA_VERSION}/record.schema.json"
    schema["title"] = "CitationRecord"
    schema["x-schema-version"] = SCHEMA_VERSION
    return schema


def export_json_schema(path: str | Path | None = None) -> Path:
    """Write the JSON Schema to ``spec/<version>/record.schema.json`` (or ``path``).

    Run as ``python -m citation_verifier.schema`` to regenerate the committed spec.
    """
    if path is None:
        root = Path(__file__).resolve().parents[2]  # repo root (src/<pkg>/schema.py)
        path = root / "spec" / f"v{SCHEMA_VERSION}" / "record.schema.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    out = export_json_schema()
    print(f"wrote {out}")
