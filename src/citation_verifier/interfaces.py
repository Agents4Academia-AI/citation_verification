"""
interfaces.py — THE CONTRACT (seams).  [FROZEN — changes require team sign-off]

Structural typing (PEP 544 ``typing.Protocol``) for the pluggable pieces, plus
the result/usage dataclasses the orchestrator returns. Modules depend ONLY on
these signatures + `schema.CitationRecord`, never on each other's internals.

Pure stdlib + local schema import. Import-safe without claude-agent-sdk and
without network.

The four seams:
  - Extractor          : source -> PaperSource + record stubs        (ingest/extract)
  - Resolver           : a cited reference -> a canonical Resolved    (grounding)
  - VerificationBackend: a PaperSource -> filled records             (backends)
  - Stage fns          : fill_correctness / fill_relevance / fill_comparison (stages)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .schema import CitationRecord, Resolved


# ───────────────────────────────────────────────────────────────
# Ingestion handoff
# ───────────────────────────────────────────────────────────────
@dataclass
class PaperSource:
    """Normalized, on-disk view of an ingested paper. Produced by `ingest`."""

    paper_id: str
    kind: str = "pdf"  # 'arxiv_latex' | 'arxiv_pdf' | 'pdf'
    pdf_path: str | None = None
    tex_dir: str | None = None  # extracted LaTeX e-print directory, if any
    bbl_path: str | None = None
    bib_path: str | None = None
    tex_available: bool = False
    arxiv_id: str | None = None
    title: str | None = None
    work_dir: str | None = None  # papers/<paper_id>/ artifact dir


# ───────────────────────────────────────────────────────────────
# Resolver candidate (grounding layer hands these to matching/stages)
# ───────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    """One canonical metadata candidate from a grounding source (pre-match)."""

    source: str  # 'crossref' | 'arxiv' | 'dblp' | 's2' | 'openalex'
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    abstract: str = ""
    extra: dict = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────
# Usage / cost accounting
# ───────────────────────────────────────────────────────────────
@dataclass
class RunUsage:
    """Per-run token/cost accounting so the two backends can be compared."""

    backend: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    tool_calls: int = 0
    wall_seconds: float = 0.0
    by_tier: dict[str, "RunUsage"] = field(default_factory=dict)  # per-ModelTier breakdown

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, other: "RunUsage") -> None:
        """Accumulate another usage into this one (orchestrator aggregation)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.cost_usd += other.cost_usd
        self.num_turns += other.num_turns
        self.tool_calls += other.tool_calls


@dataclass
class VerificationResult:
    """What `run_verification` returns: records + accounting + provenance."""

    paper_id: str
    backend: str
    records: list[CitationRecord] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    errors: list[str] = field(default_factory=list)
    source: PaperSource | None = None

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


# ───────────────────────────────────────────────────────────────
# Protocols (the pluggable seams)
# ───────────────────────────────────────────────────────────────
@runtime_checkable
class Extractor(Protocol):
    """Turn a PaperSource into record stubs (key + claim + cited_as filled).

    Stubs carry ``paper_id``, ``claim_id``, ``cite_key``, ``claim``, ``cited_as``
    and leave the judged axes at their `unresolved/inconclusive`/default values.
    """

    def extract(self, source: PaperSource) -> list[CitationRecord]: ...


@runtime_checkable
class Resolver(Protocol):
    """Match one cited reference to a canonical record (grounding, never memory)."""

    def resolve(self, cite_key: str, reference: str, /) -> Resolved | None: ...

    def candidates(self, reference: str, /, max_results: int = 4) -> list[Candidate]: ...


@runtime_checkable
class VerificationBackend(Protocol):
    """A full backend: PaperSource -> filled CitationRecords + usage.

    Two implementations share this seam and the output schema:
      - 'claude_code' : skill-driven Agent-SDK loop.
      - 'agentic'     : explicit staged pipeline (extract -> correctness -> relevance).
    """

    name: str

    def verify(self, source: PaperSource, stubs: list[CitationRecord]) -> VerificationResult: ...


# ───────────────────────────────────────────────────────────────
# Stage callable signatures (used by the 'agentic' backend)
# ───────────────────────────────────────────────────────────────
@runtime_checkable
class StageFn(Protocol):
    """One pipeline stage: take a record, return it with more axes filled.

    Stages are pure-ish (degrade-not-crash): on failure they set ``record.error``
    and leave axes at `unresolved/inconclusive` rather than raising. Concrete stages:
      - fill_correctness(record, *, resolver) -> record   (exists, resolved, metadata_issues, evidence)
      - fill_relevance(record, *, resolver)   -> record   (supports_claim, priority, evidence, confidence)
      - fill_comparison(record, ...)          -> record   (compared_against)  [reserved]
    """

    def __call__(self, record: CitationRecord, /, **kwargs) -> CitationRecord: ...
