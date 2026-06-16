"""
conftest.py — shared pytest fixtures for the citation_verifier test suite.

Everything here is OFFLINE: no claude-agent-sdk, no network. Fixtures load the
committed JSONL fixtures, expose handy paths, and provide a tiny monkeypatched
in-memory resolver so stage/orchestrator tests (when those sibling modules
exist) can run deterministically without hitting Crossref/arXiv/etc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citation_verifier.interfaces import Candidate
from citation_verifier.schema import CitationRecord, MatchMethod, Resolved

# ── repo-relative paths ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_RECORDS = FIXTURES_DIR / "sample_records.jsonl"
SMOKE_GOLD = REPO_ROOT / "evals" / "smoke" / "gold.jsonl"
SPEC_FILE = REPO_ROOT / "spec" / "v0.1" / "record.schema.json"


def _load_records(path: Path) -> list[CitationRecord]:
    """Parse a JSONL file of CitationRecords (validates against the schema)."""
    records: list[CitationRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(CitationRecord.model_validate_json(line))
    return records


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def spec_path() -> Path:
    """Absolute path to the committed JSON Schema (spec/v0.1/record.schema.json)."""
    return SPEC_FILE


@pytest.fixture(scope="session")
def sample_records() -> list[CitationRecord]:
    """The fixture records (one per row, spanning every enum value)."""
    return _load_records(SAMPLE_RECORDS)


@pytest.fixture(scope="session")
def smoke_gold() -> list[CitationRecord]:
    """The in-repo smoke gold pairs (each carries a populated ``labels`` block)."""
    return _load_records(SMOKE_GOLD)


@pytest.fixture
def tmp_work_dir(tmp_path: Path) -> Path:
    """A throwaway per-paper artifact directory (papers/<id>/ stand-in)."""
    d = tmp_path / "papers" / "testpaper"
    d.mkdir(parents=True)
    return d


class _StubResolver:
    """Offline, deterministic Resolver for tests.

    Honors the :class:`citation_verifier.interfaces.Resolver` protocol. It looks
    up a tiny in-memory table keyed by ``cite_key`` and never touches the network.
    Unknown keys resolve to ``None`` (drives ``exists=no``/``unverified`` upstream).
    """

    name = "stub"

    def __init__(self, table: dict[str, Resolved] | None = None) -> None:
        self._table = table or {
            "vaswani2017attention": Resolved(
                source="arxiv",
                match_method=MatchMethod.ARXIV,
                match_score=0.99,
                title="Attention Is All You Need",
                authors=["Ashish Vaswani"],
                year=2017,
                venue="NeurIPS",
                arxiv_id="1706.03762",
                url="https://arxiv.org/abs/1706.03762",
                url_valid=True,
                abstract="based solely on attention mechanisms",
            ),
        }

    def resolve(self, cite_key: str, reference: str, /) -> Resolved | None:
        return self._table.get(cite_key)

    def candidates(self, reference: str, /, max_results: int = 4) -> list[Candidate]:
        return []


@pytest.fixture
def stub_resolver() -> _StubResolver:
    """An offline resolver honoring the Resolver protocol (no network)."""
    return _StubResolver()
