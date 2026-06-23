"""
stages — the explicit verification pipeline (used by the 'agentic' backend).

Each stage is a :class:`citation_verifier.interfaces.StageFn`: it takes one
:class:`CitationRecord`, fills ONLY its own axes, and degrades-not-crashes
(sets ``record.error`` and leaves axes ``unresolved/inconclusive`` rather than raising).

    fill_correctness(record, *, resolver)            -> STEP 1 (exists, resolved, metadata_issues)
    fill_relevance(record, *, resolver, judge=None)  -> STEP 2 (supports_claim, priority, evidence)
    fill_comparison(record, *, paper_source=None)    -> STEP 3 (compared_against) [reserved]

All stages import only the frozen schema/interfaces; LLM/network access is lazy
and optional (inside the injected resolver/judge).
"""

from __future__ import annotations

from .comparison import fill_comparison
from .correctness import fill_correctness
from .relevance import RelevanceJudge, RelevanceVerdict, fill_relevance

__all__ = [
    "fill_correctness",
    "fill_relevance",
    "fill_comparison",
    "RelevanceJudge",
    "RelevanceVerdict",
]
