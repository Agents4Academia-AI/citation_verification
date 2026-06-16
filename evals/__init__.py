"""
evals — the scoring boundary for citation_verification.

This package scores agent output against gold labels. It is the *only* place that
turns predictions into numbers, and it is deliberately isolated:

* It imports ONLY the frozen contract — the enums from
  ``citation_verifier.schema`` (lazily, for typed records when pydantic is
  present) and the committed JSON Schema at ``spec/v0.1/record.schema.json`` —
  plus ``jsonschema`` for validation.
* It MUST NOT import the orchestrator, backends, stages, ingest, render, or the
  grounding/``paper_lookup`` layer. Reusing the agent's own resolver or judge
  model to score the agent would measure self-agreement, not accuracy
  (anti-circularity; see ``evals/README.md`` and ``tests/test_eval.py``).
* It reads agent output as JSON files on disk and never calls the network or the
  Claude Agent SDK, so ``python -m pytest evals`` and ``run_eval.py`` run fully
  offline.

Public surface::

    from evals import load_records, join, run_eval
    from evals import metrics
"""

from __future__ import annotations

from .run_eval import evaluate, join, load_dir, load_records, run_eval, schema_path

__all__ = [
    "load_records",
    "load_dir",
    "join",
    "evaluate",
    "run_eval",
    "schema_path",
]
