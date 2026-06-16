"""
chbench — CitationHallucinationBench dataset tooling (owner: phy).

The DATASET pipeline for the citation_verification project. It builds the gold
benchmark that the agent is scored against: seed natural hallucination labels
from the GPTZero NeurIPS-2025 / ICLR-2026 lists, harvest the cited papers,
parse references + claim sites, resolve them with a DIFFERENT resolver than the
agent (anti-circularity), and emit gold :class:`citation_verifier.schema.CitationRecord`
objects whose ``labels`` field IS the truth — mirroring the agent's output
schema 1:1 so agent output and gold agree by construction.

Stages (each a clear, typed, resumable function; see the matching module):
    sources  -> seed descriptors (gptzero lists, openreview)        sources.py
    harvest  -> fetch PDFs / arXiv e-print sources                  harvest.py
    parse    -> references + claim sites (record key fields)        parse.py
    resolve  -> canonical metadata via GoldResolver                 resolve.py
    label    -> gold CitationRecord(labels=...)                     label.py
    inject   -> synthetic fabrication / metadata perturbation       inject.py
    build    -> smoke + full jsonl splits                           build_splits.py
    validate -> jsonschema against spec/v0.1/record.schema.json     validate.py

Run via the ``chbench`` console script (see cli.py) or ``python -m chbench.cli``.

Import-safe and offline: importing this package and any submodule pulls in only
pydantic + stdlib (+ optional pyyaml for config). Network and any LLM are lazy
and fail-soft; nothing here imports claude-agent-sdk.

Anti-circularity (decisions-phy.md): :mod:`chbench.resolve` MUST NOT import
:mod:`citation_verifier.grounding` and MUST NOT reuse the agent's judge model.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
