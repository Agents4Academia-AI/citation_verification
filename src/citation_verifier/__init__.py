"""
citation_verifier — verify the citations in a paper draft.

Public surface (stable):
    run_verification(source, backend="agentic") -> VerificationResult
    CitationRecord, and the enums / submodels / result dataclasses.

`run_verification` is imported lazily so that `import citation_verifier`, the
schema, the renderer and the eval harness all work WITHOUT claude-agent-sdk and
WITHOUT network. The SDK is only touched when you actually call a backend that
needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .interfaces import (
    Candidate,
    Extractor,
    PaperSource,
    Resolver,
    RunUsage,
    StageFn,
    VerificationBackend,
    VerificationResult,
)
from .schema import (
    SCHEMA_VERSION,
    CitationRecord,
    CitedAs,
    Claim,
    Evidence,
    Exists,
    Labels,
    MatchMethod,
    ModelTier,
    Paper,
    Priority,
    Resolved,
    Severity,
    SupportsClaim,
    derive_severity,
    export_json_schema,
    json_schema,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    # entry point (lazy)
    "run_verification",
    # schema
    "CitationRecord",
    "Paper",
    "Claim",
    "CitedAs",
    "Resolved",
    "Evidence",
    "Labels",
    "Exists",
    "SupportsClaim",
    "Priority",
    "Severity",
    "ModelTier",
    "MatchMethod",
    "derive_severity",
    "json_schema",
    "export_json_schema",
    # interfaces
    "PaperSource",
    "Candidate",
    "RunUsage",
    "VerificationResult",
    "Extractor",
    "Resolver",
    "VerificationBackend",
    "StageFn",
]

if TYPE_CHECKING:  # for type-checkers only; not executed at import time
    from .orchestrator import run_verification


def __getattr__(name: str) -> Any:
    """Lazily resolve heavy/optional exports (PEP 562) to keep import cheap & safe."""
    if name == "run_verification":
        from .orchestrator import run_verification

        return run_verification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
