"""
base.py — the backend seam: a small ABC + a name->class registry.

A *backend* turns a :class:`~citation_verifier.interfaces.PaperSource` (plus the
record stubs the extractor produced) into a
:class:`~citation_verifier.interfaces.VerificationResult` — filled
``CitationRecord`` rows and a populated ``RunUsage``. Two implementations share
this seam and the frozen output schema:

  - ``"agentic"``     — explicit, deterministic staged pipeline.
  - ``"claude_code"`` — skill-driven Claude Agent-SDK ``query()`` loop.

:class:`BaseBackend` implements the structural
:class:`~citation_verifier.interfaces.VerificationBackend` protocol; subclasses
only supply ``name`` and :meth:`verify`. The registry (:func:`register` /
:func:`get` / :data:`REGISTRY`) is a thin string→class map so the orchestrator
and CLI can select a backend by name without importing either implementation
eagerly (which keeps the SDK out of the import path).

Pure stdlib + the contract. Import-safe without claude-agent-sdk and network.
"""

from __future__ import annotations

import abc
import time

from ..interfaces import (
    PaperSource,
    RunUsage,
    VerificationBackend,
    VerificationResult,
)
from ..schema import CitationRecord


class BaseBackend(abc.ABC):
    """Abstract base for the two verification backends.

    Concrete subclasses set the class attribute ``name`` and implement
    :meth:`verify`. The shared helpers here keep both backends consistent on the
    invariants the rest of the system relies on:

      - every returned record carries the run's ``paper_id``
        (:meth:`_stamp_paper_id`);
      - a wall-clock timer is available for usage accounting (:meth:`_timer`);
      - construction of the result envelope is uniform (:meth:`_empty_result`).
    """

    name: str = "base"

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Verify subclasses declare a concrete ``name`` (fail loud on drift)."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            if not getattr(cls, "name", None) or cls.name == "base":
                raise TypeError(f"{cls.__name__} must set a non-empty class attribute `name`")

    @abc.abstractmethod
    def verify(
        self, source: PaperSource, stubs: list[CitationRecord]
    ) -> VerificationResult:
        """Verify ``stubs`` against ``source`` and return filled records + usage.

        Implementations MUST degrade rather than crash on a single bad pair
        (set ``record.error`` and leave axes ``unverified``), and MUST return a
        :class:`VerificationResult` whose ``records`` validate against
        ``spec/record.schema.json`` and whose ``usage`` is populated.
        """
        raise NotImplementedError

    # ── shared helpers ────────────────────────────────────────────────
    def _empty_result(self, source: PaperSource) -> VerificationResult:
        """Build a fresh result envelope wired to this backend and ``source``."""
        return VerificationResult(
            paper_id=source.paper_id,
            backend=self.name,
            usage=RunUsage(backend=self.name),
            source=source,
        )

    @staticmethod
    def _stamp_paper_id(records: list[CitationRecord], paper_id: str) -> None:
        """Ensure every record carries the run's ``paper_id`` (join-key safety)."""
        for rec in records:
            if not rec.paper_id:
                rec.paper_id = paper_id

    @staticmethod
    def _timer() -> "_Stopwatch":
        """Return a context-manager stopwatch for wall-time accounting."""
        return _Stopwatch()


class _Stopwatch:
    """Tiny context manager recording elapsed wall-clock seconds."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "_Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


# ───────────────────────────────────────────────────────────────
# Registry — name -> backend class
# ───────────────────────────────────────────────────────────────
REGISTRY: dict[str, type[BaseBackend]] = {}


def register(cls: type[BaseBackend]) -> type[BaseBackend]:
    """Class decorator: register ``cls`` under its ``name`` in :data:`REGISTRY`.

    Raises ``ValueError`` on a duplicate/empty name so two backends can never
    silently shadow each other.
    """
    name = getattr(cls, "name", "")
    if not name or name == "base":
        raise ValueError(f"{cls.__name__} has no concrete `name` to register under")
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"backend name {name!r} already registered to {REGISTRY[name].__name__}")
    REGISTRY[name] = cls
    return cls


def get(name: str, *, settings: object | None = None) -> VerificationBackend:
    """Instantiate the backend registered under ``name``.

    Unknown names raise ``ValueError`` listing the available backends. The
    optional ``settings`` (a ``config.Settings``) is forwarded to backends that
    accept it (model routing, cost ceiling, pricing overrides); backends that
    ignore configuration accept it harmlessly.
    """
    cls = REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(REGISTRY)) or "(none registered)"
        raise ValueError(f"unknown backend {name!r}; available: {available}")
    try:
        return cls(settings=settings)  # type: ignore[call-arg]
    except TypeError:
        # Backend __init__ does not take `settings`; construct plainly.
        return cls()


__all__ = ["BaseBackend", "REGISTRY", "register", "get"]
