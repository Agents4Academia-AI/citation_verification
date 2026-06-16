"""
backends — the two interchangeable verification backends + their registry.

The product runs the SAME output schema through two backends and compares them
(incl. token usage / cost):

  - ``"agentic"``     — explicit, deterministic staged pipeline
                        (:class:`~citation_verifier.backends.agentic.AgenticBackend`).
  - ``"claude_code"`` — skill-driven Claude Agent-SDK loop
                        (:class:`~citation_verifier.backends.claude_code.ClaudeCodeBackend`).

Public API
----------
``get_backend(name, *, settings=None) -> VerificationBackend``
    Instantiate a backend by name; unknown name -> ``ValueError`` listing the
    available names.
``REGISTRY: dict[str, type]``
    The frozen name->class map (re-exported from :mod:`.base`).

Importing this package registers both backends but pulls in NEITHER the Claude
Agent SDK nor the network: the SDK is a lazy import inside
``ClaudeCodeBackend.verify`` (see :func:`.claude_code._require_sdk`), and the
stage/grounding siblings are imported lazily inside each backend's ``verify``.
So ``import citation_verifier.backends`` and ``get_backend('agentic')`` succeed
with claude-agent-sdk NOT installed.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import VerificationBackend
from .base import REGISTRY, get, register

# Import the implementations so their @register decorators populate REGISTRY.
# These imports are SDK-free (claude_agent_sdk is imported lazily inside verify).
from .agentic import AgenticBackend  # noqa: E402,F401  (registration side-effect)
from .claude_code import ClaudeCodeBackend  # noqa: E402,F401  (registration side-effect)


def get_backend(name: str, *, settings: Any | None = None) -> VerificationBackend:
    """Return an instance of the backend registered under ``name``.

    Parameters
    ----------
    name:
        ``"agentic"`` or ``"claude_code"``.
    settings:
        Optional ``config.Settings`` forwarded to the backend (model routing,
        cost ceiling, pricing overrides); ignored by backends that don't use it.

    Raises
    ------
    ValueError
        If ``name`` is not a registered backend (message lists the available
        names).
    """
    return get(name, settings=settings)


__all__ = [
    "get_backend",
    "REGISTRY",
    "register",
    "AgenticBackend",
    "ClaudeCodeBackend",
]
