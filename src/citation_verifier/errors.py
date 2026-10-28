"""
errors.py — failures that must not be mistaken for findings.

The pipeline degrades instead of crashing: a pair that cannot be judged becomes one
``inconclusive`` row and the run goes on. That is right for one stuck pair and wrong for a
model that cannot be reached at all. An expired login made every call fail, every pair
come back "inconclusive", and the run exit 0 with nothing on stderr — a report that reads
as "the evidence settled nothing" when the evidence was never shown to anyone.

So that one condition is named, carried to the top of the report and to the exit code,
and never retried (no retry cures a login).

Pure stdlib; imported by the judge, the table subsystem, the renderers and the CLI.
"""

from __future__ import annotations

__all__ = ["MODEL_UNAVAILABLE", "ModelUnavailable", "unavailable_message", "model_unavailable_errors"]

# Prefix of the run-level error line. Renderers and the CLI key on it.
MODEL_UNAVAILABLE = "MODEL UNAVAILABLE"


class ModelUnavailable(RuntimeError):
    """The model could not be reached AT ALL — expired login, rejected key, no credit.

    Unlike a transient failure, retrying does not help and nothing was judged.
    """


def unavailable_message(reason: str, what: str) -> str:
    """The run-level error line: what was not done, why, and how to fix it."""
    reason = " ".join(str(reason or "no reason given").split())
    return (
        f"{MODEL_UNAVAILABLE}: {what} — the model could not be reached ({reason}). "
        "Log in again (run `claude`, then /login) or set ANTHROPIC_API_KEY, and re-run."
    )


def model_unavailable_errors(errors: list[str] | None) -> list[str]:
    """The run-level errors that say the model could not be reached."""
    return [e for e in (errors or []) if str(e).startswith(MODEL_UNAVAILABLE)]
