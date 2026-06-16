"""Module entry point: ``python -m citation_verifier.bot``."""

from __future__ import annotations

from .discord_bot import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
