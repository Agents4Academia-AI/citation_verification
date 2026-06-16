"""
bot — a thin Discord front-end for the citation verifier.

A single slash command, ``/check <paper>``, pipes an arXiv reference through the
same :func:`citation_verifier.orchestrator.run_verification` the CLI uses and
posts the hallucination report back to the channel. The bot is a *front-end
only*: it owns no verification logic, depends only on the package's public
surface (``ingest.parse_arxiv_id``, ``orchestrator.run_verification``,
``render``), and degrades to a clear message on any failure.

Import-safe: importing this package does NOT import ``discord`` or touch the
network — the SDK and the gateway are reached only inside :func:`main` / the
runtime client. So ``import citation_verifier.bot`` works with discord.py absent
(the actionable error is raised only when you actually run the bot).

Run it::

    cverify-bot                       # console script (after `pip install -e '.[bot]'`)
    python -m citation_verifier.bot   # module form

Configuration (env or .env): ``DISCORD_BOT_TOKEN`` (required),
``DISCORD_GUILD_ID`` (optional; instant per-guild command sync),
``BOT_DEFAULT_BACKEND`` (``agentic`` default). See :mod:`.config`.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``cverify-bot`` console script.

    Delegates to :func:`citation_verifier.bot.discord_bot.main`. Imported lazily
    so ``import citation_verifier.bot`` stays free of the ``discord`` dependency.
    """
    from .discord_bot import main as _main

    return _main(argv)
