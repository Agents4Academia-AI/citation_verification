"""
web/ — a minimal browser front-end for citation verification.

A third front-end alongside the CLI (`cverify`) and the Discord bot
(`cverify-bot`): it accepts a PDF upload or an arXiv link, runs the SAME
``run_verification`` pipeline in a worker thread, streams progress to the page
over Server-Sent Events, and renders the report in the browser.

Reuses the core only through its public seams (``run_verification`` +
``render_report``); it adds no verification logic of its own. The heavy web deps
(FastAPI / uvicorn) are an optional ``web`` extra and are imported lazily, so
``import citation_verifier`` stays light and SDK/network-free.
"""

from __future__ import annotations

__all__ = ["create_app", "main"]


def __getattr__(name: str):  # lazy: don't import FastAPI unless the web app is used
    if name in ("create_app", "main"):
        from .app import create_app, main

        return {"create_app": create_app, "main": main}[name]
    raise AttributeError(name)
