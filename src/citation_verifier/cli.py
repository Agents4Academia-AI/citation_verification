"""
cli.py — the ``cverify`` command-line entry point.

Usage:
    cverify <source> [--backend agentic|claude_code] [--out PATH] [--format md|json]

``<source>`` is an arXiv id, an arXiv URL, or a local PDF path. The command runs
:func:`citation_verifier.orchestrator.run_verification`, prints the rendered
report (Markdown by default, JSONL with ``--format json``), and leaves the
persisted ``report.json`` / ``run.json`` under ``papers/<paper_id>/``.

Import-safe: importing this module touches neither the SDK nor the network; the
heavy work happens only inside :func:`main`.
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``cverify`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="cverify",
        description="Verify the citations in a paper draft (arXiv id/URL or local PDF).",
    )
    parser.add_argument(
        "source",
        help="arXiv id (e.g. 1706.03762), arXiv URL, or a local .pdf path.",
    )
    parser.add_argument(
        "--backend",
        default="agentic",
        choices=["agentic", "claude_code"],
        help="Verification backend: 'agentic' staged pipeline (default) or "
        "'claude_code' skill-driven Agent-SDK loop.",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="Override the per-paper artifact dir (defaults to papers/<paper_id>/).",
    )
    parser.add_argument(
        "--format",
        default="md",
        choices=["md", "json"],
        help="Output written to stdout: 'md' rendered report (default) or 'json' JSONL records.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any prior report.json and re-run from scratch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Verify only the first N (claim, citation) pairs (0 = all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``cverify`` CLI.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, non-zero on a fatal input error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Imported here (not at module top) to keep `import citation_verifier.cli`
    # cheap and SDK/network-free.
    from .orchestrator import run_verification
    from .render import render_report, to_json

    try:
        result = run_verification(
            args.source,
            backend=args.backend,
            resume=not args.no_resume,
            out_dir=args.out,
            max_citations=args.limit,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        sys.stdout.write(to_json(list(result.records)))
    else:
        print(render_report(result))

    if result.errors:
        for err in result.errors:
            print(f"note: {err}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
