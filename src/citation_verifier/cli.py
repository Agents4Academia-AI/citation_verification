"""
cli.py — the ``cverify`` command-line entry point.

Usage:
    cverify <source> [--backend agentic|claude_code] [--out PATH] [--format md|json]
                     [--tables on|off|only]

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
        "'claude_code' skill-driven, grounded, concurrent judge.",
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
        "--tables",
        default=None,
        choices=["on", "off", "only"],
        help="Related-work comparison tables: 'on' verifies each ✓/✗ cell against the "
        "cited paper alongside the citations (default, unless ENABLE_TABLE_VERIFICATION "
        "is off); 'off' skips them; 'only' verifies the tables and nothing else "
        "(writes tables.json / tables.md, leaves report.json untouched).",
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
        Process exit code: 0 on success, 2 on a fatal input error, 3 when the model
        could not be reached (the report is written, but nothing in it was judged).
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
            tables=args.tables,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.tables == "only":
        # The table findings on their own: one JSON object per table, or the section.
        if args.format == "json":
            import json

            for report in result.tables:
                sys.stdout.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False) + "\n")
        else:
            from .tables import render_table_reports

            print(render_table_reports(result.tables) or "No related-work comparison table found.")
    elif args.format == "json":
        sys.stdout.write(to_json(list(result.records)))
    else:
        print(render_report(result))

    # A model that could not be reached is not a note: the report it leaves behind has
    # every relevance verdict missing, and a script must not take that for a success.
    from .errors import model_unavailable_errors

    down = model_unavailable_errors(result.errors)
    for err in result.errors:
        print(("error: " if err in down else "note: ") + err, file=sys.stderr)

    return 3 if down else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
