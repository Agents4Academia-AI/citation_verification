"""
cli.py — the ``chbench`` console entry point.

Drives the CitationHallucinationBench pipeline stage by stage. Each stage is
resumable (reads the previous stage's checkpoint from the data dir, writes its
own) so the whole thing can be run incrementally or as ``chbench all``:

    chbench seeds            # write seeds.json (gptzero + openreview descriptors)
    chbench harvest          # fetch papers -> harvest.json  (+ --fetch to download)
    chbench parse            # parse papers  -> parsed.json
    chbench resolve          # gold-resolve  -> resolved.json (+ --fetch for network)
    chbench label            # build gold    -> gold.jsonl
    chbench inject           # add synthetic positives -> gold.jsonl (in place)
    chbench build            # smoke/full splits from gold.jsonl
    chbench validate         # jsonschema-validate a split
    chbench all              # seeds -> ... -> build (offline by default)

Default data dir comes from ``$CHBENCH_DATA_DIR`` or
``/scratch/datasets/CitationHallucinationBench`` (overridable with ``--data-dir``).
Network is OFF by default (``--fetch`` opts in); everything runs offline so
``chbench --help`` and a dry ``chbench all`` work with no network and no SDK.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from citation_verifier.schema import Exists

from . import __version__, build_splits, harvest, inject, label, parse, sources, validate
from .resolve import GoldResolver

DEFAULT_DATA_DIR = "/scratch/datasets/CitationHallucinationBench"


# ── data-dir + checkpoint helpers ─────────────────────────────────────────────


def resolve_data_dir(arg: str | None) -> Path:
    """Resolve the dataset data dir: ``--data-dir`` > ``$CHBENCH_DATA_DIR`` > default."""
    return Path(arg or os.environ.get("CHBENCH_DATA_DIR") or DEFAULT_DATA_DIR)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ── stages ────────────────────────────────────────────────────────────────────


def stage_seeds(data_dir: Path, *, fetch: bool) -> Path:
    """Collect seed descriptors -> ``seeds.json``."""
    seeds = sources.all_seed_records(fetch=fetch)
    out = sources.write_seeds(seeds, data_dir / "seeds.json")
    print(f"seeds: {len(seeds)} -> {out}")
    return out


def stage_harvest(data_dir: Path, *, fetch: bool) -> Path:
    """Harvest paper artifacts from seeds -> ``harvest.json`` (+ per-paper dirs)."""
    seeds_path = data_dir / "seeds.json"
    seeds = _read_json(seeds_path) if seeds_path.exists() else sources.all_seed_records(fetch=fetch)
    descriptors = harvest.harvest(seeds, data_dir / "papers", fetch=fetch)
    resolved_n = sum(1 for d in descriptors if d["source_kind"] != "unresolved")
    print(f"harvest: {resolved_n}/{len(descriptors)} resolved -> {data_dir / 'harvest.json'}")
    return data_dir / "harvest.json"


def stage_parse(data_dir: Path) -> Path:
    """Parse harvested papers -> ``parsed.json`` (reference+claim-site dicts).

    Two complementary inputs: (a) full-paper parses for harvested papers (the rich
    path), and (b) GPTZero-flagged references promoted directly from seeds (so
    natural-hallucination positives are captured even before/without harvest).
    """
    harvest_path = data_dir / "harvest.json"
    descriptors = _read_json(harvest_path) if harvest_path.exists() else []
    parsed: list[dict[str, Any]] = []
    harvested_pids: set[str] = set()
    for d in descriptors:
        src = d.get("tex_path") or d.get("pdf_path")
        if not src:
            continue
        items = parse.parse_paper(src, paper_id=d["paper_id"])
        for it in items:  # carry seed hint forward for labelling
            it["seed_hint"] = (d.get("seed") or {}).get("hint", {})
        if items:
            harvested_pids.add(d["paper_id"])
        parsed.extend(items)

    # Promote flagged-reference seeds that produced no full parse.
    seeds_path = data_dir / "seeds.json"
    seeds = _read_json(seeds_path) if seeds_path.exists() else []
    for i, seed in enumerate(seeds):
        item = sources.seed_to_parsed(seed, index=i)
        if item and item["paper_id"] not in harvested_pids:
            parsed.append(item)

    out = _write_json(parsed, data_dir / "parsed.json")
    print(f"parse: {len(parsed)} (claim,citation) pairs -> {out}")
    return out


def stage_resolve(data_dir: Path, *, fetch: bool) -> Path:
    """Gold-resolve each parsed reference -> ``resolved.json`` (positionally aligned)."""
    parsed_path = data_dir / "parsed.json"
    parsed = _read_json(parsed_path) if parsed_path.exists() else []
    resolver = GoldResolver(fetch=fetch)
    resolved = [resolver.resolve((p.get("cited_as") or {}).get("raw", "")) for p in parsed]
    out = _write_json(resolved, data_dir / "resolved.json")
    n_hit = sum(1 for r in resolved if r)
    print(f"resolve: {n_hit}/{len(resolved)} matched (fetch={fetch}) -> {out}")
    return out


def stage_label(data_dir: Path) -> Path:
    """Join parsed + resolved into gold records -> ``gold.jsonl``."""
    parsed = _read_json(data_dir / "parsed.json") if (data_dir / "parsed.json").exists() else []
    resolved_path = data_dir / "resolved.json"
    resolved = _read_json(resolved_path) if resolved_path.exists() else [None] * len(parsed)
    records = label.make_gold(parsed, resolved)
    out = build_splits.write_jsonl(records, data_dir / "gold.jsonl")
    print(f"label: {len(records)} gold records -> {out}")
    return out


def stage_inject(data_dir: Path) -> Path:
    """Append synthetic positives (fabrication + metadata perturbation) to gold."""
    gold_path = data_dir / "gold.jsonl"
    records = build_splits.read_jsonl(gold_path) if gold_path.exists() else []
    synthetic = []
    for i, rec in enumerate(records):
        if rec.exists != Exists.NO:  # only corrupt records that are currently clean/real
            synthetic.append(inject.inject_fabrication(rec))
            field = inject.PERTURBABLE_FIELDS[1 + (i % 4)]  # year/venue/title/authors-ish
            synthetic.append(inject.perturb_metadata(rec, field, seed=i))
    combined = records + synthetic
    out = build_splits.write_jsonl(combined, gold_path)
    print(f"inject: +{len(synthetic)} synthetic -> {len(combined)} total -> {out}")
    return out


def stage_build(data_dir: Path, *, smoke_n: int) -> dict[str, Path]:
    """Build smoke + full splits from ``gold.jsonl``."""
    records = build_splits.read_jsonl(data_dir / "gold.jsonl")
    paths = build_splits.build_splits(records, data_dir, smoke_n=smoke_n)
    print(f"build: smoke={paths['smoke']} full={paths['full']}")
    return paths


def stage_validate(data_dir: Path, target: str | None) -> int:
    """Validate a split against the spec; print errors. Returns process exit code."""
    path = Path(target) if target else data_dir / "smoke.jsonl"
    if not path.exists():
        path = data_dir / "gold.jsonl"
    errors = validate.validate_dataset(path)
    if errors:
        print(f"validate: {len(errors)} error(s) in {path}", file=sys.stderr)
        for e in errors[:50]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"validate: OK ({path})")
    return 0


# ── argument parsing ──────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``chbench`` argument parser."""
    p = argparse.ArgumentParser(
        prog="chbench",
        description="CitationHallucinationBench dataset pipeline (owner: phy).",
    )
    p.add_argument("--version", action="version", version=f"chbench {__version__}")
    p.add_argument("--data-dir", default=None, help="dataset dir (or $CHBENCH_DATA_DIR).")
    p.add_argument(
        "--fetch",
        action="store_true",
        help="enable network downloads/queries (default: offline, fail-soft).",
    )
    p.add_argument("--smoke-n", type=int, default=18, help="smoke split size (default 18).")

    sub = p.add_subparsers(dest="stage", required=True, metavar="stage")
    for name, help_text in (
        ("seeds", "collect seed descriptors (gptzero + openreview)"),
        ("harvest", "fetch paper artifacts from seeds"),
        ("parse", "parse papers into reference+claim-site pairs"),
        ("resolve", "gold-resolve references (independent of the agent)"),
        ("label", "assemble gold CitationRecords"),
        ("inject", "append synthetic hallucination positives"),
        ("build", "write smoke + full jsonl splits"),
        ("all", "run seeds -> harvest -> parse -> resolve -> label -> inject -> build"),
    ):
        sub.add_parser(name, help=help_text)
    vp = sub.add_parser("validate", help="jsonschema-validate a split against the spec")
    vp.add_argument("target", nargs="?", default=None, help="path to a .jsonl/.json split")
    return p


def main(argv: list[str] | None = None) -> int:
    """``chbench`` entry point. Returns a process exit code.

    Args:
        argv: argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, non-zero on validation failure or stage error.
    """
    args = build_parser().parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)

    try:
        if args.stage == "seeds":
            stage_seeds(data_dir, fetch=args.fetch)
        elif args.stage == "harvest":
            stage_harvest(data_dir, fetch=args.fetch)
        elif args.stage == "parse":
            stage_parse(data_dir)
        elif args.stage == "resolve":
            stage_resolve(data_dir, fetch=args.fetch)
        elif args.stage == "label":
            stage_label(data_dir)
        elif args.stage == "inject":
            stage_inject(data_dir)
        elif args.stage == "build":
            stage_build(data_dir, smoke_n=args.smoke_n)
        elif args.stage == "validate":
            return stage_validate(data_dir, args.target)
        elif args.stage == "all":
            stage_seeds(data_dir, fetch=args.fetch)
            stage_harvest(data_dir, fetch=args.fetch)
            stage_parse(data_dir)
            stage_resolve(data_dir, fetch=args.fetch)
            stage_label(data_dir)
            stage_inject(data_dir)
            stage_build(data_dir, smoke_n=args.smoke_n)
            return stage_validate(data_dir, None)
    except Exception as exc:  # surface stage errors without a traceback to the user
        print(f"chbench {args.stage}: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
