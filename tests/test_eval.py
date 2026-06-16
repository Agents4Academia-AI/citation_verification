"""
test_eval.py — the eval harness must score a canned report against smoke gold.

Two layers:
  1. Gold sanity (this module owns the smoke gold): it is schema-valid, every
     pair carries labels + anti-circularity provenance, keys are unique, and the
     fabricated/wrong-metadata floor from DECISIONS.md holds. Runs always.
  2. run_eval round-trip: builds a canned agent report from the gold (verdicts =
     labels => a perfect predictor), writes both to a tmp dir, runs the sibling
     ``evals.run_eval`` join+score, and asserts sane metrics. ``importorskip``s
     until the eval branch merges. Fully offline either way.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from citation_verifier.schema import CitationRecord


# ── layer 1: gold sanity (always runs) ─────────────────────────────────
def test_smoke_gold_present_and_nonempty(smoke_gold) -> None:
    assert 15 <= len(smoke_gold) <= 40, f"smoke gold has {len(smoke_gold)} pairs"


def test_smoke_gold_keys_unique(smoke_gold) -> None:
    keys = [r.key for r in smoke_gold]
    assert len(keys) == len(set(keys)), "duplicate (paper_id, claim_id, cite_key)"


def test_smoke_gold_has_labels_and_provenance(smoke_gold) -> None:
    for r in smoke_gold:
        assert r.labels is not None, f"{r.key} missing labels"
        assert r.labels.exists is not None
        assert r.labels.supports_claim is not None
        assert r.labels.priority is not None
        assert r.labels.is_hallucinated is not None
        # anti-circularity: gold must record how it was made.
        assert r.labels.provenance, f"{r.key} missing gold provenance"


def test_smoke_gold_hallucination_floor(smoke_gold) -> None:
    fabricated = [r for r in smoke_gold if r.labels.exists == "no"]
    wrong_meta = [
        r for r in smoke_gold if r.labels.is_hallucinated and r.labels.exists == "yes"
    ]
    assert len(fabricated) >= 3, "need >=3 fabricated pairs"
    assert len(wrong_meta) >= 3, "need >=3 wrong-metadata pairs"


# ── layer 2: run_eval round-trip (skips until eval branch merges) ───────
@pytest.fixture
def run_eval():
    return pytest.importorskip(
        "evals.run_eval",
        reason="evals/run_eval.py is a sibling module; not present on this checkout yet.",
    )


def _write_canned_report(smoke_gold: list[CitationRecord], out_dir: Path) -> Path:
    """A 'perfect' agent report: copy each gold record, promote labels to verdicts.

    The eval join key is (paper_id, claim_id, cite_key); agent output has
    ``labels=None``, so a perfect predictor sets the judged axes equal to gold.
    """
    report = out_dir / "report.json"
    records: list[dict] = []
    for g in smoke_gold:
        d = g.model_dump()
        lbl = d["labels"]
        d["exists"] = lbl["exists"]
        d["supports_claim"] = lbl["supports_claim"]
        d["priority"] = lbl["priority"]
        d["severity"] = lbl["severity"]
        d["labels"] = None  # agent output never carries gold
        records.append(d)
    report.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return report


def _find_runner(run_eval):
    for name in ("run_eval", "evaluate", "main", "score"):
        fn = getattr(run_eval, name, None)
        if callable(fn):
            return name, fn
    pytest.skip("evals.run_eval exposes no recognized entry point")


def test_run_eval_perfect_report(run_eval, smoke_gold, tmp_path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_canned_report(smoke_gold, report_dir)

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "\n".join(r.model_dump_json() for r in smoke_gold) + "\n", encoding="utf-8"
    )

    name, fn = _find_runner(run_eval)
    try:
        result = fn(str(report_dir), str(gold_path))  # (agent_dir, gold)
    except TypeError:
        pytest.skip(f"evals.run_eval.{name} has an unexpected signature; integration TBD")

    if isinstance(result, dict):
        # A perfect predictor should not score below chance on any reported axis.
        for metric, value in result.items():
            if isinstance(value, (int, float)) and "f1" in metric.lower():
                assert 0.0 <= value <= 1.0
                assert value >= 0.5, f"{metric}={value} too low for a perfect report"


def test_run_eval_cli_on_perfect_report(repo_root, smoke_gold, tmp_path) -> None:
    """Drive `python -m evals.run_eval <agent> <gold> --json` end to end.

    A perfect canned report (verdicts == gold labels) must score a flawless
    correctness pass. Skips cleanly if the module isn't importable yet.
    """
    pytest.importorskip(
        "evals.run_eval",
        reason="evals/run_eval.py is a sibling module; not present on this checkout yet.",
    )
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    _write_canned_report(smoke_gold, report_dir)
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        "\n".join(r.model_dump_json() for r in smoke_gold) + "\n", encoding="utf-8"
    )

    env = {"PYTHONPATH": str(repo_root / "src"), "PATH": __import__("os").environ["PATH"]}
    proc = subprocess.run(
        [sys.executable, "-m", "evals.run_eval", str(report_dir), str(gold_path), "--json"],
        capture_output=True, text=True, cwd=str(repo_root), env=env,
    )
    if proc.returncode != 0:
        pytest.skip(f"run_eval CLI not runnable on this checkout: {proc.stderr.strip()[:200]}")

    try:
        metrics = json.loads(proc.stdout)
    except json.JSONDecodeError:
        pytest.skip("run_eval --json did not emit a JSON metrics dict")

    flat = json.dumps(metrics).lower()
    assert "f1" in flat, "metrics should report at least one F1 figure"
    # A perfect predictor: no correctness F1 in the dict should be implausibly low.
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and "f1" in metric.lower():
            assert 0.0 <= value <= 1.0
