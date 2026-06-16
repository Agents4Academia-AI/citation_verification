"""
run_eval.py — score agent citation-verification output against gold.

The scoring harness lives on one side of a hard boundary: it imports ONLY the
frozen contract (``citation_verifier.schema`` for the enums + the JSON Schema at
``spec/v0.1/record.schema.json``) and ``jsonschema`` for validation. It reads the
agent's output as JSON files on disk and never imports the orchestrator,
backends, stages, or grounding layer (anti-circularity — see ``evals/README.md``
and the enforcing test ``tests/test_eval.py``).

Pipeline
--------
1. :func:`load_records` reads a ``.jsonl`` (or ``.json`` array / agent
   ``report.json``) and validates every record against the committed JSON Schema.
2. :func:`join` aligns predictions and gold on ``(paper_id, claim_id, cite_key)``.
3. :func:`run_eval` runs every axis in :mod:`evals.metrics` and returns a metrics
   dict whose headline is ``correctness_f1``.
4. :func:`main` is the CLI: ``run_eval <agent_dir_or_file> <gold> [--json]``.

Import-safety: pydantic is OPTIONAL here. When ``citation_verifier.schema`` (and
thus pydantic) is importable, :func:`load_records` returns typed ``CitationRecord``
objects; otherwise it returns the schema-validated ``dict``\\s unchanged. Either
way the records are validated against the same JSON Schema and the metrics treat
them uniformly, so the harness runs with no network and no SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

# jsonschema is a HARD dependency of the scoring boundary (validation gate).
from jsonschema import Draft202012Validator

# Type-only import of the contract record; never required at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from citation_verifier.schema import CitationRecord

# Public alias: a "record" is either a typed CitationRecord (pydantic present) or
# a schema-validated dict (offline floor). Metrics consume both uniformly.
Record = Any
JoinPair = tuple["Record | None", "Record"]


# ───────────────────────────────────────────────────────────────
# Schema location + validator (the committed contract)
# ───────────────────────────────────────────────────────────────
def _repo_root() -> Path:
    """Locate the repo root (the dir containing ``spec/``) from this file."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "spec").is_dir():
            return parent
    # Fallback: two levels up (evals/run_eval.py -> repo root).
    return here.parents[1]


def schema_path() -> Path:
    """Absolute path to the committed ``spec/v0.1/record.schema.json``."""
    return _repo_root() / "spec" / "v0.1" / "record.schema.json"


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    """A cached JSON-Schema validator built from the committed spec."""
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@lru_cache(maxsize=1)
def _record_model() -> type | None:
    """Return the pydantic ``CitationRecord`` class, or ``None`` if unavailable.

    Importing ``citation_verifier.schema`` pulls pydantic; that is fine when
    present (typed records) and gracefully skipped when not (dict records). We
    import ONLY the schema module — never any runtime/agent module.
    """
    try:
        from citation_verifier.schema import CitationRecord  # noqa: PLC0415

        return CitationRecord
    except Exception:  # pragma: no cover - depends on env (pydantic install)
        return None


# ───────────────────────────────────────────────────────────────
# Loading + validation
# ───────────────────────────────────────────────────────────────
def _iter_raw_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield raw record dicts from a ``.jsonl`` file or a ``.json`` document.

    Accepts:
      * JSONL — one record per non-blank line.
      * A JSON array of records.
      * An agent ``report.json`` object carrying a ``"records"`` array (the
        orchestrator's on-disk shape).
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"{path}:{lineno}: invalid JSON line: {exc}") from exc
            yield obj
        return
    # .json (array, single object, or {"records": [...]})
    doc = json.loads(text)
    if isinstance(doc, dict) and "records" in doc:
        doc = doc["records"]
    if isinstance(doc, dict):
        doc = [doc]
    if not isinstance(doc, list):
        raise ValueError(f"{path}: expected a JSON array or {{'records': [...]}}")
    yield from doc


def _validate(obj: dict[str, Any], *, where: str) -> None:
    """Validate one raw record against the committed JSON Schema, or raise."""
    errors = sorted(_validator().iter_errors(obj), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        loc = "/".join(str(p) for p in first.path) or "<root>"
        raise ValueError(f"{where}: schema violation at '{loc}': {first.message}")


def _coerce(obj: dict[str, Any]) -> Record:
    """Turn a validated dict into a typed record when pydantic is available."""
    model = _record_model()
    if model is None:
        return obj
    try:
        return model.model_validate(obj)
    except Exception:  # pragma: no cover - schema already validated; be lenient
        return obj


def load_records(path: str | Path) -> list[Record]:
    """Load + schema-validate every record in ``path`` (``.jsonl`` or ``.json``).

    Each record is validated against ``spec/v0.1/record.schema.json`` via
    ``jsonschema``; an invalid record raises ``ValueError`` naming the file and
    the offending field. Returns typed ``CitationRecord`` objects when pydantic
    is installed, else the validated dicts.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such records file: {path}")
    out: list[Record] = []
    for idx, obj in enumerate(_iter_raw_records(path)):
        if not isinstance(obj, dict):
            raise ValueError(f"{path}[{idx}]: each record must be a JSON object")
        _validate(obj, where=f"{path}[{idx}]")
        out.append(_coerce(obj))
    return out


def load_dir(path: str | Path) -> list[Record]:
    """Load + validate all records under a directory of agent outputs.

    Accepts a single file (delegates to :func:`load_records`) or a directory; in
    the directory case it concatenates every ``*.jsonl``, ``report*.json``, and
    ``*.json`` it finds (skipping the gold file is the caller's job). Useful for
    an ``agent_json_dir`` holding one ``report.json`` per paper.
    """
    path = Path(path)
    if path.is_file():
        return load_records(path)
    if not path.is_dir():
        raise FileNotFoundError(f"no such file or directory: {path}")
    files: list[Path] = []
    files += sorted(path.glob("*.jsonl"))
    files += sorted(p for p in path.glob("*.json") if p.is_file())
    out: list[Record] = []
    for f in files:
        out.extend(load_records(f))
    return out


# ───────────────────────────────────────────────────────────────
# Join on the contract key
# ───────────────────────────────────────────────────────────────
def _key(record: Record) -> tuple[str, str, str]:
    """The frozen join key ``(paper_id, claim_id, cite_key)`` for any record."""
    if isinstance(record, dict):
        return (str(record.get("paper_id", "")), str(record.get("claim_id", "")), str(record.get("cite_key", "")))
    # pydantic CitationRecord exposes a `.key` property.
    key = getattr(record, "key", None)
    if key is not None:
        return tuple(str(k) for k in key)  # type: ignore[return-value]
    return (str(getattr(record, "paper_id", "")), str(getattr(record, "claim_id", "")), str(getattr(record, "cite_key", "")))


def join(pred: list[Record], gold: list[Record]) -> list[JoinPair]:
    """Left-join predictions onto gold on ``(paper_id, claim_id, cite_key)``.

    Gold drives the evaluation: every gold record yields exactly one pair
    ``(pred_or_None, gold)``. A gold pair with no matching prediction yields
    ``(None, gold)`` (scored as a full abstention by the metrics). Predictions
    with no gold counterpart are dropped (nothing to score them against) — but
    counted in the returned report's diagnostics by the caller if needed.
    """
    by_key: dict[tuple[str, str, str], Record] = {}
    for p in pred:
        by_key[_key(p)] = p  # last wins on duplicate keys (one pair per record)
    pairs: list[JoinPair] = []
    for g in gold:
        pairs.append((by_key.get(_key(g)), g))
    return pairs


# ───────────────────────────────────────────────────────────────
# The eval
# ───────────────────────────────────────────────────────────────
def _metrics_module():
    """Import the sibling ``metrics`` module whether imported as a package or run as a script.

    ``python -m pytest evals`` / ``from evals import run_eval`` resolve the
    package import; ``python evals/run_eval.py`` does not put the repo root on
    ``sys.path``, so fall back to a direct module import.
    """
    try:
        from . import metrics as M  # noqa: PLC0415  (package import)

        return M
    except ImportError:  # pragma: no cover - script-invocation fallback
        import importlib  # noqa: PLC0415

        return importlib.import_module("metrics")


def evaluate(pred: list[Record], gold: list[Record]) -> dict[str, Any]:
    """Compute every axis from a loaded prediction + gold set; return metrics."""
    M = _metrics_module()

    pairs = join(pred, gold)

    correctness = M.correctness_prf(pairs)
    relevance = M.relevance_metrics(pairs)
    priority = M.priority_metrics(pairs)
    abstention = M.abstention_metrics(pairs)

    matched = sum(1 for p, _ in pairs if p is not None)
    result: dict[str, Any] = {
        "n_gold": len(gold),
        "n_pred": len(pred),
        "n_matched": matched,
        "n_unmatched_gold": len(pairs) - matched,
        "correctness": correctness,
        "relevance": relevance,
        "priority": priority,
        "abstention": abstention,
        # Flat headline fields for quick programmatic access / CI assertions.
        "correctness_f1": correctness["f1"],
        "correctness_precision": correctness["precision"],
        "correctness_recall": correctness["recall"],
        "relevance_macro_f1": relevance["macro_f1"],
        "priority_accuracy": priority["accuracy"],
        "priority_obligatory_f1": priority["obligatory_f1"],
        "abstention_rate": abstention["abstention_rate"],
    }
    result["headline"] = M.headline(result)
    return result


def run_eval(agent_json_dir: str | Path, gold_path: str | Path) -> dict[str, Any]:
    """Load gold + agent predictions, join, and score. Headline = correctness_f1.

    ``agent_json_dir`` may be a directory of agent outputs (``report*.json`` /
    ``*.jsonl``) or a single prediction file. ``gold_path`` is the gold
    ``.jsonl``. Both sides are validated against the committed JSON Schema. The
    gold file is excluded if it happens to live inside ``agent_json_dir``.
    """
    gold_path = Path(gold_path)
    gold = load_records(gold_path)

    agent_path = Path(agent_json_dir)
    if agent_path.is_dir():
        # Avoid double-counting the gold file when it sits in the same dir.
        gold_resolved = gold_path.resolve()
        files: list[Path] = []
        files += sorted(agent_path.glob("*.jsonl"))
        files += sorted(p for p in agent_path.glob("*.json") if p.is_file())
        pred: list[Record] = []
        for f in files:
            if f.resolve() == gold_resolved:
                continue
            pred.extend(load_records(f))
    else:
        pred = load_records(agent_path)

    return evaluate(pred, gold)


# ───────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────
def _format_human(metrics: dict[str, Any]) -> str:
    """Render a compact human-readable score summary."""
    c = metrics["correctness"]
    r = metrics["relevance"]
    p = metrics["priority"]
    a = metrics["abstention"]
    lines = [
        "citation-verification eval",
        f"  gold={metrics['n_gold']}  pred={metrics['n_pred']}  "
        f"matched={metrics['n_matched']}  unmatched_gold={metrics['n_unmatched_gold']}",
        "",
        f"  HEADLINE correctness-F1 : {metrics['headline']:.3f}",
        f"  correctness  P/R/F1     : {c['precision']:.3f} / {c['recall']:.3f} / {c['f1']:.3f}"
        f"  (tp={c['tp']} fp={c['fp']} fn={c['fn']}, n={c['n']})",
        f"  relevance    macro-F1   : {r['macro_f1']:.3f}  (acc {r['accuracy']:.3f})",
        f"  priority     acc / oblF1: {p['accuracy']:.3f} / {p['obligatory_f1']:.3f}",
        f"  abstention   rate / F1  : {a['abstention_rate']:.3f} / {a['abstention_f1']:.3f}",
    ]
    cg = a.get("calibration_gap")
    if cg is not None:
        lines.append(f"  calibration  gap        : {cg:+.3f}  (conf-acc on committed rows)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``run_eval <agent_dir_or_file> <gold> [--json]``."""
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Score agent citation-verification output against gold.",
    )
    parser.add_argument("agent", help="Agent output: a directory of report*.json/*.jsonl, or one file.")
    parser.add_argument("gold", help="Gold .jsonl (CitationRecords with labels).")
    parser.add_argument("--json", action="store_true", help="Emit the full metrics dict as JSON.")
    args = parser.parse_args(argv)

    try:
        metrics = run_eval(args.agent, args.gold)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        print(_format_human(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
