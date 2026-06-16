"""
metrics.py — per-axis scoring for citation-verification predictions vs. gold.

This module is the *scoring boundary*. It depends ONLY on the frozen output
contract: it reads the same closed enums that ``citation_verifier.schema``
defines (``exists``, ``supports_claim``, ``priority``, ``severity``) but treats a
record as a plain mapping so it can score equally whether records arrive as
pydantic ``CitationRecord`` objects or as schema-validated ``dict``\\s loaded
offline. It never imports the orchestrator, backends, stages, or grounding layer
(anti-circularity — enforced by a test).

The unit of scoring is a *pair* ``(pred, gold)`` produced by
:func:`evals.run_eval.join` on the key ``(paper_id, claim_id, cite_key)``. A
prediction may be ``None`` (the agent emitted no row for a gold pair); gold is
always present (we score against gold).

Axes scored (mirrors docs/DECISIONS.md and the Notion plan):

* **Correctness** — binary detection of a *hallucination*: a row whose cited work
  is fabricated (``exists == "no"``) or carries wrong metadata. Positive class =
  hallucination. Reported as precision / recall / F1, **de-duplicated per resolved
  paper** so a reference cited in N claim-sites does not count N times.
* **Relevance** — ``supports_claim`` as a 4-way label, scored as macro-F1 across
  ``{supports, partial, does_not, unverified}``.
* **Priority** — ``obligatory`` vs ``helpful``: accuracy plus F1 on the
  ``obligatory`` positive class.
* **Abstention / calibration** — ``unverified`` is a *first-class* label, not a
  silent gap: we report how often the agent abstains, whether it abstains when it
  should, and a coarse confidence-vs-correctness calibration gap.

A missing prediction (``pred is None``) is scored as a full abstention:
``exists = supports_claim = unverified``, ``priority`` defaulting to ``helpful``,
``confidence = 0.0``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# ── Enum tokens (mirror citation_verifier.schema EXACTLY; do not redefine the
#    pydantic enums — these are the wire/string values the schema serialises to). ──
EXISTS_YES = "yes"
EXISTS_NO = "no"
EXISTS_UNVERIFIED = "unverified"

SUPPORTS_LABELS = ("supports", "partial", "does_not", "unverified")
PRIORITY_LABELS = ("obligatory", "helpful")

Pair = tuple[Any | None, Any]  # (pred|None, gold) — gold always present


# ───────────────────────────────────────────────────────────────
# Record access helpers (work on pydantic models OR plain dicts)
# ───────────────────────────────────────────────────────────────
def _get(record: Any, field: str, default: Any = None) -> Any:
    """Read ``field`` from a record whether it's a Mapping or an attr object."""
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(field, default)
    return getattr(record, field, default)


def _enum_value(value: Any) -> Any:
    """Normalise an enum member to its string value; pass through plain values."""
    return getattr(value, "value", value)


def _axis(record: Any, field: str, default: str) -> str:
    """Return a normalised string label for an enum-valued field."""
    if record is None:
        return default
    raw = _get(record, field, default)
    val = _enum_value(raw)
    return val if isinstance(val, str) else default


def _labels(record: Any) -> Any:
    """Return the nested gold ``labels`` block (Mapping or model), or None."""
    return _get(record, "labels", None)


def _gold_axis(gold: Any, field: str, default: str) -> str:
    """Read a judged axis from gold, preferring the explicit ``labels`` block.

    A gold ``CitationRecord`` may carry its truth either in the nested ``labels``
    sub-model (the canonical place) or, for convenience, on the top-level field.
    Prefer ``labels`` and fall back to the top-level axis.
    """
    labels = _labels(gold)
    if labels is not None:
        lv = _enum_value(_get(labels, field, None))
        if isinstance(lv, str):
            return lv
    return _axis(gold, field, default)


def _metadata_issues(record: Any) -> list[str]:
    issues = _get(record, "metadata_issues", []) or []
    return list(issues)


def _resolved_id(record: Any) -> str | None:
    """A stable id for the *resolved* canonical paper, for de-duplication.

    Prefers DOI, then arXiv id, then a normalised title from the ``resolved``
    block; falls back to the ``cite_key`` when nothing resolved.
    """
    resolved = _get(record, "resolved", None)
    if resolved is not None:
        for key in ("doi", "arxiv_id"):
            v = _get(resolved, key, None)
            if v:
                return f"{key}:{str(v).strip().lower()}"
        title = _get(resolved, "title", None)
        if title:
            return "title:" + " ".join(str(title).lower().split())
    cite = _get(record, "cite_key", None)
    return f"cite:{cite}" if cite else None


# ───────────────────────────────────────────────────────────────
# Correctness: is this row a hallucination? (positive class)
# ───────────────────────────────────────────────────────────────
def is_hallucination_pred(pred: Any) -> bool:
    """Predicted-positive: the agent flagged the row as fabricated or wrong-metadata.

    True when ``exists == "no"`` OR the agent recorded any ``metadata_issues``.
    ``unverified`` is NOT a positive prediction (the agent declined to commit).
    """
    if pred is None:
        return False
    if _axis(pred, "exists", EXISTS_UNVERIFIED) == EXISTS_NO:
        return True
    return bool(_metadata_issues(pred))


def is_hallucination_gold(gold: Any) -> bool:
    """Gold-positive: the reference truly is fabricated or has wrong metadata.

    Uses the explicit ``labels.is_hallucinated`` flag when present; otherwise
    derives it from gold ``exists == "no"`` OR non-empty ``metadata_issues``.
    """
    labels = _labels(gold)
    if labels is not None:
        flag = _get(labels, "is_hallucinated", None)
        if flag is not None:
            return bool(flag)
    if _gold_axis(gold, "exists", EXISTS_UNVERIFIED) == EXISTS_NO:
        return True
    return bool(_metadata_issues(gold))


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision / recall / F1 from a confusion-count triple (0-safe)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def correctness_prf(pairs: Iterable[Pair]) -> dict[str, float]:
    """Precision / recall / F1 for hallucination detection, de-duped per paper.

    Positive class = hallucination (``exists == "no"`` OR wrong metadata). A
    reference resolving to the same canonical paper within one ``paper_id`` is
    counted once: its gold-positive iff ANY of its rows is gold-positive, and its
    predicted-positive iff ANY of its rows is predicted-positive (catching the
    hallucination anywhere is enough; missing it everywhere is the failure).

    Returns precision/recall/f1 plus the tp/fp/fn confusion counts and the number
    of de-duplicated units (``n``).
    """
    # Group rows by (paper_id, resolved-paper-id), OR-reducing the two flags.
    grouped: dict[tuple[str, str], dict[str, bool]] = {}
    for pred, gold in pairs:
        paper_id = str(_get(gold, "paper_id", ""))
        # De-dupe key uses gold's resolved identity (gold is always present and
        # is the anti-circular oracle); fall back to pred's if gold has none.
        rid = _resolved_id(gold) or _resolved_id(pred) or str(_get(gold, "cite_key", ""))
        key = (paper_id, rid)
        bucket = grouped.setdefault(key, {"gold": False, "pred": False})
        bucket["gold"] = bucket["gold"] or is_hallucination_gold(gold)
        bucket["pred"] = bucket["pred"] or is_hallucination_pred(pred)

    tp = fp = fn = 0
    for bucket in grouped.values():
        g, p = bucket["gold"], bucket["pred"]
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
    out = _prf(tp, fp, fn)
    out["n"] = len(grouped)
    return out


# ───────────────────────────────────────────────────────────────
# Relevance: supports_claim macro-F1 (4-way)
# ───────────────────────────────────────────────────────────────
def _macro_f1(
    pairs: Iterable[Pair],
    *,
    pred_axis: str,
    gold_axis: str,
    labels: Sequence[str],
    pred_default: str,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Generic macro-F1 over a closed label set; returns (macro_f1, per_label)."""
    per: dict[str, dict[str, int]] = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in labels}
    for pred, gold in pairs:
        p = _axis(pred, pred_axis, pred_default) if pred is not None else pred_default
        g = _gold_axis(gold, gold_axis, pred_default)
        if g not in per:  # gold label outside the closed set — skip defensively
            continue
        if p == g:
            per[g]["tp"] += 1
        else:
            if p in per:
                per[p]["fp"] += 1
            per[g]["fn"] += 1

    per_label: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for lab in labels:
        c = per[lab]
        prf = _prf(c["tp"], c["fp"], c["fn"])
        per_label[lab] = prf
        f1s.append(prf["f1"])
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return macro, per_label


def supports_macro_f1(pairs: Iterable[Pair]) -> float:
    """Macro-F1 of ``supports_claim`` over {supports, partial, does_not, unverified}."""
    pairs = list(pairs)
    macro, _ = _macro_f1(
        pairs,
        pred_axis="supports_claim",
        gold_axis="supports_claim",
        labels=SUPPORTS_LABELS,
        pred_default=EXISTS_UNVERIFIED,
    )
    return macro


def relevance_metrics(pairs: Iterable[Pair]) -> dict[str, Any]:
    """Full relevance breakdown: macro-F1 plus per-label P/R/F1 and accuracy."""
    pairs = list(pairs)
    macro, per_label = _macro_f1(
        pairs,
        pred_axis="supports_claim",
        gold_axis="supports_claim",
        labels=SUPPORTS_LABELS,
        pred_default=EXISTS_UNVERIFIED,
    )
    correct = total = 0
    for pred, gold in pairs:
        p = _axis(pred, "supports_claim", EXISTS_UNVERIFIED) if pred is not None else EXISTS_UNVERIFIED
        g = _gold_axis(gold, "supports_claim", EXISTS_UNVERIFIED)
        total += 1
        correct += int(p == g)
    return {
        "macro_f1": macro,
        "accuracy": correct / total if total else 0.0,
        "per_label": per_label,
        "n": total,
    }


# ───────────────────────────────────────────────────────────────
# Priority: accuracy + obligatory-F1
# ───────────────────────────────────────────────────────────────
def priority_metrics(pairs: Iterable[Pair]) -> dict[str, float]:
    """Accuracy over {obligatory, helpful} plus F1 on the ``obligatory`` class.

    Obligatory is the consequential class (a wrong obligatory cite is high
    severity), so we report its F1 separately from raw accuracy.
    """
    pairs = list(pairs)
    correct = total = 0
    tp = fp = fn = 0
    for pred, gold in pairs:
        # Priority defaults to 'helpful' (schema default) when unpredicted.
        p = _axis(pred, "priority", "helpful") if pred is not None else "helpful"
        g = _gold_axis(gold, "priority", "helpful")
        total += 1
        correct += int(p == g)
        if g == "obligatory" and p == "obligatory":
            tp += 1
        elif p == "obligatory" and g != "obligatory":
            fp += 1
        elif g == "obligatory" and p != "obligatory":
            fn += 1
    obligatory = _prf(tp, fp, fn)
    return {
        "accuracy": correct / total if total else 0.0,
        "obligatory_f1": obligatory["f1"],
        "obligatory_precision": obligatory["precision"],
        "obligatory_recall": obligatory["recall"],
        "n": total,
    }


# ───────────────────────────────────────────────────────────────
# Abstention / calibration (unverified is a first-class label)
# ───────────────────────────────────────────────────────────────
def _is_abstention(record: Any) -> bool:
    """A row 'abstains' when it leaves either judged axis at ``unverified``."""
    if record is None:
        return True  # a missing prediction is a full abstention
    ex = _axis(record, "exists", EXISTS_UNVERIFIED)
    sc = _axis(record, "supports_claim", EXISTS_UNVERIFIED)
    return ex == EXISTS_UNVERIFIED or sc == EXISTS_UNVERIFIED


def _gold_unverifiable(gold: Any) -> bool:
    """Gold says the row is genuinely unverifiable on at least one judged axis."""
    ex = _gold_axis(gold, "exists", EXISTS_UNVERIFIED)
    sc = _gold_axis(gold, "supports_claim", EXISTS_UNVERIFIED)
    return ex == EXISTS_UNVERIFIED or sc == EXISTS_UNVERIFIED


def abstention_metrics(pairs: Iterable[Pair]) -> dict[str, float]:
    """Abstention rate + abstention P/R against gold-unverifiable, + calibration.

    * ``abstention_rate`` — fraction of rows the agent left ``unverified``.
    * ``abstention_precision/recall/f1`` — treating "agent abstained" as the
      positive prediction and "gold is genuinely unverifiable" as the positive
      label: did the agent abstain *when it should* and commit otherwise?
    * ``calibration_gap`` — mean confidence on the rows the agent COMMITTED to
      (non-abstentions) minus its empirical accuracy on those rows, over the
      judged axes present in gold. A positive gap = overconfident. ``None`` when
      no committed row carries a confidence.
    """
    pairs = list(pairs)
    n = len(pairs)
    abstained = sum(1 for pred, _ in pairs if _is_abstention(pred))

    tp = fp = fn = 0
    conf_sum = 0.0
    conf_count = 0
    committed_correct = 0
    committed_total = 0
    for pred, gold in pairs:
        agent_abstained = _is_abstention(pred)
        should_abstain = _gold_unverifiable(gold)
        if agent_abstained and should_abstain:
            tp += 1
        elif agent_abstained and not should_abstain:
            fp += 1
        elif (not agent_abstained) and should_abstain:
            fn += 1

        if not agent_abstained and pred is not None:
            # Calibration sample: committed row with a confidence value.
            conf = _get(pred, "confidence", None)
            # Correctness on the committed judged axes (exists + supports_claim).
            ex_ok = _axis(pred, "exists", EXISTS_UNVERIFIED) == _gold_axis(gold, "exists", EXISTS_UNVERIFIED)
            sc_ok = _axis(pred, "supports_claim", EXISTS_UNVERIFIED) == _gold_axis(
                gold, "supports_claim", EXISTS_UNVERIFIED
            )
            row_correct = int(ex_ok and sc_ok)
            if conf is not None:
                conf_sum += float(conf)
                conf_count += 1
                committed_total += 1
                committed_correct += row_correct

    abst = _prf(tp, fp, fn)
    mean_conf = conf_sum / conf_count if conf_count else None
    committed_acc = committed_correct / committed_total if committed_total else None
    calibration_gap: float | None
    if mean_conf is None or committed_acc is None:
        calibration_gap = None
    else:
        calibration_gap = mean_conf - committed_acc

    return {
        "abstention_rate": abstained / n if n else 0.0,
        "abstention_precision": abst["precision"],
        "abstention_recall": abst["recall"],
        "abstention_f1": abst["f1"],
        "mean_confidence_committed": mean_conf,
        "accuracy_committed": committed_acc,
        "calibration_gap": calibration_gap,
        "n": n,
    }


# ───────────────────────────────────────────────────────────────
# Headline
# ───────────────────────────────────────────────────────────────
def headline(metrics: Mapping[str, Any]) -> float:
    """The single comparison number: correctness-F1 (per docs/DECISIONS.md)."""
    correctness = metrics.get("correctness", {})
    if isinstance(correctness, Mapping):
        return float(correctness.get("f1", 0.0))
    return 0.0
