# Baselines for table-cell citation verification

Refari decides "does cited work R have property D?" from text it *retrieved* from R.
A baseline answers the same question a different way. Everything else is pinned, so
the comparison isolates one variable.

## What is held fixed

| Held fixed | Why |
|---|---|
| The cell set (`out/cells.json`) | Same denominator. A re-extraction that recovers 94 cells instead of 99 makes the numbers incomparable. |
| The grounded column definition | Otherwise the baseline is *two* changes away from Refari, not one. |
| Blinding — the judge never sees the ✓/✗ | Same as Refari, so neither can rubber-stamp the table. |
| The mark-vs-answer decision table | Imported from `tables.verify._DECISION`, never copied. |
| The judge model and transport | Read from the same `config.load_settings()`. |

## Running

```bash
python scripts/baselines/build_manifest.py   # archived Refari run -> out/cells.json
python scripts/baselines/closed_book.py      # the no-retrieval baseline -> out/closed_book.json
python scripts/baselines/summarize.py --latex
python scripts/baselines/score_human.py      # both systems vs. the human labels
```

`build_manifest.py` parses the archived per-cell audit
(`/scratch/datasets/citation_verification_outputs/table_audit_percell.md`) rather than
re-running extraction, precisely so the denominator cannot drift.

## Baselines

- **`closed_book.py`** — no retrieval, no tools. The judge answers from its own
  knowledge of the cited work. This is both the headline baseline (what you get by
  just asking a model) and Refari's retrieval ablation.

## Reading the output

`summarize.py` is the gold-free view: determinacy, behaviour on cells with no
retrievable evidence, agreement, and self-consistency. It says nothing about who is
right, and is still the only view valid on cells nobody has annotated.

`score_human.py` is the correctness view. It reads the frozen annotation snapshot
`$CV_OUT/human_labels_v1.json` (default `/scratch/datasets/citation_verification_outputs/`)
and joins it on `cell_uid`. Three things it is deliberately careful about, each of
which would otherwise inflate a headline number:

| Guard | Why |
|---|---|
| Rows with a blank `checker` are dropped | They carry the sheet's default `Supported`, not a human judgement. As of the 2026-09-01 pull that is all 20 MARRS cells — the fifth table is still being annotated. Counting them would hand both systems 20 free correct answers. |
| `skipped` cells are dropped | Matches the 99-cell denominator the report uses. |
| Hybrid labels stay hybrid | Five cells carry two verdicts ("Contradicted/May Not", "Supported / May Not Support"). Held out of the exact-match number, scored generously elsewhere — never collapsed to whichever branch flatters a system. |

**Do not lead with accuracy.** 62 of the 74 unambiguously labelled cells are correct
table claims, so a constant `Supported` classifier scores 62/74 (83.8%) and beats both
systems. The script prints that constant next to the real rows for exactly this reason.
The informative comparison is the error budget: green-light precision, false
accusations, and abstention cost.

## Caveat on the archived run

`table_audit_percell.md` records no model id, so which judge model produced Refari's
verdicts is **unverified**. The baseline uses the repo's configured judge
(`Settings.model_judge`). If the archived run used a different model, that is a
confound to disclose.
