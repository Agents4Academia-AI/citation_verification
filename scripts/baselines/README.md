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
```

`build_manifest.py` parses the archived per-cell audit
(`/scratch/datasets/citation_verification_outputs/table_audit_percell.md`) rather than
re-running extraction, precisely so the denominator cannot drift.

## Baselines

- **`closed_book.py`** — no retrieval, no tools. The judge answers from its own
  knowledge of the cited work. This is both the headline baseline (what you get by
  just asking a model) and Refari's retrieval ablation.

## Reading the output

Human cell labels are not finished, so **none of this says who is right**. It measures
determinacy, behaviour on cells with no retrievable evidence, agreement, and
self-consistency — all of which need no gold labels.

## Caveat on the archived run

`table_audit_percell.md` records no model id, so which judge model produced Refari's
verdicts is **unverified**. The baseline uses the repo's configured judge
(`Settings.model_judge`). If the archived run used a different model, that is a
confound to disclose.
