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
python scripts/baselines/scitance.py          # literature baseline -> out/scitance.json
python scripts/baselines/scitance.py --gloss  # charitable variant (+ column definition)
python scripts/baselines/refari_judge_cost.py # price Refari's judging stage
python scripts/baselines/score_human.py       # every system vs. the human labels
```

Build the manifest from **`table_audit_percell_0903.md`** (the 2026-09-03 run), not
`table_audit_percell_0829.md` or the older `table_audit_percell.md`. Every rerun so far
has rewritten the column definitions (0829 changed 96 of 106; 0903 changed 100 of 106,
plus the evidence sentence on 72 cells), so a baseline judged against an earlier set is
no longer matched to Refari. **Re-run every baseline after switching audits.** The 0829
outputs are kept as `out/*_v1_0829.json`; the 0828 ones as `out/*_v0_0828.json`.

The 0903 run's own header records two things worth knowing: its judge prompt was
scrubbed of every worked example drawn from these five papers, and MARRS's verdict
column comes from a separate rerun because the judge call crashed on that paper in the
main batch (its mark-flip column is from the main batch).

`build_manifest.py` parses the archived per-cell audit rather than re-running
extraction, precisely so the denominator cannot drift.

## Baselines

- **`closed_book.py`** — no retrieval, no tools. The judge answers from its own
  knowledge of the cited work. This is both the headline baseline (what you get by
  just asking a model) and Refari's retrieval ablation.
- **`scitance.py`** — the literature baseline: Alvarez, Bennett and Wang (SDP 2024),
  prompt and label set verbatim in their best GPT-4 configuration. Chosen because it is
  a *prompting procedure*; `zhang2025atomic` and `minicheck2024` both need fine-tuning,
  so running them with our judge would measure the port rather than the method.

## Cost

`refari_judge_cost.py` replays Refari's own `build_cell_judge` over the frozen
definitions and evidence, so the system we propose has a figure beside the baselines.
Per judged cell (0903 run): SCitance $0.100, closed-book $0.044, Refari ≥$0.038.

Full-set scores on the 0903 run (99 labelled cells, 94 with a single-valued label):
exact match Refari 79.8%, closed-book 75.5%, SCitance 54.3% (64.9% with the
definition supplied). Contradictions issued / overturned by annotators: Refari 10 / 5,
closed-book 19 / 10, SCitance 15 / 9. MARRS still separates SCitance from the rest —
8/20 there against 20/20 for the others. The order on exact match is **not** stable
across runs: on the 0829 definitions closed-book (75.5%) edged Refari (74.5%), and the
95% intervals overlap almost entirely; the error profile (fewest accusations, fewest
overturned) is the finding, not the exact-match ranking.

The Refari number is a **lower bound twice over** — it omits retrieval and column
grounding, and it feeds only the evidence *sentence* the report quotes rather than the
passage the live run held. That second restriction costs a lot: the same judge under it
reproduces only 52/94 (55.3%) of the archived verdicts. Since `scitance.py` is given
that same sentence, **the literature baseline is evidence-starved relative to Refari**,
and its scores are a lower bound on the method. The full retrieved passages are not in
the archive, so closing this needs a fresh Refari run.

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
| Rows with a blank `checker` are dropped | They carry the sheet's default `Supported`, not a human judgement, so counting them hands every system free correct answers. This fired on a false positive once — the 20 MARRS rows *were* annotated (by mingye) but left unsigned. Fixed in the snapshot, which now records `checker` plus a `checker_source` noting the attribution came from the project owner and not the sheet. Fix attribution in the data; do not loosen the check. |
| `skipped` cells are dropped | Matches the 99-cell denominator the report uses. |
| Hybrid labels stay hybrid | Five cells carry two verdicts ("Contradicted/May Not", "Supported / May Not Support"). Held out of the exact-match number, scored generously elsewhere — never collapsed to whichever branch flatters a system. |

**Do not lead with accuracy.** 82 of the 94 unambiguously labelled cells are correct
table claims, so a constant `Supported` classifier scores 82/94 (87.2%) and beats all
three systems. The script prints that constant next to the real rows for exactly this reason.
The informative comparison is the error budget: green-light precision, false
accusations, and abstention cost.

## Caveat on the archived run

`table_audit_percell.md` records no model id, so which judge model produced Refari's
verdicts is **unverified**. The baseline uses the repo's configured judge
(`Settings.model_judge`). If the archived run used a different model, that is a
confound to disclose.
