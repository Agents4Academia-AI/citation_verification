# `evals/` — scoring boundary for citation verification

This package scores an agent's citation-verification output against gold labels.
It is the **only** place that turns predictions into numbers, and it lives behind
a hard wall from the agent so the scores mean what they claim to mean.

## What it scores

Every gold record is one `(claim, citation)` pair — a row of the SKILL.md table —
keyed by `(paper_id, claim_id, cite_key)`. The harness loads agent predictions and
gold, joins on that key, and reports per-axis metrics:

| Axis | Metric | Positive / labels |
| --- | --- | --- |
| **Correctness** | precision / recall / **F1** (headline) | hallucination = `exists == no` **or** wrong metadata; de-duped per resolved paper |
| **Relevance** | macro-F1 (+ accuracy, per-label) | `supports` / `partial` / `does_not` / `unresolved/inconclusive` |
| **Priority** | accuracy + **obligatory-F1** | `obligatory` / `helpful` |
| **Abstention / calibration** | abstain rate, abstain P/R/F1, calibration gap | `unresolved/inconclusive` is a **first-class** label, not a silent gap |

The headline number is **correctness-F1** (`metrics["headline"]`), per
`docs/DECISIONS.md`.

`unresolved/inconclusive` is scored, not ignored: abstaining when gold is genuinely
unverifiable is *rewarded* (abstention recall), abstaining when the answer was
knowable is *penalised* (abstention precision drops, correctness recall drops). A
missing prediction for a gold pair is treated as a full abstention.

## How to score

```bash
# Smoke set (offline, no SDK, no network) — directory or single file both work:
python evals/run_eval.py evals/smoke evals/smoke/gold.jsonl
python evals/run_eval.py papers/2310.06825/report.json evals/smoke/gold.jsonl --json

# Or from Python:
python - <<'PY'
from evals import run_eval
m = run_eval("papers/", "evals/smoke/gold.jsonl")
print(m["headline"], m["correctness_f1"], m["relevance_macro_f1"])
PY
```

* `agent_json_dir` may be a directory of `report*.json` / `*.jsonl` outputs (one
  per paper) or a single prediction file. If the gold file lives inside that
  directory it is excluded automatically.
* Agent output is read from disk as JSON. The orchestrator's `report.json` (an
  object with a `"records": [...]` array) and plain JSONL / JSON arrays are all
  accepted.

Every record loaded — predictions **and** gold — is validated against
`spec/v0.1/record.schema.json` via `jsonschema`. An invalid record fails loudly,
naming the file, the index, and the offending field. This is what keeps the
agent's output and the gold in lock-step with the frozen contract.

## The smoke gold (`smoke/gold.jsonl`)

18 hand-written gold `CitationRecord`s across three fictional `paper_id`s
(`smoke-paper-A/B/C`), each with populated `labels` and `provenance` set to
`human-adjudication`. The set spans every enum value and deliberately includes:

* **≥3 fabricated** references (`exists == "no"`, `is_hallucinated == true`);
* **≥5 wrong-metadata** references (non-empty `metadata_issues`: year, venue,
  author, and wrong-paper errors);
* `supports` / `partial` / `does_not` / `unresolved/inconclusive` relevance cases;
* `obligatory` and `helpful` priorities at every severity.

It is small on purpose: green smoke means *schema-valid + non-trivial
correctness-F1*, a fast contract-regression gate for CI. The full
CitationHallucinationBench lives off-repo under `$CHBENCH_DATA_DIR`
(`/scratch/datasets/citation_verification_benchmark`) and is built by `src/chbench/`.

## Anti-circularity (read before adding gold)

The gold oracle must be **independent of the thing it scores**. Concretely:

1. **No agent internals.** `evals/` imports only `citation_verifier.schema` (enums
   / record type, lazily) and `jsonschema`. It must **not** import the
   orchestrator, backends, stages, `ingest`, `render`, or the grounding /
   `paper_lookup` layer. If scoring
   reused the agent's own resolver, correctness P/R would measure self-agreement,
   not accuracy.
2. **No agent judge model.** Gold labels come from human adjudication or a
   *different* resolver / model than the agent under test — never the agent's own
   `MODEL_JUDGE`. Record how each gold label was produced in
   `labels.provenance` so the audit is explicit.
3. **Gold drives the join.** Evaluation iterates gold; predictions with no gold
   counterpart are dropped (nothing to score them against) and surface as
   `n_pred` vs `n_matched` in the report.

When you grow the smoke set or build the full bench (`src/chbench/`), keep these
invariants: the bench's resolver (`src/chbench/resolve.py`) is intentionally a
*different* path from the agent's `src/citation_verifier/grounding/`.
