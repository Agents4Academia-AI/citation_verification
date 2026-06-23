# Team decisions

The high-level decisions the team agreed at initialization, adapted to the
**realized** repo structure. Promoted from [`decisions-phy.md`](decisions-phy.md)
(kept verbatim as the proposal-of-record / provenance). Tags: **[frozen]** =
decided and locked at init (changing needs team sign-off); **[open]** = proposal
still under discussion (these were the `[change]` items in the proposal).

## 1. Where the contract lives — [frozen]

`CitationRecord` lives in **`src/citation_verifier/schema.py`** (pydantic v2) and
is mirrored by a committed, language-agnostic JSON Schema at
**`spec/v0.1/record.schema.json`**. Both, plus
`src/citation_verifier/interfaces.py` and
`.claude/skills/verify-citations/SKILL.md`, are the **frozen seam**. The four
modules import the schema/interfaces and nothing else of each other.
`make schema` regenerates the spec and CI fails on drift
(`tests/test_schema.py::test_committed_spec_matches_model`).

## 2. The record is keyed by `(paper_id, claim_id, cite_key)`, 1:1 with SKILL.md — [frozen]

One record = one `(claim, citation)` pair = one table row. A reference cited in N
places yields N records. The eight SKILL.md columns map 1:1 to schema fields and
no column exists that is not a field. The four enum vocabularies
(`exists` / `supports_claim` / `priority` / `severity`) are token-for-token
identical between `schema.py`, `SKILL.md`, and the renderer; the only
token-vs-string difference is `does_not` → "does not".

## 3. JSON is the source of truth; the table is rendered — [frozen]

The model emits `CitationRecord`s; `render.py` renders the SKILL.md table and the
summary **deterministically**. The model never hand-authors the table. JSON
(`report.json`) is what eval and the dataset consume.

## 4. Severity is derived deterministically — [frozen at init; was [change]]

`severity` is derived from `(exists, supports_claim, priority)` by
`schema.derive_severity()` (e.g. `exists=no → high`; obligatory + `does_not` →
high). *This overrides SKILL.md's free-judged severity* for reproducibility and
agent/gold agreement. The schema still permits a model-judged value (the field
records whichever was used), but the pipeline derives it. Rationale and the full
map are in `schema.py` and `tests/test_schema.py`.

## 5. One unit of work, degrade-not-crash, resumable — [frozen]

The orchestrator (`run_verification`) maps over `(claim, citation)` pairs with
bounded context. A stuck/failed pair degrades to a single `unresolved/inconclusive` row (its
`error` field set) and the run continues. Per-paper artifacts land under
`papers/<paper_id>/`.

## 6. Multi-source resolver; "never from memory"; gated web — [frozen core, [open] web policy]

`lookup_paper` evolved into a **multi-source resolver** (`grounding/`) with a
`source → dimension` map in `config/sources.yaml`: correctness from Crossref +
arXiv (keyless floor) + DBLP + optional OpenAlex; relevance signal (abstract /
citation-intent) from optional Semantic Scholar. Match cascade **DOI > arXiv-id >
fuzzy-title** (fuzzy gated by author overlap + year ±1); the LLM only adjudicates
the fuzzy tier against *retrieved* fields. The cited URL is validated.

- **[frozen]** The abstain rule: `unresolved/inconclusive` beats guessing; a web hit alone
  never upgrades `unresolved/inconclusive → yes` without a corroborating structured record.
- **[open]** *Demoting WebSearch/WebFetch to a gated last resort* (off by
  default, `ENABLE_WEB_SEARCH`). Proposed to narrow the baseline's broad web use;
  to confirm at merge.

## 7. Two interchangeable backends + per-run cost accounting — [frozen]

Same input, same output schema, two backends behind the `VerificationBackend`
seam: **`agentic`** (explicit staged pipeline) and **`claude_code`**
(skill-driven, grounded, concurrent judge; lazy SDK import). `backends/usage.py` records
`RunUsage` per run and per `ModelTier`, so the two are compared on quality **and
token/cost**.

## 8. Two-tier model routing — [frozen mechanism; [open] which models]

Routing happens at pass boundaries, never mid-loop: a cheap `bulk` tier for
correctness, a strong `judge` tier for relevance, logged per record
(`model_tier`). `MODEL_BULK` / `MODEL_JUDGE` are env-overridable with a per-run
USD ceiling (`COST_CEILING_USD`). Single-model is acceptable for the demo.

## 9. LaTeX-first ingestion — [frozen]

The arXiv LaTeX e-print is the primary path (`.bbl`/`.bib` + each `\cite`
call-site → deterministic `claim_id` + claim text/section); PDF-via-`Read` is the
fallback. Extraction is its own checkpoint emitting record stubs before any
verification — and gives a well-defined eval join key.

## 10. Eval is the scoring boundary — [frozen]

The agent never imports `evals/` and vice versa. `evals/run_eval.py` joins agent
`report.json` to gold on `(paper_id, claim_id, cite_key)`, validates against the
spec, and reports correctness P/R/F1 (positive class = fabricated/wrong-metadata),
relevance macro-F1, priority accuracy + obligatory-F1, and abstention/calibration
(`unresolved/inconclusive` is a first-class scored label). Headline = correctness-F1.

## 11. Anti-circularity for gold — [frozen]

The gold oracle must **not** reuse the agent's resolver (`grounding/`) or the
agent's judge model. Gold is built from a *different* resolver / human
adjudication and records its provenance (`Labels.provenance`). Otherwise
correctness P/R measures self-agreement, not accuracy. Enforced for the smoke
gold by `tests/test_eval.py::test_smoke_gold_has_labels_and_provenance`.

## 12. Two-tier eval data — [frozen]

A small in-repo **smoke** set (`evals/smoke/gold.jsonl`, ≥3 fabricated + ≥3
wrong-metadata) runs in CI for fast contract-regression; the full
CitationHallucinationBench lives off-repo on `/scratch`
(`CHBENCH_DATA_DIR`). Green smoke = schema-valid + non-trivial correctness-F1.

## 13. Dataset mirrors the schema 1:1 — [frozen]

`src/chbench/` builds gold as `CitationRecord`s whose `labels` block carries the
truth — agent output and gold agree by construction. Seeds: the GPTZero natural
hallucination lists (NeurIPS 2025, ICLR 2026) plus collected PDFs/arXiv source.
See [`DATASET.md`](DATASET.md).

## 14. STEP 3 (comparison objectiveness) is reserved — [frozen seam]

`CitationRecord.compared_against` and `stages/comparison.py` exist now (n/a for
the MVP) so comparison-objectiveness lands later without a breaking schema change.

## Open questions still on the table (from the proposal, decide together)

1. Per-pair verification: one SDK sub-agent call or a batch of N? Correctness +
   relevance one call or two? (needs a spike on a ~60-ref paper.)
2. API keys: register shared S2 / OpenAlex keys before forking, or ship the demo
   on the keyless Crossref + arXiv + DBLP floor and treat keyed sources as a
   post-demo upgrade.
3. Eval scoring constants: the abstention-aware reward/penalty for `unresolved/inconclusive`,
   confidence semantics (per-axis vs per-record), and whether correctness-F1
   splits into fabrication vs perturbed-metadata sub-scores.
4. Relevance gold source for the headline: human adjudication (small, trusted) vs
   a different-model LLM judge (scalable, weaker).
5. Cost ceiling per paper (~$0.50 for the demo?) and the total hackathon budget.

> The `[open]` items above (gated web search, exact model choices) are for team
> discussion at merge, not unilateral overrides of `SKILL.md`.
