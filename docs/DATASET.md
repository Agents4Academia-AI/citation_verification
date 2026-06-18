# CitationHallucinationBench (chbench)

> Owner: **phy**. The gold benchmark the citation-verification agent is scored
> against. Built by `src/chbench/`, run via the `chbench` CLI. This spec mirrors
> the team decisions in `docs/decisions-phy.md` and freezes the dataset's seams.

## Why this exists

An LLM asked "is this citation real?" will happily hallucinate "yes". To measure
whether the agent actually *catches* citation problems we need ground truth:
papers with known fabricated / wrong-metadata / irrelevant citations, labelled
in the **same schema the agent emits** so agent output and gold agree by
construction and the eval harness can join them on `(paper_id, claim_id,
cite_key)`.

## The contract (schema-mirrored gold)

Gold records ARE `citation_verifier.schema.CitationRecord` objects (the frozen
contract) with the nested `labels` block populated. There is **no parallel label
schema** — a gold record is just a `CitationRecord` whose verdicts are the truth.
This is the same decision recorded in `decisions-phy.md` ("the schema *is*
CitationHallucinationBench's label schema").

Labelled axes (`Labels`):

| field | how it is set |
|---|---|
| `exists` | `yes` if the gold resolver matched; `no` if a seed flags a fabricated reference or `inject` synthesizes one; `unverified` if offline / no match |
| `supports_claim` | `unverified` by default — relevance gold needs a **human or a different-model judge** (anti-circularity), never the agent's judge |
| `priority` | heuristic `obligatory`/`helpful` from claim-site cues; provenance flags it as heuristic for human override |
| `severity` | derived deterministically via the contract's `derive_severity(exists, supports_claim, priority)` |
| `is_hallucinated` | `True` when `exists=no` OR a metadata error was detected — the positive class for correctness P/R/F1 |
| `provenance` | how the label was made (`gptzero-natural`, `<source>-resolver`, `synthetic:*`, human) — for the anti-circularity audit |

## Sources (seed material)

Defined in `src/chbench/sources.py` and `config/sources.yaml`:

- **GPTZero NeurIPS 2025** — `https://gptzero.me/news/neurips/` (~100 natural hallucinations).
- **GPTZero ICLR 2026** — `https://gptzero.me/news/iclr-2026/` (50+ natural hallucinations).
- **OpenReview** — opt-in broad collection of submitted papers per venue.
- **arXiv / PDF** — the papers behind the seeds, harvested for parsing.

The GPTZero lists seed *natural* hallucination labels; `inject.py` adds
*synthetic* positives (fabrication + metadata perturbation) for balance and
control. Natural vs synthetic is distinguishable via `labels.provenance`.

## Pipeline (stages)

Each stage is a typed, resumable function writing a checkpoint into the data dir;
`chbench all` runs them end to end. Offline by default (`--fetch` opts into
network; every network body is fail-soft).

```
seeds    sources.gptzero_seed_records / openreview_sources   -> seeds.json
harvest  harvest.harvest(seeds)                              -> harvest.json + papers/<id>/
parse    parse.parse_paper(path)                             -> parsed.json   (.bbl/.bib + \cite sites)
resolve  resolve.GoldResolver.resolve(ref)                   -> resolved.json (DBLP-first, independent)
label    label.make_gold(parsed, resolved)                   -> gold.jsonl
inject   inject.inject_fabrication / perturb_metadata        -> gold.jsonl (+ synthetic positives)
build    build_splits.build_splits(records)                  -> smoke.jsonl + full.jsonl
validate validate.validate_dataset(path)                     -> [] if schema-valid + labelled
```

## Anti-circularity (frozen)

The gold oracle must **not** measure self-agreement with the agent:

- `chbench.resolve` **does not import** `citation_verifier.grounding` and uses an
  independent source order (DBLP first; the agent floor is Crossref+arXiv). A test
  asserts the non-import.
- The gold pipeline performs **no agent-judge LLM call**. Relevance gold comes
  from a human or a *different* model, recorded in `labels.provenance`.

## Splits & storage (30-day-scratch / git-split)

- **Smoke** (`smoke.jsonl`, ~18 pairs): committed for CI / fast contract
  regression. Front-loads positives (fabricated + metadata-error) so a green
  smoke run means *schema-valid + non-trivial correctness signal*. Intended to
  seed `evals/smoke/gold.jsonl`.
- **Full** (`full.jsonl`): the complete benchmark. Lives **off-repo** under
  `/scratch/datasets/citation_verification_benchmark/` (default; `$CHBENCH_DATA_DIR`
  overrides). `/scratch` is treated as ephemeral (≈30-day retention) — the
  pipeline is fully resumable from checkpoints and re-runnable to rebuild it, so
  only the small committed smoke split and the code are the durable artifacts.

## Granularity

One record per **(claim-site, citation)** pair (decisions-phy.md: "one
(claim,citation) pair per record"). A reference cited in N places yields N
records. `claim_id` is deterministic (`<paper_id>:<cite_key>#<n>`) so it is stable
across runs and aligns with the agent extractor's claim-site convention — relevance
scoring stays aligned only if both sides keep this convention.

## Running

```bash
uv pip install -e .            # or: pip install -e .
chbench all                    # offline dry-run end to end (uses committed seeds)
chbench all --fetch            # download + query live sources (fail-soft)
chbench validate path/to.jsonl # schema-validate any split
```
