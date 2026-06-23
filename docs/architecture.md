# Architecture

How `citation_verification` is layered, how a `(claim, citation)` pair flows
through it, how the two backends plug into the same seam, and where to extend.

## Design goals

1. **One frozen contract, four parallel modules.** Teammates build on separate
   branches and communicate *only* through the schema and the Protocol seams —
   never through each other's internals.
2. **Import-safe & offline-tolerant core.** `schema` + `render` + `eval` +
   the grounding floor import and run with no `claude-agent-sdk`, no API key, and
   no network. The SDK is a lazy import; network calls fail soft.
3. **JSON is the source of truth.** The model emits structured `CitationRecord`s;
   the human-readable SKILL.md table is *rendered* from them, never hand-authored.
4. **Degrade, don't crash.** A stuck pair becomes one `unresolved/inconclusive` row and the
   run continues.

## Layers

```
                 ┌─────────────────────────────────────────────┐
  CLI / API ─────│  cli.py  ·  run_verification(source, backend)│   orchestrator.py
                 └───────────────┬─────────────────────────────┘
                                 │ (claim, citation) pairs
        ┌────────────────────────┼─────────────────────────────────────┐
        │                        │                                      │
   ingest.py                 extract/                              backends/
   arxiv|url|pdf →        latex.py (primary)                  ┌── agentic.py  (staged pipeline)
   PaperSource           pdf.py   (fallback)                  └── claude_code.py (grounded judge)
        │                   → record stubs                          │  both → VerificationResult
        │                                                           │
        └───────────────► grounding/ ◄──── stages/ ────────────────┘
                          resolver +        fill_correctness  (STEP 1)
                          paper_lookup +    fill_relevance    (STEP 2)
                          validate_url      fill_comparison   (STEP 3, reserved)
                                 │
                                 ▼
                          render.py  →  SKILL.md table + report.json
                                 │
                                 ▼
                          evals/  (scoring boundary; joins report.json to gold)
```

The two **contract** files sit *under* everything:

- `schema.py` — `CitationRecord` (+ enums, submodels, `derive_severity`,
  JSON-Schema export). The single source of truth for one table row.
- `interfaces.py` — the Protocols (`Extractor`, `Resolver`,
  `VerificationBackend`, `StageFn`) and the result/usage dataclasses
  (`PaperSource`, `Candidate`, `RunUsage`, `VerificationResult`).

Every module imports from these two and nothing else of a sibling's.

## The `(claim, citation)` data flow

The unit of work is **one `(claim, citation)` pair**, keyed by
`(paper_id, claim_id, cite_key)` — *not* one paper. A reference cited in N places
becomes N records (it can be obligatory in one spot, background in another).

1. **Ingest** (`ingest.py`): an arXiv id/URL or a PDF path → a `PaperSource`
   (paper_id, work_dir, `tex_available`). Download is fail-soft.
2. **Extract** (`extract/`): prefer the LaTeX e-print — parse `.bbl`/`.bib` and
   each `\cite` call-site, anchoring the exact reference to the exact claim
   sentence/section, and minting a deterministic `claim_id`. PDF is the fallback.
   Output: **record stubs** (key + `claim` + `cited_as`, judged axes left
   `unresolved/inconclusive`). Extraction is its own checkpoint.
3. **Correctness** (`stages/correctness.py` over `grounding/`): resolve the
   reference against structured sources via the cascade **DOI > arXiv-id >
   fuzzy-title** (fuzzy gated by author overlap + year ±1), validate the cited
   URL, and fill `exists`, `resolved`, `metadata_issues`, `evidence`. Never from
   memory; if nothing resolves and web is gated off, the verdict is `unresolved/inconclusive`,
   not `no`.
4. **Relevance** (`stages/relevance.py`): only when correctness holds — compare
   the retrieved abstract/snippet against the claim and fill `supports_claim`,
   `priority`, `confidence`, and a quoted `evidence` item (the justification).
5. **Comparison** (`stages/comparison.py`): reserved seam (STEP 3) — for
   obligatory baselines, `compared_against`.
6. **Severity**: derived deterministically from
   `(exists, supports_claim, priority)` via `derive_severity` (see DECISIONS.md),
   so agent output and gold agree by construction.
7. **Render** (`render.py`): records → the exact SKILL.md 7-column table + a
   summary; plus `report.json` / `run.json` under `papers/<paper_id>/`.

## Backend abstraction (the two-baseline comparison)

Both backends satisfy the same `VerificationBackend` Protocol —
`verify(source, stubs) -> VerificationResult` — and emit the **same schema**:

- **`agentic`** (`backends/agentic.py`): an explicit staged pipeline that calls
  the `stages/` functions in order. Deterministic control flow; the LLM is used
  narrowly (fuzzy-match adjudication, relevance judgement). Two-tier model routing
  at pass boundaries: a cheap `bulk` tier for correctness, a strong `judge` tier
  for relevance (`MODEL_BULK` / `MODEL_JUDGE`).
- **`claude_code`** (`backends/claude_code.py`): a skill-driven, grounded,
  concurrent judge. Each reference is first resolved deterministically via the
  grounding layer (no LLM) to fix `exists`/`resolved`/`metadata_issues`; the
  SKILL.md method then judges the (claim, citation) pairs in bounded, concurrent
  chunks — one structured `query()` per chunk, no tools, sharing the SKILL.md
  system prompt as a prompt-cache prefix. Lazy SDK import.

`backends/usage.py` records `RunUsage` (tokens, cost, turns, tool calls) per run
and per `ModelTier`, so the two backends can be compared on **quality and
token/cost** under one report schema. `get_backend(name)` resolves the registry.

## Extension points (seams left open)

- **New grounding source** → add it to `config/sources.yaml` (params +
  `dimensions`) and a client in `grounding/`; the resolver cascade and
  `source → dimension` map pick it up. The keyless floor stays up if it is keyed
  and the key is absent.
- **STEP 3 comparison** → implement `stages/comparison.py` filling
  `compared_against`; the field already exists in the schema (no breaking change).
- **New backend** → implement `VerificationBackend` and register it; the CLI,
  orchestrator, renderer, and eval are unchanged.
- **New output sink** (Discord, etc.) → a thin adapter that calls
  `run_verification` and renders — zero core deps.
- **Schema evolution** → bump `SCHEMA_VERSION`, regenerate `spec/<v>/…` with
  `make schema`, and get team sign-off (the tests guard drift).

## Eval boundary

`evals/` is the scoring boundary and never imports agent internals: `run_eval.py`
joins the agent's `report.json` to gold on `(paper_id, claim_id, cite_key)`,
validates both against `spec/v0.1/record.schema.json`, and computes per-axis
metrics. Gold is built by a **different** resolver than the agent uses
(anti-circularity); see `docs/DATASET.md` and `docs/DECISIONS.md`.
