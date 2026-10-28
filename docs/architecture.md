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
   `unresolved/inconclusive`). Extraction is its own checkpoint. Every extractor cuts the
   claim with one segmenter (`extract/sentences.py`): the whole sentence (up to 1000
   characters, then the clauses around the citation), never across a heading or a
   paragraph break, and verbatim. The PDF hyperlink path reads the document as one flow
   in reading order — so a sentence continues across columns and pages — with headings as
   boundaries and captions, footnotes and small type set apart; the numbered-marker path
   does the same with the text's lines (captions, figure numbers and table grids moved out).
   `claim.section` is filled from the headings. LaTeX citations are written out from the
   bibliography ("Guo et al. (2017)") instead of being deleted with their sentence's subject.
3. **Correctness** (`stages/correctness.py` over `grounding/`): resolve the
   reference against structured sources via the cascade **DOI > arXiv-id >
   fuzzy-title** (fuzzy gated by author overlap + year ±1), validate the cited
   URL, and fill `exists`, `resolved`, `metadata_issues`, `evidence`. Never from
   memory; if nothing resolves and web is gated off, the verdict is `unresolved/inconclusive`,
   not `no`. One work often has several records — the article and its anniversary
   reprint, a paper and a repost under a new DOI — so among records with the same title
   and lead author the one dated as cited is preferred, and a record dated well away from
   the cited year does not end the search. The venue comparison goes through an alias table (`venues.py` over
   `data/venues.yaml`): every spelling of a venue maps to one key, so "NeurIPS", "NIPS"
   and "Adv. Neural Inf. Process. Syst." are one venue; a mismatch is asserted only when
   both sides are known venues that differ, and a preprint or proceedings-series record
   ("arXiv", "PMLR") contradicts nothing. Beyond the paper indexes, the title tier asks
   OpenAlex by fields (title + first author + year — it holds books and working papers),
   Open Library for references that read as books, and OpenReview for workshop and
   ICLR/TMLR papers (and, last, for any title nothing else placed). A record with nothing
   to read (no abstract, no arXiv id) does not end the title tier: every remaining source is
   asked and a readable record of the same work is preferred. A short abstract counts only
   when it speaks of the title's subject (a series' boilerplate is a placeholder), by one
   rule shared with the relevance judge. A title that repeats a phrase after a stray prefix
   (copied from a broken record) is resolved first as the phrase it repeats. A not-found
   note names the sources that were actually asked. An identifier is looked up at its registry (arXiv,
   Crossref) before Semantic Scholar; when arXiv's query API does not answer, the paper's
   own arXiv page supplies the record, so a printed arXiv id still matches exactly. A
   source that keeps failing — rate-limited, out of
   its daily budget, behind a bot check — is left alone for a cool-down (a per-host breaker in
   `paper_lookup`) instead of being retried on every request. When nothing placed it, the web is the last resort
   (`grounding/web_lookup.py` over `backends/web_search.py`, only in runs that use a model):
   a model with web search names the work's pages, and the code fetches the index record an
   identifier in them points to, or accepts a web-only work's own page when its title
   matches and the page names a cited author beside it; the page's own authors and year
   are then compared with the citation like a record's. For a technical report, thesis or working paper the model also searches the
   issuing institution's own site, and an entry in that institution's report list (title
   with the cited author beside it, on the institution's host) counts; the list's other
   pages — same directory, same name but for the numbering — are read too. See DECISIONS.md §6.
4. **Relevance** (`stages/relevance.py`): only when correctness holds — compare
   the retrieved abstract/snippet against the claim and fill `supports_claim`,
   `priority`, `confidence`, and a quoted `evidence` item (the justification).
   The first pass reads the abstract and the Introduction (or, with neither, the opening
   of the text); a verdict it cannot settle — inconclusive, partial, does not — is
   re-judged on the full text, the whole paper when it is short enough. The full text
   comes from one cascade (`grounding/fulltext.py`): arXiv, the record's link, the DOI's
   open-access copy. A link that is a web page is followed to the work's PDF — the page's
   `citation_pdf_url`, a link whose text is the title, a PDF link in the title's own list
   entry — and that PDF counts only when it carries the title. A publisher's preview (the
   first pages, offered when the whole is paywalled) is tried last and labelled: the
   re-judge is told it holds the first pages only, never the complete text, and table cells
   are not judged on it. A PDF that is a scan is
   transcribed by a model (`backends/scan_reader.py`, only in runs that use a model) and
   the verdict says it rests on a transcription. With no text at all, the pair stays
   inconclusive and the explanation names what was tried.
5. **Comparison** (`stages/comparison.py`): reserved seam (STEP 3) — for
   obligatory baselines, `compared_against`.
5b. **Tables** (`tables/`, run by the orchestrator after the backend): the ✓/✗ cells of
   related-work comparison tables, each verified against the paper its row cites. A
   different unit from a record — "cited method R has property D", not a
   `(claim, citation)` pair — so the findings travel beside the records
   (`VerificationResult.tables`, `tables.json`), not inside them. The pass reuses the
   prose pass's references and resolutions (`tables/pipeline.py`); a row the prose pass
   did not cover goes through the same correctness stage, with the same web last resort
   (built only when the table pass builds its model). Results tables and tables
   without cited rows are out of scope. `tables="only"` runs this step alone.
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
