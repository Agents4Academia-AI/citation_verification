# Citation Agent — high-level decisions (phy's proposal)

> **Status:** phy's input to the team-wide initialization. Merge with the other
> members' proposals into one set, commit to `main` as the shared starting point,
> *then* everyone forks to their own branch.
>
> **Builds on** Yunqiao's v0 already on `origin/main` (`e40d202`): the Claude
> Agent-SDK `query()` loop (`src/agent.py`), the `lookup_paper` grounding tool
> (`src/paper_lookup.py`, Crossref + arXiv), and the method/contract in
> `.claude/skills/verify-citations/SKILL.md`. Tags below: `[keep] [extend]
> [change] [new]` relative to that baseline. `[change]` items are proposals to
> discuss at merge, not unilateral.

## Architecture & data model
- **Thin Python orchestrator over a typed pipeline, not a monolithic prompt.**
  Keep the SDK `query()` loop as the *agent vehicle*, but wrap it in
  `run_verification(source) -> list[CitationRecord]` returning structured records,
  not a printed table. The `.md` report becomes one renderer over that object
  (alongside JSON / eval / Discord). `[change]` `(mvp)`
- **Unit of work = one `(citation, claim)` pair, not one paper.** Each pair is
  verified with bounded context; a stuck/failed pair degrades to a single
  `unverified` row and the run continues (degrade-not-crash, resumable). `[change]` `(mvp)`

## Output contract (freeze at init — this is the shared seam)
- **One versioned `CitationRecord`, keyed by `(claim_id, cite_key)`, 1:1 with the
  SKILL.md columns.** Closed enums exactly as SKILL.md: `exists`
  (yes/no/unverified), `supports_claim` (supports/partial/does-not/unverified),
  `priority` (obligatory/helpful), `severity` (high/medium/low/ok); plus
  `claim_text`, `cited_as{authors,title,year,venue}`, `resolved_id`,
  `metadata_issues[]`, `evidence[]`, `confidence`, `model_tier`, `schema_version`.
  **This schema *is* CitationHallucinationBench's label schema** — agent output and
  gold labels agree by construction. `[extend]` `(mvp)`
- **JSON is the source of truth; the SKILL.md Markdown table is rendered
  deterministically in Python**, never hand-authored by the model. No table column
  exists that isn't a JSON field. `[extend]` `(mvp)`
- **Derive `severity` deterministically** from `(exists, supports_claim, priority)`
  via a fixed map (e.g. `exists=no & obligatory -> high`); only the primary axes
  are judged/scored. *Proposal to override SKILL.md's free-judged severity —
  justification: reproducibility + agent/gold agreement.* `[change]` `(v1)`
- **Reserve a `compared_against` field now** (n/a for MVP) so comparison-
  objectiveness lands later without a breaking schema change. `[new]` `(v1)`

## Grounding & "never from memory"
- **SKILL.md stays the single source of truth for rubrics/vocab;** orchestration
  code re-encodes no rubric (Python owns only control flow, validation, routing,
  retries, rendering). `[keep]` `(mvp)`
- **Evolve `lookup_paper` into a multi-source resolver with source→dimension
  mapping:** correctness/metadata from Crossref + arXiv (keyless floor) + optional
  OpenAlex; relevance/priority signal (abstract, citation-intent) from optional
  Semantic Scholar. Match cascade DOI > arXiv-id > fuzzy-title (gated by author
  overlap + year ±1); the LLM only adjudicates the fuzzy tier against *retrieved*
  fields. `[extend]` `(mvp)`
- **Strengthen the abstain rule** (`unverified` beats guessing) and **demote
  WebSearch/WebFetch to gated last-resort** — only after structured sources
  abstain, results quoted with a URL; a web hit alone never upgrades
  `unverified -> yes` without a corroborating structured record. *Proposal to
  narrow the baseline's broad web use — justification: kills hallucination-by-
  snippet + reproducibility.* `[change]` `(mvp)`

## Parsing, cost & ops
- **arXiv LaTeX e-print is the primary ingestion path** (extract `.bib`/`.bbl` +
  each `\cite` call-site with its sentence/section → deterministic `claim_id`);
  PDF-via-`Read` is fallback. Extraction is its own checkpoint emitting record
  stubs before any verification. *(On the baseline roadmap; highest-leverage
  accuracy upgrade and required for a well-defined eval join key.)* `[extend]` `(mvp)`
- **Two-tier model routing as config, at pass boundaries (never mid-loop):** cheap
  Haiku-class for bulk correctness, strong Opus/Sonnet-class for relevance;
  `model_tier` logged per record; `MODEL_BULK`/`MODEL_JUDGE` env-overridable with a
  per-run USD ceiling. `[extend]` `(v1)`
- **Per-paper artifact dir + run log:** normalize input → `paper_id`; write
  everything under `papers/<paper_id>/` (source, refs, `report.md`,
  `report.json`/`run.json` with model/tokens/USD/tool-calls). Pin model ids,
  temperature 0 where supported. Secrets in one gitignored `.env` + `config.py`,
  S2/OpenAlex keys optional. `[new]` `(mvp)`

## Team modularity & eval
- **4 modules = 4 branches, communicating ONLY via the frozen schema on disk;** no
  module imports another's internals. Shared files (schema, orchestrator, config,
  SKILL.md) merged at init and frozen — changes need team sign-off. `[new]` `(mvp)`
- **`evals/` is the scoring boundary:** the agent never imports `evals/` and vice
  versa. `run_eval.py` joins agent `report-*.json` to gold on `(claim_id,
  cite_key)` → correctness P/R/F1 (positive class = hallucination/wrong-metadata),
  relevance macro-F1, priority accuracy + obligatory-F1, plus abstention/
  calibration (`unverified` is a first-class scored label). Headline =
  correctness-F1. `[new]` `(mvp)`
- **Anti-circularity:** the gold oracle must NOT reuse `src/paper_lookup.py` or the
  agent's judge model — build gold from a *different* resolver / human
  adjudication, and record gold provenance. Otherwise correctness P/R measures
  self-agreement, not accuracy. `[new]` `(mvp)`
- **Two-tier eval data:** a ~15–20-pair in-repo smoke set (`evals/smoke/` or
  `tests/fixtures/`, incl. ≥3 fabricated + ≥3 wrong-metadata) runnable in CI for
  fast contract-regression, plus the full CitationHallucinationBench on
  `/scratch/datasets/`. Green smoke = schema-valid + non-trivial correctness-F1.
  `[new]` `(mvp)`
- **Discord bot = thin optional adapter** that only calls `run_verification(source)`
  and posts the rendered report; own module, zero core deps, built last.
  `[new]` `(later)`

## MVP for the Fri Jun 19 demo
On 2–3 fixed arXiv papers (≥1 with a deliberately fabricated / wrong-metadata
reference so both `no` and `unverified` fire):
1. arXiv id/URL → download LaTeX source → extract record stubs with `(claim_id,
   cite_key)` + ≥1 claim site each (PDF fallback wired but LaTeX is the demo path).
2. Per-pair verification fills **correctness** (Crossref+arXiv resolver, abstain
   rule on) and **relevance** (`supports_claim` + quoted evidence); a failed pair
   degrades to `unverified` and the run continues.
3. Python deterministically renders the exact SKILL.md table + saves
   `report.json`/`run.json` under `papers/<id>/`.
4. `evals/run_eval.py` joins that JSON against the in-repo smoke gold (oracle that
   does **not** reuse `paper_lookup.py`) → prints correctness P/R/F1 + relevance
   macro-F1.

Single-model is fine for the demo; model routing, OpenAlex/S2, comparison-
objectiveness, severity-derivation, calibration, and the Discord bot are
stubbed/coarse — but their **schema fields and stage boundaries are real** so they
slot in by Jun 26 without touching siblings.

## Suggested ownership (parallel, collision-free)
| Module | Owner | Stable interface other modules depend on |
|---|---|---|
| Shared core: schema + orchestrator + config + SKILL.md (freeze after init) | Yunqiao (baseline author) | `run_verification(source) -> list[CitationRecord]`; the `CitationRecord` schema is THE contract |
| Ingestion + parsing (LaTeX source, PDF fallback) | Mingye / Luke | `extract(source) -> [CitationRecord stubs]` with `cite_key`, `cited_as`, `claim_sites[{claim_id,text,section}]` |
| Correctness + grounding (`lookup_paper` v2) | Luke / Mingye | `fill_correctness(record)` → populates `exists`, `resolved_id`, `metadata_issues[]`, `evidence[]` |
| Relevance + priority (+ later comparison) | Luke / Mingye | `fill_relevance(record)` → populates `supports_claim`, `priority`, `evidence[]`, `confidence` |
| Dataset + eval harness | **phy** | `evals/run_eval.py(agent_json_dir, gold)`; talks to agent ONLY via JSON on disk |

## Open questions for the team (decide together at merge)
1. **Where does `CitationRecord` physically live** so every branch + the dataset
   import it without circular deps — shared `src/schema.py` (pydantic) vs a
   JSON-Schema file vs both? *(Decide in the merge commit — this is the seam.)*
2. **SDK loop vs manual `messages.create` tool loop**, and **is per-pair
   verification one SDK sub-agent call or a batch** of N citations (needs a quick
   spike on a 60-ref paper)? Correctness + relevance one call or two?
3. **Claim-site granularity** (sentence-level?) must match between the LaTeX
   parser's spans and the dataset's gold, or relevance scoring is misaligned.
4. **API keys:** OpenAlex now needs a key, S2's good rate limit needs an approved
   key — register shared keys into `.env` before forking, or run the demo on the
   keyless Crossref+arXiv floor and treat S2/OpenAlex as a post-demo upgrade?
5. **Eval scoring constants:** the abstention-aware reward/penalty for
   `unverified`, confidence semantics (per-axis vs per-record), and whether
   correctness-F1 splits into fabrication vs perturbed-metadata sub-scores.
6. **Relevance gold source** for the headline: human adjudication (small, trusted)
   vs a different-model LLM judge (scalable, weaker) — which do we trust / report?
7. **Cost ceiling** (~$0.50/paper for the demo?) + total hackathon API budget.

## Housekeeping
- `phy` is behind `origin/main` (cut from `d009bf1`); rebase onto `e40d202` before
  developing.
- Proposed `[change]` items (JSON-as-truth, derived severity, gated web search) are
  for team discussion at merge, not unilateral overrides of `SKILL.md`.
