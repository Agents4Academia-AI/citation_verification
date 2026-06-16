# AGENTS.md — what an agent/contributor reads at session start

> Keep this short — every line costs attention each session. Deep rationale lives
> in `docs/architecture.md` and `docs/DECISIONS.md`; the verification *method* and
> the frozen output table live in `.claude/skills/verify-citations/SKILL.md`.

## How to run

- **Install (offline floor):** `uv pip install -e '.[dev]'`  (or `make install`).
  No `claude-agent-sdk`, no API key, no network needed for the core.
- **Test:** `make test`  (full pytest suite, offline) · `make smoke` (tests +
  `run_eval` on the in-repo smoke gold).
- **Lint:** `make lint`  (ruff).
- **Verify a paper (needs LLM backend + `.env`):**
  `cverify <arxiv-id | arxiv-url | path.pdf> --backend agentic|claude_code`.
- **Check one reference (no LLM):**
  `python -m citation_verifier.grounding.paper_lookup "<reference string>"`.
- **Regenerate / check the schema spec:** `make schema`.
- **Build the dataset:** `chbench build` (full) — see `docs/DATASET.md`.

## Conventions

- Python 3.11+. `pydantic` v2 for the schema.
- Functions are short. If something needs a paragraph of comment to explain,
  refactor it.
- Type-hint module boundaries (the Protocols in `interfaces.py` are the seams);
  internal code can be lighter.
- `snake_case` functions, `PascalCase` classes, `ALL_CAPS` constants.
- The core (schema + render + eval + grounding floor) MUST import and run
  **without** the SDK and **without** network. `claude-agent-sdk` is a lazy,
  optional import; network calls fail soft.

## What's off-limits (the frozen seams — changes need team sign-off)

- `src/citation_verifier/schema.py`, `interfaces.py`, `__init__.py`,
  `spec/v0.1/record.schema.json`, and
  `.claude/skills/verify-citations/SKILL.md` are the **contract**. The 8-column
  table and the four enum vocabularies must stay token-for-token in sync (the
  tests in `tests/test_schema.py` enforce this).
- `evals/` is the scoring boundary: the agent never imports `evals/` and vice
  versa. Gold is built by a **different** resolver than the agent uses
  (anti-circularity).
- Don't commit anything under `.env*` (except `.env.example`), `secrets/`, or
  `*.key`. Per-paper artifacts under `papers/<id>/` are gitignored.
- Don't push directly to `main` — open a PR.

## Where the important stuff lives

- `src/citation_verifier/schema.py` — `CitationRecord` (the contract) + enums +
  `derive_severity`.
- `src/citation_verifier/interfaces.py` — Extractor / Resolver /
  VerificationBackend / StageFn Protocols + `PaperSource` / `RunUsage` /
  `VerificationResult`.
- `src/citation_verifier/orchestrator.py` — `run_verification(source, backend=...)`.
- `src/citation_verifier/{ingest,extract,grounding,stages,backends,render,cli}` —
  the pipeline modules (see the structure table in `README.md`).
- `src/chbench/` — CitationHallucinationBench (dataset).
- `evals/` — `run_eval.py`, `metrics.py`, `smoke/gold.jsonl`.
- `config/{sources,venues}.yaml` — source params + venue normalization (no secrets).
- `papers/` — per-paper input/artifact dirs.

## Operating principles

1. *Think before coding* — state assumptions; surface confusion first.
2. *Respect the seam* — depend only on `schema` + `interfaces`, never another
   module's internals.
3. *Degrade, don't crash* — a stuck `(claim, citation)` pair becomes one
   `unverified` row; the run continues.
4. *Never decide correctness from memory* — ground every verdict in retrieved
   evidence, or mark it `unverified`.

## When in doubt

- Ask a clarifying question before generating code.
- Prefer surgical changes over big refactors.
- If a change touches a contract file, raise it with the team first.
