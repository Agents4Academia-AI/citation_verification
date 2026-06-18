# citation_verification

> An agent that, given an arXiv link or a PDF, **verifies the citations in a
> paper draft** and emits a verification table: which references are real, whether
> their metadata is correct, and whether each cited paper actually supports the
> claim it is attached to.

Built during [Agents4Academia](https://github.com/Agents4Academia-AI),
14–26 June 2026.

---

## What it does (the Notion 3-step plan → concrete files)

The method lives in **`.claude/skills/verify-citations/SKILL.md`** (the frozen
contract: the output table + value vocabularies). Each step maps to code:

| Notion step | What it checks | Where it lives |
|---|---|---|
| **STEP 1 — Reality & Accuracy** | Real vs fake citations; authors/venue/year/title; validate the cited URL. | `grounding/` (multi-source resolver: Crossref/arXiv/DBLP, optional S2/OpenAlex) → `stages/correctness.py` → `exists`, `metadata_issues` |
| **STEP 2 — Relevance & Justification** | Does the cited paper support the claim? Obligatory vs helpful. *Construct a justification.* | `stages/relevance.py` → `supports_claim`, `priority`; the "construct-a-skill / justification" note is in SKILL.md |
| **STEP 3 — Objectiveness of Comparison** | For obligatory baselines, were they actually compared against? | reserved seam: `stages/comparison.py` + `CitationRecord.compared_against` |

The pipeline framework is: **extract** text + references (prefer LaTeX
`.bbl`/`.bib` + `\cite` call-sites) → **check correctness** of each citation
against structured sources and validate its URL → if correct, **check relevance**
→ **justify** the relevance verdict. Every verdict is grounded in retrieved
evidence, never model memory.

### Two interchangeable backends (and a comparison)

The same input and the **same output schema** are produced by two backends so we
can compare quality *and token/cost*:

| Backend | What it is | File |
|---|---|---|
| `claude_code` | Skill-driven, grounded, concurrent judge: existence is grounded deterministically (no LLM), then the SKILL.md method judges (claim, citation) pairs in concurrent, no-tool `query()` chunks. | `backends/claude_code.py` |
| `agentic` | Explicit, staged pipeline: extract → correctness → relevance. Deterministic control flow. | `backends/agentic.py` |

Per-run token/cost accounting (`RunUsage`, `backends/usage.py`) lets us put the
two side by side. Both emit `CitationRecord`s; the renderer turns them into the
identical SKILL.md table.

## The contract (frozen seam)

Everything agrees on one schema: **`CitationRecord`**, keyed by
`(paper_id, claim_id, cite_key)`, 1:1 with the SKILL.md table
(`src/citation_verifier/schema.py`). JSON is the source of truth; the Markdown
table is rendered deterministically from it (`render.py`). The same schema is the
**gold-label format** for our dataset (`src/chbench/`), so agent output and gold
agree by construction. The canonical JSON Schema is committed at
`spec/v0.1/record.schema.json` and CI checks it never drifts (`make schema`).

## Quickstart

```bash
# 1. Install (core + dev tooling; offline floor needs no extras / no SDK / no network)
uv pip install -e '.[dev]'        # or:  make install

# 2. Run the offline contract tests + smoke eval
make test                         # full pytest suite, offline
make smoke                        # tests + run_eval on the in-repo smoke gold

# 3. Verify a paper (needs the LLM backend + a key — see .env.example)
cp .env.example .env              # then fill ANTHROPIC_API_KEY etc.
uv pip install -e '.[llm]'
cverify 1706.03762 --backend agentic --out papers/1706.03762
cverify https://arxiv.org/abs/2005.14165 --backend claude_code

# 4. Check a single reference against the grounding sources (no LLM)
python -m citation_verifier.grounding.paper_lookup "Vaswani et al. Attention Is All You Need 2017"

# 5. Run the Discord bot (slash command /check <arxiv>) — see docs/DISCORD_BOT.md
uv pip install -e '.[bot]'
cverify-bot                       # reads DISCORD_BOT_TOKEN from .env
```

The **keyless floor** — schema + render + eval + Crossref/arXiv grounding — runs
with no API keys, no `claude-agent-sdk`, and degrades softly without network.
Keyed sources (Semantic Scholar, OpenAlex) and the `claude_code` backend turn on
only when their `.env` keys are present.

## Repository structure

| Path | What |
|---|---|
| `src/citation_verifier/schema.py` | **[contract]** `CitationRecord` + enums + `derive_severity` + JSON-Schema export |
| `src/citation_verifier/interfaces.py` | **[contract]** Protocols (Extractor/Resolver/VerificationBackend/StageFn) + result/usage dataclasses |
| `src/citation_verifier/orchestrator.py` | `run_verification(source, backend=...)` — maps over `(claim, citation)` pairs, degrade-not-crash |
| `src/citation_verifier/render.py` | records → the exact SKILL.md table + summary; `to_json`/`from_json` |
| `src/citation_verifier/ingest.py` | arXiv id/url/PDF → `PaperSource` (fail-soft download) |
| `src/citation_verifier/extract/` | LaTeX (primary) + PDF (fallback) extractors → record stubs |
| `src/citation_verifier/grounding/` | multi-source resolver + `lookup_paper` + URL validation |
| `src/citation_verifier/stages/` | `fill_correctness` / `fill_relevance` / `fill_comparison` |
| `src/citation_verifier/backends/` | `claude_code` (grounded SKILL.md judge) and `agentic` (staged) + usage accounting |
| `src/citation_verifier/bot/` | Discord front-end (`cverify-bot`): `/check`, `/help`, `/ping` → [`docs/DISCORD_BOT.md`](docs/DISCORD_BOT.md) |
| `src/chbench/` | CitationHallucinationBench: harvest → parse → resolve → label → splits |
| `evals/` | scoring boundary (`run_eval.py`, `metrics.py`); `smoke/gold.jsonl` for CI |
| `spec/v0.1/record.schema.json` | **[contract]** committed JSON Schema for one record |
| `config/` | `sources.yaml` (per-source params + dimensions), `venues.yaml` (normalization + harvest scope) |
| `.claude/skills/verify-citations/SKILL.md` | **[contract]** the method + frozen output table + enums |
| `docs/` | [`architecture.md`](docs/architecture.md), [`DECISIONS.md`](docs/DECISIONS.md), [`DATASET.md`](docs/DATASET.md) |
| `papers/` | per-paper artifact dirs (gitignored content) |

## For contributors / agents

- Start here: **[`AGENTS.md`](AGENTS.md)** (run/test/conventions/off-limits) and
  **[`CLAUDE.md`](CLAUDE.md)** (Claude Code session guide).
- Design rationale: **[`docs/architecture.md`](docs/architecture.md)** and
  **[`docs/DECISIONS.md`](docs/DECISIONS.md)**.
- Four modules → four branches, communicating **only** via the frozen schema on
  disk. Shared/contract files are frozen; changing them needs team sign-off.
- Python 3.11+. Secrets only in a gitignored `.env`. PR to `main`.
