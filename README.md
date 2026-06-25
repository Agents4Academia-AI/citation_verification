# citation_verification

> Given an arXiv link or a PDF, **verify the citations in a paper draft** and emit a
> verification table: which references are real, whether their metadata is correct,
> and whether each cited paper actually supports the claim it is attached to.

Built during [Agents4Academia](https://github.com/Agents4Academia-AI), 14–26 June 2026.

---

## What it does

Three checks (the Notion plan), each grounded in **retrieved evidence, never model
memory**. The method and the frozen output table live in
[`SKILL.md`](.claude/skills/verify-citations/SKILL.md).

| Step | Question it answers | Output fields |
|---|---|---|
| **1 — Reality & Accuracy** | Is the citation real? Are authors / venue / year / title right? Does the cited URL resolve? | `exists`, `metadata_issues` |
| **2 — Relevance & Justification** | Does the cited paper actually support the claim? Is it obligatory or just helpful? | `supports_claim`, `priority` + a quoted justification |
| **3 — Objectiveness** *(reserved seam)* | For obligatory baselines, were they actually compared against? | `compared_against` |

## How it works

The unit of work is **one `(claim, citation)` pair** — a reference cited in N
places becomes N rows (it can be obligatory in one spot, background in another).
Each pair flows through a deterministic pipeline:

```
arXiv id / URL / PDF
  → ingest       → PaperSource (paper_id, work_dir, tex_available?)
  → extract      → LaTeX .bbl/.bib + \cite sites (PDF fallback) → record stubs
  → correctness  → resolve DOI > arXiv-id > fuzzy-title; validate URL → exists, metadata_issues
  → relevance    → judge the claim against the retrieved abstract / full text → supports_claim, priority
  → render       → the SKILL.md table + report.json under papers/<paper_id>/
```

A stuck pair degrades to one `unresolved` / `inconclusive` row and the run
continues. The full layered design, the `(claim, citation)` data flow, and the
extension seams are in **[`docs/architecture.md`](docs/architecture.md)**.

**Backend.** The default is **`agentic`** — the explicit, deterministic staged
pipeline above, with two-tier model routing (`MODEL_BULK` for correctness,
`MODEL_JUDGE` for relevance). A second backend, `claude_code` (a skill-driven
concurrent judge), lives behind the same `VerificationBackend` seam for a
quality/cost comparison and is opt-in via `--backend claude_code`; both emit the
identical schema.

## The contract (frozen seam)

Everything agrees on one schema: **`CitationRecord`**, keyed by
`(paper_id, claim_id, cite_key)`, 1:1 with the SKILL.md table
([`schema.py`](src/citation_verifier/schema.py)). JSON is the source of truth; the
Markdown table is rendered deterministically from it
([`render.py`](src/citation_verifier/render.py)). The same schema is the
**gold-label format** for the benchmark (`src/chbench/`), so agent output and gold
agree by construction. The canonical JSON Schema is committed at
`spec/v0.1/record.schema.json`; CI checks it never drifts (`make schema`).

Depend only on `schema.py` (records + enums) and `interfaces.py` (the Protocols) —
never another module's internals. Those seams are **frozen**: changing them needs
team sign-off.

## Quickstart

```bash
# 1. Install (core + dev tooling; the offline floor needs no extras / no SDK / no network)
make install                      # uv pip install -e '.[dev]'

# 2. Offline contract tests + smoke eval
make test                         # full pytest suite, offline
make smoke                        # tests + run_eval on the in-repo smoke gold

# 3. Verify a paper (LLM backend; auth = your Claude Code subscription, or an API key)
uv pip install -e '.[llm]'
cverify 1706.03762 --out papers/1706.03762        # 'agentic' is the default backend
cverify https://arxiv.org/abs/2005.14165 --format json

# 4. Check one reference against the grounding sources (no LLM, no key)
python -m citation_verifier.grounding.paper_lookup "Vaswani et al. Attention Is All You Need 2017"

# 5. Discord bot (slash command /check <arxiv>) — see docs/DISCORD_BOT.md
uv pip install -e '.[bot]'
cverify-bot                       # reads DISCORD_BOT_TOKEN from .env

# 6. Web UI (upload a PDF or paste an arXiv link; progress bar + report in the browser)
uv pip install -e '.[web]'
cverify-web                       # serves http://127.0.0.1:8000  (see "Web UI" below)
```

The **keyless floor** — schema + render + eval + Crossref/arXiv grounding +
open-access full-text — runs with no API keys, no `claude-agent-sdk`, and degrades
softly offline. Keyed sources and the LLM backends turn on only when their keys are
present.

### Web UI (`cverify-web`)

A browser front-end: drop a PDF or paste an arXiv link, watch the progress bar
(stage + live citation count + elapsed time), and read the rendered report in the
page. It runs the same `run_verification` pipeline and auth as the CLI (Claude
Code subscription, or `ANTHROPIC_API_KEY`). Host/port: `CVERIFY_WEB_HOST` /
`CVERIFY_WEB_PORT` (default `127.0.0.1:8000`).

**Viewing it from your laptop when the code runs on a remote server.** The server
listens on `127.0.0.1` only, so reach it over an SSH tunnel rather than exposing
the port:

```bash
# on the server
cverify-web                                   # listens on 127.0.0.1:8000

# on your laptop (a second terminal): forward local 8000 -> the server's 8000
ssh -L 8000:localhost:8000 <user>@<server>
# then open http://localhost:8000 in your local browser
```

- Already connected over SSH? Run the `ssh -L ...` in a second local terminal, or
  add `LocalForward 8000 localhost:8000` to your `~/.ssh/config` host entry.
- VS Code / Cursor Remote-SSH forwards the port automatically (or use the Ports
  panel -> forward `8000`).
- Direct access (only if you control the firewall): `CVERIFY_WEB_HOST=0.0.0.0
  cverify-web`, then `http://<server-ip>:8000` — less secure; prefer the tunnel.

### Environment variables (keys & contacts)

**Nothing here is required** — the keyless floor runs with no keys at all. Keys
only *raise rate limits* or *unlock extra sources*. Every secret lives **only**
in the process environment or a gitignored `.env` — **never in source, never
committed** (e.g. your own OpenAlex key stays in your shell, not in the repo).

Set them by copying `.env.example → .env`, **or** by `export`-ing in your shell
(a shell `export` wins over `.env`):

```bash
export OPENALEX_API_KEY=…      # your OpenAlex key — polite pool + open-access full-text
export ANTHROPIC_API_KEY=…     # only to use the API instead of the Claude Code subscription
```

| Variable | Enables (absent ⇒ skipped, fail-soft) |
|---|---|
| `ANTHROPIC_API_KEY` | LLM backends via the API instead of your Claude Code subscription. |
| `S2_API_KEY` | The Semantic Scholar grounding source. |
| `OPENALEX_API_KEY` | OpenAlex grounding **and** open-access full-text PDF lookup. Keyless works (rate-limited); a key adds the polite pool. |
| `UNPAYWALL_EMAIL` | Unpaywall as an open-access full-text source (needs a contact email, no key). |
| `CONTACT_EMAIL` | Polite `User-Agent` / OpenAlex `mailto` for full-text lookups. |
| `CROSSREF_MAILTO` | Crossref's polite pool (higher rate limits, no key). |
| `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Keys the last-resort open-web search fallback (Google CSE; else keyless DuckDuckGo). The fallback itself is **off** unless `ENABLE_WEB_SEARCH=true` — arXiv/OpenAlex/Unpaywall/DOI lookups run regardless. |
| `DISCORD_BOT_TOKEN` | Running the Discord bot (`cverify-bot` → `/check`). |

The full template — incl. model routing (`MODEL_BULK`/`MODEL_JUDGE`), the cost
ceiling, and bot options — is in [`.env.example`](.env.example).

## Layout

The full module map lives in [`docs/architecture.md`](docs/architecture.md). The
**contract** files (frozen — team sign-off to change) are the ones to know:

| Path | What |
|---|---|
| [`schema.py`](src/citation_verifier/schema.py) | `CitationRecord` + enums + `derive_severity` + JSON-Schema export |
| [`interfaces.py`](src/citation_verifier/interfaces.py) | Protocols (Extractor / Resolver / VerificationBackend / StageFn) + result/usage dataclasses |
| [`spec/v0.1/record.schema.json`](spec/v0.1/record.schema.json) | the committed JSON Schema for one record (drift-checked by `make schema`) |
| [`SKILL.md`](.claude/skills/verify-citations/SKILL.md) | the verification method + frozen output table + enum vocabularies |

Everything else hangs off those seams — `ingest` · `extract/` · `grounding/` ·
`stages/` · `backends/` · `render` · `bot/` (Discord front-end, see
[`docs/DISCORD_BOT.md`](docs/DISCORD_BOT.md)) · `chbench/` (dataset builder) ·
`evals/` (scoring boundary). Per-paper artifacts land in `papers/<paper_id>/`
(gitignored).

## For contributors / agents

- **Start here:** [`AGENTS.md`](AGENTS.md) (run/test/conventions/off-limits) and
  [`CLAUDE.md`](CLAUDE.md) (Claude Code session guide).
- **Design rationale:** [`docs/architecture.md`](docs/architecture.md) and
  [`docs/DECISIONS.md`](docs/DECISIONS.md).
- **Benchmark dataset** (CitationHallucinationBench): the gold set the agents are
  scored against lives off-repo at the shared
  `/scratch/datasets/citation_verification_benchmark/` (`$CHBENCH_DATA_DIR`
  overrides), built by `src/chbench/` — see [`docs/DATASET.md`](docs/DATASET.md).
- **Team workflow:** work on your own branch and PR to `main` — never push directly
  to `main`, it's the Discord bot's deployment branch and must stay green
  (`make smoke`). Modules communicate **only** via the frozen schema on disk.
- Python 3.11+. Secrets only in a gitignored `.env`.
