# AGENTS.md — what Claude Code reads at session start

> Replace placeholders with your team's actual commands and conventions. Keep
> this file short — every line costs attention each session.

## How to run

- Install: `pip install -r requirements.txt` (Track 3 — needs Claude Code installed/authed, or `ANTHROPIC_API_KEY`)
- Run:     `python src/agent.py <arxiv-id | arxiv-url | path-to.pdf>`
- Check tool only (no LLM): `python src/paper_lookup.py "<a reference string>"`
- Test:    `pytest -q`  (no tests yet)

## Conventions

- Python 3.11+
- Functions are short. If something needs a paragraph of comment to explain,
  refactor it.
- Type-hint module boundaries; internal code can be untyped.
- snake_case for functions, PascalCase for classes, ALL_CAPS for constants.

## What's off-limits

- Do not modify `data/raw/` — this is our golden dataset.
- Do not commit anything under `secrets/`, `.env*`, or `*.key`.
- Do not push directly to `main` — open a PR.

## Where the important stuff lives

- `src/agent.py` — Track-3 entry point: `@tool` + `query()`, arXiv/PDF → verification table
- `src/paper_lookup.py` — grounding tool: Crossref + arXiv metadata lookup (stdlib only)
- `.claude/skills/verify-citations/SKILL.md` — the verification method (rubrics + output table)
- `papers/` — downloaded / input PDFs
- `tests/` — pytest tests (none yet)
- `evals/` — evaluation harness (if/when we add one)

## Operating principles

1. *Think before coding* — state assumptions, surface confusion first.
2. *Simplicity first* — nothing beyond what was asked.
3. *Surgical changes* — touch only what the request requires.
4. *Goal-driven* — give success criteria, let it loop.

## When in doubt

- Ask a clarifying question before generating code.
- Prefer surgical changes over big refactors.
- If the task affects multiple files, sketch the diff first and ask before applying.
