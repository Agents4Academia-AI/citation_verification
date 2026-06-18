# CLAUDE.md — Claude Code session guide

Short pointer file. The real, maintained guidance lives elsewhere — read these
first and do not duplicate them here:

- **[`AGENTS.md`](AGENTS.md)** — how to run/test/lint, conventions, off-limits,
  and where everything lives. Start here every session.
- **[`docs/architecture.md`](docs/architecture.md)** — the layered design, the
  `(claim, citation)` data flow, the two backends, and the extension points.
- **[`docs/DECISIONS.md`](docs/DECISIONS.md)** — the frozen decisions (the seam,
  derived severity, anti-circularity) and the open `[change]` proposals.
- **[`docs/DATASET.md`](docs/DATASET.md)** — CitationHallucinationBench design.
- **[`.claude/skills/verify-citations/SKILL.md`](.claude/skills/verify-citations/SKILL.md)**
  — the verification *method* and the **frozen** output table + enum vocabularies.

## Team workflow & deployment

- **Team:** Agents4Academia. The benchmark dataset is **shared** at
  `/scratch/datasets/citation_verification_benchmark/` (not in the repo; built by
  `src/chbench/`, `$CHBENCH_DATA_DIR` overrides). Everyone works on their **own
  branch** and merges into `main` via PR — never push directly to `main` (see
  `AGENTS.md`).
- **`main` is the deployment branch.** It must stay deployable at all times: the
  Discord front-end bot (`/check`) is run from `main`. Keep it green
  (`make smoke`) and free of personal/machine-specific config.
- **Local bot setup (gitignored — never committed):**
  `cp .claude/settings.json.example .claude/settings.json`, then put your
  `DISCORD_BOT_TOKEN` in `.claude/channels/discord/.env`. The live
  `.claude/settings.json`, `.claude/channels/` (token + allowlist) are
  per-deployer state and stay out of git. Only `.claude/skills/` (the team
  contract) and `*.example` templates are committed.

## The one thing to remember

This repo is **four modules behind one frozen contract**. Depend only on
`citation_verifier.schema` (`CitationRecord` + enums) and
`citation_verifier.interfaces` (the Protocols) — never another module's
internals. The core must import and run **without** `claude-agent-sdk` and
**without** network; the SDK is a lazy/optional import and network calls fail
soft.

## Fast loop

```bash
make install   # uv pip install -e '.[dev]'  — offline floor
make test      # full pytest suite, offline
make smoke     # tests + run_eval on the in-repo smoke gold
make schema    # regenerate + drift-check spec/v0.1/record.schema.json
```

If a task touches a **contract** file (`schema.py`, `interfaces.py`,
`spec/…/record.schema.json`, `SKILL.md`), stop and flag it for team sign-off —
those are frozen seams the other branches build against.
