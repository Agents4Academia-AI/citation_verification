# Project progress log

> Each session, append a new entry below. Most-recent at the top.
> See `AGENTS.md` for conventions. Read more on the pattern:
> [Agents4Academia-AI/example-agents/04-operating-well/working-as-a-team.md](https://github.com/Agents4Academia-AI/example-agents/blob/main/04-operating-well/working-as-a-team.md)

---

## 2026-06-16 — session: integrate the initialization + workable demo (phy)

**Goal:** materialize the workflow-designed initialization onto `main`, refactor
the repo to the new layout, and prove a *workable* baseline runs end-to-end.

**Done:**
- Refactored `main` from the toy `src/agent.py` to a src-layout package
  (`citation_verifier` + `chbench`) + `evals/` + `tests/` + `spec/` + `config/`
  + `docs/`. Yunqiao's `paper_lookup` (Crossref+arXiv) and `SKILL.md` were carried
  forward, refactored into `grounding/` and `.claude/skills/` respectively.
- Installed + verified offline: 26 modules import clean, `pytest` → 32 passed.
- Fixed 3 real bugs found by running it:
  - `grounding/resolver.py`: the match cascade only gated the single
    highest-title candidate, so a 2025 *repost* DOI hid the real 2017 record →
    now iterates candidates and skips (not aborts on) gate failures; also drops
    "et al" → bogus surname "al" from the author gate. Real papers now resolve.
  - `backends/agentic.py`: passed the model-name *string* as the relevance
    `judge` (callable) → `TypeError`. Now `judge` is an optional seam (default
    `None` = honest deterministic abstain); judge-tier usage only charged when a
    judge actually runs.
  - `backends/claude_code.py`: `_extract_json` truncated arrays at the first
    nested `]`; and `_parse_records` rejected the model's loose output (claim/
    cited_as as strings, evidence as list-of-strings). Now parses to the closing
    fence and builds each record from the extractor stub, overlaying *coerced*
    verdicts (incl. `"does not"`→`does_not`). LLM path now yields real records.

**Verified by (live, using `.env` ANTHROPIC_API_KEY):**
- `agentic` backend on a 2-citation paper (1 real ref, 1 fabricated): real →
  `exists=yes` (grounded via arXiv), fake → `unverified` (no match), relevance
  honestly abstains; ~115 tokens, ~$0.0002, no crashes.
- `claude_code` backend, same paper, Sonnet: real → `yes/supports/ok`, fake →
  `no/does_not/high` with correct reasoning (flagged fictional authors + fake
  venue); ~6,086 tokens, ~$0.19. Demonstrates the two-backend token comparison
  the Notion plan asks for.
- `pytest tests/` → 32 passed after the fixes.

**Not done / blocked:**
- `model_judge` config default (`claude-opus-4-5`) is not a valid model id; the
  demo routed to `claude-sonnet-4-6`. Set `MODEL_JUDGE`/`MODEL_BULK` in `.env` to
  real ids before relying on the defaults.
- agentic relevance abstains (no judge wired) — by design for the deterministic
  baseline; the relevance-judge seam (`relevance_judge`) is ready to fill.

**Next session — start here:**
1. Wire an LLM `relevance_judge` into the agentic backend (the seam exists) so it
   does STEP 2 without the full Claude Code loop.
2. Run `cverify <arxiv-id> --backend agentic|claude_code` on a real paper and
   eyeball the table; point `make eval` at the written `papers/<id>/report.json`.
3. Fix the default model ids in `config.py` / `.env.example` to valid strings.

---

## 2026-06-16 — session: shared init (docs-meta module)

**Goal:** stand up the shared, frozen initialization the four module branches
fork from — repo docs, build glue, config, the frozen SKILL.md contract, and the
offline test/eval scaffolding.

**Done:**
- Froze the contract surface: `schema.py` + `interfaces.py` + `__init__.py` +
  `spec/v0.1/record.schema.json` (committed) + `.claude/skills/.../SKILL.md`.
  SKILL.md table header = the 8 columns and its enum vocab matches `schema.py`
  token-for-token (`does_not` → "does not"); added a STEP-2
  relevance-justification note.
- Refreshed repo docs for the realized layout: `README.md` (Notion 3-step →
  files + two-backend comparison + structure table), `AGENTS.md`, `CLAUDE.md`,
  `docs/architecture.md`, `docs/DECISIONS.md` (promoted from `decisions-phy.md`;
  `[change]`s marked `[open]`). `docs/DATASET.md` and `config/*.yaml` landed via
  sibling work and were left intact (verified, consistent).
- Build glue: `Makefile` (install/test/lint/smoke/eval/bench/schema/clean),
  `requirements.txt` (thin `-e .[llm]` pointer), extended `.gitignore`
  (papers/<id>, report-*, caches; `.env.example` un-ignored; spec/ & config/
  stay committed), `papers/.gitkeep`.
- Tests (all offline, no SDK/network): `tests/conftest.py`,
  `tests/test_schema.py` (enums mirror SKILL.md; key; `derive_severity`;
  round-trip; committed spec is byte-identical to the model export),
  `tests/test_render.py` (exact SKILL.md table header; `does_not`→"does not";
  determinism; JSON round-trip), `tests/test_eval.py` (smoke-gold sanity +
  `run_eval` round-trip on a perfect canned report). Sibling modules
  (`render`, `run_eval`) are reached via `importorskip` so the suite is green on
  any partial checkout and tightens to a hard contract guard once they land.
- Fixtures: `tests/fixtures/sample_records.jsonl` (every enum value covered) and
  `evals/smoke/gold.jsonl` (≥3 fabricated + ≥3 wrong-metadata, every gold
  carries `labels` + anti-circularity `provenance`).

**Verified by:**
- `PYTHONPATH=src python -m pytest tests/` → 32 passed, offline.
- Committed `spec/v0.1/record.schema.json` byte-identical to
  `json_schema()` export (`make schema` clean).
- `config/sources.yaml` + `config/venues.yaml` load via pyyaml; no secrets.
- `make test`, `make smoke`, `make schema` all green; `python -m evals.run_eval`
  on a perfect canned report scores correctness-F1 = 1.0.

**Not done / blocked:**
- `make smoke`'s `run_eval evals/smoke` shows 0.000 only because no agent
  prediction report has been written there yet; the in-test canned report scores
  1.0, proving the join.

**Next session — start here:**
1. Once an agent run writes `papers/<id>/report.json`, point `make eval` at it.
2. Register shared S2/OpenAlex keys in `.env` (or confirm keyless-floor demo).
3. Resolve the `[open]` decisions in `docs/DECISIONS.md` at the merge meeting.

---

## YYYY-MM-DD — session 0 (template)

**Goal:** [one line — what you set out to do this session]

**Done:**
- bullet
- bullet

**Verified by:**
- `pytest` / manual check / etc.

**Not done / blocked:**
- bullet

**Next session — start here:**
1. ...
2. ...
3. ...

---

## Conventions

- **Append, don't rewrite.** This is your audit trail.
- **"Done" means verified.** Not "wrote the code", not "looks fine". Verified by something runnable.
- **Always end with "next session — start here."** Three numbered, atomic tasks.
- **One log per repo.** Don't fragment across files.
