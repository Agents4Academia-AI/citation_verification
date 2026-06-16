# Project progress log

> Each session, append a new entry below. Most-recent at the top.
> See `AGENTS.md` for conventions. Read more on the pattern:
> [Agents4Academia-AI/example-agents/04-operating-well/working-as-a-team.md](https://github.com/Agents4Academia-AI/example-agents/blob/main/04-operating-well/working-as-a-team.md)

---

## 2026-06-16 — session: Discord bot front-end (`/check`) (phy)

**Goal:** ship a Discord bot that pipelines citation verification behind a
`/check <arxiv>` slash command (accepting bare id, `/abs/` and `/pdf/` URLs) and
posts the hallucination report; make it deployable on the team's test server.

**Done:**
- New `src/citation_verifier/bot/` package (import-safe; `discord` is a runtime-
  only import): `config.py` (token/guild/backend/cap from env+.env, reuses
  `load_settings`), `discord_bot.py` (`CitationBot` + `/check`, `/help`, `/ping`),
  `report.py` (compact embed: headline + counts + emoji-tagged flagged list +
  full report attached as `.md`). Console script `cverify-bot` + `[bot]` extra
  (`discord.py`, `pypdf`). `/check` defers then runs the SAME
  `run_verification` on a worker thread (`asyncio.to_thread`).
- Orchestrator/CLI gained an optional `max_citations` (CLI `--limit`) bound — a
  clean truncation at the right layer (the bot caps per-command runs so a big
  bibliography can't exceed Discord's 15-min interaction window).
- Ran an adversarial review **workflow** (6 dimensions → verify): 11/12 findings
  confirmed and fixed —
  - guarded the post-verify render+send and added a `CommandTree.on_error`
    backstop (no more hung "thinking…" interactions);
  - clipped all user/exception text under Discord's 2000-char content cap +
    `max_length` on the `paper` option;
  - per-cache-key in-flight **coalescing** (concurrent identical `/check`s share
    one run; no `report.json` write race);
  - cache key now includes a **settings fingerprint** (model/web/keys) so a
    config change never serves a stale verdict;
  - oversize-attachment guard; `allowed_mentions=none`;
  - **orchestrator resume bug:** the cache-hit path overwrote `run.json` with a
    zeroed `RunUsage` and showed `0 tok`. Added `_load_run` to restore the real
    token/cost on resume (fixes the footer + stops clobbering accounting).
- Recall bug fix in `stages/correctness.py`: `_reference_string` dropped the
  `arxiv_id`/`doi` the extractor parsed, so the resolver's DOI/arXiv exact-match
  cascade never fired. Now appended (still grounded — only accepts an id a
  fetched candidate also carries). Real papers with a DOI/surfaced-arXiv id now
  resolve to `exists=yes` instead of `unverified`.
- Docs: `docs/DISCORD_BOT.md` (invite URL + scopes/permissions, run, troubleshoot),
  README Quickstart + structure row, `.env.example` (Discord section; **fixed the
  invalid default model ids** to `claude-haiku-4-5-20251001` / `claude-sonnet-4-6`).
- Follow-ups (same session, after live use):
  - **Duplicate commands** — `/check`/`/help`/`/ping` showed twice because a
    global sync fired as a fallback (during testing, before the bot was invited)
    and then a guild sync ran. `setup_hook` now clears global commands on a
    successful guild sync (self-healing; global removal propagates in ~1h).
  - **💰 Cost (USD)** field added to the result embed (was only in the footer);
    agentic shows `$0.0000 (no LLM)`, claude_code shows real API spend.
  - **CROSSREF_MAILTO not wired**: `paper_lookup` reads it from `os.environ` at
    import, so the `.env` value was ignored (stayed in the slow anonymous pool).
    The bot now bridges `.env` → `os.environ` in `main()`; also added it to the
    cache fingerprint.
  - **All-`unverified` episode diagnosed** = transient Crossref timeouts/throttle
    after the heavy 108-citation run hammered it (no backoff in `_get`; sources
    fail-soft to empty → no match → unverified). arXiv/DBLP stayed up; arXiv
    keyword search is weak on messy reference strings. agentic recovers on
    cooldown; `claude_code` is the resilient backend meanwhile. Proper fix
    (retry/backoff + cleaner queries in the shared grounding module) left to a
    follow-up — out of bot scope.

**Verified by (live, using `.env` DISCORD_BOT_TOKEN + ANTHROPIC_API_KEY):**
- Logged in as `main_branch_agent#5388`; after the bot was invited, guild sync
  registered `/check`, `/help`, `/ping` to the test server (guild
  1516408471208202260) — confirmed via `tree.fetch_commands(guild=…)`.
- End-to-end through the bot's exact path on `2505.03335` (182 stubs / 108 cites):
  capped runs produce the embed + attached report; `guo2025deepseek` →
  `exists=yes` (DOI) after the recall fix; cap shown as a "Scope" field; resume
  restores real usage (2,635 tok / $0.0036, non-zero).
- `pytest tests/` → 32 passed; bot package imports with no network/SDK touched.

**Not done / blocked:**
- arXiv-id-only citations whose title has LaTeX escapes (e.g. T\"ULU 3) still
  read `unverified` under `agentic`: the multi-source *search* doesn't surface
  the id-bearing candidate, and a robust fix needs a direct arXiv-id fetch in the
  resolver (deliberately out of scope — "don't optimise"). `claude_code` handles
  these; it's the recommended deep-check backend.
- `agentic` grounding is ~12 s/citation (sequential multi-source fan-out); hence
  the `BOT_MAX_CITATIONS=25` default. Not optimised by design.

**Next session — start here:**
1. (Optional) direct arXiv-id/DOI fetch path in the resolver to lift `agentic`
   recall + speed on id-bearing refs (then raise/remove the bot cap).
2. Wire the LLM `relevance_judge` into `agentic` so `/check` reports STEP-2
   relevance without the full `claude_code` loop.
3. Add a `fresh:true` `/check` option (resume=False) for forced re-verification.

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
