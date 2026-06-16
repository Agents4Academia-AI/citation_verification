# Project progress log

> Each session, append a new entry below. Most-recent at the top.
> See `AGENTS.md` for conventions. Read more on the pattern:
> [Agents4Academia-AI/example-agents/04-operating-well/working-as-a-team.md](https://github.com/Agents4Academia-AI/example-agents/blob/main/04-operating-well/working-as-a-team.md)

---

## 2026-06-16 — session 1 (bootstrap the citation verifier)

**Goal:** Stand up the initial citation-verifier agent (Track 3 / Claude Agent SDK)
as the team's starting point.

**Done:**
- `src/agent.py` — Track-3 entry: resolves an arXiv id/link (or local PDF), runs
  `query()`, streams + saves the verification table.
- `src/paper_lookup.py` — grounding tool querying Crossref + arXiv (stdlib only);
  pattern adapted from PaperArena's `cross_ref_lookup`, extended to two sources.
- `.claude/skills/verify-citations/SKILL.md` — the method: correctness / relevance /
  priority rubrics + the fixed output table; injected into the system prompt.
- README / AGENTS / requirements updated for the Track-3 stack.

**Verified by:**
- `python src/paper_lookup.py "Vaswani ... 2017 NeurIPS"` → correct Crossref + arXiv match.
- Minimal `query()` run with the custom tool: agent grounded its answer and even
  flagged Crossref repost DOIs vs. the 2017 original (4 turns, ~$0.43).
- **Not yet** run end-to-end on a full multi-citation paper.

**Not done / blocked:**
- Full end-to-end run on a real paper (cost/time) — ready, not yet executed.
- Relevance is abstract-level only; comparison-objectiveness check not built.

**Next session — start here:**
1. Run `python src/agent.py 1706.03762` end-to-end; eyeball the table.
2. Tune the obligatory-vs-helpful rubric + table columns in SKILL.md on a real draft.
3. Add a pytest for `paper_lookup` (mock the Crossref/arXiv responses).

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
