# Citation Verifier

> An agent that checks the **citations in a paper draft**: whether each cited
> work is *real*, whether its *authors / venue / year* are correct, and whether
> it actually *supports the claim* it's attached to. Input an arXiv link or PDF;
> output a citation-verification table (correctness · relevance · priority · issue).

**Team:** Name 1 · Name 2 · Name 3 · Name 4
**Advisor:** TBC
**Day-5 demo targets (Fri 19 Jun):**
1. **Correctness** — for each citation, verify the paper is real and its authors/venue/year are right, grounded in Crossref + arXiv.
2. **Relevance & priority** — judge whether the cited paper supports the claim, and whether the citation is *obligatory* vs. *helpful* background.
3. **Output** — a citation-verification table from an arXiv link or PDF, saved to `report-<id>.md`.

---

## How it works

```
arXiv id / PDF ─▶ src/agent.py ─▶ query()  ── reads draft, per citation:
                                     │          ├─ lookup_paper  (Crossref + arXiv)   ← our tool
                                     │          ├─ WebSearch / WebFetch / Read        ← built in
                                     │          └─ judge: correct? relevant? priority?
                                     ▼
                            citation-verification table  +  "fix before submission"
```

**What makes it an *agent* (not just a chatbot):** a plain chat model asked "is
this citation correct?" will *guess* — the exact failure this tool exists to
catch. The difference is **grounding + a loop**:

| Piece | Where it comes from |
|---|---|
| The agent loop (read → look up → compare → judge → next) | the SDK's `query()` — free |
| Reading the PDF, web search/fetch | built-in tools — free |
| **`lookup_paper`** — Crossref + arXiv canonical metadata | `src/paper_lookup.py` (ours) |
| **The method** — rubrics + output table | `.claude/skills/verify-citations/SKILL.md` (ours) |

The first rule in the skill: *never decide correctness from memory.* Every
"this paper exists / its metadata is X" claim must come from a real lookup.

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # or just be logged in to Claude Code

python src/agent.py 1706.03762                   # arXiv id
python src/agent.py https://arxiv.org/abs/2310.06825
python src/agent.py ./papers/my_draft.pdf        # local PDF
```

The table streams to your terminal and is saved to `report-<id>.md`.
Sanity-check the grounding tool alone (no LLM needed):

```bash
python src/paper_lookup.py "Vaswani et al. Attention is all you need 2017 NeurIPS"
```

## What's in here

| File | What |
|------|------|
| `src/agent.py` | Track-3 entry point: `@tool` + `query()`; arXiv/PDF in → table out |
| `src/paper_lookup.py` | The grounding tool: Crossref + arXiv metadata (standard library only) |
| [`.claude/skills/verify-citations/SKILL.md`](.claude/skills/verify-citations/SKILL.md) | The method: correctness/relevance/priority rubrics + the output table |
| [`AGENTS.md`](AGENTS.md) | What Claude Code reads at session start — commands, conventions, off-limits |
| [`claude-progress.md`](claude-progress.md) | Session continuity log — append each session |
| `requirements.txt` | Python dependencies |

## What we borrowed from PaperArena

[PaperArena](https://github.com/ustc-ai4science/PaperArena) (USTC AI4Science) is a
benchmark for tool-augmented agents reasoning over scientific papers. Ideas we
adapted:

- **Its `tools/cross_ref_lookup.py` pattern** — take a free-text reference, hit a
  scholarly API, return structured metadata, and let an *LLM do the matching*. Our
  `src/paper_lookup.py` is this idea, improved to query **both Crossref and arXiv**
  (PaperArena hit only arXiv despite the file name).
- **A focused tool environment** — we use the SDK's built-in Read/WebSearch/WebFetch
  for PDF/search and add only the one tool that doesn't exist for free (scholarly
  metadata).
- **An operational caution from their results** — even strong agents score ~39% on
  cross-paper reasoning and "invoke more tools than necessary." So citation
  *relevance* is the genuinely hard part, and tool calls cost money — the skill
  tells the agent to use the fullest query and not over-search.

## Roadmap (beyond the Day-5 demo)

- **arXiv LaTeX source** instead of PDF: the `.bib`/`.bbl` + `\cite` locations are
  ground truth for citation keys and contexts — more reliable than PDF parsing.
- **Comparison objectiveness**: check whether obligatory baselines actually appear
  in the experiments/tables.
- **Cost**: cheaper model for bulk metadata checks, stronger model for relevance.

## Working norms

- Branches: `name/feature` — one Claude Code session per branch
- Shared files (`AGENTS.md`, `.mcp.json`, anything in `src/`) get reviewed via PR
- Personal Claude Code settings go in `.claude/settings.local.json` (gitignored)
- No auto-merge of agent PRs — a teammate reviews
- Slack-ping before starting a non-trivial Claude Code session

See [`Agents4Academia-AI/example-agents/04-operating-well/working-as-a-team.md`](https://github.com/Agents4Academia-AI/example-agents/blob/main/04-operating-well/working-as-a-team.md) for the longer version.

## Acknowledgements

Built during [Agents4Academia](https://github.com/Agents4Academia-AI), 14–26 June 2026.
