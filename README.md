# [Team Project Name]

> One-paragraph summary of what your team is building and why. Replace this
> with your real description — it's the first thing visitors and the agent
> will read.

**Team:** Name 1 · Name 2 · Name 3 · Name 4
**Advisor:** TBC
**Day-5 demo targets (Fri 19 Jun):**
1. ...
2. ...
3. ...

---

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python src/agent.py
```

## What's in here

| File | What |
|------|------|
| [`AGENTS.md`](AGENTS.md) | What the agent reads at session start — install/run/test commands, conventions, off-limits stuff |
| [`claude-progress.md`](claude-progress.md) | Session continuity log — append each session, agent reads on startup |
| `src/agent.py` | Hello-world agent. Replace this with what you're actually building. |
| `requirements.txt` | Python dependencies |

## Working norms

- Branches: `name/feature` — one Claude Code session per branch
- Shared files (`AGENTS.md`, `.mcp.json`, anything in `src/`) get reviewed via PR
- Personal Claude Code settings go in `.claude/settings.local.json` (gitignored)
- No auto-merge of agent PRs — a teammate reviews
- Slack-ping before starting a non-trivial Claude Code session

See [`Agents4Academia-AI/example-agents/04-operating-well/working-as-a-team.md`](https://github.com/Agents4Academia-AI/example-agents/blob/main/04-operating-well/working-as-a-team.md) for the longer version.

## Acknowledgements

Built during [Agents4Academia](https://github.com/Agents4Academia-AI), 14–26 June 2026.
