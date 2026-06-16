"""
Citation Verifier — entry point (Track 3 / Claude Agent SDK).

What it does, end to end:
  1. You give it an arXiv link/ID or a local PDF.
  2. The SDK runs Claude Code's agent loop: it reads the draft, pulls out each
     reference and the claim it supports, and verifies it.
  3. It grounds every check in real data via two tools:
       - lookup_paper  (our custom tool: Crossref + arXiv, see paper_lookup.py)
       - WebSearch / WebFetch / Read  (built into the Agent SDK, for free)
  4. It prints — and saves — a citation-verification table:
       correctness (real? metadata right?) · relevance · priority · issue.

The agent loop, the built-in tools, and skill loading all come from the SDK —
our job is the tool (paper_lookup) and the method (.claude/skills/...).

Run from the repo root:
    pip install -r requirements.txt        # claude-agent-sdk (+ Claude Code installed)
    python src/agent.py 1706.03762
    python src/agent.py https://arxiv.org/abs/2310.06825
    python src/agent.py ./papers/my_draft.pdf
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    query, tool, create_sdk_mcp_server,
    ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage,
)

# Works both as a script (`python src/agent.py`) and as a package import.
try:
    import paper_lookup
except ModuleNotFoundError:
    from . import paper_lookup

ROOT = Path(__file__).resolve().parent.parent          # repo root (src/ is one down)
PAPERS_DIR = ROOT / "papers"
SKILL_FILE = ROOT / ".claude" / "skills" / "verify-citations" / "SKILL.md"


# ── 1. Wrap our grounding function as a tool the agent can call ───────────────
# The description is written like an instruction: it tells the model WHEN to use
# the tool and WHAT it gets back.
@tool(
    "lookup_paper",
    "Verify a cited reference against authoritative metadata sources "
    "(Crossref for published venues, arXiv for preprints). "
    "USE WHEN: checking whether a cited paper is real and whether its authors, "
    "venue, year and title are correct. "
    "Pass the fullest reference string you have (authors + title + year + venue) "
    "for best matches — a bare title is noisy. "
    "Returns candidate records to compare against what the draft claims; it does "
    "NOT decide correctness for you — you compare and judge.",
    {"reference": str, "source": str},
)
async def lookup_paper(args: dict[str, Any]) -> dict[str, Any]:
    result = paper_lookup.lookup_paper(
        args["reference"], source=args.get("source") or "auto"
    )
    return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}


papers_server = create_sdk_mcp_server(name="papers", tools=[lookup_paper])


# ── 2. Turn the input (arXiv id/link or local path) into a local PDF ──────────
def resolve_to_pdf(arg: str) -> Path:
    """Accept an arXiv id/URL or a local .pdf path; return a local PDF path."""
    if arg.lower().endswith(".pdf") and Path(arg).exists():
        return Path(arg)

    m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", arg)  # e.g. 1706.03762, 2310.06825v2
    if not m:
        sys.exit(f"Could not parse an arXiv id or find a PDF at: {arg!r}")
    arxiv_id = m.group(1)
    dest = PAPERS_DIR / f"{arxiv_id}.pdf"
    PAPERS_DIR.mkdir(exist_ok=True)
    if not dest.exists():
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"↓ downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "citation-verifier/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
    return dest


# ── 3. The method lives in the SKILL.md (single source of truth) ──────────────
# We inject its body straight into the system prompt so the run is deterministic.
# The same file is also a drop-in Claude Code skill (it auto-loads in a terminal
# session). One method, two ways to run it.
def load_method() -> str:
    text = SKILL_FILE.read_text()
    return re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL).strip()


async def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    pdf_path = resolve_to_pdf(sys.argv[1])
    print(f"✓ paper: {pdf_path}\n")

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a meticulous citation-verification agent for academic "
            "papers. You NEVER assert from memory that a paper exists or that "
            "its metadata is correct — every such claim must be grounded in a "
            "lookup_paper or web result. Follow this method exactly:\n\n"
            + load_method()
        ),
        mcp_servers={"papers": papers_server},
        allowed_tools=[
            "mcp__papers__lookup_paper",  # our grounding tool
            "Read",        # read the PDF draft (the SDK's Read parses PDFs)
            "WebSearch",   # papers not in Crossref/arXiv; sanity checks
            "WebFetch",    # fetch a cited paper's page for relevance checks
        ],
        cwd=str(ROOT),
        max_turns=80,      # a paper can have many references
    )

    prompt = (
        f"Verify the citations in the paper at: {pdf_path}\n"
        "Read it, then produce the citation-verification table and summary "
        "exactly as the method specifies."
    )

    transcript: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
                    transcript.append(block.text)
                # Show tool activity so you can watch the agent work.
                elif getattr(block, "name", None) and hasattr(block, "input"):
                    arg = block.input.get("reference") or block.input.get("query") \
                        or block.input.get("url") or block.input.get("file_path") or ""
                    print(f"\n  → {block.name}({str(arg)[:70]})", flush=True)
        elif isinstance(message, ResultMessage):
            print(
                f"\n\n[done in {message.num_turns} turn(s), "
                f"cost ${message.total_cost_usd or 0:.4f}]"
            )

    out = ROOT / f"report-{pdf_path.stem}.md"
    out.write_text("".join(transcript))
    print(f"saved → {out}")


if __name__ == "__main__":
    asyncio.run(main())
