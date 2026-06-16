"""
claude_code.py — the "claude_code" backend: a skill-driven Agent-SDK loop.

This is the faithful refactor of the baseline ``src/agent.py`` into a
:class:`BaseBackend`. It runs Claude Code's agent loop via
``claude_agent_sdk.query()``, injecting the frozen ``SKILL.md`` method into the
system prompt and exposing the grounding ``lookup_paper`` function as an SDK
tool. The model reads the draft, grounds every correctness check in a
``lookup_paper`` / web result, and emits the structured verification — which we
parse back into :class:`CitationRecord` rows and pair with a
:class:`RunUsage` captured from the run's ``ResultMessage``.

What changed vs the baseline (behavior preserved, structure improved):
  - the printed Markdown table is no longer the product — the backend returns
    ``list[CitationRecord]`` (rendering is ``render.py``'s job);
  - the SDK is a **lazy import inside methods** (:func:`_require_sdk`) so this
    module imports with the SDK absent; calling :meth:`verify` without it raises
    a clear ``RuntimeError`` naming the ``llm`` extra;
  - token/USD usage is captured from ``ResultMessage`` via
    :func:`citation_verifier.backends.usage.usage_from_result_message`;
  - the asyncio ``query()`` loop is driven synchronously so the backend matches
    the synchronous :class:`VerificationBackend` seam.

The model is asked to emit a fenced ``json`` block of records conforming to the
contract; we parse that block and validate each row, degrading unparseable
output to ``unverified`` rather than crashing.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from ..interfaces import PaperSource, RunUsage, VerificationResult
from ..schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Evidence,
    Exists,
    ModelTier,
    Priority,
    SupportsClaim,
    derive_severity,
)
from .base import BaseBackend, register
from .usage import usage_from_result_message

# Repo root: src/citation_verifier/backends/claude_code.py -> parents[3].
_ROOT = Path(__file__).resolve().parents[3]
_SKILL_FILE = _ROOT / ".claude" / "skills" / "verify-citations" / "SKILL.md"

_MISSING_SDK_MSG = (
    "The 'claude_code' backend needs the Claude Agent SDK. Install the optional "
    "extra:  uv pip install -e '.[llm]'   (or: pip install claude-agent-sdk), "
    "and ensure Claude Code is authenticated or ANTHROPIC_API_KEY is set. "
    "The 'agentic' backend runs without the SDK."
)


def _require_sdk() -> Any:
    """Lazily import claude_agent_sdk; raise an informative error if absent.

    Returns the imported module so callers can reach its symbols. Keeping the
    import inside this function (never at module top) is what makes
    ``import citation_verifier.backends.claude_code`` succeed offline.
    """
    try:
        import claude_agent_sdk  # type: ignore
    except ImportError as exc:  # SDK not installed -> actionable message
        raise RuntimeError(_MISSING_SDK_MSG) from exc
    return claude_agent_sdk


def load_skill_body(skill_file: Path = _SKILL_FILE) -> str:
    """Return the SKILL.md body with its YAML front-matter stripped.

    The front-matter (``---\\n...\\n---``) is metadata for Claude Code's skill
    loader; the prose+table below it is the method we inject into the system
    prompt. Missing file degrades to an empty string (verify will still run, the
    model just lacks the rubric) rather than crashing at import or call time.
    """
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError:
        return ""
    return re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL).strip()


@register
class ClaudeCodeBackend(BaseBackend):
    """Skill-driven Claude Agent-SDK backend (``name='claude_code'``).

    Importing this class does NOT import the SDK. The SDK is touched only when
    :meth:`verify` runs the agent loop.
    """

    name = "claude_code"

    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings
        self.model = _setting(settings, "model_judge", "claude-opus-4-5")
        self.max_turns = int(_setting(settings, "max_turns", 80) or 80)
        self.enable_web = bool(_setting(settings, "enable_web_search", False))

    # ──────────────────────────────────────────────────────────────
    def verify(
        self, source: PaperSource, stubs: list[CitationRecord]
    ) -> VerificationResult:
        """Run the skill-driven agent loop and parse structured records.

        Raises ``RuntimeError`` (via :func:`_require_sdk`) if claude-agent-sdk is
        not installed. ``stubs`` are passed to the model as the known
        ``(claim, cite_key)`` pairs to verify so output keys line up with the
        extractor's stubs and the eval join key.
        """
        sdk = _require_sdk()
        result = self._empty_result(source)

        with self._timer() as sw:
            text, usage = asyncio.run(self._run_loop(sdk, source, stubs, result))
        usage.wall_seconds = sw.seconds
        result.usage = usage

        records = self._parse_records(text, source, stubs)
        if not records:
            result.errors.append("model produced no parseable records; emitting unverified stubs")
            records = [self._degrade(stub, source, "no model output parsed") for stub in stubs]

        self._stamp_paper_id(records, source.paper_id)
        result.records = records
        return result

    # ──────────────────────────────────────────────────────────────
    async def _run_loop(
        self,
        sdk: Any,
        source: PaperSource,
        stubs: list[CitationRecord],
        result: VerificationResult,
    ) -> tuple[str, RunUsage]:
        """Drive ``query()`` to completion; return (transcript, usage)."""
        tool = sdk.tool
        create_sdk_mcp_server = sdk.create_sdk_mcp_server
        AssistantMessage = sdk.AssistantMessage
        TextBlock = sdk.TextBlock
        ResultMessage = sdk.ResultMessage
        ClaudeAgentOptions = sdk.ClaudeAgentOptions

        lookup_paper_tool = self._build_lookup_tool(tool)
        papers_server = create_sdk_mcp_server(name="papers", tools=[lookup_paper_tool])

        allowed = ["mcp__papers__lookup_paper", "Read"]
        if self.enable_web:
            allowed += ["WebSearch", "WebFetch"]

        options = ClaudeAgentOptions(
            system_prompt=self._system_prompt(),
            mcp_servers={"papers": papers_server},
            allowed_tools=allowed,
            cwd=str(_ROOT),
            max_turns=self.max_turns,
            model=self.model,
        )

        transcript: list[str] = []
        usage = RunUsage(backend=self.name, model=self.model)

        async for message in sdk.query(prompt=self._user_prompt(source, stubs), options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        transcript.append(block.text)
                    elif getattr(block, "name", None) and hasattr(block, "input"):
                        usage.tool_calls += 1
            elif isinstance(message, ResultMessage):
                usage = usage_from_result_message(message, self.model, self.name)
                usage.tool_calls = max(usage.tool_calls, _count_tool_calls(transcript))

        return "".join(transcript), usage

    # ──────────────────────────────────────────────────────────────
    def _build_lookup_tool(self, tool: Any) -> Any:
        """Wrap grounding.lookup_paper as an SDK @tool (lazy grounding import)."""
        try:
            from ..grounding import lookup_paper  # type: ignore
        except Exception:  # noqa: BLE001 — grounding not on this branch yet
            lookup_paper = _lookup_unavailable

        @tool(
            "lookup_paper",
            "Verify a cited reference against authoritative metadata sources "
            "(Crossref for published venues, arXiv for preprints, DBLP/S2/OpenAlex "
            "when configured). USE WHEN checking whether a cited paper is real and "
            "whether its authors, venue, year and title are correct. Pass the "
            "fullest reference string you have. Returns candidate records to "
            "compare against the draft's claim; it does NOT decide correctness.",
            {"reference": str, "source": str},
        )
        async def _lookup(args: dict[str, Any]) -> dict[str, Any]:
            payload = lookup_paper(args["reference"], source=args.get("source") or "auto")
            return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}

        return _lookup

    # ──────────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        """System prompt = grounding rule + the injected SKILL.md method."""
        return (
            "You are a meticulous citation-verification agent for academic papers. "
            "You NEVER assert from memory that a paper exists or that its metadata "
            "is correct — every such claim must be grounded in a lookup_paper or "
            "web result you actually retrieved. Follow this method exactly:\n\n"
            + load_skill_body()
            + "\n\n## Output for this backend\n"
            "After your analysis, emit ONE fenced ```json code block containing a "
            "JSON array of CitationRecord objects (one per (claim_id, cite_key) "
            "pair). Use the machine enum tokens: exists yes|no|unverified; "
            "supports_claim supports|partial|does_not|unverified; priority "
            "obligatory|helpful. Include claim, cited_as, metadata_issues, and "
            "evidence. Do not hand-author the Markdown table — the JSON is the "
            "source of truth."
        )

    def _user_prompt(self, source: PaperSource, stubs: list[CitationRecord]) -> str:
        """User prompt: where the paper is + the stub pairs to verify."""
        where = source.tex_dir or source.pdf_path or source.work_dir or source.paper_id
        lines = [
            f"Verify the citations in the paper at: {where}",
            f"paper_id: {source.paper_id}",
        ]
        if stubs:
            pairs = [
                {"claim_id": s.claim_id, "cite_key": s.cite_key, "claim": s.claim.text}
                for s in stubs
            ]
            lines.append(
                "Verify exactly these (claim_id, cite_key) pairs and key your "
                "output records by them:\n" + json.dumps(pairs, ensure_ascii=False, indent=2)
            )
        else:
            lines.append("Extract the references and claim sites yourself, then verify each.")
        lines.append(
            "Read the draft, ground every check, then emit the JSON array of "
            "CitationRecords as specified."
        )
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    def _parse_records(
        self, text: str, source: PaperSource, stubs: list[CitationRecord]
    ) -> list[CitationRecord]:
        """Parse the model's fenced JSON block into validated CitationRecords.

        Robust to surrounding prose and to a bare array vs an object with a
        ``records`` key. Rows that fail validation are skipped (the run degrades
        for those pairs) rather than failing the whole parse.
        """
        blob = _extract_json(text)
        if blob is None:
            return []
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return []
        rows = data.get("records", []) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return []

        stub_by_key = {s.cite_key: s for s in stubs}
        out: list[CitationRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("cite_key") or "").strip()
            stub = stub_by_key.get(key)
            # Build from the extractor's stub (authoritative claim/cited_as/key);
            # the model is trusted ONLY for the verdicts. The model frequently
            # returns claim/cited_as as plain strings and evidence as a list of
            # strings — coercing here keeps every well-keyed row instead of
            # dropping it on a schema-validation error.
            if stub is not None:
                rec = stub.model_copy(deep=True)
            elif key:
                cid = str(row.get("claim_id") or f"{key}-0")
                rec = CitationRecord(
                    paper_id=source.paper_id,
                    claim_id=cid,
                    cite_key=key,
                    claim=Claim(claim_id=cid, text=_as_text(row.get("claim"))),
                    cited_as=CitedAs(raw=_as_text(row.get("cited_as"))),
                )
            else:
                continue  # unkeyable row — cannot join to a citation

            rec.paper_id = rec.paper_id or source.paper_id
            rec.exists = _coerce_enum(Exists, row.get("exists"), Exists(rec.exists)).value
            rec.supports_claim = _coerce_enum(
                SupportsClaim, row.get("supports_claim"), SupportsClaim(rec.supports_claim)
            ).value
            rec.priority = _coerce_enum(Priority, row.get("priority"), Priority(rec.priority)).value
            rec.metadata_issues = _coerce_str_list(row.get("metadata_issues"))
            evidence = _coerce_evidence(row.get("evidence"))
            if evidence:
                rec.evidence = evidence
            conf = row.get("confidence")
            if isinstance(conf, (int, float)):
                rec.confidence = max(0.0, min(1.0, float(conf)))
            notes = row.get("notes")
            if isinstance(notes, str) and notes.strip():
                rec.notes = notes.strip()
            rec.model_tier = ModelTier.JUDGE.value
            rec.severity = derive_severity(
                Exists(rec.exists), SupportsClaim(rec.supports_claim), Priority(rec.priority)
            ).value
            out.append(rec)
        return out

    def _degrade(self, stub: CitationRecord, source: PaperSource, reason: str) -> CitationRecord:
        """Return ``stub`` as an unverified, error-stamped record."""
        rec = stub.model_copy(deep=True)
        rec.paper_id = rec.paper_id or source.paper_id
        rec.exists = Exists.UNVERIFIED.value
        rec.supports_claim = SupportsClaim.UNVERIFIED.value
        rec.model_tier = ModelTier.JUDGE.value
        rec.error = reason
        rec.severity = derive_severity(
            Exists.UNVERIFIED, SupportsClaim.UNVERIFIED, Priority(rec.priority)
        ).value
        return rec


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _extract_json(text: str) -> str | None:
    """Pull the JSON payload out of the model transcript.

    Prefers a fenced ```json block; falls back to the outermost ``[...]`` /
    ``{...}`` span. Returns ``None`` when nothing JSON-shaped is present.
    """
    if not text:
        return None
    # Fenced block first: capture to the closing FENCE (not a bracket), then take
    # the outermost [..]/{..} inside it — so nested arrays (authors, evidence,
    # metadata_issues) don't truncate the payload at their first ']'.
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    if fence:
        inner = fence.group(1).strip()
        s, e = inner.find("["), inner.rfind("]")
        if s == -1:
            s, e = inner.find("{"), inner.rfind("}")
        if s != -1 and e > s:
            return inner[s : e + 1]
    # No fence: take the outermost array/object span in the whole transcript.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            return text[start : end + 1]
    return None


def _count_tool_calls(transcript: list[str]) -> int:
    """Best-effort tool-call tally from the transcript (fallback metric)."""
    return sum(t.count("lookup_paper") for t in transcript)


def _lookup_unavailable(reference: str, source: str = "auto") -> dict[str, Any]:
    """Stand-in for grounding.lookup_paper when the grounding module is absent."""
    return {
        "query": reference,
        "found": 0,
        "candidates": [],
        "note": "grounding module unavailable on this branch; cannot ground lookup.",
    }


def _as_text(v: Any) -> str:
    """Coerce a loose claim/cited_as value (str | dict | None) to plain text."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return str(v.get("text") or v.get("raw") or v.get("claim") or "")
    return "" if v is None else str(v)


def _coerce_str_list(v: Any) -> list[str]:
    """Coerce ``metadata_issues`` to a list of non-empty strings."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


# Loose model spellings -> canonical enum tokens.
_ENUM_ALIASES = {
    "does not": "does_not",
    "doesnt": "does_not",
    "n/a": "unverified",
    "na": "unverified",
    "unknown": "unverified",
}


def _coerce_enum(enum_cls: Any, value: Any, default: Any) -> Any:
    """Best-effort map a loose string to ``enum_cls``; ``default`` on failure."""
    if value is None:
        return default
    raw = str(value).strip().lower()
    token = _ENUM_ALIASES.get(raw, raw.replace(" ", "_").replace("-", "_"))
    try:
        return enum_cls(token)
    except ValueError:
        return default


def _coerce_evidence(v: Any) -> list[Evidence]:
    """Coerce model evidence (list[str] | list[dict] | str) to ``Evidence`` rows."""
    items = v if isinstance(v, list) else ([v] if v else [])
    out: list[Evidence] = []
    for it in items:
        if isinstance(it, dict):
            out.append(
                Evidence(
                    kind=str(it.get("kind") or "snippet"),
                    source=str(it.get("source") or "model"),
                    quote=str(it.get("quote") or it.get("text") or "")[:600],
                    url=it.get("url") if isinstance(it.get("url"), str) else None,
                )
            )
        elif isinstance(it, str) and it.strip():
            out.append(Evidence(kind="snippet", source="model", quote=it.strip()[:600]))
    return out


def _setting(settings: Any | None, attr: str, default: Any) -> Any:
    """Read ``attr`` from a Settings object or mapping, else ``default``."""
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)


__all__ = ["ClaudeCodeBackend", "load_skill_body"]
