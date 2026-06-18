"""
claude_code.py — the "claude_code" backend: a skill-driven, grounded, concurrent
judge.

History: this started as a single Agent-SDK ``query()`` loop over the WHOLE
bibliography (``max_turns=80``, an in-loop ``lookup_paper`` tool). That design
let one accumulating context grow across dozens of turns — every turn re-read the
whole transcript, so a 25-citation run burned ~710k cache-read tokens, ~$5, and
~14 min, *and* (because the model rarely called the tool) judged relevance from
memory rather than retrieved text.

This module keeps the backend's identity — the **SKILL.md method drives the
model, which emits the full ``CitationRecord`` verdicts** — but removes the
structural waste, on two axes the team asked for (accuracy floor + lower cost):

  1. **Ground first, deterministically (no LLM).** Each reference is resolved
     against the grounding layer (:func:`citation_verifier.stages.fill_correctness`):
     that fixes ``exists`` / ``resolved`` / ``metadata_issues`` from *retrieved*
     metadata, never model memory. We additionally retrieve the cited work's
     abstract (+ its introduction, best-effort) as the relevance evidence.
  2. **Judge in bounded, concurrent chunks (one ``query()`` each, no tools, low
     ``max_turns``).** The pairs are sharded; each chunk is judged in a single
     structured call whose context is just {claim, reference, resolved record,
     evidence} — not an accumulating transcript. Chunks run concurrently, and the
     identical SKILL.md system prompt is shared across them (prompt-cache prefix).

Accuracy guards: the model never sets ``exists`` (it stays the grounded value, so
existence is never asserted from memory); and a ``supports`` verdict is downgraded
to ``unverified`` when no abstract/introduction was retrieved to ground it. When
the abstract(+intro) lacks the specific claimed evidence, the model is asked to
abstain (``partial`` / ``unverified``) rather than guess.

The SDK is a lazy import inside methods (:func:`_require_sdk`) so this module
imports with the SDK absent; the grounding/stages siblings are imported lazily too.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import apply_auth
from ..interfaces import PaperSource, RunUsage, VerificationResult
from ..schema import (
    CitationRecord,
    Evidence,
    Exists,
    MatchMethod,
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
    """Lazily import claude_agent_sdk; raise an informative error if absent."""
    try:
        import claude_agent_sdk  # type: ignore
    except ImportError as exc:  # SDK not installed -> actionable message
        raise RuntimeError(_MISSING_SDK_MSG) from exc
    return claude_agent_sdk


def load_skill_body(skill_file: Path = _SKILL_FILE) -> str:
    """Return the SKILL.md body with its YAML front-matter stripped."""
    try:
        text = skill_file.read_text(encoding="utf-8")
    except OSError:
        return ""
    return re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL).strip()


@dataclass
class _CiteGroup:
    """One UNIQUE citation + every claim-site that cites it (the unit of work).

    Correctness/evidence is grounded once for the citation; relevance is judged
    once per claim against the SHARED evidence — so the cited work's abstract is
    sent to the model once per citation, not once per (claim, citation) pair.
    """

    cite_key: str
    stubs: list[CitationRecord]   # all claim-sites citing this key
    evidence: str                 # abstract (+ "[Introduction]" …) retrieved once
    has_evidence: bool


@register
class ClaudeCodeBackend(BaseBackend):
    """Skill-driven, grounded, concurrent backend (``name='claude_code'``).

    Importing this class does NOT import the SDK; the SDK is touched only when
    :meth:`verify` runs the chunked judging.
    """

    name = "claude_code"

    def __init__(self, *, settings: Any | None = None) -> None:
        self.settings = settings
        self.model = _setting(settings, "model_judge", "claude-opus-4-6")
        # No agent loop: each chunk is one structured call. A small ceiling is
        # plenty (the model answers in one turn) and caps a runaway chunk.
        self.max_turns = int(_setting(settings, "claude_code_max_turns", 4) or 4)
        # Sharding + concurrency (bounded context per call; calls run in parallel).
        self.chunk_size = int(_setting(settings, "claude_code_chunk_size", 8) or 8)
        self.concurrency = int(_setting(settings, "claude_code_concurrency", 5) or 5)
        self.ground_concurrency = int(
            _setting(settings, "claude_code_ground_concurrency", 5) or 5
        )
        # L1 = abstract + the cited work's introduction (best-effort, arXiv only).
        self.use_intro = bool(_setting(settings, "claude_code_use_intro", True))
        self.evidence_cap = int(_setting(settings, "claude_code_evidence_cap", 4000) or 4000)

    # ──────────────────────────────────────────────────────────────
    def verify(
        self, source: PaperSource, stubs: list[CitationRecord]
    ) -> VerificationResult:
        """Ground every pair deterministically, then judge in concurrent chunks.

        Raises ``RuntimeError`` (via :func:`_require_sdk`) if claude-agent-sdk is
        not installed. Requires extracted ``stubs`` (the supported, low-cost path);
        with none, it returns an honest empty result rather than reading the whole
        paper into an accumulating context.
        """
        sdk = _require_sdk()
        apply_auth(self.settings)  # API key if configured, else the subscription
        result = self._empty_result(source)

        if not stubs:
            result.errors.append(
                "claude_code: no extracted (claim, citation) stubs to verify"
            )
            result.records = []
            return result

        with self._timer() as sw:
            records, usage = asyncio.run(self._run(sdk, source, stubs, result))
        usage.wall_seconds = sw.seconds
        result.usage = usage

        self._stamp_paper_id(records, source.paper_id)
        result.records = records
        return result

    # ──────────────────────────────────────────────────────────────
    async def _run(
        self,
        sdk: Any,
        source: PaperSource,
        stubs: list[CitationRecord],
        result: VerificationResult,
    ) -> tuple[list[CitationRecord], RunUsage]:
        """Pre-ground (Python) → shard → judge chunks concurrently → aggregate."""
        resolver = self._build_resolver()

        # 1) Pre-ground concurrently — deterministic, no LLM. Grounds each UNIQUE
        #    citation once (exists/metadata from retrieved sources + abstract/intro
        #    for relevance) and carries all of its claim-sites.
        groups = await self._preground_all(resolver, stubs)

        # 2) Shard by CITATION (not by pair): chunk_size citations per chunk, so a
        #    citation's evidence is sent once and all its claims ride along.
        chunks = [
            groups[i : i + self.chunk_size]
            for i in range(0, len(groups), self.chunk_size)
        ]

        # 3) Judge chunks concurrently (each is one query(); identical system
        #    prompt → shared prompt-cache prefix). Bounded concurrency.
        sem = asyncio.Semaphore(max(1, self.concurrency))

        async def _run_chunk(chunk: list[_CiteGroup]) -> tuple[list[CitationRecord], RunUsage]:
            async with sem:
                return await self._judge_chunk(sdk, source, chunk)

        outs = await asyncio.gather(
            *(_run_chunk(c) for c in chunks), return_exceptions=True
        )

        # 4) Aggregate records + usage; a failed chunk degrades its claims only.
        records: list[CitationRecord] = []
        usage = RunUsage(backend=self.name, model=self.model)
        for chunk, out in zip(chunks, outs, strict=False):
            if isinstance(out, Exception):
                result.errors.append(f"claude_code: chunk judge failed: {out!r}")
                for grp in chunk:
                    records.extend(
                        self._finalize_degraded(s, source, "chunk judge failed")
                        for s in grp.stubs
                    )
                continue
            recs, u = out
            usage.add(u)
            records.extend(recs)
        return records, usage

    # ──────────────────────────────────────────────────────────────
    def _build_resolver(self) -> Any:
        """Construct the grounding resolver (fail-soft to None)."""
        try:
            from ..grounding import MultiSourceResolver  # type: ignore

            return MultiSourceResolver(settings=self.settings)
        except Exception:  # noqa: BLE001 — grounding not on this branch / no ctor kw
            try:
                from ..grounding import MultiSourceResolver  # type: ignore

                return MultiSourceResolver()
            except Exception:  # noqa: BLE001
                return None

    async def _preground_all(
        self, resolver: Any, stubs: list[CitationRecord]
    ) -> list[_CiteGroup]:
        """Ground each UNIQUE citation once, then share across its claim-sites.

        Correctness (exists/metadata) and the relevance evidence (abstract+intro)
        are properties of the cited PAPER, not of the (claim, citation) pair — so a
        citation cited N times needs ONE resolver lookup, not N. Deduping by
        cite_key cuts the API fan-out from ``len(stubs)`` to the unique-citation
        count (e.g. 166 -> ~53 on 2505.13447), which is what stops the grounding
        sources from rate-limiting at scale. Relevance stays per-pair (downstream).
        """
        by_key: dict[str, list[CitationRecord]] = {}
        for s in stubs:
            by_key.setdefault(s.cite_key, []).append(s)

        sem = asyncio.Semaphore(max(1, self.ground_concurrency))

        async def _ground(cite_key: str, group: list[CitationRecord]) -> _CiteGroup:
            async with sem:
                return await asyncio.to_thread(self._preground_group, resolver, cite_key, group)

        return list(await asyncio.gather(*(_ground(k, g) for k, g in by_key.items())))

    def _preground_group(
        self, resolver: Any, cite_key: str, group: list[CitationRecord]
    ) -> _CiteGroup:
        """Ground ONE citation (its representative), share it across its claim-sites.

        Runs in a worker thread (network-bound). The representative is grounded via
        :func:`fill_correctness` (exists/resolved/metadata_issues/evidence) and its
        abstract+intro is fetched once; every other stub with the same cite_key
        copies that grounding — only the per-claim priority is recomputed.
        """
        try:
            from ..stages import fill_correctness  # public stage API
            from ..stages.relevance import _infer_priority
        except Exception:  # noqa: BLE001 — siblings absent: degrade to unverified
            fill_correctness = None  # type: ignore[assignment]
            _infer_priority = None  # type: ignore[assignment]

        rep = group[0]
        if fill_correctness is not None:
            try:
                fill_correctness(rep, resolver=resolver)
            except Exception as exc:  # noqa: BLE001 — degrade-not-crash
                rep.error = (rep.error + "; " if rep.error else "") + f"preground: {exc!r}"
                rep.exists = Exists.UNVERIFIED.value
        evidence = self._evidence_text(rep)

        for stub in group:
            if stub is not rep:
                # Share the cited-paper grounding; do NOT re-resolve (the point).
                stub.resolved = rep.resolved
                stub.exists = rep.exists
                stub.metadata_issues = list(rep.metadata_issues)
                stub.evidence = list(rep.evidence)
                if rep.error and not stub.error:
                    stub.error = rep.error
            if _infer_priority is not None:
                try:
                    stub.priority = _infer_priority(stub.claim.text).value
                except Exception:  # noqa: BLE001
                    pass
        return _CiteGroup(
            cite_key=cite_key, stubs=group, evidence=evidence, has_evidence=bool(evidence.strip())
        )

    def _evidence_text(self, stub: CitationRecord) -> str:
        """Assemble relevance evidence: abstract (L0) + introduction (L1, arXiv)."""
        resolved = stub.resolved
        abstract = (resolved.abstract if resolved and resolved.abstract else "") or ""
        text = abstract.strip()
        if self.use_intro and resolved is not None:
            try:
                from .relevance_judge import _fetch_intro  # shared L1 retrieval

                intro = _fetch_intro(resolved)
            except Exception:  # noqa: BLE001
                intro = ""
            if intro:
                text = (text + "\n\n[Introduction]\n" + intro).strip()
        return text[: self.evidence_cap]

    # ──────────────────────────────────────────────────────────────
    async def _judge_chunk(
        self, sdk: Any, source: PaperSource, chunk: list[_CiteGroup]
    ) -> tuple[list[CitationRecord], RunUsage]:
        """Judge one chunk of citations in a single structured query() (no tools)."""
        options = sdk.ClaudeAgentOptions(
            system_prompt=self._system_prompt(),
            allowed_tools=[],            # not an agent loop — a grounded judgement
            cwd=str(_ROOT),
            max_turns=self.max_turns,
            model=self.model,
        )

        transcript: list[str] = []
        usage = RunUsage(backend=self.name, model=self.model)
        async for message in sdk.query(
            prompt=self._chunk_user_prompt(source, chunk), options=options
        ):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        transcript.append(block.text)
            elif isinstance(message, sdk.ResultMessage):
                usage = usage_from_result_message(message, self.model, self.name)

        records = self._apply_verdicts(chunk, "".join(transcript), source)
        return records, usage

    # ──────────────────────────────────────────────────────────────
    def _system_prompt(self) -> str:
        """System prompt = grounding rule + injected SKILL.md + I/O contract."""
        return (
            "You are a meticulous citation-verification agent for academic papers. "
            "You are given a list of CITATIONS. Each citation comes with the retrieved "
            "evidence ONCE — a resolved canonical record and the cited work's abstract "
            "(sometimes plus its introduction) — and a list of CLAIMS in the citing "
            "paper that reference it. Judge each claim ONLY from that citation's "
            "evidence; NEVER assert anything from memory, and do NOT call any tools. "
            "Follow this method exactly:\n\n"
            + load_skill_body()
            + "\n\n## Output for this backend\n"
            "Existence is already determined for you from retrieval — do not invent it. "
            "Per citation you may ADD metadata issues; per CLAIM you judge relevance + "
            "priority:\n"
            "- supports_claim: supports | partial | does_not | unverified. Use ONLY the "
            "citation's evidence text. If the specific claimed fact/result/method is not "
            "present in it, use 'unverified' (or 'partial' if clearly related but not "
            "confirming). Do NOT guess 'supports'. If the evidence is empty, you MUST "
            "use 'unverified'.\n"
            "- priority: obligatory (the claim depends on this source — a method "
            "used/extended, a baseline, a dataset, a specific result/quote) or helpful "
            "(background / see-also).\n"
            "- metadata_issues (per citation): ONLY genuine mismatches between the "
            "reference string and the resolved record (missing diacritic, wrong "
            "venue/year, truncated title). Omit if none.\n"
            "- notes (per claim): one short sentence justifying the verdict from the "
            "evidence.\n\n"
            "Emit ONE JSON array, exactly one object per input CITATION, echoing its "
            "cite_key and one verdict per claim_id:\n"
            '[{"cite_key":"...","metadata_issues":["..."],"claims":['
            '{"claim_id":"...","supports_claim":"...","priority":"...",'
            '"confidence":0.0-1.0,"notes":"..."}]}]\n'
            "Output JSON only — no prose, no Markdown table."
        )

    def _chunk_user_prompt(self, source: PaperSource, chunk: list[_CiteGroup]) -> str:
        """User prompt: the grounded citations (evidence once + their claims)."""
        items = []
        for grp in chunk:
            rep = grp.stubs[0]
            items.append(
                {
                    "cite_key": grp.cite_key,
                    "reference": _reference_text(rep),
                    "resolved": _resolved_brief(rep.resolved),
                    "evidence": grp.evidence,
                    "claims": [
                        {"claim_id": s.claim_id, "claim": s.claim.text} for s in grp.stubs
                    ],
                }
            )
        return (
            f"paper_id: {source.paper_id}\n"
            "Judge each citation's claims using ONLY that citation's evidence text. "
            "Return the JSON array specified in the system prompt — one object per "
            "citation, one verdict per claim_id:\n"
            + json.dumps(items, ensure_ascii=False, indent=2)
        )

    # ──────────────────────────────────────────────────────────────
    def _apply_verdicts(
        self, chunk: list[_CiteGroup], text: str, source: PaperSource
    ) -> list[CitationRecord]:
        """Overlay the model's verdicts onto the grounded stubs for this chunk.

        The model returns one object per citation
        (``{cite_key, metadata_issues, claims:[{claim_id, ...}]}``). Existence stays
        the grounded value (never model-set); a ``supports`` with no retrieved
        evidence is downgraded to ``unverified``; a claim the model omitted degrades
        to ``unverified`` (1:1 with the chunk's claim-sites).
        """
        by_key: dict[str, dict] = {}
        for row in _parse_rows(text):
            if isinstance(row, dict):
                ck = str(row.get("cite_key") or "").strip()
                if ck:
                    by_key.setdefault(ck, row)

        out: list[CitationRecord] = []
        for grp in chunk:
            cit = by_key.get(grp.cite_key) or {}
            extra_meta = _coerce_str_list(cit.get("metadata_issues"))
            verdicts = {
                str(cv.get("claim_id") or "").strip(): cv
                for cv in (cit.get("claims") or [])
                if isinstance(cv, dict) and str(cv.get("claim_id") or "").strip()
            }
            for stub in grp.stubs:
                cv = verdicts.get(stub.claim_id)
                if cv is None:
                    out.append(self._finalize_degraded(stub, source, "no model verdict for claim"))
                    continue
                self._merge_verdict(stub, cv, grp.has_evidence)
                if extra_meta:
                    stub.metadata_issues = _dedupe_keep_order(
                        list(stub.metadata_issues) + extra_meta
                    )
                if grp.has_evidence:
                    stub.evidence = _add_abstract_evidence(stub.evidence, stub.resolved, grp.evidence)
                stub.paper_id = stub.paper_id or source.paper_id
                stub.model_tier = ModelTier.JUDGE.value
                stub.severity = derive_severity(
                    Exists(stub.exists), SupportsClaim(stub.supports_claim), Priority(stub.priority)
                ).value
                out.append(stub)
        return out

    def _merge_verdict(self, rec: CitationRecord, row: dict, has_evidence: bool) -> None:
        """Apply one model row's verdicts onto a grounded record (exists untouched)."""
        rec.supports_claim = _coerce_enum(
            SupportsClaim, row.get("supports_claim"), SupportsClaim(rec.supports_claim)
        ).value
        rec.priority = _coerce_enum(Priority, row.get("priority"), Priority(rec.priority)).value

        extra = _coerce_str_list(row.get("metadata_issues"))
        if extra:
            rec.metadata_issues = _dedupe_keep_order(list(rec.metadata_issues) + extra)

        conf = row.get("confidence")
        if isinstance(conf, int | float):
            rec.confidence = max(0.0, min(1.0, float(conf)))

        notes = row.get("notes")
        if isinstance(notes, str) and notes.strip():
            rec.notes = (rec.notes + "\n" if rec.notes else "") + notes.strip()

        model_ev = _coerce_evidence(row.get("evidence"))
        if model_ev:
            rec.evidence = list(rec.evidence) + model_ev

        # Accuracy guard: support requires retrieved text to rest on.
        if rec.supports_claim == SupportsClaim.SUPPORTS.value and not has_evidence:
            rec.supports_claim = SupportsClaim.UNVERIFIED.value
            rec.notes = (rec.notes + "\n" if rec.notes else "") + (
                "abstained: no abstract/introduction was retrieved to ground support"
            )

    def _finalize_degraded(
        self, stub: CitationRecord, source: PaperSource, reason: str
    ) -> CitationRecord:
        """Return a grounded stub as an unverified, error-stamped record."""
        stub.paper_id = stub.paper_id or source.paper_id
        stub.supports_claim = SupportsClaim.UNVERIFIED.value
        stub.model_tier = ModelTier.JUDGE.value
        stub.error = (stub.error + "; " if stub.error else "") + reason
        stub.severity = derive_severity(
            Exists(stub.exists), SupportsClaim.UNVERIFIED, Priority(stub.priority)
        ).value
        return stub


# ───────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────
def _parse_rows(text: str) -> list:
    """Parse the model transcript into a list of verdict rows (fail-soft)."""
    blob = _extract_json(text)
    if blob is None:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    rows = data.get("records", []) if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


def _extract_json(text: str) -> str | None:
    """Pull the JSON payload out of the model transcript (fenced or bare)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, flags=re.DOTALL)
    if fence:
        inner = fence.group(1).strip()
        s, e = inner.find("["), inner.rfind("]")
        if s == -1:
            s, e = inner.find("{"), inner.rfind("}")
        if s != -1 and e > s:
            return inner[s : e + 1]
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            return text[start : end + 1]
    return None


def _resolved_brief(resolved: Any) -> dict | None:
    """Compact the canonical record for the prompt (drop empties / None match)."""
    if resolved is None or getattr(resolved, "match_method", MatchMethod.NONE) is MatchMethod.NONE:
        return None
    fields = {
        "title": resolved.title,
        "authors": resolved.authors,
        "year": resolved.year,
        "venue": resolved.venue,
        "doi": resolved.doi,
        "arxiv_id": resolved.arxiv_id,
        "url": resolved.url,
        "source": resolved.source,
    }
    return {k: v for k, v in fields.items() if v}


def _add_abstract_evidence(items: list[Evidence], resolved: Any, text: str) -> list[Evidence]:
    """Append the retrieved abstract(+intro) as an ``abstract`` evidence row.

    This is the text the relevance verdict was judged against — recording it keeps
    the verdict auditable (grounded in retrieved evidence, never model memory).
    Deduped against any existing abstract row.
    """
    snippet = re.sub(r"\s+", " ", (text or "")).strip()[:600]
    if not snippet:
        return items
    src = "structured"
    url = None
    if resolved is not None:
        src = resolved.doi or resolved.arxiv_id or resolved.url or (resolved.source or "structured")
        url = resolved.url
    new = Evidence(kind="abstract", source=str(src), quote=snippet, url=url)
    for e in items:
        if e.kind == new.kind and e.source == new.source and e.quote == new.quote:
            return items
    return list(items) + [new]


def _as_text(v: Any) -> str:
    """Coerce a loose value (str | dict | None) to plain text."""
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


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Stable de-dup for merged metadata issues."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


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


def _reference_text(rec: CitationRecord) -> str:
    """Best reference string for a stub: raw bib line + parsed ids (arXiv/DOI)."""
    c = rec.cited_as
    raw = (c.raw or "").strip()
    parts = (
        [raw]
        if raw
        else [", ".join(a for a in c.authors if a), c.title or "", str(c.year or ""), c.venue or ""]
    )
    low = raw.lower()
    if c.arxiv_id and "arxiv" not in low:
        parts.append(f"arXiv:{c.arxiv_id}")
    if c.doi and c.doi.lower() not in low:
        parts.append(f"doi:{c.doi}")
    return " ".join(p for p in parts if p).strip()


def _setting(settings: Any | None, attr: str, default: Any) -> Any:
    """Read ``attr`` from a Settings object or mapping, else ``default``."""
    if settings is None:
        return default
    if isinstance(settings, dict):
        return settings.get(attr, default)
    return getattr(settings, attr, default)


__all__ = ["ClaudeCodeBackend", "load_skill_body"]
