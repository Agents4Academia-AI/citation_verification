"""
label.py — assemble GOLD :class:`CitationRecord` objects (the benchmark truth).

Joins parsed reference+claim-site dicts (:mod:`chbench.parse`) with the gold
resolver's output (:mod:`chbench.resolve`) into full
:class:`citation_verifier.schema.CitationRecord` objects whose ``labels`` field
carries the ground truth. Because the gold record IS a ``CitationRecord``, agent
output and gold labels agree by construction (decisions-phy.md) and the eval
harness joins them on ``(paper_id, claim_id, cite_key)``.

What gets labelled (``Labels``):
    exists          : yes if the gold resolver matched; no if a seed flagged the
                      reference as fabricated; unverified if offline/no match.
    supports_claim  : left ``unverified`` here — relevance gold needs human or a
                      DIFFERENT-model judge (anti-circularity); this module only
                      sets it when a seed hint provides it. (HONEST: no LLM call.)
    priority        : heuristic (obligatory/helpful) from claim-site cues, marked
                      in provenance as heuristic so a human can override.
    severity        : derived deterministically via the contract's
                      :func:`citation_verifier.schema.derive_severity`.
    is_hallucinated : True when exists=no OR a metadata error was detected.
    provenance      : records HOW the label was made (anti-circularity audit).

This module performs NO network and NO LLM calls itself: it consumes the
resolver's already-fetched output. Import-safe and offline.
"""

from __future__ import annotations

from typing import Any

from citation_verifier.schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Exists,
    Labels,
    Priority,
    Resolved,
    SupportsClaim,
    derive_severity,
)

# Lexical cues that a citation is obligatory (method/baseline/dataset/result use)
# vs merely background. Heuristic only; provenance flags it for human review.
_OBLIGATORY_CUES = (
    "we use",
    "we adopt",
    "we extend",
    "based on",
    "following",
    "baseline",
    "outperform",
    "compared to",
    "compared with",
    "dataset",
    "showed that",
    "proposed by",
    "introduced by",
    "achieves",
    "state-of-the-art",
)


def _priority_for(claim_text: str) -> Priority:
    """Heuristic obligatory/helpful from the claim sentence (human-overridable)."""
    low = claim_text.lower()
    return Priority.OBLIGATORY if any(c in low for c in _OBLIGATORY_CUES) else Priority.HELPFUL


def _detect_metadata_issues(cited_as: CitedAs, resolved: Resolved | None) -> list[str]:
    """List metadata discrepancies between the claimed reference and the canonical.

    Deterministic and conservative: only flags clear mismatches. Empty when no
    canonical record is available (cannot judge).
    """
    if resolved is None:
        return []
    issues: list[str] = []
    if cited_as.year and resolved.year and cited_as.year != resolved.year:
        issues.append(f"year: claimed {cited_as.year}, found {resolved.year}")
    if cited_as.title and resolved.title:
        from .resolve import _title_similarity  # local, offline helper

        if _title_similarity(cited_as.title, resolved.title) < 0.6:
            issues.append("title: claimed title does not match canonical record")
    if cited_as.authors and resolved.authors:
        claimed_last = {a.split()[-1].lower() for a in cited_as.authors if a.split()}
        found_last = {a.split()[-1].lower() for a in resolved.authors if a.split()}
        if claimed_last and found_last and not (claimed_last & found_last):
            issues.append("authors: no overlap with canonical first authors")
    return issues


def _build_one(parsed: dict[str, Any], resolved_dict: dict[str, Any] | None) -> CitationRecord:
    """Build a single gold CitationRecord from one parsed item + its resolution."""
    cited_as = CitedAs(**parsed["cited_as"]) if parsed.get("cited_as") else CitedAs()
    claim = Claim(**parsed["claim"])
    resolved = Resolved(**resolved_dict) if resolved_dict else None

    seed_hint: dict[str, Any] = parsed.get("seed_hint", {}) or {}
    flagged_fabricated = seed_hint.get("label") == "fabricated_reference"

    if flagged_fabricated:
        exists = Exists.NO
        provenance = "gold: gptzero-natural-hallucination (fabricated)"
    elif resolved is not None:
        exists = Exists.YES
        provenance = f"gold: {resolved.source}-resolver ({resolved.match_method})"
    else:
        exists = Exists.UNVERIFIED
        provenance = "gold: offline/no-match (needs resolution)"

    metadata_issues = _detect_metadata_issues(cited_as, resolved)
    priority = _priority_for(claim.text)

    supports = SupportsClaim(seed_hint["supports_claim"]) if seed_hint.get(
        "supports_claim"
    ) else SupportsClaim.UNVERIFIED

    severity = derive_severity(exists, supports, priority)
    is_hallucinated = exists is Exists.NO or bool(metadata_issues)

    labels = Labels(
        exists=exists,
        supports_claim=supports,
        priority=priority,
        severity=severity,
        is_hallucinated=is_hallucinated,
        provenance=provenance,
    )

    return CitationRecord(
        paper_id=parsed["paper_id"],
        claim_id=parsed["claim_id"],
        cite_key=parsed["cite_key"],
        claim=claim,
        cited_as=cited_as,
        exists=exists,
        resolved=resolved,
        metadata_issues=metadata_issues,
        supports_claim=supports,
        priority=priority,
        severity=severity,
        notes="chbench gold record",
        labels=labels,
    )


def make_gold(
    parsed: list[dict[str, Any]],
    resolved: list[dict[str, Any] | None],
) -> list[CitationRecord]:
    """Assemble gold :class:`CitationRecord` objects (labels populated).

    Args:
        parsed: parsed reference+claim-site dicts from :mod:`chbench.parse`
            (each may also carry a ``"seed_hint"`` from the originating seed).
        resolved: positionally-aligned resolver outputs from
            :class:`chbench.resolve.GoldResolver.resolve` — ``None`` where the
            reference did not resolve. Must be the same length as ``parsed``.

    Returns:
        Gold records, each with ``labels`` set (``exists``, ``priority``,
        ``severity``, ``is_hallucinated``, ``provenance``; ``supports_claim`` only
        when a seed hint provides it — relevance gold is added by a human / a
        different-model judge, never the agent's judge).

    Raises:
        ValueError: if ``parsed`` and ``resolved`` differ in length (a join bug).
    """
    if len(parsed) != len(resolved):
        raise ValueError(
            f"parsed ({len(parsed)}) and resolved ({len(resolved)}) must align 1:1"
        )
    return [_build_one(p, r) for p, r in zip(parsed, resolved, strict=True)]
