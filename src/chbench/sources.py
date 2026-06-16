"""
sources.py — the chbench source registry.

Where the dataset's raw material comes from:

  - the GPTZero natural-hallucination lists (NeurIPS 2025 ~100, ICLR 2026 50+)
    that seed real, naturally-occurring hallucination labels;
  - OpenReview venues for broad paper collection;
  - (downstream) arXiv / PDF collection driven by the seed descriptors.

A *seed descriptor* is a plain ``dict`` (not a pydantic model — these are loose,
pre-schema collection artifacts) describing one candidate paper or one flagged
hallucination, with enough provenance for the later stages and for the
anti-circularity audit. Shape:

    {
      "seed_kind":   "gptzero_hallucination" | "openreview_paper",
      "source":      "gptzero_neurips_2025" | "openreview" | ...,
      "venue":       "NeurIPS 2025",
      "paper_title": str | None,
      "paper_url":   str | None,     # landing page / openreview forum
      "arxiv_id":    str | None,
      "pdf_url":     str | None,
      "hint":        {...},          # natural-label hint (e.g. flagged reference)
      "provenance":  str,            # human-readable origin, for gold audit
    }

Network access is OPTIONAL and lazy: the GPTZero pages are fetched via stdlib
``urllib`` only when :func:`gptzero_seed_records` is asked to (``fetch=True``)
AND ``requests``-free; on any failure it returns ``[]`` (fail-soft) so the
pipeline degrades instead of crashing. The default is offline: it returns the
small, committed set of hand-curated seed descriptors that ship with the repo so
``import chbench`` and a dry-run work with no network.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# Authoritative GPTZero hallucination-list URLs (Notion plan, frozen).
GPTZERO_NEURIPS_2025_URL: str = "https://gptzero.me/news/neurips/"
GPTZERO_ICLR_2026_URL: str = "https://gptzero.me/news/iclr-2026/"

_USER_AGENT = "chbench/0.1 (CitationHallucinationBench; +https://github.com/Agents4Academia)"
_TIMEOUT = 20


# A tiny committed seed set so the pipeline is runnable offline and CI-green.
# These are illustrative descriptors mirroring the GPTZero-list shape; real
# entries are appended by the (network) fetch path. Keeping a handful in-repo
# lets `chbench harvest`/`build` produce a non-empty smoke split with no network.
_OFFLINE_GPTZERO_SEEDS: list[dict[str, Any]] = [
    {
        "seed_kind": "gptzero_hallucination",
        "source": "gptzero_neurips_2025",
        "venue": "NeurIPS 2025",
        "paper_title": None,
        "paper_url": None,
        "arxiv_id": None,
        "pdf_url": None,
        "hint": {
            "flagged_reference": "A. Nonexistent et al. Imaginary Transformers for "
            "Real Tasks. NeurIPS 2024.",
            "label": "fabricated_reference",
        },
        "provenance": "gptzero_neurips_2025_list (offline seed example)",
    },
    {
        "seed_kind": "gptzero_hallucination",
        "source": "gptzero_iclr_2026",
        "venue": "ICLR 2026",
        "paper_title": None,
        "paper_url": None,
        "arxiv_id": None,
        "pdf_url": None,
        "hint": {
            "flagged_reference": "J. Doe, K. Roe. Self-Supervised Mirage Networks. "
            "ICLR 2025.",
            "label": "fabricated_reference",
        },
        "provenance": "gptzero_iclr_2026_list (offline seed example)",
    },
]


def _http_get(url: str, *, timeout: int = _TIMEOUT) -> bytes | None:
    """Fetch ``url`` with stdlib urllib. Returns bytes, or None on any failure.

    Fail-soft by contract: callers MUST treat ``None`` as "no network / source
    unavailable" and degrade, never crash.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted urls)
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _parse_gptzero_html(html: str, *, source: str, venue: str) -> list[dict[str, Any]]:
    """Extract seed descriptors from a fetched GPTZero hallucination-list page.

    HONEST STUB. The live GPTZero pages render their hallucination tables from
    JSON/HTML whose exact structure must be pinned against the real markup (and
    may change). Rather than silently fabricate rows, this parser extracts only
    what is unambiguous from raw HTML — arXiv ids and ``href`` links — and wraps
    each as a coarse seed descriptor flagged ``parser="coarse_html"`` so the
    audit trail shows these need refinement. Returns ``[]`` if nothing is found.

    Refinement task (see docs/DATASET.md): replace the coarse scan with a parser
    pinned to the real DOM / embedded JSON once the page structure is captured.
    """
    import re

    seeds: list[dict[str, Any]] = []
    seen: set[str] = set()

    for arxiv_id in re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", html):
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        seeds.append(
            {
                "seed_kind": "gptzero_hallucination",
                "source": source,
                "venue": venue,
                "paper_title": None,
                "paper_url": f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_id": arxiv_id,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "hint": {"label": "flagged_in_gptzero_list", "parser": "coarse_html"},
                "provenance": f"{source}_list (coarse_html scan)",
            }
        )
    return seeds


def gptzero_seed_records(*, fetch: bool = False) -> list[dict[str, Any]]:
    """Return seed descriptors from the GPTZero hallucination lists.

    Args:
        fetch: when False (default), return the committed offline seed set so the
            pipeline runs with no network. When True, additionally fetch the live
            NeurIPS-2025 / ICLR-2026 pages and append any descriptors the coarse
            HTML parser can extract (fail-soft: network failures are ignored and
            only the offline seeds are returned).

    Returns:
        A list of seed-descriptor dicts (see module docstring for the shape).
    """
    seeds: list[dict[str, Any]] = [dict(s) for s in _OFFLINE_GPTZERO_SEEDS]
    if not fetch:
        return seeds

    for url, source, venue in (
        (GPTZERO_NEURIPS_2025_URL, "gptzero_neurips_2025", "NeurIPS 2025"),
        (GPTZERO_ICLR_2026_URL, "gptzero_iclr_2026", "ICLR 2026"),
    ):
        raw = _http_get(url)
        if raw is None:
            continue
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - decode is defensive
            continue
        seeds.extend(_parse_gptzero_html(html, source=source, venue=venue))
    return seeds


def openreview_sources(venue: str, *, fetch: bool = False) -> list[dict[str, Any]]:
    """Return seed descriptors for papers submitted to an OpenReview ``venue``.

    Args:
        venue: an OpenReview venue id, e.g. ``"ICLR.cc/2026/Conference"``.
        fetch: when False (default), return ``[]`` (offline; OpenReview is an
            opt-in broad-collection source). When True, query the OpenReview v2
            API and map each note to a seed descriptor.

    HONEST STUB for ``fetch=True``: this issues the documented OpenReview-v2
    ``/notes`` query and maps the returned notes' ``title``/``pdf`` fields to seed
    descriptors. It is fail-soft (returns whatever it could fetch, ``[]`` on
    error) and unauthenticated, so it only sees public notes; venue-specific
    invitation ids and pagination must be pinned per call for completeness.

    Returns:
        A list of seed-descriptor dicts (``seed_kind="openreview_paper"``).
    """
    if not fetch:
        return []

    base = "https://api2.openreview.net/notes"
    query = f"{base}?content.venueid={urllib.parse.quote(venue)}&limit=100"
    raw = _http_get(query)
    if raw is None:
        return []
    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    seeds: list[dict[str, Any]] = []
    for note in payload.get("notes", []):
        content = note.get("content", {}) or {}

        def _val(field: str, _content: dict[str, Any] = content) -> Any:
            v = _content.get(field)
            return v.get("value") if isinstance(v, dict) else v

        pdf = _val("pdf")
        seeds.append(
            {
                "seed_kind": "openreview_paper",
                "source": "openreview",
                "venue": venue,
                "paper_title": _val("title"),
                "paper_url": f"https://openreview.net/forum?id={note.get('id', '')}",
                "arxiv_id": None,
                "pdf_url": (f"https://openreview.net{pdf}" if pdf else None),
                "hint": {"label": "collected", "openreview_id": note.get("id")},
                "provenance": f"openreview:{venue}",
            }
        )
    return seeds


def seed_to_parsed(seed: dict[str, Any], *, index: int = 0) -> dict[str, Any] | None:
    """Convert a GPTZero-flagged seed directly into a parsed-record dict.

    The GPTZero lists flag *specific references* as hallucinated; that flagged
    reference is itself a labelable gold positive (``exists='no'``) even before
    the full paper is harvested/parsed. This lets the offline pipeline produce
    real natural-hallucination positives from the seed alone.

    Returns ``None`` for seeds that carry no flagged reference (nothing to label
    without harvesting the paper).

    The returned dict matches the :mod:`chbench.parse` output shape and carries
    the seed's ``hint`` under ``"seed_hint"`` so :mod:`chbench.label` sets
    ``exists='no'`` for fabricated references.
    """
    hint = seed.get("hint") or {}
    flagged = hint.get("flagged_reference")
    if not flagged:
        return None
    paper_id = seed.get("arxiv_id") or f"{seed.get('source', 'seed')}-{index}"
    cite_key = f"flagged{index}"
    claim_id = f"{paper_id}:{cite_key}#1"
    return {
        "paper_id": paper_id,
        "claim_id": claim_id,
        "cite_key": cite_key,
        "cited_as": {"raw": flagged},
        "claim": {
            "claim_id": claim_id,
            "text": "",  # claim site unknown without the harvested paper body
            "section": None,
            "char_span": None,
        },
        "seed_hint": hint,
    }


def all_seed_records(*, fetch: bool = False) -> list[dict[str, Any]]:
    """Convenience aggregator: every registered seed source, concatenated.

    The harvest stage calls this to get its full work list. Offline by default.
    """
    return list(gptzero_seed_records(fetch=fetch))


def write_seeds(seeds: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Persist seed descriptors to a JSON file (the seeds checkpoint). Resumable.

    Args:
        seeds: seed descriptors to write.
        out_path: destination JSON path; parent dirs are created.

    Returns:
        The written :class:`pathlib.Path`.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seeds, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out
