"""
extract/pdf_links.py — author-year citation extraction for PDFs.

Numbered PDFs anchor each citation with a ``[n]`` marker (scanned in
:mod:`extract.pdf`). Author-year PDFs ("(Yang et al., 2023)") have no such
marker, so the ``[n]`` scanner produces only reference stubs (no claim). Two
tiers recover the citation *sites* — the in-text location + its sentence:

  1. **Hyperlink anchors (primary).** A hyperref-generated PDF carries each
     citation as an internal link whose named destination is ``cite.<bibkey>``
     (e.g. ``cite.yang2023leandojo``). The link rectangle locates the citation
     in the body — its surrounding sentence is the claim — and the bibkey names
     the reference. This is exact: no name/year guessing.
  2. **Author-year text regex (fallback).** When a PDF has no such links, scan
     the body text for ``(Surname et al., 2023)`` / ``Surname et al. (2023)``
     forms and synthesize a ``surname+year`` key.

Both tiers return ``[{cite_key, claim, span}]``; the caller binds ``cite_key``
to a parsed reference (by surname + year). Fail-soft: PyMuPDF is optional;
hyperlink extraction returns ``[]`` when it is absent or the PDF has no links.
"""

from __future__ import annotations

import re
import unicodedata

from .pdf_refs import _clean as _normalize  # NFKC + diacritic heal (shared with refs)

__all__ = ["extract_link_citations", "extract_text_citations"]

# Sentence boundary: ./!/? then whitespace then a capital or an opening paren
# (author-year citations often open the next clause with "(Surname …"), BUT not
# after a single-capital initial ("Kingma D. P.") or the abbreviations "et al.",
# "e.g.", "i.e." — those otherwise chop a claim down to "(2022)" / "Hindle et al.".
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])(?<![A-Z]\.)(?<!et\sal\.)(?<!e\.g\.)(?<!i\.e\.)\s+(?=[A-Z(])"
)
_MAX_CLAIM_CHARS = 360
_CITE_PREFIX = "cite."

# Parenthetical "(Surname et al., 2023)" / "(Surname and Other, 2023)" / "(Surname, 2023)".
_AY_PAREN_RE = re.compile(
    r"\(\s*([A-Z][A-Za-z'’.\-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'’.\-]+|&\s*[A-Z][A-Za-z'’.\-]+))?)"
    r"\s*,?\s+((?:19|20)\d{2}[a-z]?)"
)
# Narrative "Surname et al. (2023)" / "Surname and Other (2023)".
_AY_NARRATIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z'’.\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z'’.\-]+)?)\s+\(\s*((?:19|20)\d{2}[a-z]?)\s*\)"
)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def extract_link_citations(pdf_path) -> list[dict]:
    """Tier 1: ``[{cite_key, claim, span}]`` from ``cite.*`` hyperlink anchors.

    ``span`` is a stable, page-offset char range usable as a claim-id seed. The
    author-name and year halves of one citation each carry a link, so sites that
    share a ``(cite_key, claim)`` are de-duplicated.
    """
    try:
        import fitz  # PyMuPDF — optional
    except Exception:
        return []
    try:
        sites: list[dict] = []
        with fitz.open(str(pdf_path)) as doc:
            for pno, page in enumerate(doc):
                anchors = [
                    (lnk["from"], (lnk.get("nameddest") or "")[len(_CITE_PREFIX) :])
                    for lnk in page.get_links()
                    if (lnk.get("nameddest") or "").startswith(_CITE_PREFIX)
                ]
                if not anchors:
                    continue
                words = page.get_text("words")
                width = page.rect.width
                for rect, cite_key in anchors:
                    if not cite_key:
                        continue
                    claim, (cs, ce) = _sentence_at_rect(words, rect, width)
                    if claim:
                        base = pno * 1_000_000
                        sites.append(
                            {"cite_key": cite_key, "claim": claim, "span": (base + cs, base + ce)}
                        )
        return _dedupe(sites)
    except Exception:  # noqa: BLE001 — fail soft on any PyMuPDF error
        return []


def extract_text_citations(text: str) -> list[dict]:
    """Tier 2: ``[{cite_key, claim, span}]`` from author-year text patterns.

    A conservative fallback for PDFs with no hyperlinks: matches the common
    parenthetical/narrative forms and synthesizes a ``surname+year`` key.
    """
    sites: list[dict] = []
    for rx in (_AY_PAREN_RE, _AY_NARRATIVE_RE):
        for m in rx.finditer(text or ""):
            surname = _fold(re.sub(r"[^A-Za-z'’\-]", "", m.group(1).split()[0]))
            if len(surname) < 2:
                continue
            cite_key = f"{surname}{m.group(2)}"
            claim, span = _sentence_around(text, m.start())
            if claim:
                sites.append({"cite_key": cite_key, "claim": claim, "span": span})
    return _dedupe(sites)


def _order_reading(words: list) -> list:
    """Order words in reading order by clustering into lines, then top-to-bottom,
    left-to-right. Clustering by baseline tolerance (not a flat ``round(y)`` sort)
    is robust to small baseline jitter — inline code/math/sub-superscripts — that
    would otherwise scramble a line (``"Lean.Expr that into the which"``)."""
    lines: list[dict] = []
    for w in sorted(words, key=lambda w: w[1]):  # by y0 (top down)
        h = (w[3] - w[1]) or 10.0
        for ln in lines:
            if abs(w[1] - ln["y0"]) <= 0.6 * h:  # same baseline within tolerance
                ln["words"].append(w)
                break
        else:
            lines.append({"y0": w[1], "words": [w]})
    out: list = []
    for ln in lines:
        out.extend(sorted(ln["words"], key=lambda w: w[0]))
    return out


def _sentence_at_rect(words, rect, page_width: float) -> tuple[str, tuple[int, int]]:
    """Sentence (claim) containing the citation at ``rect``; ``""`` when not found.

    Words are read in the rect's column, ordered by line (clustered), then the
    sentence covering the word the rect overlaps is returned (capped + normalized).
    """
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    mid = page_width / 2
    col = 0 if (x0 + x1) / 2 < mid else 1
    cw = _order_reading([w for w in words if (0 if (w[0] + w[2]) / 2 < mid else 1) == col])
    if not cw:
        return "", (0, 0)
    offsets: list[int] = []
    pos = 0
    for w in cw:
        offsets.append(pos)
        pos += len(w[4]) + 1
    text = " ".join(w[4] for w in cw)
    ti = next(
        (i for i, w in enumerate(cw) if not (w[3] < y0 or w[1] > y1 or w[2] < x0 or w[0] > x1)),
        None,
    )
    if ti is None:
        return "", (0, 0)
    return _sentence_around(text, offsets[ti])


def _sentence_around(text: str, pos: int, window: int = 460) -> tuple[str, tuple[int, int]]:
    """The sentence containing offset ``pos``, normalized + capped at word boundaries."""
    lo = max(0, pos - window)
    hi = min(len(text), pos + window)
    chunk = text[lo:hi]
    rel = pos - lo
    starts = [0] + [m.end() for m in _SENT_SPLIT.finditer(chunk)]
    ends = [m.start() for m in _SENT_SPLIT.finditer(chunk)] + [len(chunk)]
    s0, s1 = 0, len(chunk)
    for st, en in zip(starts, ends, strict=False):
        if st <= rel <= en:
            s0, s1 = st, en
            break
    raw = chunk[s0:s1]
    start_off = s0
    if len(raw) > _MAX_CLAIM_CHARS:
        half = _MAX_CLAIM_CHARS // 2
        a = max(0, (rel - s0) - half)
        b = min(len(raw), a + _MAX_CLAIM_CHARS)
        if a > 0:  # snap the start forward to a word boundary (never mid-word)
            sp = raw.find(" ", a)
            a = sp + 1 if 0 <= sp < b else a
        if b < len(raw):  # snap the end back to a word boundary
            sp = raw.rfind(" ", a, b)
            b = sp if sp > a else b
        start_off = s0 + a
        raw = raw[a:b]
    raw = re.sub(r"(?<=[A-Za-z])-\s+(?=[a-z])", "", raw)  # join line-break hyphenation
    return _normalize(raw), (lo + start_off, lo + start_off + len(raw))


def _dedupe(sites: list[dict]) -> list[dict]:
    """Collapse sites of one cite_key whose spans OVERLAP (the name + year link
    halves of a citation, or the same sentence reached from two rects) — keep the
    longest claim. Distinct citation locations (non-overlapping spans) are kept."""
    by_key: dict[str, list[dict]] = {}
    for s in sites:
        by_key.setdefault(s["cite_key"], []).append(s)
    out: list[dict] = []
    for group in by_key.values():
        group.sort(key=lambda s: (s["span"][0], -len(s["claim"])))
        kept: list[dict] = []
        for s in group:
            a0, a1 = s["span"]
            if not any(a0 < k["span"][1] and a1 > k["span"][0] for k in kept):
                kept.append(s)
        out.extend(kept)
    return out
