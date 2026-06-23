"""
extract/pdf_refs.py — layout-aware reference extraction (one robust path).

A single bibliography parser that works across templates by reading *layout*, not
a flat text stream:

  1. **Layout pass** — PyMuPDF ``dict`` mode gives every line a bbox + font size.
     Fragments on the same baseline (PyMuPDF splits long lines / URLs) are merged,
     and lines are read in column order (``page -> column -> y -> x``) so a
     two-column journal and a single-column conference paper both come out in
     reading order.
  2. **Locate the section** — the *last* ``References`` / ``Bibliography`` heading
     (a "References" column header inside a comparison table appears first and
     must be skipped), stopping at the Appendix / publisher boilerplate.
  3. **Segment, style-aware** — a NUMBERED bibliography (``[12]`` / ``12.`` markers
     forming a dense ``1..N`` run) is split on its markers; an UNNUMBERED
     author-year bibliography is split on the **hanging indent** (a new entry
     starts at the column-left margin; continuation lines are indented).
  4. **Join lines** — de-hyphenate word-breaks, keep URL/DOI continuations glued.

This unifies the two failure modes: a marker-only split breaks author-year lists,
and a hanging-indent-only split breaks (and over-counts) two-column journals.

Network-free; PyMuPDF is an optional import (absent -> ``[]``, caller degrades).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["extract_reference_entries", "extract_reference_strings"]


@dataclass
class _Line:
    page: int
    text: str
    x0: float
    y0: float
    size: float
    col: int


_REF_HEAD_RE = re.compile(r"(?:references|bibliography|works cited)\s*:?", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^(?:Appendix|APPENDIX|Supplementary Material|SUPPLEMENTARY|"
    r"Publisher'?s Note|Springer Nature)\b"
)
_NUM_MARKER_RE = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+")
_URL_CONT_RE = re.compile(r"^(?:https?://|doi:|URL\b|arXiv:)", re.IGNORECASE)


def _clean(s: str) -> str:
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"\s+", " ", s).strip()


def _line_text(line: dict) -> str:
    """Reconstruct a rawdict line's text from its glyphs, re-inserting spaces.

    Some PDFs encode inter-word spaces as positional (kerning) gaps with no space
    glyph; joining span text verbatim then glues words ("Bird JJ" -> "BirdJJ"),
    which wrecks author/title parsing and grounding. We walk the glyphs left to
    right and insert a space wherever the gap to the previous glyph exceeds a
    small fraction of the font size — well above intra-word kerning (~0) but below
    a real space — while leaving existing space glyphs untouched.
    """
    out: list[str] = []
    prev_x1: float | None = None
    for span in line.get("spans", []):
        size = span.get("size", 0.0) or 10.0
        for ch in span.get("chars", []):
            c = ch.get("c", "")
            x0, _, x1, _ = ch["bbox"]
            if (
                prev_x1 is not None
                and c != " "
                and out
                and out[-1] != " "
                and (x0 - prev_x1) > 0.1 * size
            ):
                out.append(" ")
            out.append(c)
            prev_x1 = x1
    text = _clean("".join(out))
    # A '.'/',' glyph carries inflated trailing space, so the gap rule over-splits
    # numbers ("GPT-3.5" -> "GPT-3. 5", "1,000" -> "1, 000"). Re-join digit groups.
    return re.sub(r"(?<=\d[.,])\s+(?=\d)", "", text)


def extract_reference_entries(pdf_path) -> list[tuple[str, str]]:
    """Return ``[(cite_key, reference_text)]``, one pair per bibliography entry.

    A dense numbered list yields ``ref-<n>`` keys aligned to the in-text ``[n]``
    markers; an author-year list yields sequential ``ref-1``, ``ref-2`` … keys.
    Returns ``[]`` when no reference section is found or PyMuPDF is unavailable,
    so the caller can fall back to the flat-text path.
    """
    ref_lines = _reference_section(_layout_lines(pdf_path))
    if not ref_lines:
        return []
    if _is_numbered(ref_lines):
        pairs = _split_numbered(ref_lines)
    else:
        pairs = [(f"ref-{i}", body) for i, body in enumerate(_split_by_indent(ref_lines), start=1)]
    return [(key, body) for key, body in pairs if len(body) >= 10]


def extract_reference_strings(pdf_path) -> list[str]:
    """Reference strings only (inspection/debugging); see :func:`extract_reference_entries`."""
    return [body for _, body in extract_reference_entries(pdf_path)]


# ── layout pass ───────────────────────────────────────────────────────────
def _layout_lines(pdf_path) -> list[_Line]:
    try:
        import fitz  # PyMuPDF — optional
    except Exception:
        return []
    try:
        out: list[_Line] = []
        with fitz.open(str(pdf_path)) as doc:
            for pi, page in enumerate(doc, start=1):
                w, h = page.rect.width, page.rect.height
                # rawdict (not dict) so we have per-glyph bboxes: some PDFs render
                # inter-word spaces as kerning gaps with NO space glyph, which makes
                # dict/words span text glue ("Bird JJ" -> "BirdJJ"); _line_text
                # re-inserts spaces from the glyph gaps.
                for block in page.get_text("rawdict").get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    raw = []
                    for ln in block.get("lines", []):
                        text = _line_text(ln)
                        if not text:
                            continue
                        x0, y0, x1, y1 = ln["bbox"]
                        size = max((s.get("size", 0) for s in ln["spans"]), default=0.0)
                        raw.append((text, x0, y0, x1, y1, size))
                    for text, x0, y0, _x1, _y1, size in _merge_baselines(raw):
                        if y0 < 55:  # running header
                            continue
                        if re.fullmatch(r"\d+", text) and y0 > h - 60:  # page-number footer
                            continue
                        out.append(_Line(pi, text, x0, y0, size, 0 if x0 < w * 0.5 else 1))
        out.sort(key=lambda z: (z.page, z.col, round(z.y0), z.x0))
        return out
    except Exception:  # noqa: BLE001 — fail soft on any PyMuPDF error
        return []


def _merge_baselines(raw, y_tol: float = 1.5):
    """Merge line fragments that share a baseline (PyMuPDF splits long lines/URLs)."""
    raw = sorted(raw, key=lambda z: (z[2], z[1]))
    groups: list[list] = []
    for r in raw:
        for g in groups:
            if abs(r[2] - g[0][2]) <= y_tol:
                g.append(r)
                break
        else:
            groups.append([r])
    merged = []
    for g in groups:
        g = sorted(g, key=lambda z: z[1])
        merged.append(
            (
                _clean(" ".join(i[0] for i in g)),
                min(i[1] for i in g),
                min(i[2] for i in g),
                max(i[3] for i in g),
                max(i[4] for i in g),
                max(i[5] for i in g),
            )
        )
    return merged


# ── locate the references section ─────────────────────────────────────────
def _reference_section(lines: list[_Line]) -> list[_Line]:
    start = None
    for i, ln in enumerate(lines):
        if _REF_HEAD_RE.fullmatch(ln.text.strip()):
            start = i + 1  # LAST heading wins (skip a table's "References" column header)
    if start is None:
        return []
    out = []
    for ln in lines[start:]:
        if _is_stop(ln):
            break
        out.append(ln)
    return out


def _is_stop(ln: _Line) -> bool:
    t = ln.text.strip()
    if _STOP_RE.match(t):
        return True
    if re.fullmatch(r"[A-Z]", t):  # ICLR: a lone "A" line before APPENDIX
        return True
    # "A. Something" / "A.1 Proof" appendix headings — font-gated so a reference's
    # own "S. Scaling laws …" continuation is not mistaken for a section heading.
    if re.match(r"^[A-Z](?:\.\d+)*\.?\s+[A-Z]", t) and ln.size >= 11.5:
        return True
    return False


# ── segmentation ──────────────────────────────────────────────────────────
def _is_numbered(ref_lines: list[_Line]) -> bool:
    nums = sorted(
        int(m.group(1) or m.group(2))
        for ln in ref_lines
        if (m := _NUM_MARKER_RE.match(ln.text))
    )
    if len(nums) < 3:
        return False
    return len(nums) >= 0.6 * (nums[-1] - nums[0] + 1)  # dense 1..N, not a few strays


def _split_numbered(ref_lines: list[_Line]) -> list[tuple[str, str]]:
    """Split a numbered list on its markers; key each entry ``ref-<n>`` (marker stripped)."""
    refs: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_num: str | None = None
    for ln in ref_lines:
        m = _NUM_MARKER_RE.match(ln.text)
        if m:
            if cur and cur_num is not None:
                refs.append((f"ref-{cur_num}", _join(cur)))
            cur_num = m.group(1) or m.group(2)
            cur = [ln.text[m.end() :]]  # drop the "[12] " / "12. " marker
        elif cur_num is not None:
            cur.append(ln.text)
        # lines before the first marker are pre-list noise — skip
    if cur and cur_num is not None:
        refs.append((f"ref-{cur_num}", _join(cur)))
    return refs


def _split_by_indent(ref_lines: list[_Line]) -> list[str]:
    left: dict[tuple[int, int], float] = {}
    for ln in ref_lines:
        key = (ln.page, ln.col)
        left[key] = min(left.get(key, ln.x0), ln.x0)
    refs: list[str] = []
    cur: list[str] = []
    for ln in ref_lines:
        at_left = ln.x0 <= left[(ln.page, ln.col)] + 6.0
        starts_new = at_left and len(ln.text) >= 3 and not _URL_CONT_RE.match(ln.text)
        if starts_new and cur:
            refs.append(_join(cur))
            cur = [ln.text]
        else:
            cur.append(ln.text)
    if cur:
        refs.append(_join(cur))
    return refs


def _join(parts: list[str]) -> str:
    """Join a reference's lines: glue URL/DOI continuations, de-hyphenate word breaks."""
    s = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not s:
            s = part
            continue
        last = s.split()[-1] if s.split() else ""
        urlish = last.startswith(("http://", "https://")) or last.lower().startswith("doi:")
        if (urlish and re.match(r"^(?://|[\w.~/%?#=&+-])", part)) or s.endswith(("=", "/")):
            s += part
        elif re.search(r"(?:doi:\s*\S*|https?://\S*)\.$", s, re.IGNORECASE) and part[:1].isdigit():
            s += part  # wrapped DOI: "…acl-long.\n427" -> "…acl-long.427"
        elif s.endswith("-") and part[:1].islower():
            s = s[:-1] + part  # word-break hyphen (real compounds keep their hyphen + space)
        else:
            s += " " + part
    return re.sub(r"\s+([,.;:])", r"\1", s).strip()
