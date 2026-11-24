"""
extract/pdf_refs.py — layout-aware reference extraction (one robust path).

A single bibliography parser that works across templates by reading *layout*, not
a flat text stream:

  1. **Layout pass** — PyMuPDF ``dict`` mode gives every line a bbox + font size.
     Fragments on the same baseline (PyMuPDF splits long lines / URLs) are merged,
     and lines are read in column order (``page -> column -> y -> x``) so a
     two-column journal and a single-column conference paper both come out in
     reading order.
  2. **Locate the sections** — EVERY ``References`` / ``Bibliography`` heading opens a
     candidate, read up to the Appendix / publisher boilerplate or the next such
     heading, and a candidate is kept when what follows it parses as a reference list.
     A paper with supplementary material often carries two lists, each numbered from 1:
     taking only the last one bound every ``[n]`` of the main text to the supplement's
     entries. A "References" column header inside a comparison table is a heading with
     no list after it, and drops out on that test.
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
import unicodedata
from dataclasses import dataclass

__all__ = ["extract_reference_sections", "extract_reference_entries", "extract_reference_strings"]


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

# A flush-left author-year bibliography (e.g. AAAI) has NO hanging indent — every
# entry line sits at the column-left margin — so the indent split treats each
# visual line as its own entry (one real reference becomes 3-4). Segment it on
# content instead: the reliable anchor is the author-list -> year transition. A
# name token is "Surname, I." — initials may be hyphenated ("P.-Y."), spaced
# ("S.- T."), or lowercase (Korean romanization "Jung, J.-w.", "Kim, j.-h."); each
# initial keeps its period, which (with the year anchor below) stops the run from
# eating title words. Names are ";"/"," separated with an optional trailing "and".
# We split the joined stream just before each fresh author run that leads into a
# 4-digit year.
_AY_NAME_TOK = r"[A-Z][A-Za-z'’-]+,\s+[A-Za-z]\.(?:[-\s]*[A-Za-z]\.)*"
_AY_ENTRY_BOUNDARY = re.compile(
    r"(?<=[.\d)])\s+"
    r"(?=(?:" + _AY_NAME_TOK + r"[;,]?\s+(?:and\s+)?)+(?:19|20)\d{2}[a-z]?[.,])"
)


# A combining mark (U+0300–U+036F) stranded after a space — a despacing artifact.
_ORPHAN_COMBINING_RE = re.compile("\\s+[̀-ͯ]+")
# Spacing/standalone diacritic glyphs a PDF emits where an accent belongs, plus a
# mangled umlaut rendered as a straight/curly double-quote (U+0022/U+201C/U+201D).
# Drop one when it sits *between two letters* — a clear name artifact, e.g.
# "Vuli<acute>c" -> "Vulic", "Hakkani-T<dq>ur" -> "Hakkani-Tur",
# "Yeti<cedilla>stiren" -> "Yetistiren", "Rozi<grave>ere" -> "Roziere".
# Apostrophes (U+0027 / U+2019) are excluded so "O'Brien" survives.
_STRAY_DIACRITIC_RE = re.compile(
    "(?<=[A-Za-z])"
    '[¨¯´¸`ˆ˘˙˚˛˜˝"“”̀-ͯ]+'
    "(?=[A-Za-z])"
)


def _clean(s: str) -> str:
    # NFKC folds ligatures (ﬁ->fi) and compatibility glyphs; then heal the
    # diacritic artifacts that break author names + matching.
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("­", "")  # soft hyphen
    s = _ORPHAN_COMBINING_RE.sub("", s)
    s = _STRAY_DIACRITIC_RE.sub("", s)
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


def extract_reference_sections(pdf_path) -> list[list[tuple[str, str]]]:
    """Every reference list in the document, in document order.

    Each list is ``[(cite_key, reference_text)]`` with keys LOCAL to that list: a dense
    numbered list yields ``ref-<n>`` aligned to its own ``[n]`` markers, an author-year
    list sequential ``ref-1``, ``ref-2`` …. The first list is the paper's own; a later
    one belongs to an appendix or supplement, and the citations it answers are the ones
    in the text that precedes it. Returns ``[]`` when no reference section is found or
    PyMuPDF is unavailable, so the caller can fall back to the flat-text path.
    """
    return _select_sections(_candidate_blocks(_layout_lines(pdf_path)))


def extract_reference_entries(pdf_path) -> list[tuple[str, str]]:
    """The paper's own reference list — the first of :func:`extract_reference_sections`."""
    sections = extract_reference_sections(pdf_path)
    return sections[0] if sections else []


def _entries(ref_lines: list[_Line]) -> list[tuple[str, str]]:
    """Segment one block of reference-section lines into ``(cite_key, text)`` entries."""
    if not ref_lines:
        return []
    if _is_numbered(ref_lines):
        pairs = _split_numbered(ref_lines)
    elif _is_flush_left(ref_lines):
        bodies = _split_author_year(ref_lines)
        if len(bodies) < 3:  # not a surname-first author-year block — don't trust it
            bodies = _split_by_indent(ref_lines)
        pairs = [(f"ref-{i}", body) for i, body in enumerate(bodies, start=1)]
    else:
        pairs = [(f"ref-{i}", body) for i, body in enumerate(_split_by_indent(ref_lines), start=1)]
    return [(key, body) for key, body in pairs if len(body) >= 10 and _has_words(body)]


# A reference names people and a work; a row of a table that follows the bibliography on
# the next page ("0.961 0", "k = 5 k = 10 k = 20") names nothing.
_WORD_RE = re.compile(r"[^\W\d_]{3,}")


def _has_words(text: str) -> bool:
    return len(_WORD_RE.findall(text)) >= 2


_ENTRY_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _is_reference_list(entries: list[tuple[str, str]]) -> bool:
    """True when a block of entries reads as a bibliography rather than as prose.

    The one property every reference style shares is a year in (almost) every entry;
    body text segmented the same way — which is what follows a "References" heading that
    is really a table's column header — carries a year in a small minority of its
    pieces. Three entries are the least that can show the pattern.
    """
    if len(entries) < 3:
        return False
    dated = sum(1 for _key, text in entries if _ENTRY_YEAR_RE.search(text))
    return dated / len(entries) >= 0.6


def _candidate_blocks(lines: list[_Line]) -> list[list[_Line]]:
    """The lines that follow each ``References``-style heading, one block per heading.

    A block ends at the Appendix / boilerplate stop or at the next such heading,
    whichever comes first.
    """
    heads = [i for i, ln in enumerate(lines) if _REF_HEAD_RE.fullmatch(ln.text.strip())]
    blocks: list[list[_Line]] = []
    for n, head in enumerate(heads):
        limit = heads[n + 1] if n + 1 < len(heads) else len(lines)
        block: list[_Line] = []
        for ln in lines[head + 1 : limit]:
            if _is_stop(ln):
                break
            block.append(ln)
        blocks.append(block)
    return blocks


def _select_sections(blocks: list[list[_Line]]) -> list[list[tuple[str, str]]]:
    """Keep the candidate blocks that are reference lists, in document order.

    When none passes the test, the block after the LAST heading is returned as it always
    was: the test exists to tell several candidates apart, never to discard the only
    bibliography a document has.
    """
    parsed = [_entries(block) for block in blocks]
    real = [entries for entries in parsed if _is_reference_list(entries)]
    if real:
        return real
    return [parsed[-1]] if parsed and parsed[-1] else []


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
                from .pdf_margin import content_clip

                # Without a review template's line numbers (see pdf_margin).
                for block in page.get_text("rawdict", clip=content_clip(page)).get("blocks", []):
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
                        if y0 < 48:  # running header (venue banner sits ~y36; body starts ~y52+)
                            continue
                        if re.fullmatch(r"\d+", text) and y0 > h - 76:  # page-number footer
                            continue
                        out.append(_Line(pi, text, x0, y0, size, 0 if x0 < w * 0.5 else 1))
        out.sort(key=lambda z: (z.page, z.col, round(z.y0), z.x0))
        return out
    except Exception:  # noqa: BLE001 — fail soft on any PyMuPDF error
        return []


def _merge_baselines(raw, y_tol: float = 1.5, gap_tol: float = 24.0):
    """Merge line fragments that share a baseline (PyMuPDF splits long lines/URLs).

    Fragments separated by a wide horizontal gap (a column gutter) are NOT merged:
    on a two-column page a left-column line and a right-column line can share a
    baseline, and merging them fuses a heading into a reference
    ("Acknowledgments Gu, T.; …"), which then trips the running-header filter and
    drops the entry. Splitting each baseline group at gutter-width gaps keeps the
    long-line/URL re-joining (small gaps) while keeping columns apart.
    """
    raw = sorted(raw, key=lambda z: (z[2], z[1]))
    groups: list[list] = []
    for r in raw:
        for g in groups:
            if abs(r[2] - g[0][2]) <= y_tol:
                g.append(r)
                break
        else:
            groups.append([r])

    def _combine(run: list) -> tuple:
        return (
            _clean(" ".join(i[0] for i in run)),
            min(i[1] for i in run),
            min(i[2] for i in run),
            max(i[3] for i in run),
            max(i[4] for i in run),
            max(i[5] for i in run),
        )

    merged = []
    for g in groups:
        g = sorted(g, key=lambda z: z[1])  # left to right
        run = [g[0]]
        for prev, cur in zip(g, g[1:], strict=False):
            if cur[1] - prev[3] > gap_tol:  # a column gutter — don't merge across it
                merged.append(_combine(run))
                run = [cur]
            else:
                run.append(cur)
        merged.append(_combine(run))
    return merged


# ── section boundaries ────────────────────────────────────────────────────
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


def _is_flush_left(ref_lines: list[_Line], thresh: float = 0.85) -> bool:
    """True when nearly every entry line sits at the column-left margin.

    A flush-left (no hanging indent) author-year bibliography gives the indent
    split no boundary to work with, so it emits one entry per visual line. When
    that's the case, :func:`_split_author_year` segments on content instead.
    """
    left: dict[tuple[int, int], float] = {}
    for ln in ref_lines:
        key = (ln.page, ln.col)
        left[key] = min(left.get(key, ln.x0), ln.x0)
    at_left = sum(1 for ln in ref_lines if ln.x0 <= left[(ln.page, ln.col)] + 6.0)
    return bool(ref_lines) and at_left >= thresh * len(ref_lines)


def _split_author_year(ref_lines: list[_Line]) -> list[str]:
    """Segment a flush-left author-year bibliography on the author-list -> year
    transition (see :data:`_AY_ENTRY_BOUNDARY`).

    The lines are joined into one stream and split just before each fresh author
    run that leads into a 4-digit year, so a multi-line entry stays whole instead
    of fragmenting at every visual line break.
    """
    joined = _join([ln.text for ln in ref_lines])
    return [p.strip() for p in _AY_ENTRY_BOUNDARY.split(joined) if p.strip()]


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
