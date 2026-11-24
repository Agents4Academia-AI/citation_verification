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
from collections import Counter

from .pdf_refs import _clean as _normalize  # NFKC + diacritic heal (shared with refs)
from .sentences import heading_title, is_heading_line, section_at, sentence_span

__all__ = ["extract_link_citations", "extract_text_citations"]

_CITE_PREFIX = "cite."
# A block that is not part of the running text: a caption, or a footnote (small type low on
# the page). Read where it stands, it lands in the middle of the sentence that runs past it
# into the next column; it is set apart as a paragraph of its own instead.
_CAPTION_RE = re.compile(r"^(?:Figure|Fig\.|Table|Tab\.|Algorithm|Listing)\s*[A-Z]?\d+(?:\.\d+)?\s*[:.|]")
_HYPHEN_BREAK_RE = re.compile(r"(?<=[A-Za-z])-\s+(?=[a-z])")  # "knowl- edge" -> "knowledge"

# A parenthetical citation block — any "(...)" carrying a year. Split internally on ";"
# into individual citations; each may list several years for one author
# ("(Li et al. 2022b; Chen et al. 2025a; Jung et al. 2019, 2020, 2022)").
_PAREN_BLOCK_RE = re.compile(r"\(([^()]*(?:19|20)\d{2}[a-z]?[^()]*)\)")
# Narrative "Surname et al. (2023)" / "Surname and Other (2024, 2025)".
_AY_NARRATIVE_RE = re.compile(
    r"\b([A-Z][A-Za-z'’.\-]+(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-z'’.\-]+|\s*&\s*[A-Z][A-Za-z'’.\-]+)?)"
    r"\s*\(\s*((?:19|20)\d{2}[a-z]?(?:\s*,\s*(?:19|20)\d{2}[a-z]?)*)\s*\)"
)
# A citation chunk's leading first-author surname, and every year token within a chunk.
_CHUNK_SURNAME_RE = re.compile(r"^[\s,]*([A-Z][A-Za-z'’.\-]+)")
_YEAR_TOKEN_RE = re.compile(r"\b((?:19|20)\d{2}[a-z]?)\b")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def extract_link_citations(pdf_path) -> list[dict]:
    """Tier 1: ``[{cite_key, claim, span, anchor, section}]`` from ``cite.*`` hyperlink anchors.

    The claim is the sentence the link sits in, read from the whole document as one flow of
    text (see :func:`_document_flow`): a sentence that runs past a column or a page is read
    to its end, and a section heading is a boundary no sentence crosses. ``span`` is the
    sentence's char range in that flow, usable as a claim-id seed. The author-name and year
    halves of one citation each carry a link, so sites that share a ``(cite_key, claim)``
    are de-duplicated.
    """
    try:
        import fitz  # PyMuPDF — optional
    except Exception:
        return []
    try:
        sites: list[dict] = []
        with fitz.open(str(pdf_path)) as doc:
            flow = _document_flow(doc)
            text = flow.text
            for pno, page in enumerate(doc):
                anchors = [
                    (lnk["from"], (lnk.get("nameddest") or "")[len(_CITE_PREFIX) :])
                    for lnk in page.get_links()
                    if (lnk.get("nameddest") or "").startswith(_CITE_PREFIX)
                ]
                words = flow.page_words.get(pno, [])
                for rect, cite_key in anchors:
                    if not cite_key:
                        continue
                    best = _covered_word(words, rect)
                    if best is None or (pno, best) not in flow.offsets:
                        continue
                    pos = flow.offsets[(pno, best)]
                    claim, span = _claim_at(text, pos)
                    if claim:
                        sites.append(
                            {"cite_key": cite_key, "claim": claim, "span": span,
                             "anchor": _text_under(words, rect),
                             "section": section_at(flow.headings, pos)}
                        )
        return _dedupe(sites)
    except Exception:  # noqa: BLE001 — fail soft on any PyMuPDF error
        return []


class _Flow:
    """A document's running text, with where each word of each page landed in it."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.offsets: dict[tuple[int, int], int] = {}
        self.page_words: dict[int, list] = {}
        self.headings: list[tuple[int, str]] = []

    def add(self, piece: str) -> int:
        at = self.length
        self.parts.append(piece)
        self.length += len(piece)
        return at

    def paragraph_break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n\n"):
            self.add("\n\n")

    @property
    def text(self) -> str:
        return "".join(self.parts)


def _document_flow(doc) -> _Flow:
    """The running text of a PDF, in reading order, as one string.

    Per page the blocks are put in reading order — on a two-column page, the full-width
    blocks at the top, the left column, the right column, then the remaining full-width
    blocks; on a one-column page, top to bottom — and each block's words in line order.
    Running text is joined with single spaces, across lines, blocks, columns and pages, so
    a sentence is read to its end wherever it continues. A heading is a paragraph of its own
    (and recorded with its offset, for the claim's section). Captions, footnotes and other
    small type are taken out of the flow and set, each as its own paragraph, after the last
    page.
    """
    from .pdf_margin import content_clip

    flow = _Flow()
    pages = []
    sizes: Counter = Counter()
    for page in doc:
        clip = content_clip(page)
        # Without a review template's line numbers: read with the text, they land inside
        # sentences ("ques- 068 tion") and shift sentence boundaries.
        words = page.get_text("words", clip=clip)
        blocks = _text_blocks(page, clip, sizes)
        pages.append((page, words, blocks))
    body = sizes.most_common(1)[0][0] if sizes else 10.0

    asides: list[tuple[int, list[tuple[int, object]]]] = []
    for pno, (page, words, blocks) in enumerate(pages):
        flow.page_words[pno] = words
        members: dict[int, list[int]] = {i: [] for i in range(len(blocks))}
        loose: list[int] = []
        for wi, w in enumerate(words):
            bi = _block_of(blocks, w)
            (members[bi] if bi is not None else loose).append(wi)
        order = _block_order(blocks, words, page.rect.width)
        for bi in order:
            kind = _block_kind(blocks[bi], body)
            idx = members[bi]
            if not idx:
                continue
            if kind == "aside":
                asides.append((pno, [(wi, words[wi]) for wi in idx]))
                continue
            ordered = _order_reading([words[wi] + (wi,) for wi in idx])
            if kind == "heading":
                flow.paragraph_break()
                at = flow.length
                _emit(flow, pno, ordered)
                flow.headings.append((at, heading_title(" ".join(w[4] for w in ordered))))
                flow.paragraph_break()
            else:
                _emit(flow, pno, ordered)
        if loose:
            _emit(flow, pno, _order_reading([words[wi] + (wi,) for wi in loose]))
    for pno, members in asides:
        flow.paragraph_break()
        _emit(flow, pno, _order_reading([w + (wi,) for wi, w in members]))
    return flow


def _emit(flow: _Flow, pno: int, ordered: list) -> None:
    """Append words (each carrying its page index last) to the flow, space-separated."""
    for w in ordered:
        if flow.parts and not flow.parts[-1].endswith((" ", "\n")):
            flow.add(" ")
        flow.offsets[(pno, w[-1])] = flow.add(w[4])


def _text_blocks(page, clip, sizes: Counter) -> list[dict]:
    """A page's text blocks: bounding box, text, mean font size, and share of bold type.

    Each span's character count is added to ``sizes`` (rounded to half points), whose most
    common value is the body type size.
    """
    out = []
    for b in page.get_text("dict", clip=clip).get("blocks", []):
        if b.get("type") != 0:
            continue
        lines, chars, bold, weighted = [], 0, 0, 0.0
        for ln in b.get("lines", []):
            text = "".join(sp.get("text", "") for sp in ln.get("spans", []))
            if text.strip():
                lines.append(text)
            for sp in ln.get("spans", []):
                n = len(sp.get("text", "").strip())
                chars += n
                weighted += (sp.get("size") or 0.0) * n
                bold += n if (sp.get("flags", 0) & 16) else 0
                sizes[round((sp.get("size") or 0.0) * 2) / 2] += n
        if lines:
            out.append({"bbox": tuple(b["bbox"]), "lines": lines,
                        "size": weighted / max(chars, 1), "bold": bold / max(chars, 1)})
    return out


def _block_kind(block: dict, body: float) -> str:
    """``heading``, ``aside`` (caption, footnote, small type) or ``flow`` (running text)."""
    text = " ".join(" ".join(block["lines"]).split())
    if _CAPTION_RE.match(text):
        return "aside"
    short = len(block["lines"]) <= 2 and len(text) <= 120
    if short and is_heading_line(text):
        return "heading"
    if (
        short
        and text[:1].isupper()
        and text[-1:] not in ".,;:"
        and len(text.split()) <= 12
        and (block["bold"] >= 0.6 or block["size"] >= body + 0.9)
    ):
        return "heading"  # set apart by its type, not its wording
    if block["size"] <= body - 0.8:
        # Smaller type than the running text: a footnote, a figure's labels, a table set
        # small. Left in place, "CONCAT, and" from a figure opened the sentence below it.
        return "aside"
    return "flow"


def _block_of(blocks: list[dict], w) -> int | None:
    """The block whose box holds the centre of word ``w``."""
    cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
    for i, b in enumerate(blocks):
        x0, y0, x1, y1 = b["bbox"]
        if x0 - 1 <= cx <= x1 + 1 and y0 - 1 <= cy <= y1 + 1:
            return i
    return None


def _block_order(blocks: list[dict], words: list, page_width: float) -> list[int]:
    """Block indices in reading order (see :func:`_document_flow`)."""
    idx = list(range(len(blocks)))
    if not _is_two_column(words, page_width):
        return sorted(idx, key=lambda i: (round(blocks[i]["bbox"][1]), blocks[i]["bbox"][0]))
    mid = page_width / 2
    width = lambda i: blocks[i]["bbox"][2] - blocks[i]["bbox"][0]  # noqa: E731
    full = [i for i in idx if width(i) >= 0.72 * page_width]
    cols = [i for i in idx if width(i) < 0.72 * page_width]
    by_y = lambda i: (round(blocks[i]["bbox"][1]), blocks[i]["bbox"][0])  # noqa: E731
    left = sorted([i for i in cols if (blocks[i]["bbox"][0] + blocks[i]["bbox"][2]) / 2 < mid], key=by_y)
    right = sorted([i for i in cols if (blocks[i]["bbox"][0] + blocks[i]["bbox"][2]) / 2 >= mid], key=by_y)
    first_col = min((blocks[i]["bbox"][1] for i in cols), default=0.0)
    top = sorted([i for i in full if blocks[i]["bbox"][3] <= first_col + 1], key=by_y)
    rest = sorted([i for i in full if i not in top], key=by_y)
    return top + left + right + rest


def _covered_word(words: list, rect) -> int | None:
    """Index of the word a link box covers MOST — not the first it touches: a link box is
    a point or two taller than its line and grazes the bottom of the line above, and the
    citation was then paired with the sentence before it."""
    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1

    def overlap(w) -> float:
        return max(0.0, min(w[2], x1) - max(w[0], x0)) * max(0.0, min(w[3], y1) - max(w[1], y0))

    if not words:
        return None
    best = max(range(len(words)), key=lambda i: overlap(words[i]))
    return best if overlap(words[best]) > 0 else None


def _claim_at(text: str, pos: int, breaks: list[int] | None = None) -> tuple[str, tuple[int, int]]:
    """The sentence of ``text`` at ``pos``, normalized, with its span."""
    start, end = sentence_span(text, pos, breaks=breaks)
    raw = _HYPHEN_BREAK_RE.sub("", text[start:end])  # join line-break hyphenation
    return _normalize(raw), (start, end)


def extract_text_citations(
    text: str, breaks: list[int] | None = None, headings: list[tuple[int, str]] | None = None
) -> list[dict]:
    """Tier 2: ``[{cite_key, claim, span}]`` from author-year text patterns.

    A conservative fallback for PDFs with no hyperlinks. Group-aware: a grouped
    parenthetical ("(Li et al. 2022b; Chen et al. 2025a; Jung et al. 2019, 2020)")
    yields one site per individual citation AND per year — not just the first — so a
    claim that cites several works produces several ``(claim, cite_key)`` pairs.
    """
    text = text or ""
    sites: list[dict] = []

    def add(surname: str, year: str, pos: int) -> None:
        s = _fold(re.sub(r"[^A-Za-z'’\-]", "", surname))
        if len(s) < 2:
            return
        claim, span = _claim_at(text, pos, breaks)
        if claim:
            sites.append({"cite_key": f"{s}{year}", "claim": claim, "span": span,
                          "section": section_at(headings or [], pos)})

    # Parenthetical groups: ";"-separated citations, each possibly multi-year.
    for mb in _PAREN_BLOCK_RE.finditer(text):
        for chunk in re.split(r"\s*;\s*", mb.group(1)):
            ms = _CHUNK_SURNAME_RE.match(chunk)
            if ms:
                for y in _YEAR_TOKEN_RE.findall(chunk):
                    add(ms.group(1), y, mb.start())

    # Narrative "Surname et al. (2019, 2020)".
    for mn in _AY_NARRATIVE_RE.finditer(text):
        for y in _YEAR_TOKEN_RE.findall(mn.group(2)):
            add(mn.group(1).split()[0], y, mn.start())

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


def _is_two_column(words: list, page_width: float) -> bool:
    """Detect a genuine two-column page (vs a single-column one with wide lines).

    A two-column layout leaves a near-empty central gutter — very few words cross
    the page midline — with substantial text on both sides. A single-column page
    has many full-width lines straddling the midline; splitting it at the midpoint
    would bisect every line and interleave the halves, scrambling the sentence. So
    only split when the gutter is real. Per page: layouts mix within one document.
    """
    if len(words) < 40:
        return False  # too little text to judge — treat as single-column (no split)
    mid = page_width / 2
    straddle = sum(1 for w in words if w[0] < mid < w[2])
    left = sum(1 for w in words if (w[0] + w[2]) / 2 < mid)
    right = len(words) - left
    return straddle < 0.02 * len(words) and min(left, right) > 0.25 * len(words)


def _sentence_at_rect(words, rect, page_width: float) -> tuple[str, tuple[int, int]]:
    """Sentence (claim) containing the citation at ``rect`` on one page; ``""`` when not found.

    Words are read in the rect's column (only when the page is genuinely two-column),
    ordered by line, and the sentence covering the word the rect overlaps most is
    returned. :func:`extract_link_citations` reads the whole document instead, so a
    sentence that continues on the next column or page is not cut; this is the one-page
    form of the same reading.
    """
    if _is_two_column(words, page_width):
        mid = page_width / 2
        col = 0 if (rect.x0 + rect.x1) / 2 < mid else 1
        cw = _order_reading([w for w in words if (0 if (w[0] + w[2]) / 2 < mid else 1) == col])
    else:
        cw = _order_reading(words)  # single column: read full width, never split mid-line
    best = _covered_word(cw, rect)
    if best is None:
        return "", (0, 0)
    offsets, pos = [], 0
    for w in cw:
        offsets.append(pos)
        pos += len(w[4]) + 1
    return _claim_at(" ".join(w[4] for w in cw), offsets[best])


def _sentence_around(text: str, pos: int) -> tuple[str, tuple[int, int]]:
    """The sentence containing offset ``pos``, normalized (see :mod:`.sentences`)."""
    return _claim_at(text, pos)


def _text_under(words: list, rect) -> str:
    """The words a link box covers — what the reader sees as the citation ("Bengio &
    Grandvalet", "(2011)"). Mostly-inside only, so a word on the line above is not taken."""
    out = []
    for w in words:
        ix = max(0.0, min(w[2], rect.x1) - max(w[0], rect.x0))
        iy = max(0.0, min(w[3], rect.y1) - max(w[1], rect.y0))
        area = max(1e-6, (w[2] - w[0]) * (w[3] - w[1]))
        if ix * iy >= 0.5 * area:
            out.append(w[4])
    return " ".join(out)


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
            same = next((k for k in kept if a0 < k["span"][1] and a1 > k["span"][0]), None)
            if same is None:
                kept.append(dict(s))
            elif s.get("anchor") and s["anchor"] not in (same.get("anchor") or ""):
                # The name half and the year half of one citation: keep both texts.
                same["anchor"] = f"{same.get('anchor') or ''} {s['anchor']}".strip()
        out.extend(kept)
    return out
