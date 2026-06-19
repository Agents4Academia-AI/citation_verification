"""
extract/pdf.py — the PDF fallback ingestion path.

Used when no LaTeX e-print is available (``source.tex_available`` is ``False``).
PDFs are noisier than LaTeX (no exact ``\\cite`` anchors, OCR/column artifacts),
so this extractor is best-effort: it splits the document into a *body* and a
*reference list*, parses numbered/keyed reference entries from the list, scans
the body for in-text citation markers, and pairs each marker with its
surrounding sentence. It emits the SAME ``CitationRecord`` stub shape as the
LaTeX extractor (key + ``claim`` + ``cited_as``; judged axes at defaults).

Text extraction:
  * Preferred: ``pypdf`` (lazy import; in the ``sources`` story it ships, but it
    is optional — absence degrades, never crashes).
  * Documented minimal path when ``pypdf`` is missing: a sidecar ``<pdf>.txt``
    (e.g. produced by ``pdftotext`` or the Claude Code ``Read`` tool dumping the
    PDF to text) is used if present. With neither, ``extract`` returns ``[]``.

No network and no heavy imports at module import time.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ..interfaces import PaperSource
from ..schema import CitationRecord, CitedAs, Claim, Paper
from .latex import _coerce_year, _split_authors, make_claim_id

__all__ = ["PdfExtractor", "extract_pdf_text"]


# ───────────────────────────────────────────────────────────────
# Text extraction (pypdf, else sidecar .txt)
# ───────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str | Path) -> str:
    """Return the full text of a PDF, or ``""`` if it cannot be read.

    Tries ``pypdf`` first (lazy import). If unavailable or it fails, falls back
    to a sidecar ``<pdf>.txt`` file. Always fails soft.
    """
    path = Path(pdf_path)
    if not path.is_file():
        return _read_sidecar(path)
    # Best available extractor wins, then we normalize. PyMuPDF reads columns and
    # reading-order far better than pypdf and avoids the per-glyph spacing
    # artifact; pypdf is the always-installed floor. (MinerU — what PaperArena
    # uses — is a heavier opt-in tier that can slot in above PyMuPDF.)
    for extract in (_text_via_pymupdf, _text_via_pypdf):
        text = extract(path)
        if text.strip():
            return _normalize_pdf_text(text)
    return _read_sidecar(path)


def _text_via_pymupdf(path: Path) -> str:
    """Extract text with PyMuPDF in **reading order**. ``""`` if unavailable.

    Naive ``sort=True`` interleaves two-column layouts (reference 1 from the left
    column lands on the same line as reference 23 from the right), which destroys
    the numbered reference list. :func:`_reading_order` instead separates full-width
    spans (title, abstract, single-column section headers, wide tables) from the two
    body columns and emits ``top spans → left column → right column → bottom spans``
    — correct order for the common scholarly layout, so claim sentences and the
    reference list are not scrambled.
    """
    try:
        import fitz  # PyMuPDF — optional; far better than pypdf when present
    except Exception:
        return ""
    try:
        pages: list[str] = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                blocks = [b for b in page.get_text("blocks") if b[4].strip()]
                pages.append("\n".join(_reading_order(blocks, page.rect.width)))
        return "\n".join(pages).strip()
    except Exception:
        return ""


def _reading_order(blocks: list, page_width: float) -> list[str]:
    """Order PyMuPDF text blocks for a 1-or-2-column scholarly page.

    Blocks at least 72% of the page wide are treated as full-width spans (title,
    abstract, single-column section headers, wide tables) and kept around the two
    columns: those in the top fifth of the page come first, the rest last. The
    remaining blocks are split into left/right columns by their horizontal centre
    and read top-to-bottom. Returns the block texts in reading order.
    """
    if not blocks:
        return []
    mid = page_width / 2
    full = [b for b in blocks if (b[2] - b[0]) >= 0.72 * page_width]
    cols = [b for b in blocks if (b[2] - b[0]) < 0.72 * page_width]
    left = sorted([b for b in cols if (b[0] + b[2]) / 2 < mid], key=lambda b: (round(b[1]), b[0]))
    right = sorted([b for b in cols if (b[0] + b[2]) / 2 >= mid], key=lambda b: (round(b[1]), b[0]))
    max_y = max(b[3] for b in blocks)
    top_full = sorted([b for b in full if b[3] < 0.22 * max_y], key=lambda b: (round(b[1]), b[0]))
    bottom_full = sorted([b for b in full if b[3] >= 0.22 * max_y], key=lambda b: (round(b[1]), b[0]))
    return [b[4] for b in (top_full + left + right + bottom_full)]


def _text_via_pypdf(path: Path) -> str:
    """Extract text with pypdf (the always-available floor). ``""`` on failure."""
    try:
        from pypdf import PdfReader  # lazy: optional dependency
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return ""


# Line-break hyphenation: "knowl- edge" / "Lan-\nguage" -> "knowledge" / "Language".
# Only join when a lowercase letter follows (real compounds like "state-of-the-art"
# or "GPT-3" have no following lowercase-after-whitespace and are preserved).
_DEHYPHEN_RE = re.compile(r"(?<=[A-Za-z])-\s+(?=[a-z])")
# A combining diacritic stranded after whitespace ("Yeti ̧stiren" for the surname
# "Yetiştiren") — a despacing/encoding artifact that splits a name and blocks the
# author parser. Drop the stray space+mark so the two halves of the name rejoin.
_ORPHAN_COMBINING_RE = re.compile(r"\s+[\u0300-\u036f]+")
# Restore a space after a separator inside a de-spaced run ("11.YangZ" -> "11. YangZ").
_SEP_RESPACE_RE = re.compile(r"([.,;])(?=[A-Za-z0-9])")


def _normalize_pdf_text(text: str) -> str:
    """Repair four common PDF text artifacts that wreck downstream grounding.

    1. **Ligatures / compatibility glyphs** — NFKC folds ``ﬁ``/``ﬂ`` etc. back to
       ``fi``/``fl`` so a cited title ("Efﬁcient") matches its canonical form.
    2. **Orphaned combining marks** — a diacritic stranded after a space
       ("Yeti ̧stiren") is dropped so the split surname rejoins ("Yetistiren").
    3. **De-hyphenation** — pypdf keeps the hyphen of a word split across a line
       break ("Lan- guage"), so cited titles never match their canonical form.
    4. **Character-spacing** — some lines come out with a space between every
       glyph ("1 1 .Y a n gZ ,G a nZ"), which hides the reference number from the
       entry parser and garbles the authors. Collapsed per line (only on lines
       that are clearly char-spaced), so normal prose is untouched.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _ORPHAN_COMBINING_RE.sub("", text)
    text = _DEHYPHEN_RE.sub("", text)
    return "\n".join(_despace_line(line) for line in text.split("\n"))


def _despace_line(line: str) -> str:
    """Collapse pypdf character-spacing on a single line; leave normal lines as-is.

    A line is treated as char-spaced only when it has many tokens and at least
    half are single characters — a signature normal academic prose never has.
    """
    toks = [t for t in line.split(" ") if t]
    singles = sum(len(t) == 1 for t in toks)
    # Char-spaced lines have many single-char tokens; the comma/period merges
    # (",G", ".Y") keep the ratio below 0.5, so gate on an absolute floor too.
    if len(toks) < 6 or singles < 5 or singles / len(toks) < 0.4:
        return line
    return _SEP_RESPACE_RE.sub(r"\1 ", "".join(line.split(" "))).strip()


def _read_sidecar(pdf_path: Path) -> str:
    """Read a ``<pdf>.txt`` sidecar (the documented pypdf-free path)."""
    sidecar = pdf_path.with_suffix(".txt")
    if sidecar.is_file():
        try:
            return sidecar.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    return ""


# ───────────────────────────────────────────────────────────────
# Reference-list parsing
# ───────────────────────────────────────────────────────────────
_REF_HEADING_RE = re.compile(
    r"\n\s*(?:references|bibliography|works cited)\s*\n", re.IGNORECASE
)
# A numbered reference entry start: "[12] " or "12. " at line start.
_NUM_ENTRY_RE = re.compile(r"(?m)^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+")
# Trailing journal boilerplate that follows the last reference and would otherwise
# be absorbed into it ("Publisher's Note …", "Springer Nature remains neutral …").
_REF_TAIL_RE = re.compile(
    r"\b(?:Publisher['’]?s\s+Note|Springer\s+Nature\s+remains|author\s+self-archiving)",
    re.IGNORECASE,
)
_ARXIV_RE = re.compile(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7})", re.IGNORECASE)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")


def split_body_and_references(text: str) -> tuple[str, str]:
    """Split full text into ``(body, references_block)`` at the last heading.

    If no References heading is found, the whole text is treated as body and the
    references block is empty.
    """
    matches = list(_REF_HEADING_RE.finditer(text))
    if not matches:
        return text, ""
    split_at = matches[-1].end()
    return text[: matches[-1].start()], text[split_at:]


def parse_reference_block(ref_block: str) -> dict[str, CitedAs]:
    """Parse a PDF reference list into ``{cite_key: CitedAs}``.

    Numbered entries get keys ``ref-<n>`` matching the in-text ``[n]`` marker.
    If the list is not numbered, entries get sequential ``ref-1``, ``ref-2`` …
    keys (best-effort; in-text pairing will be weaker). Fails soft to ``{}``.
    """
    if not ref_block.strip():
        return {}
    # Drop trailing publisher/copyright boilerplate so it isn't glued onto the
    # last reference's body (e.g. ref-89 absorbing the "Publisher's Note …" trailer).
    tail = _REF_TAIL_RE.search(ref_block)
    if tail:
        ref_block = ref_block[: tail.start()]

    starts = list(_NUM_ENTRY_RE.finditer(ref_block))
    out: dict[str, CitedAs] = {}

    if starts:
        for i, m in enumerate(starts):
            num = m.group(1) or m.group(2)
            body_start = m.end()
            body_end = starts[i + 1].start() if i + 1 < len(starts) else len(ref_block)
            body = re.sub(r"\s+", " ", ref_block[body_start:body_end]).strip()
            if body:
                out[f"ref-{num}"] = _parse_ref_entry(body)
    else:
        # Unnumbered: split on blank lines, sequential keys.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", ref_block) if c.strip()]
        for i, chunk in enumerate(chunks, start=1):
            body = re.sub(r"\s+", " ", chunk)
            out[f"ref-{i}"] = _parse_ref_entry(body)
    return out


# Connectors that join names in an author list — a name followed by one of these
# is still inside the authors, not the start of the title.
_NAME_CONNECTORS = {"and", "&", "et"}

# One surname-initial author, e.g. "Ferrucci DA", "Chu-Carroll J", "Hakkani-Tur D"
# (a single, optionally hyphenated, surname followed by 1–4 capital initials).
_VANCOUVER_AUTHOR = r"[A-Z][A-Za-z'’-]+\s+(?:[A-Z]\.?-?){1,4}"
# A full surname-initial author run: authors joined by commas/"and", optionally
# closed by "et al", then the period that opens the title. "et al" is captured here
# (and stripped below) so it never leaks into a name, and the title-opening period
# is left for the boundary — so a title may legitimately start with a capital
# ("Building Watson:") or acronym ("GPT-3.5-turbo:") without leaking into authors.
_AUTHOR_RUN_RE = re.compile(
    rf"^\s*(?P<authors>{_VANCOUVER_AUTHOR}"
    rf"(?:\s*(?:,|\band\b)\s*{_VANCOUVER_AUTHOR})*(?:\s*,?\s+et\s+al)?)"
    r"\.\s+(?=[A-Z0-9])(?P<rest>.+)",
    re.DOTALL,
)
_ETAL_TAIL_RE = re.compile(r"\s*,?\s+et\s+al\.?$", re.IGNORECASE)


def _title_from(title_raw: str) -> str | None:
    """Trim a reference title at the next sentence boundary or ``In:`` venue marker."""
    title = re.split(r"\.\s+(?=[A-Z0-9])|\s+In:\s", title_raw, maxsplit=1)[0].strip(" .,")
    return title or None


def _split_author_title(body: str) -> tuple[list[str], str | None]:
    """Split a reference's leading ``Authors. Title`` into ``(authors, title)``.

    Three tiers, most-reliable first:

    1. **Surname-initial author run** — the dominant numbered/Vancouver style
       (``Surname I, Surname I … [et al]. Title``). Cutting at the period that
       closes the run keeps "et al" out of the names and lets a title open with a
       capital or acronym (``Building Watson:``, ``GPT-3.5-turbo:``) without the
       title leaking into the author list.
    2. **Capital→lowercase boundary** for given-name-first / lowercase-opening
       titles (``M. Jadeja and N. Varia, Perspectives …``): the title is the first
       capitalized word immediately followed by a lowercase content word (not a
       name connector like "and").
    3. **Vancouver period fallback** ([[_vancouver_split]]) for the remainder.

    Returns ``([], None)`` when no boundary is found (e.g. a reference whose words
    got merged with no spaces).
    """
    head = re.split(r"(?i)\b(?:arxiv|doi|https?://)", body)[0]

    m = _AUTHOR_RUN_RE.match(head)
    if m:
        author_str = _ETAL_TAIL_RE.sub("", m.group("authors")).strip(" .,")
        if " " in author_str:  # a genuine list, not a stray leading initial
            return _split_authors(author_str.replace(",", " and ")), _title_from(m.group("rest"))

    toks = head.split()
    cut = None
    for i in range(1, len(toks) - 1):
        w = toks[i].strip(".,;:")
        nxt = toks[i + 1].strip(".,;:()")
        if (
            w and w[:1].isupper() and len(w) >= 2
            and nxt and nxt[:1].islower() and len(nxt) >= 2
            and nxt.lower() not in _NAME_CONNECTORS
        ):
            cut = i
            break
    if cut is None:
        return _vancouver_split(head)
    author_str = " ".join(toks[:cut]).strip(" .,")
    authors = _split_authors(author_str.replace(",", " and ")) if author_str else []
    return authors, _title_from(" ".join(toks[cut:]))


def _vancouver_split(head: str) -> tuple[list[str], str | None]:
    """Vancouver fallback: split ``Surname I, Surname I. Title`` on the author period.

    In numbered/Vancouver styles the author list ends at the first initials group
    followed by ``. `` and a capitalized title — a reliable boundary even when the
    title opens with proper nouns/acronyms. Guards against a leading single initial
    (given-name-first styles, which the primary heuristic already handles) by
    requiring the author block to span multiple words. ``([], None)`` if no match.
    """
    m = re.match(r"\s*(.+?\b[A-Z]{1,4})\.\s+(?=[A-Z0-9])(.+)", head, re.DOTALL)
    if not m:
        return [], None
    author_str = m.group(1).strip(" .,")
    if " " not in author_str or len(author_str) < 4:
        return [], None  # too short to be an author list (e.g. a leading "X.")
    title = re.split(r"\.\s+(?=[A-Z0-9])|\s+In:\s", m.group(2), maxsplit=1)[0].strip(" .,")
    authors = _split_authors(author_str.replace(",", " and ")) if author_str else []
    return authors, (title or None)


def _parse_ref_entry(body: str) -> CitedAs:
    """Best-effort structured fields from a single PDF reference line."""
    arxiv = _ARXIV_RE.search(body)
    doi = _DOI_RE.search(body)
    url = _URL_RE.search(body)
    # Strip the arXiv-id / DOI spans before reading the year, so their digits
    # (e.g. the 1911 in "arXiv:1911.03688") aren't mistaken for a publication year.
    year = _YEAR_RE.search(_DOI_RE.sub(" ", _ARXIV_RE.sub(" ", body)))
    authors, title = _split_author_title(body)
    return CitedAs(
        raw=body,
        authors=authors,
        title=title,
        year=_coerce_year(year.group(0)) if year else None,
        venue=None,
        doi=doi.group(0) if doi else None,
        arxiv_id=arxiv.group(1) if arxiv else None,
        url=url.group(0) if url else None,
    )


# ───────────────────────────────────────────────────────────────
# In-text marker scanning
# ───────────────────────────────────────────────────────────────
# Numeric in-text markers: [12] or [3, 5, 9] or [3-5] -> expanded to ref keys.
# pypdf often emits a space just inside the brackets ("[ 2]", "[ 82–84]"), so we
# tolerate whitespace after "[" and before "]" — otherwise every spaced marker is
# silently dropped (it was missing ~half the citations on real two-column PDFs).
_INTEXT_NUM_RE = re.compile(r"\[\s*(\d{1,3}(?:\s*[-–,]\s*\d{1,3})*)\s*\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _expand_num_marker(group: str) -> list[str]:
    """Expand "3, 5-7" -> ['ref-3','ref-5','ref-6','ref-7']."""
    keys: list[str] = []
    for part in re.split(r"\s*,\s*", group):
        rng = re.match(r"(\d+)\s*[-–]\s*(\d+)", part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if b - a < 50:  # guard against absurd ranges (likely not a citation)
                keys.extend(f"ref-{n}" for n in range(a, b + 1))
        elif part.strip().isdigit():
            keys.append(f"ref-{part.strip()}")
    return keys


def _sentence_around(text: str, pos: int, window: int = 500) -> tuple[str, tuple[int, int]]:
    """Sentence containing offset ``pos`` plus its ``(start, end)`` span."""
    lo = max(0, pos - window)
    hi = min(len(text), pos + window)
    chunk = text[lo:hi]
    rel = pos - lo
    starts = [0] + [m.end() for m in _SENT_SPLIT.finditer(chunk)]
    ends = [m.start() for m in _SENT_SPLIT.finditer(chunk)] + [len(chunk)]
    s_start, s_end = 0, len(chunk)
    for st, en in zip(starts, ends, strict=False):
        if st <= rel <= en:
            s_start, s_end = st, en
            break
    sentence = re.sub(r"\s+", " ", chunk[s_start:s_end]).strip()
    return sentence, (lo + s_start, lo + s_end)


# ───────────────────────────────────────────────────────────────
# The extractor
# ───────────────────────────────────────────────────────────────
class PdfExtractor:
    """Fallback extractor for PDFs (satisfies the ``Extractor`` Protocol).

    Produces the same ``CitationRecord`` stub shape as :class:`LatexExtractor`,
    keyed by ``(paper_id, claim_id, cite_key)`` with ``cite_key`` of the form
    ``ref-<n>``. Anchoring is weaker than LaTeX (no exact ``\\cite`` sites), so
    callers should prefer LaTeX when available.
    """

    def extract(self, source: PaperSource) -> list[CitationRecord]:
        """Extract reference + claim-site stubs from a PDF. Fail soft to ``[]``."""
        if not source.pdf_path:
            return []
        text = extract_pdf_text(source.pdf_path)
        if not text:
            return []

        body, ref_block = split_body_and_references(text)
        references = parse_reference_block(ref_block)
        paper = self._build_paper(source)
        records: list[CitationRecord] = []
        seen: set[tuple[str, str, str]] = set()

        for mm in _INTEXT_NUM_RE.finditer(body):
            cite_keys = _expand_num_marker(mm.group(1))
            if not cite_keys:
                continue
            sentence, span = _sentence_around(body, mm.start())
            for cite_key in cite_keys:
                claim_id = make_claim_id(source.paper_id, None, span, cite_key)
                dedup = (source.paper_id, claim_id, cite_key)
                if dedup in seen:
                    continue
                seen.add(dedup)
                cited_as = references.get(cite_key) or CitedAs(raw="")
                records.append(
                    CitationRecord(
                        paper_id=source.paper_id,
                        claim_id=claim_id,
                        cite_key=cite_key,
                        paper=paper,
                        claim=Claim(claim_id=claim_id, text=sentence, section=None, char_span=span),
                        cited_as=cited_as,
                        notes=None if cite_key in references else "no reference-list entry for marker",
                    )
                )

        # If the PDF had no resolvable in-text markers but did have a reference
        # list, still emit one stub per reference (claim text empty) so the
        # citations are visible to downstream stages rather than lost.
        if not records and references:
            for cite_key, cited_as in references.items():
                claim_id = make_claim_id(source.paper_id, None, None, cite_key)
                records.append(
                    CitationRecord(
                        paper_id=source.paper_id,
                        claim_id=claim_id,
                        cite_key=cite_key,
                        paper=paper,
                        claim=Claim(claim_id=claim_id, text="", section=None, char_span=None),
                        cited_as=cited_as,
                        notes="reference parsed from PDF list; no in-text marker located",
                    )
                )
        return records

    @staticmethod
    def _build_paper(source: PaperSource) -> Paper:
        """Construct the ``Paper`` context block from the source."""
        return Paper(
            paper_id=source.paper_id,
            title=source.title,
            arxiv_id=source.arxiv_id,
            source_kind=source.kind or "pdf",
            tex_available=False,
        )
