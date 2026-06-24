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
from .latex import TABLE_NOTE, _coerce_year, _split_authors, make_claim_id
from .pdf_links import extract_link_citations, extract_text_citations
from .pdf_refs import extract_reference_entries

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
    # Try every available extractor and keep the text that parses the reference
    # list best. PyMuPDF often has better two-column reading order, while pypdf
    # sometimes preserves intra-reference spaces better; either can be the right
    # answer for a given PDF.
    candidates = [
        _normalize_pdf_text(text)
        for text in (_text_via_pymupdf(path), _text_via_pypdf(path))
        if text.strip()
    ]
    if candidates:
        return max(candidates, key=_reference_parse_score)
    return _read_sidecar(path)


def _reference_parse_score(text: str) -> tuple[int, int, int, int, int]:
    """Score extracted text by how well its reference list parses.

    This is deliberately source-agnostic: it rewards complete structured fields
    and penalizes glued-token artifacts, so the extractor choice generalizes
    beyond a single paper.
    """
    _, ref_block = split_body_and_references(text)
    refs = parse_reference_block(ref_block)
    complete = sum(bool(c.authors and c.title and c.year) for c in refs.values())
    has_id = sum(bool(c.doi or c.arxiv_id or c.url) for c in refs.values())
    title_chars = sum(len(c.title or "") for c in refs.values())
    glued_penalty = sum(_glued_reference_penalty(c.raw or "") for c in refs.values())
    return (complete, len(refs), has_id, -glued_penalty, title_chars)


def _glued_reference_penalty(raw: str) -> int:
    """Count no-space author/title artifacts common in PDF extraction."""
    return len(
        re.findall(
            r"(?:[A-Za-zÀ-ÖØ-öø-ÿ]{3,}[A-Z]{1,4}[,.](?=[A-ZÀ-ÖØ-Þ])|"
            r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Z]\.[A-ZÀ-ÖØ-Þ])",
            raw or "",
        )
    )


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
_GLUED_AUTHOR_INITIAL_RE = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ]{1,})([A-Z]{1,3})(?=\b|[,.;])"
)


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
    joined = _SEP_RESPACE_RE.sub(r"\1 ", "".join(line.split(" "))).strip()
    return _split_glued_author_initials(joined)


def _split_glued_author_initials(line: str) -> str:
    """Repair char-spaced author tokens like ``YangZ`` -> ``Yang Z``."""
    m = re.match(r"^(\s*(?:\d+\.\s*|\[\d+\]\s*)?)([^.]{1,260})(\.\s*)(.*)$", line)
    if not m:
        return line
    prefix, authors, sep, rest = m.groups()
    if "," not in authors and len(authors.split()) > 3:
        return line
    authors = _GLUED_AUTHOR_INITIAL_RE.sub(r"\1 \2", authors)
    return f"{prefix}{authors}{sep}{rest}"


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
# arXiv id in either the "arXiv:2102.00182" form or a URL form
# "arxiv.org/abs/2102.00182" / "arxiv.org/pdf/2102.00182(.pdf)" (optional v-suffix,
# or old-style "math/0211159"). Many bibliographies cite only the URL.
_ARXIV_RE = re.compile(
    r"(?:arXiv:\s*|arxiv\.org/(?:abs|pdf)/)"
    r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7})",
    re.IGNORECASE,
)
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


def _is_dense_numbering(starts: list[re.Match]) -> bool:
    """True when ``[N]`` / ``N.`` line-starts form a real numbered reference list.

    An unnumbered (author-year) bibliography can still contain a few stray ``N.``
    line-starts — a wrapped DOI (``…acl-long.\\n427. URL``) or an appendix list
    (``1. … 2. …``). A genuine numbered list is a DENSE sequence (most of ``1..N``
    present); a handful of strays across a wide span is not, so we fall back to the
    blank-line split instead of collapsing the whole bibliography into junk entries.
    """
    nums = sorted(int(m.group(1) or m.group(2)) for m in starts)
    if len(nums) < 3:
        return False
    span = nums[-1] - nums[0] + 1
    return len(nums) >= 0.6 * span


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

    if _is_dense_numbering(starts):
        for i, m in enumerate(starts):
            num = m.group(1) or m.group(2)
            body_start = m.end()
            body_end = starts[i + 1].start() if i + 1 < len(starts) else len(ref_block)
            body = _clean_ref_body(ref_block[body_start:body_end])
            if body:
                out[f"ref-{num}"] = _parse_ref_entry(body)
    else:
        # Unnumbered: split on blank lines, sequential keys.
        chunks = [c.strip() for c in re.split(r"\n\s*\n", ref_block) if c.strip()]
        for i, chunk in enumerate(chunks, start=1):
            body = _clean_ref_body(chunk)
            out[f"ref-{i}"] = _parse_ref_entry(body)
    return out


def _references_from_layout(pdf_path: str | Path) -> dict[str, CitedAs]:
    """Primary bibliography parse via the layout-aware extractor (:mod:`pdf_refs`).

    Reads the PDF's geometry — column reading order, hanging-indent segmentation,
    font-gated section stops, baseline-merged lines — so numbered (journal) *and*
    author-year (conference) bibliographies are parsed by one path. Each entry's
    text is then read into structured fields by :func:`_parse_ref_entry`.

    Returns ``{}`` when PyMuPDF is unavailable or no reference section is found;
    :meth:`PdfExtractor.extract` then falls back to :func:`parse_reference_block`
    over the flat text (the always-available floor).
    """
    out: dict[str, CitedAs] = {}
    for cite_key, body in extract_reference_entries(pdf_path):
        body = _clean_ref_body(body)
        if body:
            out[cite_key] = _parse_ref_entry(body)
    return out


def _clean_ref_body(text: str) -> str:
    """Normalize whitespace inside one reference entry without changing content."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    # Rejoin a surname the glyph-spacer split across a transliteration apostrophe
    # ("Murakhovs’ ka" -> "Murakhovs’ka"): the stray space made the lowercase tail
    # look like a title word, so the author/title boundary landed inside the name.
    text = re.sub(r"(?<=[A-Za-z])(['’])\s+(?=[a-z])", r"\1", text)
    # Drop a trailing "cited on pages" back-reference some templates print after the
    # year ("…, 2017.2,17" / "…, 2025.6") — anchored to the end so it never touches
    # a mid-string arXiv id (e.g. "2503.18892").
    return re.sub(r"(?<=\d)\.\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*$", "", text).strip()


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

# Author-year style ("Surname, I., Surname, I., … [and Surname, I.] [et al]. Title"),
# the dominant conference bibliography form (ICLR/ICML/NeurIPS). It differs from
# Vancouver in that a comma sits *between the surname and its initials* — so the
# whole-comma → " and " split Vancouver relies on would tear "Azerbayev, Z." into
# two names. Each initial must carry its period: a *bare* capital begins the title
# ("Huet, G. P. The calculus …"), so requiring the period keeps the author run from
# swallowing the first title word.
# Initials like "Z.", "G. P.", "M.-A.", and "K.- F." (a despacing artifact puts a
# space after the hyphen) — tolerate optional whitespace around the hyphen.
# An initials group after the surname: "J.", "P. J.", "S.-T.", and the lowercase
# Korean romanization "J.-w." / "j.-h." — each initial keeps its period (so a title
# word can't pose as one). The surname still starts uppercase, so a title fragment
# ("Method, e.g. …") can't open an author run.
_AY_INITS = r"(?:[A-Za-z]\.(?:\s*-\s*[A-Za-z]\.?)*\s*){1,4}"
_AY_NAME = rf"[A-Z][\w'’`-]+,\s*{_AY_INITS}"
# Names are comma- OR semicolon-separated ("Cai, H.; Zhang, P.; and Yuan, T." —
# the AAAI/IEEE style); the comma-after-surname in _AY_NAME keeps this from firing
# on Vancouver ("Surname I"), so allowing ";" only widens it to semicolon lists.
_AY_RUN_RE = re.compile(
    rf"^\s*(?P<authors>{_AY_NAME}(?:\s*[,;]?\s*(?:and\s+)?{_AY_NAME})*"
    rf"(?:\s*[,;]?\s+et\s+al\.?)?)\s*(?P<rest>[A-Z0-9].+)$",
    re.DOTALL,
)


def _split_ay_authors(author_str: str) -> list[str]:
    """Split an author-year run into ``["Surname, I.", …]`` (surname+initials kept whole)."""
    author_str = _ETAL_TAIL_RE.sub("", author_str)
    units = re.findall(rf"[A-Z][\w'’`-]+,\s*{_AY_INITS}", author_str)
    return [re.sub(r"\s+", " ", u).strip().strip(",").strip() for u in units]


# Given-name-first authors ("Serina Chang, Ashton Anderson, and Jake M Hofman.
# Title" — the NeurIPS/arXiv style with full first names). A token is a capitalized
# word — accent-tolerant ("Jörg", "Loáiciga"), hyphen-joined even when the second
# part is lowercase ("Wen-tau"), apostrophe-joined ("O’Neil"), or a bare initial
# ("M.", "D."). A name is 2–4 tokens, allowing a lowercase nobiliary particle
# ("Niels van Berkel"). `_NAME_LETTER` is any Unicode letter (not just A–Z).
_NAME_LETTER = r"[^\W\d_]"
_GNF_TOKEN = rf"[A-ZÀ-Þ]{_NAME_LETTER}*(?:[-’']{_NAME_LETTER}+)*\.?"
_GNF_PARTICLE = r"(?:van|von|der|den|del|della|de|di|da|du|dos|la|le|bin|al)"
_GNF_NAME = rf"{_GNF_TOKEN}(?:\s+(?:{_GNF_PARTICLE}\s+)*{_GNF_TOKEN}){{1,3}}"
# The last author after "and"/"&" may be a mononym/handle ("… and Vinci. Title").
_GNF_MONONYM = rf"[A-ZÀ-Þ]{_NAME_LETTER}+(?:[-’']{_NAME_LETTER}+)*"
_GNF_LAST = rf"(?:{_GNF_NAME}|{_GNF_MONONYM})"
_TITLE_START = r"[A-Z0-9“”\"‘’'(]"  # a title may open with a capital, digit, quote, or paren
# Author lists end at an unambiguous boundary before the title — "et al." or
# ", and <Name>." — so the title may start with any capital/digit without leaking.
# Comma separators keep each name from greedily eating the next, so neither the
# leading list nor the title overshoots.
_GNF_ETAL_RE = re.compile(
    rf"^\s*(?P<authors>{_GNF_NAME}(?:\s*,\s*{_GNF_NAME})*)\s*,?\s+et\s+al\.\s+(?P<rest>{_TITLE_START}.+)$",
    re.DOTALL,
)
_GNF_AND_RE = re.compile(
    rf"^\s*(?P<authors>{_GNF_NAME}(?:\s*,\s*{_GNF_NAME})*\s*,?\s+(?:and|&)\s+{_GNF_LAST})"
    rf"\.\s+(?P<rest>{_TITLE_START}.+)$",
    re.DOTALL,
)
# A lone given-name-first author ("Harrison Chase. Langchain …"): 2–3 full-word
# Title-case tokens (optional middle initial) then ". " + title. Full words (each
# [A-Z][a-z]+) distinguish it from a Vancouver "Surname I." (whose last token is a
# bare initial), so this never fires on the numbered/Vancouver style.
_GNF_SOLO_RE = re.compile(
    r"^\s*(?P<authors>[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z][A-Za-z'’-]*){1,2})"
    r"\.\s+(?P<rest>[A-Z0-9].+)$",
    re.DOTALL,
)


def _split_gnf_authors(author_str: str) -> list[str]:
    """Split a given-name-first run into individual names, dropping ``and``/``et al``."""
    author_str = _ETAL_TAIL_RE.sub("", author_str)
    author_str = re.sub(r"\s*,?\s+(?:and|&)\s+", ", ", author_str)  # ", and X" / " & X" -> ", X"
    parts = [p.strip(" .,") for p in author_str.split(",")]
    return [p for p in parts if p and not re.fullmatch(r"(?i)et\s+al\.?", p)]


def _split_given_name_first(head: str) -> tuple[list[str], str | None] | None:
    """Authors as ``First [M.] Last, …, and First Last. Title`` (or ``…, et al. Title``).

    Recognized by the ``et al.`` / ``, and <Name>.`` terminator, which Vancouver and
    author-year tiers miss (no surname-comma-initials, no ``Surname I.`` run). ``None``
    when the head isn't this shape, so the caller falls through to the next tier.
    """
    for rx in (_GNF_ETAL_RE, _GNF_AND_RE):
        m = rx.match(head)
        if m:
            authors = _split_gnf_authors(m.group("authors"))
            if len(authors) >= 2:  # a real list (a lone "Firstname Lastname. Title" isn't GNF-safe)
                return authors, _title_from(m.group("rest"))
    # Single author ("Harrison Chase. Langchain …") — needs the stricter full-word rule.
    m = _GNF_SOLO_RE.match(head)
    if m:
        authors = _split_gnf_authors(m.group("authors"))
        if len(authors) == 1:
            return authors, _title_from(m.group("rest"))
    return None


def _title_from(title_raw: str) -> str | None:
    """Trim a reference title at the next sentence boundary or ``In:`` venue marker.

    Also drop a trailing ``, 2024`` year suffix — author-year references write
    ``Title, YEAR.`` so the year leaks into the title otherwise, which then
    pollutes the resolver's title query (e.g. "… applications, 2025" never matched).
    """
    # Author-year references put the year *between* authors and title ("… Ginis.
    # 2025. The relationship …"); strip a leading year token so the title isn't
    # read as "2025" (the ACL/EMNLP and AAAI "Authors. YEAR. Title." structure).
    title_raw = re.sub(r"^\s*(?:19|20)\d{2}[a-z]?\.\s+(?=\S)", "", title_raw)
    title = re.split(r"\.\s+(?=[A-Z0-9])|\s+In:\s", title_raw, maxsplit=1)[0].strip(" .,")
    # Cut a journal trailer that started lowercase so the split above missed it
    # ("… knowledge. nature, 550(7676):354–359, 2017") — keyed on a volume(issue):page.
    title = re.split(r"\.\s+\S.*?\b\d{1,4}\s*\(\d{1,4}\)\s*:\s*\d", title, maxsplit=1)[0].strip(" .,")
    title = re.sub(r",\s*(?:19|20)\d{2}[a-z]?$", "", title).strip(" .,")
    return title or None


# Springer/LNCS colon style ("Surname, I., …, Surname, I.: Title. In: Venue (Year)").
# The author list — the same surname-comma-initials run as author-year — terminates
# at a COLON, not a period+year, so the period tiers eat the last author into the
# title. Anchored on a real author run immediately before ":", so a title's own
# colon ("Llemma: An open …", whose prefix is not "Surname, I.") never triggers it.
_LNCS_COLON_RE = re.compile(
    rf"^\s*(?P<authors>{_AY_NAME}(?:\s*,\s*(?:and\s+)?{_AY_NAME})*"
    rf"(?:\s*,\s*et\s+al\.?)?)\s*:\s+(?P<rest>{_TITLE_START}.+)$",
    re.DOTALL,
)


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

    # Tier 0a: Springer/LNCS colon style — the author run ends at ":", not a period,
    # so the period tiers below would steal the last author into the title. Tried
    # first because the colon is an unambiguous boundary when a real author run
    # precedes it.
    m = _LNCS_COLON_RE.match(head)
    if m:
        ay_authors = _split_ay_authors(m.group("authors"))
        if ay_authors:
            return ay_authors, _title_from(m.group("rest"))

    # Tier 0: author-year ("Surname, I., …. Title"). Tried first because its names
    # carry an internal comma that the Vancouver tiers below would mis-split. The
    # regex requires a comma right after a surname, which Vancouver ("Surname I")
    # never has — so numbered/Vancouver references fall straight through.
    m = _AY_RUN_RE.match(head)
    if m:
        ay_authors = _split_ay_authors(m.group("authors"))
        if ay_authors:
            return ay_authors, _title_from(m.group("rest"))

    # Tier 1: given-name-first run ("First Last, …, and First Last. Title"). Tried
    # before Vancouver because Vancouver would mis-read a given-name + middle-initial
    # ("Justin D. Weisz") as a single "Surname Initial" author and steal the surname
    # into the title.
    gnf = _split_given_name_first(head)
    if gnf:
        return gnf

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
    cut = _rewind_colon_title_prefix(toks, cut)
    author_str = " ".join(toks[:cut]).strip(" .,")
    authors = _split_authors(author_str.replace(",", " and ")) if author_str else []
    return authors, _title_from(" ".join(toks[cut:]))


def _rewind_colon_title_prefix(toks: list[str], cut: int) -> int:
    """Move a fallback title boundary back over ``Acronym:`` title prefixes."""
    if cut >= 2 and toks[cut - 1].strip(".,;").endswith(":"):
        prev = toks[cut - 2].strip(".,;:")
        if prev and prev[:1].isupper() and not re.fullmatch(r"[A-Z]\.?", prev):
            return cut - 2
    return cut


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
        doi=doi.group(0).rstrip(".,;)") if doi else None,
        arxiv_id=arxiv.group(1) if arxiv else None,
        url=url.group(0).rstrip(".,;)") if url else None,
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
# A claim that opens with a table caption ("Table 3 Comparison of …") is a table
# cell/caption, not a prose claim — its relevance is not assessed (in_table).
_TABLE_CAPTION_RE = re.compile(r"^\s*(?:Table|TABLE|Tab\.)\s*\d+\b")
# A comparison-table *body* row dumped into one claim: several "[n]" markers each
# followed by a year or a capitalized cell (table columns), e.g.
# "ELIZA [19] 1966 Chatbot … SHRDLU [20] 1970 Task-Oriented …". In prose a marker
# is followed by lowercase/punctuation ("[19] showed", "[19], which"), so this
# does NOT fire on an ordinary multi-citation sentence ("works [1], [2], [3] …").
_TABLE_ROW_RE = re.compile(r"\[\s*\d+\s*\]\s+(?:(?:19|20)\d{2}\b|[A-Z])")


def _looks_like_table_dump(text: str) -> bool:
    """True when a claim reads as a comparison-table row dump (skip its relevance)."""
    return len(_TABLE_ROW_RE.findall(text or "")) >= 3
_MAX_CLAIM_CHARS = 360
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_AFFILIATION_START_RE = re.compile(
    r"^\s*\d+\s+(?:Department|School|Faculty|Institute|College|University|"
    r"APPCAIR|Mohamed\s+Bin\s+Zayed)\b",
    re.IGNORECASE,
)
_AFFILIATION_HINT_RE = re.compile(
    r"\b(?:Department|School|Faculty|Institute|College|University|Campus|"
    r"Singapore|Scotland|United Arab Emirates|BITS-Pilani|Nanyang|Napier|MBZUAI)\b",
    re.IGNORECASE,
)
_FOOTER_RE = re.compile(
    r"^(?:\d{1,4}|©.*|.*Springer.*|.*Cognitive Computation.*\d{4}.*|"
    r"The Author\(s\).*|under exclusive licence.*)$",
    re.IGNORECASE,
)
# A standalone numbered section heading ("1 Introduction", "2.1 Method", "3 Experiments
# and Results") — short, title-cased, no terminal punctuation. Dropped so it doesn't
# glue onto an adjacent sentence ("…unlock this potential. 1 Introduction The development
# …"); the sentence splitter can't break before a digit, so the heading would otherwise
# fuse two sentences into one claim.
_SECTION_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z][\w &,'/-]{0,48}$")
# A footnote / author-contribution line ("∗ Equal Contribution. † Project Lead.", or a
# "• Yang Yue led the project …" bullet): starts with a footnote/bullet glyph. Not prose.
_FOOTNOTE_RE = re.compile(r"^[*∗†‡§¶•●◦▪]+\s*\S")
# Repeating venue/proceedings header-footer boilerplate ("39th Conference on Neural
# Information Processing Systems (NeurIPS 2025).", "Proceedings of …", "Preprint. Under
# review.", "Published as a conference paper at ICLR") — never a claim.
_VENUE_HEADER_RE = re.compile(
    r"^(?:\d+(?:st|nd|rd|th)\s+(?:Conference|Workshop|Annual|International)\b"
    r"|(?:Proceedings|Advances)\s+(?:of|in)\b"
    r"|Published\s+as\s+a\s+(?:conference|workshop)\s+paper\b"
    r"|(?:Preprint|Under\s+review|Work\s+in\s+progress)\b"
    r"|To\s+appear\s+in\b)",
    re.IGNORECASE,
)
# CJK ideographs/kana. In an English-language paper a body line carrying CJK is an
# author-name gloss / affiliation / contribution note ("… Yang Yue (乐阳) …"), not a
# prose claim — drop it so it can't fuse into a page-1 claim.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


_CJK_NAME_GLOSS_RE = re.compile(r"[A-Za-z]\)?\s*[（(]\s*[぀-ヿ㐀-䶿一-鿿]")


def _is_cjk_boilerplate(line: str) -> bool:
    """A CJK-bearing line that is author-block boilerplate, not a prose claim.

    Conservative on purpose: an English-paper author note glosses names in CJK
    ("… Yang Yue (乐阳) …") or is written predominantly in CJK (an affiliation), but a
    multilingual/NLP paper may carry a genuine claim quoting a Chinese example or
    dataset. So drop a CJK line ONLY when it has a parenthesized CJK name gloss after
    a Latin token, or is mostly CJK — never a Latin-majority sentence.
    """
    if not _CJK_RE.search(line):
        return False
    if _CJK_NAME_GLOSS_RE.search(line):
        return True
    cjk = len(_CJK_RE.findall(line))
    latin = sum(1 for c in line if c.isascii() and c.isalpha())
    return cjk >= max(1, latin)  # predominantly CJK → affiliation/note, not English prose


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


def _valid_marker_key(key: str, references: dict[str, CitedAs]) -> bool:
    """A numeric ``[n]`` marker is a real citation only if it resolves to a ref entry.

    ``ref-0`` is never valid (bibliographies are 1-indexed). When a reference list
    was parsed, the key must be in it — this drops math/interval false positives
    like ``Si ∈ [0, 100]`` (→ ref-0/ref-100) that aren't real markers. With no
    parsed list we can't validate, so we keep the marker (best-effort floor).
    """
    if key == "ref-0":
        return False
    return key in references if references else True


def _has_numbered_citations(claim_body: str, references: dict[str, CitedAs]) -> bool:
    """True when the body's ``[n]`` markers densely resolve to the reference list.

    This is the citation-style discriminator: a numbered-citation paper has many
    ``[n]`` that map to real entries; an author-year paper only has stray ``[n]``
    (math intervals, footnotes) that don't. Gating the numeric path on this keeps a
    lone ``[0, 100]`` from short-circuiting a hyperlink/author-year paper (its 100+
    ``cite.*`` sites would otherwise be dropped). Needs ≥3 markers, ≥60% resolving.
    """
    if not references:
        return False
    total = hits = 0
    for mm in _INTEXT_NUM_RE.finditer(claim_body):
        keys = [k for k in _expand_num_marker(mm.group(1)) if k != "ref-0"]
        if not keys:
            continue
        total += 1
        if all(k in references for k in keys):
            hits += 1
    return total >= 3 and hits >= 0.6 * total


def _claim_scan_body(text: str) -> str:
    """Drop PDF header/footer/contact/affiliation lines before claim windowing."""
    lines = text.splitlines()
    keep: list[str] = []
    in_affiliation = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            in_affiliation = False
            keep.append(line)
            continue

        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _EMAIL_RE.search(stripped) or _EMAIL_RE.search(next_line):
            continue
        if _FOOTER_RE.match(stripped):
            continue
        if _SECTION_HEADING_RE.match(stripped) or _FOOTNOTE_RE.match(stripped):
            continue
        if _VENUE_HEADER_RE.match(stripped) or _is_cjk_boilerplate(stripped):
            continue
        if _AFFILIATION_START_RE.match(stripped):
            in_affiliation = True
            continue
        if in_affiliation:
            if _AFFILIATION_HINT_RE.search(stripped) or re.fullmatch(r"[A-Z][A-Za-z .,&()-]+", stripped):
                continue
            in_affiliation = False
        if _AFFILIATION_HINT_RE.search(stripped) and len(stripped) < 90:
            continue
        keep.append(line)
    return "\n".join(keep)


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
    raw = chunk[s_start:s_end]
    rel_in_sentence = max(0, rel - s_start)
    if len(raw) > _MAX_CLAIM_CHARS:
        half = _MAX_CLAIM_CHARS // 2
        cut_start = max(0, rel_in_sentence - half)
        cut_end = min(len(raw), rel_in_sentence + half)
        if cut_start:
            ws = raw.find(" ", cut_start)
            cut_start = ws + 1 if ws != -1 and ws < cut_end else cut_start
        if cut_end < len(raw):
            ws = raw.rfind(" ", cut_start, cut_end)
            cut_end = ws if ws != -1 and ws > cut_start else cut_end
        raw = raw[cut_start:cut_end]
        s_start += cut_start
        s_end = s_start + len(raw)
    sentence = re.sub(r"\s+", " ", raw).strip()
    return sentence, (lo + s_start, lo + s_end)


# ───────────────────────────────────────────────────────────────
# Author-year citation binding (hyperlink / text → reference)
# ───────────────────────────────────────────────────────────────
# A BibTeX-style key: a leading surname, an optional separator, a 4-digit year,
# then an optional suffix/keyword. Tolerates CamelCase and separators:
# "yang2023leandojo", "Zhao_2024", "MartinLofTypeTheory1998", "kaplan2020scaling".
_BIBKEY_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9]*?)(?:[-_ ]?et[-_ ]?al\d*)?[_:.\- ]?((?:19|20)\d{2})([a-z]?)(.*)$"
)


def _fold(s: str) -> str:
    """Lowercase + strip diacritics, for robust surname comparison."""
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def _ay_surnames(author: str) -> list[str]:
    """Candidate first-author surnames (folded), across the author shapes a bibkey
    might key on — the binder matches against ANY candidate:

    * author-year comma — ``'Martin-Lof, P.'`` -> ``['martinlof']`` (before the comma);
    * Vancouver — ``'Yang Z'`` / ``'Devlin J'`` -> first token (last is initials);
    * given-name-first — ``'Philipp Brauner'`` -> last token, but an org name
      (``'Qwen Team'``) keys on the first — so both ends are returned for either.

    Vancouver is told apart by a last token that is an initials group (≤2 letters,
    all uppercase like ``Z``/``ZQ``); a real short surname (``He``/``Wu``) carries a
    lowercase letter and so is kept.
    """
    a = (author or "").strip()
    if "," in a:
        cands = [a.split(",")[0]]
    else:
        toks = a.split()
        if len(toks) <= 1:
            cands = toks or [a]
        else:
            last = re.sub(r"[^A-Za-z]", "", toks[-1])
            cands = [toks[0]] if (1 <= len(last) <= 2 and last.isupper()) else [toks[-1], toks[0]]
    out: list[str] = []
    for c in cands:
        # Fold (NFKD + drop diacritics) BEFORE removing non-letters, so "Konrád"
        # folds to "konrad" — stripping first would delete the accented "á" → "konrd".
        f = re.sub(r"[^A-Za-z]", "", _fold(c))
        if f and f not in out:
            out.append(f)
        # Hyphenated or apostrophe surname: also offer the left part, since bibkeys
        # often key on it ("Zamfirescu-Pereira" -> "zamfirescu" for key
        # "zamfirescu2023johnny"; "Murakhovs’ka" -> "murakhovs" for "murakhovs2023…").
        if split := re.split(r"[-'’]", c, maxsplit=1):
            left = re.sub(r"[^A-Za-z]", "", _fold(split[0]))
            if len(split) > 1 and len(left) >= 4 and left not in out:
                out.append(left)
    return out


def _surname_matches(key_surname: str, ref_surname: str) -> bool:
    """Surnames equal, or a CamelCase key prepends the surname to title words
    ('martinloftypetheory' starts with the ref surname 'martinlof')."""
    if not key_surname or not ref_surname:
        return False
    return key_surname == ref_surname or (
        len(ref_surname) >= 5 and key_surname.startswith(ref_surname)
    )


def _bind_author_year(cite_key: str, references: dict[str, CitedAs]) -> CitedAs | None:
    """Match a bibkey ('yang2023leandojo') to a parsed reference by surname + year.

    Returns the matching :class:`CitedAs`, or ``None`` when there is no confident
    match (no surname/year match, or an unresolved tie between same-author/year
    entries — we never guess). The cite_key's trailing keyword breaks ties.
    """
    m = _BIBKEY_RE.match(cite_key)
    if m:
        # strip any digits from the surname token ("qwen3" -> "qwen") for matching
        surname = _fold(re.sub(r"[^A-Za-z]", "", m.group(1)))
        # group(3) is the a/b/c disambiguator letter; group(4) the keyword tail —
        # keep them apart (the old code glued "b" onto the keyword).
        year, suffix, kw = int(m.group(2)), m.group(3).lower(), m.group(4).lower()
        by_surname = [
            c for c in references.values()
            if c.authors and any(_surname_matches(surname, rs) for rs in _ay_surnames(c.authors[0]))
        ]
        if len(by_surname) == 1:
            return by_surname[0]  # a unique surname binds (cited year may differ slightly)
        if by_surname:
            # Author-year citations carry the EXACT year, so disambiguate among
            # same-surname refs by exact year FIRST. The ±1 tolerance (for arXiv-vs-
            # published drift) is a last resort below — applied up front it wrongly
            # ties "Cai 2024" with the neighbouring "Cai 2023"/"Cai 2025".
            exact = [c for c in by_surname if c.year == year]
            if len(exact) == 1:
                return exact[0]
            # Same surname AND same year: the cite-key's a/b/c suffix is an ordinal in
            # the reference list's document order ("2022a" -> 1st, "2022b" -> 2nd).
            # Only when there's no keyword tail, so a real keyword ("…best") isn't read
            # as the letter "b".
            if len(exact) > 1 and suffix and not kw:
                idx = ord(suffix) - ord("a")
                if 0 <= idx < len(exact):
                    return exact[idx]
            # The bibkey keyword as a WHOLE title word unique within the pool ("you" ->
            # "Are you sure?…"). Word-boundary + uniqueness avoids substring false hits.
            pool = exact or by_surname
            if len(kw) >= 3:
                hits = [c for c in pool if kw in re.findall(r"[a-z0-9]+", (c.title or "").lower())]
                if len(hits) == 1:
                    return hits[0]
            if len(kw) >= 4:  # a CamelCase keyword ("BeyondPL") as a unique title prefix
                hits = [c for c in pool if kw[:5] in re.sub(r"[^a-z0-9]", "", (c.title or "").lower())]
                if len(hits) == 1:
                    return hits[0]
            # Last resort: a UNIQUE ±1-year match (publication-year drift).
            near = [c for c in by_surname if c.year and abs(c.year - year) <= 1]
            if len(near) == 1:
                return near[0]
            return None  # a genuine same-surname ambiguity — never guess
    # No surname match (incl. keyless keys like "2024bfcl" and title-name keys like
    # "brokenmath2025"): bind by a UNIQUE title acronym (key initials == the title's
    # leading initials) or a distinctive (>=4-char) title word. The candidate token is
    # the bibkey's keyword tail, then its leading slot (a title word, not an author, for
    # short-name keys). Only when exactly one reference matches — else stay unresolved.
    cand_tokens = [m.group(4), m.group(1)] if m else [cite_key]
    for raw_tok in cand_tokens:
        token = _fold(re.sub(r"[^A-Za-z]", "", raw_tok or ""))
        if len(token) < 3:
            continue
        hits = []
        for c in references.values():
            words = re.findall(r"[a-z0-9]+", (c.title or "").lower())
            initials = "".join(w[0] for w in words)
            if token == initials[: len(token)] or (len(token) >= 4 and token in words):
                hits.append(c)
        if len(hits) == 1:
            return hits[0]
    # Fallback for non-standard keys (e.g. DBLP ".../CoquandH88"): a long, distinctive
    # surname embedded in the key and unique in the reference list.
    folded = _fold(re.sub(r"[^A-Za-z]", "", cite_key))
    embedded = [
        c for c in references.values()
        if c.authors and any(len(rs) >= 6 and rs in folded for rs in _ay_surnames(c.authors[0]))
    ]
    return embedded[0] if len(embedded) == 1 else None


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
        claim_body = _claim_scan_body(body)
        # Primary: layout-aware parse straight from the PDF geometry. Fall back to
        # the flat-text reference block only when PyMuPDF can't run (sidecar/.txt).
        references = _references_from_layout(source.pdf_path) or parse_reference_block(ref_block)
        paper = self._build_paper(source)
        records: list[CitationRecord] = []
        seen: set[tuple[str, str, str]] = set()

        # Only mine [n] markers when this is actually a numbered-citation paper.
        # Otherwise a stray math interval (e.g. "Si ∈ [0, 100]") would be read as a
        # citation and suppress the author-year/hyperlink path entirely.
        numbered = _has_numbered_citations(claim_body, references)
        for mm in _INTEXT_NUM_RE.finditer(claim_body) if numbered else ():
            cite_keys = [k for k in _expand_num_marker(mm.group(1)) if _valid_marker_key(k, references)]
            if not cite_keys:
                continue
            sentence, span = _sentence_around(claim_body, mm.start())
            in_table = bool(_TABLE_CAPTION_RE.match(sentence)) or _looks_like_table_dump(sentence)
            for cite_key in cite_keys:
                claim_id = make_claim_id(source.paper_id, None, span, cite_key)
                dedup = (source.paper_id, claim_id, cite_key)
                if dedup in seen:
                    continue
                seen.add(dedup)
                cited_as = references.get(cite_key) or CitedAs(raw="")
                notes = None if cite_key in references else "no reference-list entry for marker"
                if in_table:
                    notes = f"{notes}; {TABLE_NOTE}" if notes else TABLE_NOTE
                records.append(
                    CitationRecord(
                        paper_id=source.paper_id,
                        claim_id=claim_id,
                        cite_key=cite_key,
                        paper=paper,
                        claim=Claim(claim_id=claim_id, text=sentence, section=None, char_span=span),
                        cited_as=cited_as,
                        in_table=in_table,
                        notes=notes,
                    )
                )

        # Author-year citations (no [n] markers) — recover sites from cite.* link
        # anchors (hyperref PDFs), else from author-year text patterns, and pair
        # each with its surrounding sentence + the surname/year-matched reference.
        if not records:
            sites = extract_link_citations(source.pdf_path) or extract_text_citations(claim_body)
            for site in sites:
                cite_key = site["cite_key"]
                span = tuple(site["span"])
                claim_id = make_claim_id(source.paper_id, None, span, cite_key)
                dedup = (source.paper_id, claim_id, cite_key)
                if dedup in seen:
                    continue
                seen.add(dedup)
                cited_as = _bind_author_year(cite_key, references) or CitedAs(raw="")
                matched = bool(cited_as.title or cited_as.authors)
                records.append(
                    CitationRecord(
                        paper_id=source.paper_id,
                        claim_id=claim_id,
                        cite_key=cite_key,
                        paper=paper,
                        claim=Claim(
                            claim_id=claim_id, text=site["claim"], section=None, char_span=span
                        ),
                        cited_as=cited_as,
                        notes=None
                        if matched
                        else "citation hyperlink not matched to a bibliography entry",
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
