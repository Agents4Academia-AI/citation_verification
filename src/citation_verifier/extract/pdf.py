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

    try:
        from pypdf import PdfReader  # lazy: optional dependency
    except Exception:
        return _read_sidecar(path)

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        return _normalize_pdf_text(text) if text else _read_sidecar(path)
    except Exception:
        return _read_sidecar(path)


# Line-break hyphenation: "knowl- edge" / "Lan-\nguage" -> "knowledge" / "Language".
# Only join when a lowercase letter follows (real compounds like "state-of-the-art"
# or "GPT-3" have no following lowercase-after-whitespace and are preserved).
_DEHYPHEN_RE = re.compile(r"(?<=[A-Za-z])-\s+(?=[a-z])")
# Restore a space after a separator inside a de-spaced run ("11.YangZ" -> "11. YangZ").
_SEP_RESPACE_RE = re.compile(r"([.,;])(?=[A-Za-z0-9])")


def _normalize_pdf_text(text: str) -> str:
    """Repair two common pypdf artifacts that wreck downstream grounding.

    1. **De-hyphenation** — pypdf keeps the hyphen of a word split across a line
       break ("Lan- guage"), so cited titles never match their canonical form.
    2. **Character-spacing** — some lines come out with a space between every
       glyph ("1 1 .Y a n gZ ,G a nZ"), which hides the reference number from the
       entry parser and garbles the authors. Collapsed per line (only on lines
       that are clearly char-spaced), so normal prose is untouched.
    """
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


def _parse_ref_entry(body: str) -> CitedAs:
    """Best-effort structured fields from a single PDF reference line."""
    arxiv = _ARXIV_RE.search(body)
    doi = _DOI_RE.search(body)
    url = _URL_RE.search(body)
    year = _YEAR_RE.search(body)
    # Authors heuristic: text up to the first year or first period-group.
    authors: list[str] = []
    head = body.split(".")[0] if "." in body else ""
    if head and len(head) < 200 and ("," in head or " and " in head):
        authors = _split_authors(head.replace(",", " and "))
    return CitedAs(
        raw=body,
        authors=authors,
        title=None,
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
