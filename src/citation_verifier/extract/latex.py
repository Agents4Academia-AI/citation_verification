"""
extract/latex.py — the primary (preferred) ingestion path.

Given a :class:`~citation_verifier.interfaces.PaperSource` that points at an
extracted arXiv LaTeX e-print (``tex_dir`` + ``bbl_path`` / ``bib_path``), build
:class:`~citation_verifier.schema.CitationRecord` *stubs*: one per
``(claim_site, cite_key)`` pair, with the join key, the surrounding claim
sentence/section, and the ``cited_as`` metadata filled in. The judged axes
(``exists`` / ``supports_claim`` / ``priority`` / ``severity``) are deliberately
left at their schema defaults (``unresolved`` / ``helpful`` / ``ok``) — they are
filled later by the grounding/relevance stages or a backend.

Why LaTeX is preferred over PDF (see docs/decisions-phy.md):
  * ``.bib`` / ``.bbl`` give *exact* reference metadata (no OCR/column noise);
  * ``\\cite``/``\\citep``/``\\citet`` call-sites anchor each citation to the
    precise sentence and section that makes the claim — the eval join key.

The e-print itself is materialized HERE (the task's "download the arXiv e-print
tarball"): ``ingest`` only cheaply probes ``tex_available``; if a LaTeX source is
expected but not yet on disk, :meth:`LatexExtractor.extract` downloads and
extracts the e-print tarball into ``source.work_dir/tex`` and fills the LaTeX
paths on the source. All networking is stdlib ``urllib`` and FAILS SOFT (a
download/extract failure degrades to fewer/zero records, never an exception).

This module is import-safe with no network and no heavy deps at import time:
  * networking happens only inside :meth:`LatexExtractor.extract` (never at
    import);
  * ``bibtexparser`` is imported lazily inside the ``.bib`` parser only;
  * a tolerant, dependency-free ``.bbl`` parser handles the common
    ``\\bibitem`` formats (``natbib`` and plain) when no ``.bib`` is present.

Nothing here performs I/O at import; everything degrades softly (a missing or
malformed file yields fewer records, never an exception that aborts the run).
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

from ..interfaces import PaperSource
from ..schema import CitationRecord, CitedAs, Claim, Paper

__all__ = [
    "LatexExtractor", "make_claim_id", "load_references", "compiled_cite_keys",
    "cite_sites", "included_sources", "split_author_block", "decode_tex_accents",
    "publication_year",
]

_USER_AGENT = "citation-verifier/0.1 (+https://github.com/Agents4Academia/citation_verification)"
_TIMEOUT = 60


# ───────────────────────────────────────────────────────────────
# Deterministic claim-id (stable across runs)
# ───────────────────────────────────────────────────────────────
def make_claim_id(
    paper_id: str,
    section: str | None,
    char_span: tuple[int, int] | None,
    cite_key: str,
) -> str:
    """Return a deterministic, stable id for a single claim site.

    The id is a short blake2b digest over the fields that uniquely locate a
    claim site within a paper. It is stable across runs (no randomness, no
    timestamps) so that agent output and CitationHallucinationBench gold join on
    the same ``(paper_id, claim_id, cite_key)`` key.

    Args:
        paper_id: The citing paper's stable id.
        section: The section/heading the claim appears in (``None`` tolerated).
        char_span: ``(start, end)`` offsets of the claim in the concatenated
            source text, if known. Included in the digest so two citations of
            the same key in different places get distinct ids.
        cite_key: The BibTeX / ``\\cite`` key.

    Returns:
        A string of the form ``"claim-<12 hex chars>"``.
    """
    span_part = f"{char_span[0]}:{char_span[1]}" if char_span else "?"
    payload = "␟".join((paper_id, section or "", span_part, cite_key))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()
    return f"claim-{digest}"


# ───────────────────────────────────────────────────────────────
# Reference parsing — .bib (bibtexparser) and .bbl (tolerant)
# ───────────────────────────────────────────────────────────────
def _split_authors(raw: str | None) -> list[str]:
    """Split a BibTeX/bbl author string ("A and B and C") into a clean list."""
    if not raw:
        return []
    raw = re.sub(r"[{}]", "", raw.replace("~", " "))  # LaTeX tie ~ -> space
    parts = re.split(r"\s+and\s+", raw)
    return [p.strip().strip(",").strip() for p in parts if p.strip()]


def _coerce_year(raw: Any) -> int | None:
    """Pull a 4-digit year out of a noisy field; ``None`` if absent."""
    if raw is None:
        return None
    m = re.search(r"(19|20)\d{2}", str(raw))
    return int(m.group(0)) if m else None


def parse_bib(bib_path: str | Path) -> dict[str, CitedAs]:
    """Parse a ``.bib`` file into ``{cite_key: CitedAs}`` (bibtexparser, lazy).

    Fails soft: a missing file or any parse error yields an empty mapping rather
    than raising, so the extractor can fall back to ``.bbl`` or to call-sites
    alone.
    """
    path = Path(bib_path)
    if not path.is_file():
        return {}
    try:
        import bibtexparser  # lazy: only needed on the .bib path
        from bibtexparser.bparser import BibTexParser
    except Exception:
        return {}
    try:
        parser = BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        db = bibtexparser.loads(path.read_text(encoding="utf-8", errors="replace"), parser=parser)
    except Exception:
        return {}

    out: dict[str, CitedAs] = {}
    for entry in getattr(db, "entries", []):
        key = (entry.get("ID") or "").strip()
        if not key:
            continue
        # Accents are decoded in every field, not only in the rendered string: the
        # structured fields are what the metadata comparison reads, and an undecoded
        # ``Schr\"odinger`` or ``M\"uller`` never equals the canonical "Schrödinger" /
        # "Müller" — every such reference was reported as a title or author mismatch.
        title = _bib_text(entry.get("title"))
        venue = _bib_text(
            entry.get("journal") or entry.get("booktitle") or entry.get("publisher")
        )
        out[key] = CitedAs(
            raw=_render_bib_raw(entry),
            authors=_split_authors(decode_tex_accents(entry.get("author") or "")),
            title=title,
            year=_coerce_year(entry.get("year")),
            venue=venue,
            doi=_bib_doi(entry),
            arxiv_id=_bib_arxiv_id(entry),
            url=(entry.get("url") or None),
        )
    return out


def _bib_text(value: str | None) -> str | None:
    """A ``.bib`` field as plain text: accents decoded, braces and runs of space gone."""
    if not value:
        return None
    text = re.sub(r"[{}]", "", decode_tex_accents(value))
    return re.sub(r"\s+", " ", text).strip() or None


# Where a ``.bib`` entry says which arXiv paper it is. ``eprint`` is the dedicated field,
# but the entries people actually paste (Google Scholar, DBLP exports) put the id in the
# journal — ``journal = {arXiv preprint arXiv:1706.03762}`` — or only in the URL.
_BIB_ARXIV_FIELDS = ("eprint", "arxiv", "journal", "note", "howpublished", "url", "volume", "number")
_BIB_ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:\s/]*|abs/|pdf/)?\b(\d{4}\.\d{4,5})(?:v\d+)?\b|\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})\b",
    re.IGNORECASE,
)


def _bib_arxiv_id(entry: dict[str, str]) -> str | None:
    """The arXiv id a ``.bib`` entry carries, wherever it put it.

    The dedicated ``eprint`` field is trusted as given. Any other field must also say
    "arxiv" for a ``NNNN.NNNNN`` number in it to count — a DOI suffix or a report number
    has the same shape.
    """
    for field in _BIB_ARXIV_FIELDS:
        value = (entry.get(field) or "").strip()
        if not value:
            continue
        if field not in ("eprint", "arxiv") and "arxiv" not in value.lower():
            continue
        if field == "eprint":
            kind = (entry.get("archiveprefix") or entry.get("eprinttype") or "arxiv").lower()
            if "arxiv" not in kind:
                continue
        m = _BIB_ARXIV_ID_RE.search(value)
        if m:
            return m.group(1) or m.group(2)
    return None


def _bib_doi(entry: dict[str, str]) -> str | None:
    """The DOI of a ``.bib`` entry, bare — ``https://doi.org/10.…`` is a URL, not a DOI."""
    for field in ("doi", "url"):
        m = _DOI_IN_TEXT.search(entry.get(field) or "")
        if m:
            return m.group(0).rstrip(".,;")
    return None


def _render_bib_raw(entry: dict[str, str]) -> str:
    """Build a compact human-readable reference string from a bib entry.

    Accent commands are decoded first: a surviving ``Roth\\"orl`` splits the reference at
    the backslash, and every field extracted downstream shifts — one nine-author entry had
    "orl, Thomas, Hadsell, Raia, …" taken as its title, so the paper never resolved even
    though it was sitting in the arXiv results.
    """
    authors = ", ".join(_split_authors(decode_tex_accents(entry.get("author") or ""))) or "?"
    title = re.sub(r"[{}]", "", decode_tex_accents(entry.get("title", ""))).strip() or "?"
    year = entry.get("year", "")
    venue = re.sub(
        r"[{}]", "", decode_tex_accents(entry.get("journal") or entry.get("booktitle") or "")
    ).strip()
    bits = [authors, f'"{title}"']
    if venue:
        bits.append(venue)
    if year:
        bits.append(str(year))
    return ". ".join(bits)


# .bbl: \bibitem[label]{key} <free text>  (natbib) OR \bibitem{key} <free text>
_BIBITEM_RE = re.compile(
    r"\\bibitem\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}",
    re.DOTALL,
)
_ARXIV_IN_TEXT = re.compile(
    # "arXiv:1706.03762", "arxiv.org/abs/…", and the DBLP / Semantic-Scholar BibTeX form
    # "ArXiv, abs/1706.04599" / "CoRR, abs/1412.6980".
    r"(?:arXiv:\s*|arxiv\.org/(?:abs|pdf)/|\b(?:arXiv|CoRR)\s*[,.]?\s*(?:preprint\s*)?abs/)"
    r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7})",
    re.IGNORECASE,
)
_DOI_IN_TEXT = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_URL_IN_TEXT = re.compile(r"https?://[^\s{}]+")
_YEAR_IN_TEXT = re.compile(r"\((19|20)\d{2}[a-z]?\)|\b(19|20)\d{2}\b")


# cite commands and their braced key argument, removed wholesale from prose.
_CITE_CMD_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|citenum|"
    r"parencite|textcite|autocite|footcite)\*?\s*(?:\[[^\]]*\]\s*){0,2}\{[^}]*\}"
)
# section/heading commands and their title argument, removed from prose.
_SECTION_CMD_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|chapter|paragraph)\*?\s*\{[^}]*\}"
)


# LaTeX accent commands -> the Unicode combining mark they apply. Composed with the base
# letter and normalised, so `Roth\"orl` reads as "Rothörl" rather than surviving as
# `Roth\"orl`. Left undecoded, the backslash splits the reference mid-name and everything
# downstream shifts: the title extractor took "orl, Thomas, Hadsell, Raia, …" as the
# title, and the paper — which was sitting in the arXiv results — never resolved.
# European names in bibliographies make this routine, not exotic.
_ACCENT_MARKS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303",
    "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030C", "H": "\u030B",
    "r": "\u030A", "c": "\u0327", "k": "\u0328", "d": "\u0323", "b": "\u0331",
}
# Letters that are their own command rather than a base plus a mark.
_TEX_LETTERS = {
    "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "ss": "ß", "aa": "å", "AA": "Å",
    "ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "i": "i", "j": "j",
}
_ACCENT_RE = re.compile(
    r"\\([\"'`^~=.]|(?:u|v|H|r|c|k|d|b)(?=[{\s]))\s*(?:\{\s*([A-Za-z])\s*\}|([A-Za-z]))"
)
_TEX_LETTER_RE = re.compile(r"\\(ss|aa|AA|ae|AE|oe|OE|[oOlLij])(?![A-Za-z])\s*\{?\}?")


def decode_tex_accents(text: str) -> str:
    r"""Turn LaTeX accent commands into the characters they denote.

    ``\"o`` and ``\"{o}`` both become ``ö``; ``\ss`` becomes ``ß``. Any base letter works
    because the accent is applied as a combining mark and then normalised, so no table of
    letter-accent pairs is needed.
    """
    if not text or "\\" not in text:
        return text

    def _mark(m: re.Match[str]) -> str:
        base = m.group(2) or m.group(3) or ""
        return unicodedata.normalize("NFC", base + _ACCENT_MARKS.get(m.group(1), ""))

    text = _ACCENT_RE.sub(_mark, text)
    return _TEX_LETTER_RE.sub(lambda m: _TEX_LETTERS.get(m.group(1), m.group(1)), text)


def _strip_tex(text: str) -> str:
    """Crudely de-TeX a reference body / claim sentence into readable prose.

    Drops cite-commands and section-headings (argument included), unwraps common
    text-formatting commands keeping their content, then removes any remaining
    bare commands and brace/format noise.
    """
    # Before any command-dropping: an accent is a command whose argument is part of a
    # word, and removing it silently corrupts the name it belongs to.
    text = decode_tex_accents(text)
    text = _CITE_CMD_RE.sub(" ", text)
    text = _SECTION_CMD_RE.sub(" ", text)
    text = re.sub(r"\\newblock", " ", text)
    text = re.sub(r"\\(emph|textit|textbf|texttt|url|href)\s*\{([^}]*)\}", r"\2", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)  # drop remaining commands
    text = text.replace("~", " ")  # LaTeX tie ~ is a non-breaking space, NOT a deletion
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,")


def _is_initials_only(segment: str) -> bool:
    """True for a segment that is nothing but initials: "A.", "N. M.", "J.-P.", "D P".

    Every token must be a SINGLE letter. Allowing an optional period on a run of letters
    instead read the short surnames a bibliography is full of — Ren, Sun, He, Wu, Li — as
    initials of the name before them, and "X. Zhang, S. Ren, J. Sun" became one author.
    """
    tokens = [t for t in re.split(r"[\s\-]+", segment.strip()) if t]
    return 1 <= len(tokens) <= 4 and all(
        len(t.rstrip(".")) == 1 and t[0].isalpha() for t in tokens
    )


# What separates names in a formatted author list: ", and", ",", "and", "&", ";".
_AUTHOR_SEP_RE = re.compile(r"\s*[,;]\s*(?:(?:and|&)\s+)?|\s+(?:and|&)\s+")
_AUTHOR_TAIL_RE = re.compile(
    r"[\s,;.]*(?:\(?(?:19|20)\d{2}[a-z]?\)?|\bet\s+al\b\.?|\band\s+others\b|\(eds?\.?\)|\beds?\b\.?)[\s,;.]*$",
    re.IGNORECASE,
)
# IEEE-style entries have no \\newblock: ``A. Author and B. Author, ``Title,'' in …``.
_TEX_QUOTED_TITLE_RE = re.compile(r"``(.{8,}?)(?:,?\s*''|,?\s*\")", re.DOTALL)


def split_author_block(block: str) -> list[str]:
    """Names out of the author block a bibliography style printed.

    A ``.bst`` formats the author list one of three ways, and all three are separated by
    the same punctuation, so the shape has to be read off the segments:

    * given-first — ``Ashish Vaswani, Noam Shazeer, and Niki Parmar``;
    * initials-first — ``A. Vaswani, N. Shazeer, and N. Parmar``;
    * surname-first — ``Vaswani, A., Shazeer, N., and Parmar, N.``: here every second
      segment is initials only, and is re-attached to the surname before it.

    Returns ``[]`` rather than a guess when the block does not look like names at all
    (a corporate author with commas, a title that leaked in): a wrong author list is
    reported downstream as an author MISMATCH, which is an accusation, while an empty
    one only means the author check is skipped.
    """
    text = re.sub(r"\s+", " ", block or "").strip()
    for _ in range(3):  # "…, et al. (2017)." carries more than one trailer
        text = _AUTHOR_TAIL_RE.sub("", text).strip(" .,;")
    if not text:
        return []
    parts = [seg.strip() for seg in _AUTHOR_SEP_RE.split(text) if seg and seg.strip()]
    names: list[str] = []
    for seg in parts:
        if _is_initials_only(seg) and names and "," not in names[-1]:
            names[-1] = f"{names[-1]}, {seg}"  # "Vaswani" + "A." -> "Vaswani, A."
        else:
            names.append(seg)
    ok = all(
        1 <= len(n.split()) <= 6 and len(n) <= 60 and not re.search(r"\d", n)
        and re.search(r"[^\W\d_]{2,}", n)
        for n in names
    )
    return names if ok else []


def _parse_bbl_body(body: str) -> CitedAs:
    """Best-effort structured fields from one ``\\bibitem`` free-text body.

    The common ``.bst`` layout is ``Authors. \\newblock Title. \\newblock Venue, year.``;
    IEEE-style entries run everything together and quote the title instead.
    """
    raw = _strip_tex(body)
    arxiv = _ARXIV_IN_TEXT.search(raw)
    doi = _DOI_IN_TEXT.search(raw)
    url = _URL_IN_TEXT.search(body)
    title: str | None = None
    author_block = ""
    blocks = re.split(r"\\newblock", body)
    quoted = _TEX_QUOTED_TITLE_RE.search(blocks[0])
    if quoted:
        title = _strip_tex(quoted.group(1)) or None
        author_block = blocks[0][: quoted.start()]
    elif len(blocks) >= 2:
        cand = _strip_tex(blocks[1])
        if len(cand) > 8:
            title = cand
        author_block = blocks[0]
    return CitedAs(
        raw=raw,
        # The author block is the list as PRINTED, which is what a metadata check must
        # compare. Left empty (as it was), no .bbl-sourced reference could ever be
        # checked for a wrong or invented author — and most arXiv sources ship a .bbl.
        authors=split_author_block(_strip_tex(author_block)),
        title=title,
        year=publication_year(raw, title),
        venue=None,
        doi=doi.group(0).rstrip(".,;") if doi else None,
        arxiv_id=arxiv.group(1) if arxiv else None,
        url=url.group(0) if url else None,
    )


# A year printed straight after an address, the space lost: "…/code-r1,2025." (PDF text).
_YEAR_GLUED_TO_URL_RE = re.compile(r"[,;]((?:19|20)\d{2})[.,;)\]]*$")


def _strip_ids(text: str) -> str:
    """``text`` without arXiv ids, DOIs and URLs — their digits are not years."""
    text = _ARXIV_IN_TEXT.sub(" ", text or "")
    text = _DOI_IN_TEXT.sub(" ", text)

    def without_url(m: re.Match[str]) -> str:
        # The address runs to the next space, and so swallows a year glued to it. That
        # year is the reference's own, and for a repository or a blog post the only one.
        glued = _YEAR_GLUED_TO_URL_RE.search(m.group(0))
        return f" {glued.group(1)} " if glued else " "

    return _URL_IN_TEXT.sub(without_url, text)


# Digits in a reference that look like a year and are not one.
_BARE_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")  # "abs/2004.12345"
_PAGE_RANGE_RE = re.compile(
    r"\b(?:pp?\.?|pages?)\s*\d+(?:\s*[-–—]+\s*\d+)?|\b\d+\s*[-–—]+\s*\d+\b|\b\d+\s*\(\d+\)\s*:\s*\d+"
)
_ACCESS_NOTE_RE = re.compile(
    r"\b(?:last\s+)?(?:accessed|retrieved|visited|viewed)\b.*$", re.IGNORECASE | re.DOTALL
)
_PLAIN_YEAR_RE = re.compile(r"(?<![\d.])((?:19|20)\d{2})(?!\d|s\b)")  # not "the 2020s"


def publication_year(reference: str, title: str | None = None) -> int | None:
    """The year a reference says it was PUBLISHED, from its printed text.

    The first four-digit number is the wrong rule: titles carry years ("A ConvNet for the
    2020s", "ImageNet … Challenge 2012", "Lessons from the 2008 crisis") and so do page
    ranges, arXiv ids and "accessed" notes, and a year read from any of those is then
    reported as a year MISMATCH against the true record. Every style prints the
    publication year after the title or beside the venue, so: drop the title and the
    look-alikes, and take the LAST year that remains.
    """
    whole = _strip_ids(reference or "")
    outside = whole
    if title:
        # Whitespace- and case-insensitive: a PDF line-wraps the title it printed.
        pat = r"\s+".join(re.escape(w) for w in title.split())
        outside = re.sub(pat, " ", whole, count=1, flags=re.IGNORECASE)
    # Outside the title first. Only when nothing is left there, the whole reference: a
    # reference with no venue after its title ("MNIST handwritten digit database (2010)",
    # "Langchain, October 2022. URL …") has nothing to mark where the title ends, the
    # title parse takes the year with it, and dropping the title then drops the year.
    for text in (outside, whole):
        text = _ACCESS_NOTE_RE.sub(" ", text)
        text = _BARE_ARXIV_ID_RE.sub(" ", text)
        text = _PAGE_RANGE_RE.sub(" ", text)
        years = [int(y) for y in _PLAIN_YEAR_RE.findall(text) if 1800 < int(y) <= _MAX_YEAR]
        if years:
            return years[-1]
    return None


_MAX_YEAR = 2100  # a sanity bound, not a calendar: "2020s" and page 2401 are caught above


def _parse_bibitems(text: str) -> dict[str, CitedAs]:
    """Parse every ``\\bibitem`` entry in ``text`` into ``{cite_key: CitedAs}``."""
    matches = list(_BIBITEM_RE.finditer(text))
    out: dict[str, CitedAs] = {}
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        if not key:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # Stop at \end{thebibliography} if present in the last block.
        body = re.split(r"\\end\{thebibliography\}", body)[0]
        out[key] = _parse_bbl_body(body)
    return out


def parse_bbl(bbl_path: str | Path) -> dict[str, CitedAs]:
    """Tolerant ``.bbl`` parser -> ``{cite_key: CitedAs}``.

    Handles the common ``\\bibitem[..]{key} ... \\bibitem ...`` layout produced by
    BibTeX/natbib. Fails soft (missing/garbled file -> ``{}``).
    """
    path = Path(bbl_path)
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    return _parse_bibitems(text)


# Inline bibliography pasted directly into a .tex file (no separate .bbl/.bib).
_THEBIB_RE = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)


def parse_inline_bib(tex_dir: str | Path | None) -> dict[str, CitedAs]:
    """Parse ``\\bibitem`` entries from inline ``thebibliography`` blocks in .tex.

    Many arXiv e-prints ship NO separate ``.bbl``/``.bib`` — the compiled
    bibliography is pasted into the main ``.tex`` as a ``thebibliography``
    environment (e.g. arXiv:1706.03762). Without this those references are
    invisible and every citation resolves to an empty ``cited_as``, so nothing
    grounds. Fails soft (no blocks -> ``{}``).
    """
    out: dict[str, CitedAs] = {}
    if not tex_dir:
        return out
    for _path, text in _read_tex_files(tex_dir):
        for block in _THEBIB_RE.findall(text):
            out.update(_parse_bibitems(block))
    return out


def load_references(source: PaperSource) -> dict[str, CitedAs]:
    """Load references from inline ``thebibliography`` + every ``.bbl`` + every ``.bib``.

    Inline (.tex) entries are the base; ``.bbl`` files override them, and the richer
    ``.bib`` files (structured authors/venue) override both.

    EVERY bibliography file under the source tree is read, not one of each. A paper
    routinely splits its references (``anthology.bib`` + ``custom.bib``, a ``refs/``
    directory per section) and ships more than one ``.bbl`` from different builds; reading
    only the alphabetically first left every key defined in the others reported as "no
    bibliography entry", i.e. a citation that could not be checked at all. A cite key is
    only ever a lookup into this map, so a stale extra file costs nothing. ``.bbl`` files
    are merged smallest first, so on a key both define the fuller build wins.
    """
    refs: dict[str, CitedAs] = {}
    refs.update(parse_inline_bib(source.tex_dir))
    bbls, bibs = _bibliography_files(source)
    parsed_bbls = sorted((parse_bbl(b) for b in bbls), key=len)
    for entries in parsed_bbls:
        refs.update(entries)
    for bib in bibs:
        # .bib is richer (structured authors/venue); let it override .bbl/inline.
        refs.update(parse_bib(bib))
    return refs


def compiled_cite_keys(source: PaperSource) -> set[str]:
    """The cite keys of the paper AS COMPILED — its ``.bbl`` / inline bibliography.

    A ``.bbl`` lists exactly the works the built document cites, which makes it the one
    ground truth in a source tree for "is this citation in the paper". Empty when the
    tree ships neither (a ``.bib``-only source), and then nothing can be concluded.
    """
    keys: set[str] = set(parse_inline_bib(source.tex_dir))
    for bbl in _bibliography_files(source)[0]:
        keys.update(parse_bbl(bbl))
    return keys


def _bibliography_files(source: PaperSource) -> tuple[list[Path], list[Path]]:
    """``(bbl_files, bib_files)`` for a source: everything under ``tex_dir``, plus the
    explicit ``bbl_path`` / ``bib_path`` when they sit outside it."""
    bbls: list[Path] = []
    bibs: list[Path] = []
    if source.tex_dir and Path(source.tex_dir).is_dir():
        bbls = sorted(Path(source.tex_dir).rglob("*.bbl"))
        bibs = sorted(Path(source.tex_dir).rglob("*.bib"))
    for explicit, bucket in ((source.bbl_path, bbls), (source.bib_path, bibs)):
        if explicit and Path(explicit) not in bucket:
            bucket.append(Path(explicit))
    return bbls, bibs


# ───────────────────────────────────────────────────────────────
# \cite call-site scanning
# ───────────────────────────────────────────────────────────────
# Any command whose NAME contains "cite", with optional star and [..][..] arguments, then
# {keys}. A closed list of spellings is what this replaces: it knew ``\\citep`` but not
# ``\\citep*``, ``\\Citet`` (sentence-initial natbib), ``\\citeyearpar``, ACL's
# ``\\newcite`` / ``\\shortcite``, biblatex's ``\\parencites`` / ``\\supercite`` /
# ``\\smartcite``, or the wrapper every third paper defines for itself (``\\tcite``,
# ``\\mycite``) — and a citation the scanner does not see is a citation never checked.
# The commands that contain "cite" but do not cite are few and listed below.
_CITE_RE = re.compile(
    r"\\([A-Za-z]*[cC]ite[A-Za-z]*)\*?\s*"
    r"(?:\[[^\]]*\]\s*){0,2}"
    r"\{([^{}]+)\}"
)
# A further ``[..]{keys}`` group — biblatex multicite: ``\\cites{a}{b}``, ``\\parencites[..]{a}[..]{b}``.
_MORE_CITE_GROUP_RE = re.compile(r"\s*(?:\[[^\]]*\]\s*){0,2}\{([^{}]+)\}")
_NOT_A_CITATION = frozenset(
    "nocite bibcite citation citestyle setcitestyle citenumfont citepunct citeform "
    "citeleft citeright citemid citedash citename citeindextrue citeindexfalse "
    "citeindextype declarecitecommand declaremulticitecommand declarecitealias "
    "defcitealias citereset citesetup citecolor citebordercolor recite".split()
)
# What a BibTeX key can look like. A "key" with a space, a backslash or a macro parameter
# is the argument of something else (``\\citeform{\\textbf{#1}}``, a macro definition).
_CITE_KEY_RE = re.compile(r"[^\s\\#{}$%&~^,]+")


def cite_sites(text: str) -> list[tuple[int, list[str]]]:
    """Every citation call-site in ``text`` as ``(offset, [cite_key, …])``, in order."""
    out: list[tuple[int, list[str]]] = []
    for cm in _CITE_RE.finditer(text):
        name = cm.group(1).lower()
        if name in _NOT_A_CITATION:
            continue
        groups = [cm.group(2)]
        if name.endswith("s"):  # multicite: every following brace group is more keys
            pos = cm.end()
            while (more := _MORE_CITE_GROUP_RE.match(text, pos)) is not None:
                groups.append(more.group(1))
                pos = more.end()
        keys = [
            k.strip()
            for group in groups
            for k in group.split(",")
            if k.strip() and _CITE_KEY_RE.fullmatch(k.strip())
        ]
        if keys:
            out.append((cm.start(), keys))
    return out
_SECTION_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|chapter|paragraph)\*?\s*\{([^}]*)\}"
)
# Table float environments. A \cite inside one is a table cell/caption, not a
# prose claim, so its relevance is not assessed (see CitationRecord.in_table).
_TABLE_ENV_RE = re.compile(
    r"\\begin\{(table\*?|tabular\*?|tabularx|longtable|wraptable|sidewaystable|threeparttable)\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)
# Set on a record (extraction) when its \cite sits inside a table; the relevance
# stage / backend leaves such records inconclusive and surfaces this note.
TABLE_NOTE = "cited inside a table — relevance not assessed"
# Sentence boundary: a period/!/? followed by whitespace + capital/backslash, applied to
# lightly-cleaned text — but never after an abbreviation. The backslash in the lookahead
# is what lets a sentence open with a command, and it is also what follows "et al." in
# the most common citing construction there is: ``Smith et al. \\cite{smith} propose …``
# was cut at "al.", so the claim attached to the citation read "propose …" with its
# subject — the very thing being attributed — left in the previous sentence.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])"
    r"(?<!\bal\.)(?<!e\.g\.)(?<!i\.e\.)(?<!\bcf\.)(?<!\bvs\.)(?<!etc\.)(?<!\bresp\.)"
    r"(?<!\bFig\.)(?<!\bfig\.)(?<!Figs\.)(?<!\bEq\.)(?<!\bEqs\.)(?<!\bSec\.)(?<!\bTab\.)"
    r"(?<!\bRef\.)(?<!Refs\.)(?<!\bAlg\.)(?<!\bApp\.)(?<!\bapprox\.)"
    r"\s+(?=[A-Z\\])"
)


def _table_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of table-float environments (a ``\\cite`` inside one is a table cell)."""
    return [(m.start(), m.end()) for m in _TABLE_ENV_RE.finditer(text)]


def _in_table(spans: list[tuple[int, int]], pos: int) -> bool:
    """True when ``pos`` falls inside any table-float span."""
    return any(lo <= pos < hi for lo, hi in spans)


def _strip_line_comments(text: str) -> str:
    """Drop LaTeX line comments so commented-out ``\\cite`` sites aren't counted.

    An unescaped ``%`` starts a comment to end of line; an escaped ``\\%`` is
    kept. Applied only to the citation-site scan (not bibliography parsing, where
    a ``%`` may legitimately appear inside a URL).
    """
    out: list[str] = []
    for line in text.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line if m is None else line[: m.start()])
    return "\n".join(out)


def _read_tex_files(tex_dir: str | Path) -> list[tuple[Path, str]]:
    """Read all ``.tex`` files under ``tex_dir`` (sorted for determinism)."""
    root = Path(tex_dir)
    if not root.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for p in sorted(root.rglob("*.tex")):
        try:
            out.append((p, p.read_text(encoding="utf-8", errors="replace")))
        except Exception:
            continue
    return out


# How one source file pulls in another: ``\input{f}`` / ``\include{f}`` / ``\subfile{f}``
# and friends take the file as their (last) braced argument; the ``import`` package's
# ``\import{dir/}{f}`` / ``\subimport{dir/}{f}`` take it as the second.
_INPUT_RE = re.compile(
    r"\\(?:input|include|subfile|subfileinclude|InputIfFileExists|includestandalone)\s*"
    r"(?:\[[^\]]*\]\s*)?\{\s*([^{}]+?)\s*\}"
    r"|\\(?:sub)?(?:import|includefrom|inputfrom)\*?\s*\{[^{}]*\}\s*\{\s*([^{}]+?)\s*\}"
    r"|\\input\s+([^\s{}%\\]+)"  # plain-TeX form, no braces
)
_DOCUMENTCLASS_RE = re.compile(r"\\documentclass\b")


def included_sources(paths: list[Path]) -> list[Path]:
    r"""The subset of a paper's ``.tex`` files the document actually pulls in.

    An arXiv source tree routinely ships drafts, a previous submission's version and
    author backups alongside the live files. Reading all of them means reading a paper
    that was not submitted: one tree held a five-column draft of a table under the same
    ``\label`` as the published six-column one, and citations in an abandoned section are
    reported as claims the paper makes.

    Walks the include graph from the file carrying ``\documentclass``, ignoring
    commented-out includes (a commented ``%\input{old}`` is exactly how a draft is
    retired). Returns every path unchanged when there is no root to walk from — following
    the graph may only ever drop files, never invent a reason to.
    """
    texts: dict[Path, str] = {}
    for p in paths:
        try:
            texts[p] = _strip_line_comments(p.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    roots = [p for p, t in texts.items() if _DOCUMENTCLASS_RE.search(t)]
    if not roots:
        return paths
    by_stem: dict[str, list[Path]] = {}
    for p in texts:
        by_stem.setdefault(p.stem, []).append(p)

    def closure(root: Path) -> set[Path]:
        seen: set[Path] = set()
        queue = [root]
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for groups in _INPUT_RE.findall(texts.get(cur, "")):
                raw = next((g for g in groups if g), "")
                for cand in by_stem.get(Path(raw).stem, []):
                    if cand not in seen:
                        queue.append(cand)
        return seen

    # A tree can hold several files with `\documentclass` — a standalone snippet compiled
    # on its own is one, and so is a leftover from a previous submission. The real document
    # is the one that pulls in the most, so take the largest closure rather than giving up.
    best = max((closure(r) for r in roots), key=len)
    return [p for p in paths if p in best] or paths


def _live_tex_files(source: PaperSource) -> list[tuple[Path, str]]:
    """The ``.tex`` files that make up the paper as submitted, with their text.

    The include graph decides (:func:`included_sources`), and the compiled bibliography
    checks the decision: a ``.bbl`` lists exactly what the built paper cites, so if
    restricting to the graph loses a key the ``.bbl`` has — an include form the walker
    does not know, a file pulled in by a macro — the graph is wrong, not the ``.bbl``,
    and every file is read after all. Losing a real citation is worse than keeping a
    stale one.
    """
    files = _read_tex_files(source.tex_dir or "")
    if len(files) <= 1:
        return files
    live = set(included_sources([p for p, _t in files]))
    if len(live) == len(files):
        return files
    kept = [(p, t) for p, t in files if p in live]
    compiled = compiled_cite_keys(source)
    if compiled:

        def cited(subset: list[tuple[Path, str]]) -> set[str]:
            return {
                k
                for _p, text in subset
                for _pos, keys in cite_sites(_strip_line_comments(text))
                for k in keys
            } & compiled

        if cited(kept) != cited(files):
            return files
    return kept


def _section_at(text: str, pos: int) -> str | None:
    """Return the title of the nearest preceding section heading before ``pos``."""
    last: str | None = None
    for m in _SECTION_RE.finditer(text, 0, pos):
        last = m.group(1).strip()
    if last:
        return re.sub(r"\s+", " ", re.sub(r"\\[a-zA-Z]+", "", last)).strip()
    return None


def _sentence_around(text: str, pos: int, window: int = 600) -> tuple[str, tuple[int, int]]:
    """Extract the sentence containing offset ``pos`` and its char span.

    Returns a lightly de-TeX'd sentence string and the ``(start, end)`` span in
    the ORIGINAL ``text`` (so spans are stable and comparable).
    """
    lo = max(0, pos - window)
    hi = min(len(text), pos + window)
    chunk = text[lo:hi]
    rel = pos - lo
    # Find sentence boundaries within the chunk around the relative position.
    starts = [0] + [m.end() for m in _SENT_SPLIT.finditer(chunk)]
    ends = [m.start() for m in _SENT_SPLIT.finditer(chunk)] + [len(chunk)]
    s_start, s_end = 0, len(chunk)
    for st, en in zip(starts, ends, strict=False):
        if st <= rel <= en:
            s_start, s_end = st, en
            break
    sentence = _strip_tex(chunk[s_start:s_end])
    return sentence, (lo + s_start, lo + s_end)


# ───────────────────────────────────────────────────────────────
# E-print materialization (download + extract the arXiv tarball)
# ───────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> bool:
    """Download ``url`` to ``dest`` (fail-soft). Return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read()
        if not data:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Extract a tar archive, refusing path-traversal members (zip-slip guard)."""
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            continue  # skip members that would escape dest
        tf.extract(member, dest)


def _extract_eprint(tarball: Path, tex_dir: Path) -> bool:
    """Extract an arXiv e-print archive into ``tex_dir`` (fail-soft).

    arXiv e-prints are usually gzipped tarballs but a single-file submission may
    be a bare ``.gz`` holding one ``.tex``. Both are handled; anything else
    degrades to ``False``.
    """
    tex_dir.mkdir(parents=True, exist_ok=True)
    try:
        if tarfile.is_tarfile(tarball):
            with tarfile.open(tarball) as tf:
                _safe_extract(tf, tex_dir)
            return True
    except Exception:
        return False
    try:
        import gzip

        with gzip.open(tarball, "rb") as fh:
            (tex_dir / "main.tex").write_bytes(fh.read())
        return True
    except Exception:
        return False


def _find_bib_files(tex_dir: Path) -> tuple[str | None, str | None]:
    """Return ``(bbl_path, bib_path)`` found under ``tex_dir`` (or ``None``s)."""
    bbls = sorted(tex_dir.rglob("*.bbl"))
    bibs = sorted(tex_dir.rglob("*.bib"))
    return (str(bbls[0]) if bbls else None, str(bibs[0]) if bibs else None)


def ensure_eprint(source: PaperSource) -> PaperSource:
    """Make sure the LaTeX e-print is on disk; populate ``source`` LaTeX paths.

    If ``source.tex_dir`` already has ``.tex`` files, nothing is downloaded. Else
    — when ``source.arxiv_id`` is known and ``source.work_dir`` is set — the
    e-print tarball is fetched from ``arxiv.org/e-print/<id>`` and extracted into
    ``<work_dir>/tex``. Mutates and returns ``source``. Fails soft: on any
    network/extract error the LaTeX paths are simply left/unset.
    """
    work_dir = Path(source.work_dir) if source.work_dir else None

    # Already extracted (this run or a prior one)?
    existing = Path(source.tex_dir) if source.tex_dir else (work_dir / "tex" if work_dir else None)
    if existing and existing.is_dir() and any(existing.rglob("*.tex")):
        _apply_tex_paths(source, existing)
        return source

    if not source.arxiv_id or work_dir is None:
        return source  # nothing to download / nowhere to put it

    tex_dir = work_dir / "tex"
    tarball = work_dir / "eprint.tar"
    if not tarball.is_file():
        _download(f"https://arxiv.org/e-print/{source.arxiv_id}", tarball)
    if tarball.is_file() and _extract_eprint(tarball, tex_dir) and any(tex_dir.rglob("*.tex")):
        _apply_tex_paths(source, tex_dir)
    return source


def _apply_tex_paths(source: PaperSource, tex_dir: Path) -> None:
    """Fill ``tex_dir`` / ``bbl_path`` / ``bib_path`` / flags on ``source``."""
    bbl, bib = _find_bib_files(tex_dir)
    source.tex_dir = str(tex_dir)
    source.bbl_path = bbl
    source.bib_path = bib
    source.kind = "arxiv_latex"
    source.tex_available = True


# ───────────────────────────────────────────────────────────────
# The extractor
# ───────────────────────────────────────────────────────────────
class LatexExtractor:
    """Extractor for arXiv LaTeX e-prints (satisfies the ``Extractor`` Protocol).

    Usage::

        extractor = LatexExtractor()
        stubs = extractor.extract(paper_source)  # list[CitationRecord]

    The returned records are *stubs*: key + ``claim`` + ``cited_as`` populated,
    judged axes at their defaults. Verification stages/backends fill the rest.
    """

    def extract(self, source: PaperSource) -> list[CitationRecord]:
        """Parse references + ``\\cite`` call-sites into record stubs.

        Degrade-not-crash: any unreadable file is skipped; a cite key with no
        matching bibliography entry still yields a record (with an empty
        ``cited_as.raw``) so the citation is not silently dropped.

        If the e-print is not yet on disk (``ingest`` only probed availability),
        it is downloaded and extracted first via :func:`ensure_eprint`.
        """
        ensure_eprint(source)  # fail-soft; no-op if already on disk or offline
        references = load_references(source)
        # What the compiled paper cites. A key cited in the source but absent from the
        # .bbl is not a citation of this paper — it sits in a draft the document no longer
        # includes — and reporting it as "no bibliography entry" accuses the authors of a
        # dangling citation that is not in their paper.
        compiled = compiled_cite_keys(source)
        paper = self._build_paper(source)
        records: list[CitationRecord] = []
        seen: set[tuple[str, str, str]] = set()

        for _path, text in _live_tex_files(source):
            text = _strip_line_comments(text)  # ignore commented-out \cite call-sites
            table_spans = _table_spans(text)
            for pos, keys in cite_sites(text):
                if compiled:
                    keys = [k for k in keys if k in compiled or k in references]
                if not keys:
                    continue
                section = _section_at(text, pos)
                sentence, span = _sentence_around(text, pos)
                in_table = _in_table(table_spans, pos)
                for cite_key in keys:
                    claim_id = make_claim_id(source.paper_id, section, span, cite_key)
                    dedup = (source.paper_id, claim_id, cite_key)
                    if dedup in seen:
                        continue
                    seen.add(dedup)
                    cited_as = references.get(cite_key) or CitedAs(raw="")
                    notes = None if cite_key in references else "no bibliography entry for cite_key"
                    if in_table:
                        notes = f"{notes}; {TABLE_NOTE}" if notes else TABLE_NOTE
                    records.append(
                        CitationRecord(
                            paper_id=source.paper_id,
                            claim_id=claim_id,
                            cite_key=cite_key,
                            paper=paper,
                            claim=Claim(
                                claim_id=claim_id,
                                text=sentence,
                                section=section,
                                char_span=span,
                            ),
                            cited_as=cited_as,
                            in_table=in_table,
                            notes=notes,
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
            source_kind=source.kind or "arxiv_latex",
            tex_available=True,
        )
