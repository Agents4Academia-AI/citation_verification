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

__all__ = ["LatexExtractor", "make_claim_id"]

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
        title = entry.get("title")
        if title:
            title = re.sub(r"\s+", " ", re.sub(r"[{}]", "", title)).strip()
        venue = entry.get("journal") or entry.get("booktitle") or entry.get("publisher")
        if venue:
            venue = re.sub(r"\s+", " ", re.sub(r"[{}]", "", venue)).strip()
        # eprint is the arXiv id when archivePrefix/eprinttype say so, but BibTeX
        # in the wild often omits that field while still meaning arXiv.
        eprint = entry.get("eprint") or entry.get("arxiv") or None
        out[key] = CitedAs(
            raw=_render_bib_raw(entry),
            authors=_split_authors(entry.get("author")),
            title=title,
            year=_coerce_year(entry.get("year")),
            venue=venue,
            doi=(entry.get("doi") or None),
            arxiv_id=eprint,
            url=(entry.get("url") or None),
        )
    return out


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
_ARXIV_IN_TEXT = re.compile(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+/\d{7})", re.IGNORECASE)
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


def _parse_bbl_body(body: str) -> CitedAs:
    """Best-effort structured fields from one ``\\bibitem`` free-text body."""
    raw = _strip_tex(body)
    arxiv = _ARXIV_IN_TEXT.search(raw)
    doi = _DOI_IN_TEXT.search(raw)
    url = _URL_IN_TEXT.search(body)
    year = _YEAR_IN_TEXT.search(raw)
    # Title heuristic: the segment inside \newblock blocks or the longest
    # clause; we keep it conservative and leave None when unsure.
    title: str | None = None
    blocks = re.split(r"\\newblock", body)
    if len(blocks) >= 2:
        cand = _strip_tex(blocks[1])
        if len(cand) > 8:
            title = cand
    return CitedAs(
        raw=raw,
        authors=[],  # author parsing from free .bbl text is unreliable; leave empty
        title=title,
        year=_coerce_year(year.group(0)) if year else None,
        venue=None,
        doi=doi.group(0) if doi else None,
        arxiv_id=arxiv.group(1) if arxiv else None,
        url=url.group(0) if url else None,
    )


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
    """Load references from inline ``thebibliography`` + ``.bbl`` + ``.bib``.

    Inline (.tex) entries are the base; a separate ``.bbl`` overrides them, and
    the richer ``.bib`` (structured authors/venue) overrides both.
    """
    refs: dict[str, CitedAs] = {}
    refs.update(parse_inline_bib(source.tex_dir))
    if source.bbl_path:
        refs.update(parse_bbl(source.bbl_path))
    if source.bib_path:
        # .bib is richer (structured authors/venue); let it override .bbl/inline.
        refs.update(parse_bib(source.bib_path))
    return refs


# ───────────────────────────────────────────────────────────────
# \cite call-site scanning
# ───────────────────────────────────────────────────────────────
# All natbib/biblatex cite variants; capture optional [..][..] then {keys}.
_CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear|citenum|"
    r"parencite|textcite|autocite|footcite|cite\*)\s*"
    r"(?:\[[^\]]*\]\s*){0,2}"
    r"\{([^}]+)\}"
)
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
# Sentence boundary: a period/!/? followed by whitespace + capital/backslash,
# applied to lightly-cleaned text. Kept simple and deterministic.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")


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
        paper = self._build_paper(source)
        records: list[CitationRecord] = []
        seen: set[tuple[str, str, str]] = set()

        for _path, text in _read_tex_files(source.tex_dir or ""):
            text = _strip_line_comments(text)  # ignore commented-out \cite call-sites
            table_spans = _table_spans(text)
            for cm in _CITE_RE.finditer(text):
                keys = [k.strip() for k in cm.group(1).split(",") if k.strip()]
                if not keys:
                    continue
                section = _section_at(text, cm.start())
                sentence, span = _sentence_around(text, cm.start())
                in_table = _in_table(table_spans, cm.start())
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
