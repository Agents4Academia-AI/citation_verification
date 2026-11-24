"""
extract/sentences.py — the sentence a citation sits in, for every extractor.

The claim attached to a citation is the sentence around it. The PDF marker scanner, the
hyperlink reader, the author-year text scanner and the LaTeX reader each used to cut that
sentence their own way, and the reports showed what that cost: more than half the claims of
a ten-paper batch were not a sentence. They began mid-clause and stopped mid-word, at a
360-character cap that a sentence listing four methods with their citations easily passes,
or they ran a section heading into the next sentence ("…under gradual drift. 2 RELATED WORK
ACI (Gibbs & Candes, 2021) tracks…") because a full stop followed by a digit was never read
as an end. This module is the one place that decides where a sentence starts and ends.

* **Hard boundaries** are paragraph breaks — a blank line. A caller that knows where the
  headings, captions and footnotes are puts each in a paragraph of its own; nothing crosses
  them. Everything else (a line wrap, a column or page break) is a soft join.
* **Soft boundaries** are a ``.``/``!``/``?`` (closing quotes or brackets allowed) and white
  space before an opener — a capital, a quote, a bracket, a backslash (a LaTeX sentence may
  open with ``\\citet``) — unless the word before the stop is an abbreviation (et al., e.g.,
  Fig., Eq., Sec., …) or a single initial, or the stop sits inside a parenthesis that closes
  nearby ("(see Fig. 3. for …)").
* **Length.** A claim is the whole sentence, up to :data:`MAX_CLAIM_CHARS`. Past that it is
  the clauses (split at ``;``, ``:`` and spaced dashes) around the citation, and only when a
  single clause is longer still, a word-bounded window around it. The text is never edited,
  so a claim is a verbatim stretch of what was read and can be found in it again.
"""

from __future__ import annotations

import bisect
import re

__all__ = [
    "MAX_CLAIM_CHARS",
    "heading_title",
    "is_heading_line",
    "section_at",
    "sentence_span",
]

# A whole sentence fits; a run of table cells or a mis-joined page does not.
MAX_CLAIM_CHARS = 1000
# How far either side of the citation the boundaries are looked for. Wider than any claim,
# so the sentence that holds the citation always starts and ends inside it.
_WINDOW = MAX_CLAIM_CHARS + 500

# A word whose trailing full stop does not end the sentence. Matched case-sensitively on
# the token before the stop; multi-dot forms ("e.g.", "i.e.", "U.S.", "w.r.t.") are caught
# by shape below, so only their spellings that drop a dot need listing.
_ABBREVIATIONS = frozenset(
    {
        "al", "cf", "vs", "etc", "resp", "approx", "viz", "ca", "eg", "ie",
        "Fig", "Figs", "fig", "figs", "Eq", "Eqs", "eq", "eqs", "Eqn", "Eqns",
        "Sec", "Secs", "sec", "Sect", "Tab", "Tabs", "Ref", "Refs", "Alg", "Algs",
        "App", "Appx", "Thm", "Def", "Prop", "Lem", "Cor", "Rem", "Ex", "Exp",
        "Ch", "Chap", "Vol", "vol", "No", "Nos", "pp", "Dr", "Prof", "Mr", "Mrs",
        "Ms", "St", "Jr", "Sr", "Inc", "Ltd", "Corp", "Co", "Std", "Approx",
    }
)
_DOTTED_ABBREVIATION_RE = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")  # e.g / i.e / U.S / w.r.t

# A candidate end: the stop, any closing quotes/brackets, the white space, and what opens
# the next sentence. Group 1 spans the white space (the next sentence starts at its end).
_END_RE = re.compile(r"[.!?][\"'”’)\]]*(\s+)(?=[A-Z\"“‘'(\[\\])")
_PARAGRAPH_RE = re.compile(r"\n[ \t\r\f\v]*\n\s*")
_CLAUSE_RE = re.compile(r"(?:[;:]|\s[—–]|\s--)\s+")
_PAREN_REACH = 250  # a parenthesis open longer than this is a stray, not an aside

# ── headings ─────────────────────────────────────────────────────────────────
# "3 Method", "2.1 Setup", "A Proofs", "B.2 Extra results" — a number or an appendix
# letter, then a short title.
_NUMBERED_HEADING_RE = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z](?:\.\d{1,2}){1,3}|[A-Z])\.?\s+(\S.*)$")
_UNNUMBERED_HEADINGS = re.compile(
    r"^(?:abstract|introduction|related\s+work|background|preliminaries|"
    r"(?:proposed\s+)?methods?|methodology|approach|experiments?|experimental\s+(?:setup|results)|"
    r"results|evaluation|discussion|analysis|conclusions?|limitations|future\s+work|"
    r"(?:conclusions?|discussion)\s+and\s+future\s+work|acknowledge?ments?|appendix(?:\s+[a-z])?|"
    r"appendices|references|bibliography|broader\s+impacts?|ethics\s+statement|"
    r"reproducibility\s+statement|supplementary\s+material)$",
    re.IGNORECASE,
)
# A title never ends on a word that needs a continuation — a wrapped line of prose does.
_DANGLING_END = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or",
     "the", "to", "via", "with", "we", "that", "which", "are", "be"}
)
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or",
     "over", "the", "to", "via", "vs", "with", "without", "under", "through", "versus"}
)


def is_heading_line(line: str) -> bool:
    """Whether a line, read on its own, is a section heading.

    Numbered or lettered ("3 Method", "2.1 Setup", "A Proofs"), one of the standard section
    names ("Related Work", "ACKNOWLEDGMENTS"), or a short line all in capitals. A numbered
    title may be in sentence case ("2 Related work") when it is short; a lettered one must be
    in title case, since a wrapped line of prose can open with "A". No heading ends in a
    full stop, a comma, or a word that needs a continuation ("…the effect of").
    """
    s = " ".join((line or "").split())
    if not (2 <= len(s) <= 90) or s[-1] in ".,;":
        return False
    if _UNNUMBERED_HEADINGS.match(s.rstrip(":")):
        return True
    m = _NUMBERED_HEADING_RE.match(s)
    if m:
        title = m.group(1).rstrip(":")
        words = title.split()
        if not words or len(words) > 12 or not words[0][0].isupper():
            return False
        if words[-1].lower() in _DANGLING_END or re.search(r"[=<>]|\d{4}", title):
            return False
        if _UNNUMBERED_HEADINGS.match(title) or _all_caps(title) or _title_case(words):
            return True
        # Sentence case is how many templates set headings, but only short ones: a wrapped
        # line that happens to open with a number runs on.
        return s[0].isdigit() and len(words) <= 5
    words = s.split()
    return _all_caps(s) and len(words) <= 8 and any(len(w) >= 3 for w in words)


def heading_title(line: str) -> str:
    """A heading without its number: "2.1  Experimental setup" -> "Experimental setup"."""
    s = " ".join((line or "").split())
    m = _NUMBERED_HEADING_RE.match(s)
    return (m.group(1) if m and not _UNNUMBERED_HEADINGS.match(s) else s).rstrip(":").strip()


def _all_caps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _title_case(words: list[str]) -> bool:
    content = [w for w in words if w.lower() not in _SMALL_WORDS and w[0].isalpha()]
    return bool(content) and all(w[0].isupper() for w in content)


def section_at(headings: list[tuple[int, str]], pos: int) -> str | None:
    """The title of the last heading before ``pos``, from ``[(offset, title)]`` in order."""
    if not headings:
        return None
    i = bisect.bisect_right([h[0] for h in headings], pos) - 1
    return headings[i][1] if i >= 0 else None


# ── sentences ────────────────────────────────────────────────────────────────
def sentence_span(
    text: str, pos: int, *, max_chars: int = MAX_CLAIM_CHARS, breaks: list[int] | None = None
) -> tuple[int, int]:
    """``(start, end)`` of the sentence of ``text`` that holds offset ``pos``.

    The span never crosses a paragraph break and is at most ``max_chars`` long (see the
    module docstring for how a longer sentence is cut). Leading and trailing white space is
    excluded. ``(0, 0)`` when ``text`` is empty.

    ``breaks`` replaces the blank-line rule with explicit boundary offsets, for text whose
    blank lines are not paragraphs — a PDF's text blocks are separated by one wherever a
    column or page breaks, or a figure interrupts a sentence.
    """
    if not text:
        return (0, 0)
    pos = max(0, min(pos, len(text) - 1))
    if breaks is None:
        lo, hi = _paragraph(text, pos)
    else:
        i = bisect.bisect_right(sorted(breaks), pos)
        ordered = sorted(breaks)
        lo = ordered[i - 1] if i > 0 else 0
        hi = ordered[i] if i < len(ordered) else len(text)
    lo, hi = max(lo, pos - _WINDOW), min(hi, pos + _WINDOW)
    starts = [lo] + [m.end(1) for m in _END_RE.finditer(text, lo, hi) if _ends_sentence(text, m, lo, hi)]
    i = bisect.bisect_right(starts, pos) - 1
    start = starts[max(i, 0)]
    end = starts[i + 1] if i + 1 < len(starts) else hi
    start, end = _trim(text, start, end)
    if end - start > max_chars:
        start, end = _clause_window(text, start, end, pos, max_chars)
    return _trim(text, start, end)


def _paragraph(text: str, pos: int) -> tuple[int, int]:
    """The paragraph (between blank lines) that holds ``pos``."""
    lo, hi = 0, len(text)
    for m in _PARAGRAPH_RE.finditer(text):
        if m.end() <= pos:
            lo = m.end()
        elif m.start() > pos:
            hi = m.start()
            break
        else:  # pos sits in the break itself: take what follows
            lo = m.end()
    return lo, hi


def _ends_sentence(text: str, m: re.Match[str], lo: int, hi: int) -> bool:
    """Is this stop the end of a sentence (not an abbreviation, initial or aside)?"""
    stop = m.start()
    if text[stop] == ".":
        word = re.search(r"([^\s(\[\"“]*)$", text[max(lo, stop - 30) : stop]).group(1)
        if word in _ABBREVIATIONS or _DOTTED_ABBREVIATION_RE.match(word):
            return False
        if len(word) == 1 and word.isupper():  # an initial: "D. P. Kingma"
            return False
    return not _inside_parenthesis(text, stop, lo, hi)


def _inside_parenthesis(text: str, at: int, lo: int, hi: int) -> bool:
    """A "(" opened shortly before ``at`` and closed shortly after it, with no other
    bracket between — the stop is part of an aside, not the end of the sentence."""
    before = text[max(lo, at - _PAREN_REACH) : at]
    after = text[at + 1 : min(hi, at + 1 + _PAREN_REACH)]
    o, c = before.rfind("("), before.rfind(")")
    if o == -1 or c > o:
        return False
    close, reopen = after.find(")"), after.find("(")
    return close != -1 and (reopen == -1 or close < reopen)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _clause_window(text: str, start: int, end: int, pos: int, max_chars: int) -> tuple[int, int]:
    """Cut an over-long sentence down to the clauses around ``pos``, then — when a single
    clause is too long — to a window of whole words around it."""
    cuts = [start] + [m.end() for m in _CLAUSE_RE.finditer(text, start, end)] + [end]
    i = max(0, bisect.bisect_right(cuts, pos) - 1)
    a, b = cuts[i], cuts[min(i + 1, len(cuts) - 1)]
    left, right = i - 1, i + 2
    while True:  # grow by whole clauses, alternating sides, while it fits
        grown = False
        if right < len(cuts) and cuts[right] - a <= max_chars:
            b, right, grown = cuts[right], right + 1, True
        if left >= 0 and b - cuts[left] <= max_chars:
            a, left, grown = cuts[left], left - 1, True
        if not grown:
            break
    if b - a <= max_chars:
        return a, b
    half = max_chars // 2
    a = max(a, pos - half)
    b = min(b, a + max_chars)
    if a > start:  # snap to word boundaries, never mid-word
        sp = text.find(" ", a, b)
        a = sp + 1 if sp != -1 else a
    if b < end:
        sp = text.rfind(" ", a, b)
        b = sp if sp > a else b
    return a, b
