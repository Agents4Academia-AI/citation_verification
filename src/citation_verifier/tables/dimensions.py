"""
tables/dimensions.py — recover what a column actually MEANS.

A comparison-table header is compressed to fit the column ("Task-aware", "TDA Free",
"Retrain-free"), so on its own it is not a checkable proposition. The meaning lives
somewhere else in the paper — and often not where you would expect:

  * the caption ("… ✓ indicates the method requires no retraining"),
  * a legend/footnote under the table,
  * or a sentence far away in the body — in the ATU paper the caption is only
    "Summary of existing task augmentation strategies" while "Task-aware" is defined
    in a ``\\begin{definition}`` block in a *different file*.

So this module does two things:

  1. :func:`find_definition_snippets` — deterministically gather and RANK the passages
     that plausibly define a header (definition environments and "is defined as" /
     "refers to" / "indicates whether" phrasing score highest). Pure text work.
  2. :func:`resolve_dimensions` — turn the best passages into a one-sentence checkable
     gloss plus the yes/no ``test_question`` each cited paper will be asked. The model
     is injected (``glosser``); with none supplied the deterministic snippet is used
     and, when nothing is found at all, the column is marked ``header_only`` so its
     cells are reported as ``undefined`` instead of being guessed at.

Anti-invention rule: a gloss is only ever derived from text found IN the paper. If the
paper never defines a column, that is itself the finding.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from .model import Dimension, GlossSource

__all__ = [
    "header_variants",
    "find_definition_snippets",
    "resolve_dimensions",
    "DEFAULT_TEST_QUESTION",
]

DEFAULT_TEST_QUESTION = "Does the cited work have this property?"

# Phrasing that marks a sentence as definitional, with weights.
_DEFINITIONAL = (
    (re.compile(r"\bis defined (?:as|to be)\b", re.I), 4),
    (re.compile(r"\bif and only if\b", re.I), 4),
    (re.compile(r"\bwe (?:define|say|call|denote)\b", re.I), 4),
    (re.compile(r"\brefers? to\b", re.I), 3),
    (re.compile(r"\bdenotes?\b", re.I), 3),
    (re.compile(r"\bindicates? (?:whether|if|that)\b", re.I), 3),
    (re.compile(r"\bmeans? that\b", re.I), 3),
    (re.compile(r"\b(?:column|property|criterion|desideratum|desiderata|attribute)\b", re.I), 2),
    (re.compile(r"\bi\.e\.\b|\bthat is\b|\bnamely\b", re.I), 1),
    (re.compile(r"\bable to\b|\brequires?\b|\bsupports?\b|\bwithout\b", re.I), 1),
)
_DEFINITION_ENV = re.compile(
    r"\\begin\{(definition|property|criterion|desiderata)\*?\}(?:\[[^\]]*\])?(.*?)\\end\{\1\*?\}",
    re.DOTALL | re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\(])")

# LaTeX comments (reviewer notes like "%%% YING: ..." mention the term without defining
# it) and the table floats themselves must not count as the paper "defining" a column.
_TEX_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")
# Only LaTeX source has comments; PDF-extracted text does not.
_LOOKS_LIKE_LATEX = re.compile(r"\\(?:begin|section|cite|textit|textbf|item)\b")
_TABLE_FLOAT_RE = re.compile(
    r"\\begin\{(table\*?|wraptable|sidewaystable|threeparttable|tabular\*?|tabularx|longtable)\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)

# Papers most often define a comparison-table column NOT with "is defined as" but by
# enumerating desiderata: `(3) \textit{model-adaptive}: the augmented tasks are …`,
# `\item \textbf{TDA Free}: …`, `(iii) Retrain-free — …`. These carry the definition in
# the punctuation, so they need their own high-weight patterns.
# NOTE: no `^` anchor — `Pattern.match(text, pos)` already anchors at `pos`, and `^`
# would additionally require the true start of the string (matching nothing here).
_AFTER_TERM_RE = re.compile(
    r"\s*\}?\s*(?:[:—–]|,?\s+(?:which|where|meaning)\b)\s*(?P<def>[^.;]{15,400})",
    re.IGNORECASE,
)
# Text that follows the same punctuation but says nothing about the property itself.
_META_TEXT_RE = re.compile(
    r"^(?:we\s+(?:abbreviate|denote\s+it|refer\s+to\s+it|use|report|show|present|discuss|list)|"
    r"(?:see|cf\.?|as\s+shown|as\s+discussed|described|summari[sz]ed|reported|listed)\b|"
    r"table\b|figure\b|section\b|appendix\b|column\s+\d|was\s+first\b|the\s+focus\s+of\b|"
    r"our\s+reviewers?\b|in\s+the\s+(?:rebuttal|appendix|supplement))",
    re.IGNORECASE,
)
# Cross-reference / bookkeeping language anywhere in the captured text, not just its start.
_NON_DEFINITION_RE = re.compile(
    r"\b(?:in\s+Table\s+\d|in\s+Figure\s+\d|see\s+(?:Table|Figure|Section)\s|"
    r"is\s+column\s+\d|the\s+rebuttal)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=512)
def _term_regex(variant: str) -> re.Pattern[str]:
    """Word-boundary matcher for one spelling, tolerating space/hyphen interchange.

    Cached: this is called once per sentence per variant while scanning a whole paper.
    """
    parts = [re.escape(p) for p in re.split(r"[\s-]+", variant) if p]
    return re.compile(r"\b" + r"[\s\-]+".join(parts) + r"\b", re.IGNORECASE)


def _definitions_by_punctuation(variant: str, text: str, *, limit: int = 6) -> list[str]:
    """Definitions of the form "<term><separator> definition …".

    Papers usually define comparison-table columns by enumerating desiderata rather than
    by saying "is defined as" — ``(3) \\textit{model-adaptive}: the augmented tasks are …``
    or ``\\item \\textbf{TDA Free}: …``. The definition lives in the punctuation.

    Anchored on the term and then looking only at the ~450 characters that follow it, so
    cost is proportional to the number of occurrences — a single regex carrying optional
    prefixes backtracks catastrophically over a full paper.
    """
    out: list[str] = []
    for m in _term_regex(variant).finditer(text):
        tail = _AFTER_TERM_RE.match(text, m.end())
        if tail:
            body = re.sub(r"\s+", " ", tail.group("def")).strip(" ,;:")
            # Reject cross-references and naming asides — "…, which we abbreviate MA in
            # the tables" is not a definition, but it matches the same punctuation.
            if len(body) >= 15 and not _META_TEXT_RE.match(body) \
                    and not _NON_DEFINITION_RE.search(body):
                out.append(body)
                if len(out) >= limit:
                    break
    return out


_ENUMERATOR_RE = re.compile(r"^\(?\s*(?:[ivxIVX]{1,4}|[a-hA-H]|\d{1,2})\s*\)?[.)]?$")
# The opening of a contributions list ("(i) We propose …", "(ii) Our method …"), which is
# about the CITING paper and must never be adopted as a column name.
_SELF_CLAIM_RE = re.compile(
    r"^(?:we\b|our\b|this\s+(?:paper|work)\b|I\s)", re.IGNORECASE
)


def enumerator_name(header: str, body_text: str) -> tuple[str, str]:
    """Resolve a placeholder header like ``(i)`` to the name the body gives it.

    Some tables label columns only by an enumerator and put the names in a list —
    USEEK's Table I is headed ``(i) (ii) (iii) (iv)`` while the body says
    ``\\item (i) \\textit{Anti-occlusion.} …``. Without this the columns are unnameable
    and every cell would be reported as an undefined column.

    Args:
        header: the printed header.
        body_text: the citing paper's text.

    Returns:
        ``(name, quote)`` — the recovered column name and the passage it came from, or
        ``("", "")`` when ``header`` is not an enumerator or no name is found.
    """
    tag = (header or "").strip()
    if not _ENUMERATOR_RE.match(tag):
        return "", ""
    # Only bracketed roman/letter tags: a header of "1" or "A" is far more likely to be a
    # real (numeric) column than a pointer into an enumerated list.
    if not (tag.startswith("(") and tag.endswith(")")):
        return "", ""
    token = re.sub(r"[^\w]", "", tag)
    # The name, then everything up to the end of that list item — the definition sits
    # right after the name (``\item (i) \textit{Anti-occlusion.} The detector must …``),
    # so the item body is the gloss. `[^\\&]` stops at the next macro or a table cell,
    # which keeps the row `(i) & (ii) & …` and trailing `\\` out of the name.
    pat = re.compile(
        rf"\(\s*{re.escape(token)}\s*\)\s*"
        rf"(?:\\(?:textit|textbf|emph|texttt|textsc)\s*\{{)?\s*"
        rf"(?P<name>[A-Z][^.;:}}\n\\&]{{2,60}})"
        rf"(?P<rest>[^\\]{{0,400}})",
    )
    for m in pat.finditer(body_text or ""):
        name = re.sub(r"\s+", " ", m.group("name")).strip(" .,;:}")
        if not name or "&" in name:
            continue
        # "(i) We propose …" is the paper's own contributions list, not a column name.
        # Adopting it would name the column after this paper's contribution and then
        # accuse every cited work of lacking it.
        if _SELF_CLAIM_RE.match(name):
            continue
        # A column name is a short noun phrase, not a sentence.
        if len(name.split()) > 6:
            continue
        rest = re.sub(r"^[\s.}{]+", ": ", m.group("rest") or "")
        quote = re.sub(r"\s+", " ", name + rest).strip(" .,;:}{")
        return name, quote[:400]
    return "", ""


def header_variants(header: str) -> list[str]:
    """Spelling variants of a header to search for in the body.

    ``"Task-aware"`` -> ``["task-aware", "task aware", "taskaware", "task-awareness", …]``
    so hyphenation and the noun form both match. Returns lowercase forms, longest first.
    """
    base = re.sub(r"[^\w\s-]", " ", (header or "")).strip().lower()
    base = re.sub(r"\s+", " ", base)
    if not base:
        return []
    forms = {base, base.replace("-", " "), base.replace("-", ""), base.replace(" ", "-")}
    # noun/adjective morphology: "aware" <-> "awareness", "adaptive" <-> "adaptation"
    for f in list(forms):
        if f.endswith("aware"):
            forms.add(f + "ness")
        if f.endswith("ive"):
            forms.add(f[:-3] + "ion")
    # Keep 2-character forms: real columns are abbreviated that hard ("SP", "CT Free").
    # Safe because matching is word-boundary anchored, not substring.
    return sorted({f for f in forms if len(f) >= 2}, key=len, reverse=True)


def _score_sentence(sentence: str, variants: list[str], *, in_definition_env: bool) -> int:
    """How strongly a sentence looks like it DEFINES one of ``variants``.

    Matching is word-boundary anchored, never substring: a header like "SP" must not be
    considered "mentioned" because the text happens to contain "SPRIN".
    """
    if not any(_term_regex(v).search(sentence) for v in variants):
        return 0
    score = 2 + (5 if in_definition_env else 0)
    for pat, w in _DEFINITIONAL:
        if pat.search(sentence):
            score += w
    if len(sentence) > 600:
        score -= 2  # a whole paragraph is weaker evidence than a crisp sentence
    return score


def _recites_the_table(quote: str, siblings: list[str], *, threshold: int = 2) -> bool:
    """True when a passage is the table's own header row rather than a definition.

    In a PDF the extracted body text CONTAINS the flattened table, so the string
    ``"Method TDA Free CT Free TDL Free SP T Free String-match …"`` matches every header
    and would be handed to the judge as all five columns' definition. (The LaTeX path
    strips table floats; extracted PDF text has no such structure to strip.) A real
    definition explains one column — reciting two or more of the others gives it away.
    A single mention is fine: ATU defines "Task-imaginary" partly by contrast with
    "task-awareness".
    """
    hits = 0
    for sib in siblings:
        for variant in header_variants(sib)[:2]:
            if _term_regex(variant).search(quote):
                hits += 1
                break
        if hits >= threshold:
            return True
    return False


def find_definition_snippets(
    header: str,
    body_text: str,
    *,
    caption: str = "",
    legend: list[str] | None = None,
    limit: int = 4,
    siblings: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Passages that plausibly define ``header``, best first.

    Args:
        header: the column header as printed.
        body_text: the paper's text (LaTeX source or extracted PDF text).
        caption: the table caption — searched first and scored highest.
        legend: footnote/legend lines under the table.
        limit: max snippets to return.

    Returns:
        ``[(source, quote, score), …]`` where ``source`` is a :class:`GlossSource`
        value. Empty when the paper never mentions the header outside the table.
    """
    variants = header_variants(header)
    if not variants:
        return []
    found: list[tuple[str, str, int]] = []

    for sent in _SENT_SPLIT.split(caption or ""):
        s = _score_sentence(sent, variants, in_definition_env=False)
        if s:
            found.append((GlossSource.CAPTION.value, sent.strip(), s + 4))

    for line in legend or []:
        s = _score_sentence(line, variants, in_definition_env=False)
        if s:
            found.append((GlossSource.LEGEND.value, line.strip(), s + 3))

    # Reviewer comments and the table floats themselves mention the term without ever
    # defining it — drop both before deciding whether the paper defines this column.
    # Comment stripping is LaTeX-only: in text extracted from a PDF a `%` is a percent
    # sign ("Min-k% Prob"), and treating it as a comment deletes the rest of the line —
    # silently destroying any definition that shares a line with a number.
    rest = body_text or ""
    if _LOOKS_LIKE_LATEX.search(rest):
        rest = _TEX_COMMENT_RE.sub(" ", rest)
    rest = _TABLE_FLOAT_RE.sub(" ", rest)

    # Definition environments are the strongest body signal; harvest them first, then
    # BLANK them out so the general scan neither re-reports them nor mis-attributes the
    # sentence that follows one (a match starting inside a stale span would be dropped).
    for m in _DEFINITION_ENV.finditer(rest):
        for sent in _SENT_SPLIT.split(m.group(2)):
            s = _score_sentence(sent, variants, in_definition_env=True)
            if s:
                found.append((GlossSource.BODY.value, sent.strip(), s))
    rest = _DEFINITION_ENV.sub(lambda m: " " * (m.end() - m.start()), rest)

    # Enumerated / emphasised desiderata: `(3) \textit{model-adaptive}: <definition>`.
    # This is how papers most often define comparison-table columns, and it carries no
    # "is defined as" phrasing, so it needs its own pass.
    # Ranked BELOW an explicit "X is defined as Y" (2 + 4 = 6) and below a definition
    # environment (2 + 5 = 7): punctuation is the weakest of the three definition
    # shapes, and a tie let bookkeeping prose win on a stable sort.
    for variant in variants[:3]:
        for body in _definitions_by_punctuation(variant, rest):
            found.append((GlossSource.BODY.value, f"{header}: {body}", 5))

    for m in re.finditer(r"[^.!?]{0,400}[.!?]", rest):
        sent = m.group(0).strip()
        s = _score_sentence(sent, variants, in_definition_env=False)
        if s >= 4:  # definitional phrasing
            found.append((GlossSource.BODY.value, sent, s))
        elif s:  # a plain mention — weak, but proof the paper does discuss the term
            found.append((GlossSource.MENTION.value, sent, s))

    others = [s for s in (siblings or []) if s and s.strip().lower() != (header or "").strip().lower()]
    if others:
        found = [f for f in found if not _recites_the_table(f[1], others)]
    found.sort(key=lambda t: t[2], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, str, int]] = []
    for src, quote, score in found:
        key = re.sub(r"\W+", "", quote.lower())[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append((src, re.sub(r"\s+", " ", quote)[:600], score))
        if len(out) >= limit:
            break
    return out


def resolve_dimensions(
    dimensions: list[Dimension],
    body_text: str,
    *,
    caption: str = "",
    legend: list[str] | None = None,
    glosser: Callable[[list[dict]], list[dict]] | None = None,
) -> list[Dimension]:
    """Fill ``gloss`` / ``gloss_source`` / ``gloss_quote`` / ``test_question`` in place.

    Deterministic first: each header is matched against the caption, the legend and the
    body. When a ``glosser`` is supplied it turns the retrieved passages into a crisp
    checkable sentence; otherwise the best retrieved passage IS the gloss.

    A column with no supporting passage anywhere keeps ``gloss_source='header_only'``
    and an empty gloss — :mod:`citation_verifier.tables.verify` then reports its cells
    as ``undefined`` rather than inventing a meaning.

    Args:
        dimensions: the columns to resolve (mutated and returned).
        body_text: the citing paper's text.
        caption: the table caption.
        legend: legend/footnote lines under the table.
        glosser: optional callable taking ``[{header, caption, snippets}, …]`` and
            returning ``[{gloss, test_question}, …]`` aligned by index.

    Returns:
        The same list, with the gloss fields populated.
    """
    payload: list[dict] = []
    for dim in dimensions:
        # A placeholder header ("(i)") carries no meaning on its own; recover the name
        # the body gives it and search on that instead.
        name, name_quote = enumerator_name(dim.header, body_text)
        if name:
            dim.header = f"{dim.header} {name}".strip()
        # A merged two-level header reads "Efficiency — Fast"; the paper defines "Fast",
        # never that exact phrase, so search each part too or a defined column would be
        # reported as one the paper never defined.
        terms = [name] if name else [p.strip() for p in re.split(r"\s+—\s+", dim.header)][::-1]
        terms = [t for t in dict.fromkeys([*terms, dim.header]) if t]
        siblings = [d.header for d in dimensions if d is not dim]
        snips = []
        for term in terms:
            snips = find_definition_snippets(
                term, body_text, caption=caption, legend=legend, siblings=siblings
            )
            if snips:
                break
        # The list item that names the column is its definition by construction, so it
        # outranks whatever else in the paper happens to contain the word.
        if name and len(name_quote) > len(name) + 15:
            snips = [(GlossSource.BODY.value, name_quote, 10), *snips]
        elif name and not snips:
            snips = [(GlossSource.BODY.value, name_quote, 5)]
        if snips:
            src, quote, _ = snips[0]
            dim.gloss_source = src
            dim.gloss_quote = quote
            dim.gloss = quote  # deterministic fallback; refined by the glosser below
        else:
            dim.gloss_source = GlossSource.HEADER_ONLY.value
            dim.gloss_quote = ""
            dim.gloss = ""
        dim.test_question = (
            f"Does the cited work satisfy '{dim.header}'?" if dim.header else DEFAULT_TEST_QUESTION
        )
        payload.append(
            {
                "header": dim.header,
                "caption": caption,
                "kind": dim.kind,
                "snippets": [{"source": s, "quote": q} for s, q, _ in snips],
            }
        )

    if glosser is None or not payload:
        return dimensions

    try:
        refined = glosser(payload)
    except Exception:  # noqa: BLE001 — degrade to the deterministic gloss
        return dimensions
    for dim, got in zip(dimensions, refined or [], strict=False):
        if not isinstance(got, dict):
            continue
        gloss = (got.get("gloss") or "").strip()
        question = (got.get("test_question") or "").strip()
        # Never let the model invent meaning for a column the paper never defines.
        if gloss and dim.gloss_source != GlossSource.HEADER_ONLY.value:
            dim.gloss = gloss
        if question:
            dim.test_question = question
    return dimensions


def dimension_is_checkable(dim: Dimension | Any) -> bool:
    """True when a column carries a proposition worth verifying per cited paper.

    Excludes only columns that occur nowhere outside the table (nothing to check
    against) and free-form numeric/label columns, which are descriptive rather than
    assertive. A merely-``mention``ed column IS checked — weakly grounded is not the
    same as undefined.
    """
    kind = getattr(dim, "kind", "")
    source = getattr(dim, "gloss_source", GlossSource.NONE.value)
    if source == GlossSource.HEADER_ONLY.value:
        return False
    return kind in ("binary", "graded", "categorical")
