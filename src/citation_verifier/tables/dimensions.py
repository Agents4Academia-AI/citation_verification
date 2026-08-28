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
    "parse_symbol_legend",
    "header_variants",
    "find_definition_snippets",
    "resolve_dimensions",
    "DEFAULT_TEST_QUESTION",
]

DEFAULT_TEST_QUESTION = "Does the cited work have this property?"

# A symbol only means what THIS paper says it means. RaCoT's caption reads
# "(\cmark: strong, \tmark: medium, \xmark: weak)" and the jailbreak survey's reads
# "'-' indicates the method ... lacks that capability; $\circ$ denotes white-box attack".
# Reading the legend beats any hard-coded glyph table: it covers ▲/○/●/- and tells the
# judge what a middle grade actually stands for in this paper.
_LEGEND_ENTRY_RE = re.compile(
    # The symbol may be wrapped in any LaTeX/plain quoting: ``-'' , `-' , "-" , '-'.
    r"(?:``|`|\"|')?"
    r"(?P<sym>\\[A-Za-z@]+|\$\\[A-Za-z@]+\$|[✓✔✗✘✕×▲◐◑○●◦•√\-–—])"
    r"(?:''|'|\")?"
    r"\s*(?::|denotes?|indicates?|means?|represents?|stands?\s+for|=)\s*"
    r"(?P<meaning>[^;.,)]{2,80})",
    re.IGNORECASE,
)
_STRONG_WORDS = re.compile(r"\b(strong|full|fully|yes|support(s|ed)?|satisf(y|ies|ied)|has|available)\b", re.I)
_MID_WORDS = re.compile(r"\b(medium|partial(ly)?|moderate|limited|some|weakly|mid)\b", re.I)
_WEAK_WORDS = re.compile(r"\b(weak|no|not|none|lacks?|without|absent|unsupported|does not)\b", re.I)


def parse_symbol_legend(caption: str, legend_lines: list[str] | None = None) -> dict[str, str]:
    """Symbol -> the meaning the paper gives it, read from the caption/legend.

    Args:
        caption: the table caption.
        legend_lines: footnote lines printed under the table.

    Returns:
        ``{symbol: meaning}`` with the symbol normalised (``\\cmark``, ``▲``, ``-``).
        Empty when the paper states no legend — the caller then falls back to the
        built-in glyph vocabulary.
    """
    out: dict[str, str] = {}
    for text in [caption or "", *(legend_lines or [])]:
        for m in _LEGEND_ENTRY_RE.finditer(text):
            sym = m.group("sym").strip().strip("`'\"").replace("$", "")
            meaning = re.sub(r"\s+", " ", m.group("meaning")).strip(" .,;:")
            if sym and meaning and sym not in out:
                out[sym] = meaning
    return out


_ABSTENTION_WORDS = re.compile(
    r"\b(?:not\s+applicable|n/?\.?a\.?|not\s+(?:reported|measured|evaluated|available|"
    r"studied|tested)|unknown|unclear|undetermined|to\s+be\s+determined)\b",
    re.IGNORECASE,
)


def grade_from_meaning(meaning: str) -> str | None:
    """Map a legend meaning onto ``yes`` / ``partial`` / ``no``; ``None`` when unclear.

    Only used to interpret a symbol the built-in vocabulary does not know — the paper's
    own words decide, so "medium" becomes a partial mark rather than an empty cell.
    """
    if not meaning:
        return None
    if _ABSTENTION_WORDS.search(meaning):
        # "not applicable" / "not reported" is the paper declining to say, not a negative
        # claim — and it contains "not", which the weak-word test would read as one.
        return None
    if _MID_WORDS.search(meaning):
        return "partial"
    if _WEAK_WORDS.search(meaning):
        return "no"
    if _STRONG_WORDS.search(meaning):
        return "yes"
    return None

# Phrasing that marks a sentence as definitional, with weights.
_DEFINITIONAL = (
    (re.compile(r"\bis defined (?:as|to be)\b", re.I), 4),
    (re.compile(r"\bif and only if\b", re.I), 4),
    # "The function is SE(3)-equivariant IF FOR ANY point cloud P …" — the standard way a
    # formal property is stated in maths and CS writing, where the bare "if" means "iff".
    # Anchored on the universal quantifier so an ordinary conditional ("is fast if you use
    # a GPU") does not qualify. Measured: USEEK states a column's definition this way and
    # eight cells were reported as a column the paper never defines.
    (re.compile(r"\bif\s+(?:for\s+)?(?:any|all|every|each)\b", re.I), 4),
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
# STRUCTURAL markup left after `normalize_latex_prose` has run: a float, an include or an
# environment boundary inside what should be one sentence means the passage straddles a
# structure and what survived is a fragment. Deliberately not "any backslash command" —
# `\cite` and friends are pervasive in ordinary academic prose and say nothing about
# whether a passage is intact; penalising them cost four real column definitions.
_RESIDUAL_MARKUP_RE = re.compile(
    r"\\(?:begin|end|input|include\w*|label|caption|multicolumn|multirow|resizebox|"
    r"subfigure|includegraphics|tabular|figure|table)\b"
)
# Only LaTeX source has comments; PDF-extracted text does not.
_LOOKS_LIKE_LATEX = re.compile(
    r"\\(?:begin|end|section|subsection|paragraph|cite\w*|textit|textbf|textcolor|item|"
    r"label|ref|autoref|input|include\w*|caption|footnote|emph|multicolumn|multirow)\b"
)
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
    # Up to two words may sit between the term and the separator. A heading routinely
    # appends a category word to the column name — MARRS heads the passage that defines
    # its "Anaphora" column `\paragraph{Anaphora Resolution}`, and anchoring strictly on
    # the term left that definition unreachable.
    r"\s*\}?(?:\s+[A-Za-z][\w-]*){0,2}\s*(?:[:—–]|,?\s+(?:which|where|meaning)\b)"
    r"\s*(?P<def>[^.;]{15,400})",
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
# A column definition must describe the PROPERTY, not the citing paper's own method and
# not the gap that motivates the column. Both shapes are common right next to the term
# and were being adopted as the definition:
#   self-referential — "In contrast, \AlgName supports all four aspects by leveraging …"
#       The judge is then effectively asked whether a competitor implements THIS paper's
#       mechanism, which nothing but this paper can satisfy, so every cell reads unclear.
#   problem statement — "Many conventional methods primarily target the first type …"
#       Describes the shortcoming the column exists to expose, never the criterion, so the
#       judge has to guess what earns a ✓.
_SELF_REFERENTIAL_RE = re.compile(
    r"\b(?:we|our|ours|this\s+(?:paper|work|study))\b|\bin\s+contrast\b|\bunlike\s+(?:prior|existing)",
    re.IGNORECASE,
)
_PROBLEM_STATEMENT_RE = re.compile(
    r"^\W*(?:most|many|existing|prior|previous|current|conventional|traditional)\s+"
    r"\w*\s*(?:methods?|approaches?|works?|systems?|techniques?|models?)\b"
    r"|\b(?:however|unfortunately|but)\b.{0,40}\b(?:fail|cannot|struggle|lack|limited)\b",
    re.IGNORECASE,
)

_NON_DEFINITION_RE = re.compile(
    r"\b(?:in\s+Table\s+\d|in\s+Figure\s+\d|see\s+(?:Table|Figure|Section)\s|"
    r"is\s+column\s+\d|the\s+rebuttal)\b",
    re.IGNORECASE,
)


# Classical plurals, both directions. A header reads "Ellipses" while the section that
# defines it is headed "Ellipsis Resolution"; the regular -s/-es rule reaches neither.
# These endings are pervasive in academic prose (analysis/analyses, basis/bases,
# hypothesis/hypotheses, matrix/matrices, index/indices, criterion/criteria,
# phenomenon/phenomena, corpus/corpora). Measured: four cells were reported as a column
# MARRS never defines, in a paper that defines it under a `\\paragraph` heading.
_CLASSICAL_PLURALS = (
    ("es", "is"), ("is", "es"),          # ellipses / ellipsis
    ("ices", "ix"), ("ices", "ex"),      # matrices / matrix, indices / index
    ("ix", "ices"), ("ex", "ices"),
    ("a", "on"), ("on", "a"),            # criteria / criterion
    ("a", "um"), ("um", "a"),            # data / datum
    ("ora", "us"), ("us", "ora"),        # corpora / corpus
)


def _word_stems(word: str) -> set[str]:
    """Spellings of one word that should be considered the same term.

    The regular ``-s`` stem plus the classical plural pairs above. Kept deliberately
    small: every extra form widens what counts as a mention of the column.
    """
    out = {word}
    if len(word) > 3 and word.endswith("s"):
        out.add(word[:-1])
    for suffix, other in _CLASSICAL_PLURALS:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            out.add(word[: -len(suffix)] + other)
    return out


@lru_cache(maxsize=512)
def _term_regex(variant: str) -> re.Pattern[str]:
    """Word-boundary matcher for one spelling, tolerating space/hyphen interchange.

    Cached: this is called once per sentence per variant while scanning a whole paper.
    """
    # Allow the ordinary inflections a paper uses when it discusses a column in prose:
    # a header reads "Train Data" while the body says "training data", "Weights" vs
    # "weight". Without this the column looks like one the paper never defined.
    parts = []
    for word in re.split(r"[\s-]+", variant):
        if not word:
            continue
        stems = sorted(_word_stems(word), key=len, reverse=True)
        alt = "|".join(re.escape(x) for x in stems)
        parts.append(f"(?:{alt})" + r"(?:s|es|ing|ed)?")
    # Allow punctuation BETWEEN the words: a header reads "Seq. Len." while the caption
    # writes "sequence length (Seq. Len)" — requiring only space/hyphen misses it.
    return re.compile(r"\b" + r"[\s\-.,/]+".join(parts) + r"\b", re.IGNORECASE)


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
                    and not _NON_DEFINITION_RE.search(body) \
                    and not _RESIDUAL_MARKUP_RE.search(body):
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
    raw = re.sub(r"\s+", " ", (header or "")).strip().lower()
    base = re.sub(r"[^\w\s-]", " ", raw).strip()
    base = re.sub(r"\s+", " ", base)
    if not base:
        return []
    forms = {base, base.replace("-", " "), base.replace("-", ""), base.replace(" ", "-")}
    # Keep the header as written too. Stripping every non-word character turns the group
    # name "SE(3)-equivariant" into "se 3 -equivariant", which matches nothing — the
    # parentheses are part of the term, not punctuation to be tolerated. Measured: USEEK
    # defines that column with an equation and eight cells were still reported as a column
    # the paper never defines.
    if raw and raw != base:
        forms.add(raw)
    # noun/adjective morphology: "aware" <-> "awareness", "adaptive" <-> "adaptation"
    for f in list(forms):
        if f.endswith("aware"):
            forms.add(f + "ness")
        if f.endswith("ive"):
            forms.add(f[:-3] + "ion")
    # Keep 2-character forms: real columns are abbreviated that hard ("SP", "CT Free").
    # Safe because matching is word-boundary anchored, not substring.
    return sorted({f for f in forms if len(f) >= 2}, key=len, reverse=True)


def _score_sentence(sentence: str, variants: list[str], *, in_definition_env: bool,
                    own_names: list[str] | None = None) -> int:
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
    if _RESIDUAL_MARKUP_RE.search(sentence):
        # Still carrying LaTeX after normalisation: the passage straddles a float, a
        # macro or a figure include, so what survived is a fragment rather than a
        # sentence — and a fragment read as a definition sends the judge after the wrong
        # property.
        score -= 3
    if _SELF_REFERENTIAL_RE.search(sentence):
        score -= 4  # about the citing paper's own method, not about the property
    if _PROBLEM_STATEMENT_RE.search(sentence):
        score -= 4  # states the gap the column exposes, not what earns a ✓
    if own_names and any(_term_regex(n).search(sentence) for n in own_names if len(n) > 2):
        score -= 4  # names the citing paper's method: a self-description, not a definition
    return score


# A finite verb or modal — the minimum a sentence needs to assert a CONDITION rather than
# just name a thing. Deliberately small: it only has to separate "example order in the
# prompt (Arrangement)" from "the watermarked content quality shall not be compromised".
_PREDICATE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|can|cannot|could|may|might|must|"
    r"shall|should|will|would|does|do|did|need|needs|requires?|ensures?|allows?|enables?|"
    r"means?|refers?|denotes?|indicates?|covers?|maps?|uses?|holds?)\b",
    re.IGNORECASE,
)


# "Thus, we prefer raw 3D inputs to multi-view images." — the authors' own design choice
# following from the property, not a condition a competitor must meet. Read as part of the
# criterion it convicts every method that made the other choice: USEEK's Anti-occlusion
# asks for repeatability under self-occlusion, and a multi-view detector was marked wrong
# purely for being multi-view.
_DESIGN_RATIONALE_RE = re.compile(
    r"^\s*(?:thus|hence|therefore|so|accordingly|as\s+such|for\s+this\s+reason)?[,\s]*"
    r"(?:we|our\s+\w+)\s+(?:prefer|choose|opt\s+for|adopt|use|favou?r|rely\s+on)\b",
    re.IGNORECASE,
)


def strip_design_rationale(gloss: str) -> str:
    """Drop the citing authors' own design choices from a column's definition.

    A desideratum is often stated and then followed by what the authors did about it, and
    only the first half is a condition on anyone else's method. Dropped whole sentences at
    a time — clause-level surgery trips over the abbreviation dots ("i.e.") these
    sentences are full of.
    """
    kept = [
        part for part in _SENT_SPLIT.split(gloss or "")
        if not _DESIGN_RATIONALE_RE.match(part.strip())
    ]
    out = re.sub(r"\s+", " ", " ".join(kept)).strip(" ,;")
    return out if len(out) >= 20 else (gloss or "")


def _states_a_criterion(quote: str, header: str) -> bool:
    """True when a gloss says what EARNS a mark, not merely what the column is called.

    A caption often just expands the abbreviation — "…example composition (Composition),
    and example order in the prompt (Arrangement)". That names the topic and stops. Handed
    to the judge as a definition it invites free association: "Arrangement" became "does the
    method model example ORDER", and a paper that merely *studied* order effects was
    reported as contradicting the table.

    A criterion shows up as a predicate ("the keypoints SHOULD be repeatable"), as a
    definitional cue ("X REFERS TO …"), or as the ``Header: …`` shape papers use for
    desiderata lists.
    """
    text = (quote or "").strip()
    if not text:
        return False
    if re.search(rf"{re.escape(header.strip())}\s*[:—–-]\s*\S", text, re.IGNORECASE):
        return True
    if any(pat.search(text) for pat, _w in _DEFINITIONAL):
        return True
    return bool(_PREDICATE_RE.search(text))


# The pivot of a "prior work does X. In contrast, ours does Y" construction. The column's
# meaning is on the LEFT of it; the citing paper's own claim is on the right.
_CONTRAST_PIVOT_RE = re.compile(
    r"\s*(?:\.\s+)?\b(?:in\s+contrast|by\s+contrast|however|whereas|unlike|"
    r"on\s+the\s+other\s+hand)\b[,;]?\s*",
    re.IGNORECASE,
)


def _before_own_claim(sentence: str, own_names: list[str] | None) -> str:
    """The part of a contrastive sentence that is about prior work, not about ours.

    "LLM-based methods generate simple features or refine only a single rule. In contrast,
    LLM-FE supports all four aspects." — the first clause is what the table's columns
    mean, and dropping the whole sentence because the second clause names the authors'
    method loses the only place the paper says it.

    Returns the sentence unchanged when it is not contrastive, and "" when the own-method
    claim is not confined to the right-hand side.
    """
    if not own_names:
        return sentence
    m = _CONTRAST_PIVOT_RE.search(sentence or "")
    if not m:
        return ""
    head = sentence[: m.start()]
    if not head.strip() or _describes_own_method(head, own_names):
        return ""
    return head.strip()


def _describes_own_method(sentence: str, own_names: list[str] | None) -> bool:
    """True when a sentence is about the citing paper's own contribution.

    Either it names one of the paper's own method names, or it opens with the
    first-person framing a contribution sentence uses.
    """
    if _SELF_CLAIM_RE.match((sentence or "").strip()):
        return True
    return bool(
        own_names and any(_term_regex(n).search(sentence) for n in own_names if len(n) > 2)
    )


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


# `\paragraph{Anaphora Resolution}` and its siblings are how a paper most often names the
# term it is about to define — the heading IS the definiendum. Rewritten to the
# `Term: …` shape the punctuation pass already scores, or the definition is invisible.
_HEADING_RE = re.compile(
    r"\\(?:paragraph|subsubsection|subsection|section)\*?\s*\{([^{}]{1,80})\}\s*\.?",
)
# Structure with no prose content. Left in, these dominate a snippet and the glosser
# rightly reports that the passages do not pin the term down — MARRS defines five columns
# by worked dialogues inside `verbatim` blocks, and all five came back undefined.
_TEX_NOISE_RE = re.compile(
    r"\\(?:label|ref|autoref|eqref|cref|vspace|hspace|centering|small|footnotesize|"
    r"tiny|scriptsize|noindent|clearpage|newpage|hline|toprule|midrule|bottomrule|"
    r"item(?:sep)?|linewidth|textwidth|columnwidth|input|include(?:graphics)?|"
    r"resizebox|caption\*?|footnote(?:size)?|url)\b\s*(?:\{[^{}]*\})?\*?"
    r"|\\(?:begin|end)\s*\{[^{}]*\}"
    r"|\$[^$]{0,120}\$"
)
# A worked example is a definition by demonstration; keep it as one unit so the sentence
# splitter cannot cut it after the first line.
# Citation and cross-reference commands: noise for our purposes, and universal, so they
# are removed rather than treated as a defect in the passage.
_CITATION_RE = re.compile(
    r"~?\\(?:cite|citep|citet|citealp|citeauthor|citeyear|nocite)\w*\s*"
    r"(?:\[[^\]]*\])*\s*\{[^{}]*\}"
)
_VERBATIM_RE = re.compile(
    r"\\begin\{(verbatim|lstlisting|quote|quotation)\}(.*?)\\end\{\1\}", re.DOTALL
)


def normalize_latex_prose(text: str) -> str:
    """Turn LaTeX source into prose the definition search can read.

    Three transformations, in order: worked examples are flattened to a single line so
    they survive sentence splitting; sectioning commands become ``Term:`` so a heading
    that names a column reads as its definition; the remaining structural markup is
    dropped. Text that is not LaTeX passes through untouched.
    """
    if not text or not _LOOKS_LIKE_LATEX.search(text):
        return text
    text = _VERBATIM_RE.sub(lambda m: " " + re.sub(r"\s+", " ", m.group(2)).strip() + " ", text)
    text = _CITATION_RE.sub(" ", text)
    text = _HEADING_RE.sub(lambda m: f" {m.group(1).strip().rstrip('.')}: ", text)
    text = _TEX_NOISE_RE.sub(" ", text)
    return text


def find_definition_snippets(
    header: str,
    body_text: str,
    *,
    caption: str = "",
    legend: list[str] | None = None,
    limit: int = 4,
    siblings: list[str] | None = None,
    own_names: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Passages that plausibly define ``header``, best first.

    Args:
        header: the column header as printed.
        body_text: the paper's text (LaTeX source or extracted PDF text).
        caption: the table caption — searched first and scored highest.
        legend: footnote/legend lines under the table.
        limit: max snippets to return.
        siblings: the other column headers — a passage reciting several of them is the
            table's own header row, not a definition.
        own_names: the citing paper's method name(s). A sentence that names its own
            method is describing that method, not the property, and using it as the
            definition asks every competitor whether it implements THIS paper's design.

    Returns:
        ``[(source, quote, score), …]`` where ``source`` is a :class:`GlossSource`
        value. Empty when the paper never mentions the header outside the table.
    """
    variants = header_variants(header)
    if not variants:
        return []
    found: list[tuple[str, str, int]] = []
    # Passages that use the term while describing the citing paper's own system. Excluded
    # from `found` — adopting one as the definition is how a table's punchline became a
    # column's criterion — but handed to the glosser as context.
    context: list[str] = []

    # Score the caption both sentence-by-sentence AND as a whole: an abbreviated header
    # ("Seq. Len.") makes the sentence splitter cut inside the very phrase being defined
    # — "…sequence length (Seq." | "Len), example composition…" — so the term is never
    # found in any single sentence.
    caption_units = [*_SENT_SPLIT.split(caption or ""), (caption or "")]
    for sent in caption_units:
        # A caption clause about the AUTHORS' OWN system is never a column definition —
        # it is the table's punchline ("CaT unifies prior work on program rewriting, code
        # generation, resource allocation … without needing a new DSL"). Adopted as the
        # gloss for "Program Rewriting" it turns the column into "does the cited method
        # avoid introducing a new DSL", which is a different question and one every
        # language-based baseline fails. Dropped outright rather than merely penalised:
        # the caption bonus is large enough that a penalty still leaves it on top.
        if _describes_own_method(sent, own_names):
            continue
        s = _score_sentence(sent, variants, in_definition_env=False, own_names=own_names)
        if s > 0:
            found.append((GlossSource.CAPTION.value, sent.strip(), s + 4))

    for line in legend or []:
        s = _score_sentence(line, variants, in_definition_env=False, own_names=own_names)
        if s > 0:
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
    rest = normalize_latex_prose(rest)

    # Definition environments are the strongest body signal; harvest them first, then
    # BLANK them out so the general scan neither re-reports them nor mis-attributes the
    # sentence that follows one (a match starting inside a stale span would be dropped).
    for m in _DEFINITION_ENV.finditer(rest):
        for sent in _SENT_SPLIT.split(m.group(2)):
            s = _score_sentence(sent, variants, in_definition_env=True, own_names=own_names)
            if s > 0:
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
            if _SELF_REFERENTIAL_RE.search(body) or _PROBLEM_STATEMENT_RE.search(body):
                continue
            if own_names and any(_term_regex(n).search(body) for n in own_names if len(n) > 2):
                continue
            found.append((GlossSource.BODY.value, f"{header}: {body}", 5))

    for m in re.finditer(r"[^.!?]{0,400}[.!?]", rest):
        sent = m.group(0).strip()
        if _describes_own_method(sent, own_names):
            # A contrastive sentence carries the column's meaning before the pivot and the
            # authors' own claim after it; keep the half that is about prior work.
            sent = _before_own_claim(sent, list(own_names or [])) or sent
        s = _score_sentence(sent, variants, in_definition_env=False, own_names=own_names)
        if s <= 0 and any(_term_regex(v).search(sent) for v in variants):
            # Uses the term but does not define it — a self-description ("we instruct the
            # LLM to generate complex features") or the gap the column exposes ("existing
            # methods generate simple features"). Both say what the column is ABOUT, and
            # neither may be the gloss: as a definition the first asks competitors whether
            # they implement this paper's design, the second states a problem rather than
            # a criterion. Passed to the glosser as background instead, which is told to
            # read them for the subject and never adopt them as the definition.
            context.append(sent)
            continue
        if s >= 4:  # definitional phrasing
            found.append((GlossSource.BODY.value, sent, s))
        elif s > 0:  # a plain mention — weak, but proof the paper does discuss the term
            found.append((GlossSource.MENTION.value, sent, s))

    others = [s for s in (siblings or []) if s and s.strip().lower() != (header or "").strip().lower()]
    if others:
        # BODY text only. A caption legitimately names every column ("query dependence
        # (Dynamic), sequence length (Seq. Len), example composition (Composition) …") —
        # that IS the definition, not a recital of the table, and rejecting it left
        # abbreviated headers looking undefined.
        found = [
            f for f in found
            if f[0] in (GlossSource.CAPTION.value, GlossSource.LEGEND.value)
            or not _recites_the_table(f[1], others)
        ]
    # A gloss that only NAMES the column is not a definition of it, whatever it was found
    # in. Demoted rather than dropped: it is still the best pointer the paper offers, and
    # the caller decides what a weak gloss may license.
    found = [
        f if _states_a_criterion(f[1], header) else (GlossSource.MENTION.value, f[1], f[2] - 3)
        for f in found
    ]
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
    for sent in context[:2]:
        out.append((_SELF_CONTEXT, re.sub(r"\s+", " ", sent)[:400], 0))
    return out


def _qualify_by_parent(
    snips: list[tuple[str, str, int]], parent: str
) -> list[tuple[str, str, int]]:
    """Keep leaf-term snippets that stay inside the group header's context.

    A stacked header means the leaf ALONE is not the column. ``Retargetability — Resource
    Constraints`` asks whether the compiler can be retargeted to a machine with different
    resource constraints; searching the body for the bare leaf finds "failing to meet any
    of the resource constraints means the program cannot be run" — a true statement about
    the domain that says nothing about retargeting, and one that turns every baseline into
    a false accusation.

    Snippets that mention the parent are kept as-is. When none does, the best guess is
    kept but demoted to ``mention``: the paper never defined this column in its group
    context, and the caller must not treat a guess as a definition.
    """
    if not parent:
        return snips
    kept = [t for t in snips if t[0] == _SELF_CONTEXT or _term_regex(parent).search(t[1])]
    if any(t[0] not in (_SELF_CONTEXT,) for t in kept):
        return kept
    return [
        t if t[0] == _SELF_CONTEXT else (GlossSource.MENTION.value, t[1], t[2])
        for t in snips
    ]


# Marks a payload passage as background about the citing paper rather than a candidate
# definition. Not a GlossSource: it never labels a gloss, only a passage.
_SELF_CONTEXT = "self-context"

# Gloss grades an empty answer from the glosser may overturn: exactly those the keyword
# search itself flagged as uncertain. A definition the paper states outright stands.
_GLOSSER_MAY_VETO = frozenset({GlossSource.MENTION.value, GlossSource.RECOVERED.value})

_STOPWORDS = frozenset(
    "that this with from which have been they their there them when what where must "
    "than then some such only also into over under about because while whether does "
    "each other more most many both same able upon".split()
)


def _content_words(text: str) -> set[str]:
    """Substantive words of a passage, for checking one text against another."""
    return {
        w for w in re.findall(r"[A-Za-z][A-Za-z-]{3,}", (text or "").lower())
        if w not in _STOPWORDS
    }


def _is_grounded_in(gloss: str, col: dict) -> bool:
    """True when a model-written gloss reuses the material it was given.

    The glosser is told to define a column only from the caption, the legend and the
    retrieved passages. This checks that it did: a gloss sharing no substantive vocabulary
    with any of them was written from the model's own knowledge of what the term usually
    means, which is exactly the failure the deterministic path exists to avoid.
    """
    material = " ".join(
        [
            col.get("caption") or "",
            " ".join(col.get("legend") or []),
            " ".join(s.get("quote") or "" for s in col.get("snippets") or []),
            col.get("header") or "",
        ]
    )
    shared = _content_words(gloss) & _content_words(material)
    return len(shared) >= 2


def resolve_dimensions(
    dimensions: list[Dimension],
    body_text: str,
    *,
    caption: str = "",
    legend: list[str] | None = None,
    glosser: Callable[[list[dict]], list[dict]] | None = None,
    own_names: set[str] | list[str] | None = None,
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
        own_names: the citing paper's own method name(s), so a sentence describing that
            method is not adopted as the column's definition.

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
        siblings = [d.header for d in dimensions if d is not dim]
        # A stacked header merges a group with its leaf ("Efficiency — Fast"); the paper
        # defines the LEAF, so the leaf is searched too — but only in the BODY. Against
        # the caption a lone generic leaf ("… — Features") matches any sentence that
        # happens to use the word ("Comparison of existing feature engineering methods"),
        # and that sentence then becomes the column's definition.
        parts = [p.strip() for p in re.split(r"\s+—\s+", dim.header) if p.strip()]
        full_terms = [name] if name else [dim.header]
        leaf_terms = [] if name else [p for p in reversed(parts) if p != dim.header]
        snips = []
        for term in full_terms:
            snips = find_definition_snippets(
                term, body_text, caption=caption, legend=legend, siblings=siblings,
                own_names=list(own_names or []),
            )
            if any(t[0] != _SELF_CONTEXT for t in snips):
                break
        if not any(t[0] != _SELF_CONTEXT for t in snips):
            parent = parts[0] if len(parts) > 1 else ""
            for term in leaf_terms:
                snips = find_definition_snippets(
                    term, body_text, caption="", legend=None, siblings=siblings,
                    own_names=list(own_names or []),
                )
                if any(t[0] != _SELF_CONTEXT for t in snips):
                    snips = _qualify_by_parent(snips, parent)
                    break
        # The list item that names the column is its definition by construction, so it
        # outranks whatever else in the paper happens to contain the word.
        if name and len(name_quote) > len(name) + 15:
            snips = [(GlossSource.BODY.value, name_quote, 10), *snips]
        elif name and not any(t[0] != _SELF_CONTEXT for t in snips):
            snips = [(GlossSource.BODY.value, name_quote, 5)]
        # Context passages ride along for the glosser only — never as the gloss itself.
        best = next((t for t in snips if t[0] != _SELF_CONTEXT), None)
        if best:
            src, quote, _ = best
            dim.gloss_source = src
            dim.gloss_quote = quote
            # deterministic fallback; refined by the glosser below
            dim.gloss = strip_design_rationale(quote)
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
                # The other columns of the same table. A gloss is only useful if it
                # DISTINGUISHES this column from its neighbours — MARRS lists "Anaphora",
                # "Ellipses" and "Conversational Entity Resolution" side by side, and a
                # gloss that fits all three convicts a cited work of missing a capability
                # that a different column is about.
                "siblings": siblings,
                "legend": list(legend or []),
                "snippets": [{"source": s, "quote": q} for s, q, _ in snips],
            }
        )

    if glosser is None or not payload:
        return dimensions

    try:
        refined = glosser(payload)
    except Exception:  # noqa: BLE001 — degrade to the deterministic gloss
        return dimensions
    for dim, col, got in zip(dimensions, payload, refined or [], strict=False):
        if not isinstance(got, dict):
            continue
        gloss = (got.get("gloss") or "").strip()
        question = (got.get("test_question") or "").strip()
        if gloss and dim.gloss_source != GlossSource.HEADER_ONLY.value:
            dim.gloss = gloss
        elif gloss and _is_grounded_in(gloss, col):
            # The glosser can reach a column the keyword search could not — a header the
            # caption expands ("Seq. Len."), or one whose meaning is carried by a symbol
            # legend rather than by prose. Refusing its answer for exactly those columns
            # left the best-informed source unused where it was needed most. Graded
            # `mention`, never `caption`: recovered by a model from indirect material is
            # good enough to check a cell against, not good enough to accuse the authors.
            dim.gloss = gloss
            dim.gloss_source = GlossSource.RECOVERED.value
        elif not gloss and dim.gloss_source in _GLOSSER_MAY_VETO:
            # An EMPTY gloss is the glosser's verdict that the passages do not pin the
            # term down, and it outranks whatever the keyword search guessed. Without this
            # the guess survives: LLM-FE's four columns were "defined" by ablation results
            # ("Without domain knowledge, the performance drops to 0"), which the judge
            # then applied to twenty cells of other people's methods.
            #
            # The veto is scoped to the grades the deterministic path was already unsure
            # of. An explicit definition — a `\\paragraph{Term}` heading, an "X: …" list
            # item, a definition environment — is a fact about the paper, and one
            # nondeterministic call declining to restate it must not erase it. Measured:
            # a run where the glosser returned nothing for two columns MARRS defines under
            # such headings moved twelve cells to "the paper never defines this column",
            # while the previous run glossed the same two columns without trouble.
            dim.gloss = ""
            dim.gloss_quote = ""
            dim.gloss_source = GlossSource.HEADER_ONLY.value
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
