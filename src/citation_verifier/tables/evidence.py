"""
tables/evidence.py — retrieve the CITED paper's own text for cell verification.

A table cell asserts something about another paper, so the evidence has to come from
that paper. This wires the existing grounding layer into the ``evidence_for`` seam of
:func:`citation_verifier.tables.verify.verify_table`.

Why a dedicated builder rather than "just use the abstract":

  * **Abstracts are truncated.** The lookup layer caps a fetched abstract at 600
    characters, roughly half of a typical one, and the sentence that settles a property
    is as likely to be in the second half as the first.
  * **Table properties are rarely in the abstract at all.** "Needs no retraining",
    "requires access to the training data", "is SE(3)-equivariant" are method details
    that live in the body. Judging them on an abstract yields honest-but-useless
    ``unverifiable`` for most cells.
  * **The query is the COLUMN, not the row.** Excerpts must be selected against what the
    property means; searching the cited paper for its own name finds nothing useful.

Retrieval order per reference: resolve the citation, then arXiv full text (HTML → LaTeX
e-print → PDF) or, for non-arXiv work, the open-access text behind its URL/DOI. Every
step is fail-soft — with nothing retrieved the cells become ``unverifiable``, which is
the honest outcome, never a refutation.

Both external dependencies (``lookup``, ``resolver``) are injected, so this module is
testable offline.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

__all__ = ["build_evidence_provider", "compose_evidence"]

# Enough for every column of a wide table to carry several excerpts. The judge reads this
# once per row, and a formal column ("the function is SE(3)-equivariant if …") can only be
# decided from the one or two sentences that state the mechanism — starving it is how a
# cell becomes unverifiable for want of a paragraph the paper does contain.
_MAX_EVIDENCE_CHARS = 40000
# A cited paper is sent WHOLE rather than as excerpts. Retrieval is a source of error in
# its own right — a property the paper does discuss is settled by a passage the search did
# not rank, and a property it never discusses cannot be told apart from one whose passages
# were missed. A whole paper removes both failure modes.
#
# One request carries at most `_MAX_REQUEST_CHARS`. Papers in the corpora measured so far
# run 25k-130k characters and fit in one; a longer one is cut into at most
# `_MAX_SEGMENTS` parts and the same question is asked of each part, the answers combined
# afterwards. Beyond three parts the excerpt path applies instead.
_MAX_REQUEST_CHARS = 120_000
_MAX_SEGMENTS = 3
_MAX_WHOLE_PAPER_CHARS = _MAX_REQUEST_CHARS * _MAX_SEGMENTS
# Passages pulled in per property by the recall pass, whatever their rank. Raised with the
# wider net: a property searched under ten wordings can be mentioned in more places.
_RECALL_PER_PROPERTY = 6
# Passages describing HOW the cited work works, included whatever the properties are.
_CONSTRUCTION_PASSAGES = 8


def _dimension_queries(dimensions: list[Any]) -> list[str]:
    """One retrieval query per column: what the column asserts, not its terse header."""
    out: list[str] = []
    for d in dimensions or []:
        gloss = (getattr(d, "gloss", "") or "").strip()
        header = (getattr(d, "header", "") or "").strip()
        q = f"{header}. {gloss}" if gloss else header
        # The other wordings widen the lexical overlap the ranking runs on: a passage
        # about "rotation invariance" scores for an "SE(3)-equivariant" column only if
        # the query carries those words.
        terms = [str(t) for t in (getattr(d, "search_terms", None) or []) if str(t).strip()]
        if terms:
            q = f"{q} ({'; '.join(terms)})"
        if q.strip():
            out.append(q.strip())
    return out


def split_paper(text: str, *, max_chars: int = _MAX_REQUEST_CHARS,
                max_segments: int = _MAX_SEGMENTS) -> list[str]:
    """``text`` in as few parts as fit ``max_chars``, cut at paragraph boundaries.

    Returns one part for a paper that fits. Cuts never fall mid-paragraph, so no part
    starts or ends inside a sentence the judge then has to guess at. ``[]`` when even
    ``max_segments`` parts would not fit — the caller then falls back to excerpts.
    """
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text else []
    want = min(max_segments, -(-len(text) // max_chars))
    if want > max_segments or len(text) > max_chars * max_segments:
        return []
    target = -(-len(text) // want)  # even parts, so none is a sliver
    out: list[str] = []
    start = 0
    while start < len(text) and len(out) < want - 1:
        cut = text.find("\n\n", start + target)
        if cut == -1 or cut - start > max_chars:
            cut = text.rfind("\n\n", start, start + max_chars)
        if cut == -1 or cut <= start:
            cut = min(start + max_chars, len(text))
        out.append(text[start:cut].strip())
        start = cut
    out.append(text[start:].strip())
    return [p for p in out if p]


def compose_whole_paper(
    title: str, abstract: str, text: str, *, part: int = 1, of: int = 1
) -> str:
    """The cited paper — entire, or one part of it — as the judge's evidence.

    No excerpts and no retrieval: the judge is looking at everything the paper says, so
    "the paper never claims this" is something it can establish by reading rather than
    something the retrieval code has to vouch for. A paper too long for one request is
    split (:func:`split_paper`) and each part carries its own coverage line, because
    "not claimed" then means "not claimed IN THIS PART" and only the caller, holding
    every part's answer, can conclude anything about the paper.
    """
    if of > 1:
        head = (
            f"COVERAGE: PART {part} OF {of} of the complete text of the cited paper "
            f"({len(text):,} characters of it). The paper was too long for one request "
            f"and was cut into {of} parts at paragraph boundaries; you are being asked "
            f"the same question about each part separately, and the answers are combined "
            f"afterwards. So judge THIS PART only: 'absent' here means this part carries "
            f"no such claim, not that the paper carries none."
        )
    else:
        head = (
            "COVERAGE: the COMPLETE text of the cited paper follows "
            f"({len(text):,} characters). Nothing has been selected, ranked or left out, "
            "so what the paper does not say here, it does not say anywhere."
        )
    parts = [head, f"TITLE: {title}".strip()]
    if abstract and abstract.strip() not in text:
        parts.append(f"ABSTRACT: {abstract}")
    parts.append(f"FULL TEXT:\n{text}")
    return "\n\n".join(p for p in parts if p.strip())


def compose_evidence(
    title: str,
    abstract: str,
    chunks: list[tuple[str, str]],
    *,
    max_chars: int = _MAX_EVIDENCE_CHARS,
    total_chars: int = 0,
    named: dict[int, bool] | None = None,
    search: dict[int, dict[str, Any]] | None = None,
) -> str:
    """Assemble the evidence block sent to the judge, tagged by provenance.

    Section tags are kept so the judge can say where it read something, and duplicate
    excerpts (the same passage selected by two columns) are dropped.

    The block opens with how much of the paper it covers, because that changes what
    silence means — and it must say so TRUTHFULLY. An earlier version announced "full
    text" whenever any excerpt existed; the judge then reasoned "across the full text the
    paper never discusses occlusion" from five excerpts covering a quarter of a paper in
    which "occluded" appears four times, and a ✗ was confirmed on that basis. Now the line
    gives the excerpt count against the paper's length, and states per property whether
    the paper names it anywhere at all — the one fact about absence the code can vouch for,
    because every passage naming a property is pulled in (see ``_full_text_chunks``).
    """
    if chunks:
        n_named = sum(1 for h, _t in chunks if h.startswith(_NAMES_TAG))
        n_built = sum(1 for h, _t in chunks if h.startswith(_CONSTRUCTION_TAG))
        coverage = (
            f"COVERAGE: {len(chunks)} excerpts selected from a {total_chars:,}-character "
            f"paper — NOT the whole paper. Passages in which the paper mentions a "
            f"property — by its column name or by any of the other wordings listed "
            f"below — are included first ({n_named} such passages). A property marked "
            f"'mentioned in paper: no' is one the paper discusses under none of those "
            f"wordings anywhere; 'all shown' means every passage that mentions it is in "
            f"front of you. Excerpts tagged '{_CONSTRUCTION_TAG}' ({n_built}) describe how "
            f"the work is built and what it was trained on, and are included whatever the "
            f"properties are — a property is often settled by the construction rather than "
            f"by anything the paper says about the property itself."
        )
        if named:
            lines = []
            for k, v in sorted(named.items()):
                rec = (search or {}).get(k) or {}
                terms = ", ".join(rec.get("terms") or [])
                n, shown = int(rec.get("mentions") or 0), int(rec.get("shown") or 0)
                count = ""
                if v and n:
                    count = f"; {n} mention{'s' if n != 1 else ''}, " + (
                        "all shown" if shown >= n else f"{shown} shown"
                    )
                lines.append(
                    f"mentioned in paper: {'yes' if v else 'no'} — property #{k}"
                    + (f" (searched for: {terms}{count})" if terms else "")
                )
            coverage += "\n" + "\n".join(lines)
    else:
        coverage = "COVERAGE: title and abstract only — the body was not retrieved"
    parts = [coverage, f"TITLE: {title}".strip()]
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    seen: set[str] = set()
    body: list[str] = []
    for heading, text in chunks:
        key = re.sub(r"\W+", "", (text or "").lower())[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        body.append(f"[{heading or 'body'}] {text}")
    if body:
        head = "\n\n".join(p for p in parts if p.strip())
        budget = max(0, max_chars - len(head) - len("\n\nFULL TEXT EXCERPTS:\n"))
        parts.append("FULL TEXT EXCERPTS:\n" + "\n".join(_fit(body, budget)))
    out = "\n\n".join(p for p in parts if p.strip())
    return out[:max_chars]


def _fit(excerpts: list[str], budget: int) -> list[str]:
    """As many excerpts as fit in ``budget``, in the order given.

    Fairness lives in that order: :func:`_full_text_chunks` interleaves the columns, so
    truncating the tail drops everyone's Nth-best excerpt rather than the last columns'
    only one.
    """
    kept: list[str] = []
    used = 0
    for x in excerpts:
        if used + len(x) + 1 > budget:
            break
        kept.append(x)
        used += len(x) + 1
    return kept or excerpts[:1]


def build_evidence_provider(
    *,
    lookup: Callable[[str | None], Any],
    resolver: Any,
    dimensions: list[Any],
    whole_paper: bool = True,
    use_full_text: bool = True,
    chunks_per_query: int = 5,
    chunk_chars: int = 700,
    retries: int = 3,
    pace_seconds: float = 1.0,
) -> Callable[[str | None, str], tuple[str, str, dict]]:
    """Build an ``evidence_for(cite_key, row_label) -> (evidence, source, meta)`` callable.

    By default the cited paper is sent WHOLE (``whole_paper``): retrieval is a source of
    error in its own right, and a judge holding the entire paper can establish "this work
    never claims that" by reading rather than on the retrieval code's word. ``meta`` then
    carries ``{"complete": True}``.

    When the paper is too long, or only its abstract could be fetched, the excerpt path
    applies instead and ``meta["named"]`` maps column index to whether the cited paper
    mentions that property anywhere — what lets the verdict stage refuse a judge's
    "absent" for a property that appears in a passage the judge was not shown.

    Args:
        lookup: ``cite_key`` (or PDF marker) -> the reference as written, with a ``raw``
            attribute. Return ``None`` when the key is not in the bibliography.
        resolver: object with ``resolve(key, reference)`` returning a canonical record
            (``title`` / ``abstract`` / ``arxiv_id`` / ``url``), or ``None``.
        dimensions: the table's columns — their glosses become the retrieval queries
            when excerpts are used.
        whole_paper: send the entire cited paper when it fits in
            :data:`_MAX_WHOLE_PAPER_CHARS`. Off restores the excerpt path unconditionally.
        use_full_text: fetch the cited paper's body. Off makes every cell abstract-only.
        chunks_per_query: excerpts to keep per column. Two was too few for a column whose
            criterion is an equation: whether the cell could be decided at all came down to
            whether chunk scoring happened to surface the sentence carrying it, and the
            same paper judged three different ways across three runs.
        chunk_chars: max characters per excerpt.
        retries: attempts before giving up on a reference. The metadata APIs rate-limit
            a burst of lookups and simply return nothing, which is indistinguishable
            from "this paper does not exist" — measured: the same key failed when
            queried back-to-back and resolved on every attempt once spaced out. Without
            retrying, a fifth of the rows silently become ``unverifiable``.
        pace_seconds: delay before each retry (doubled each time).

    Returns:
        A cached provider. Never raises: any failure yields ``("", key)`` and the cells
        become ``unverifiable``.
    """
    cache: dict[str | None, tuple[str, str, dict]] = {}

    def evidence_for(cite_key: str | None, row_label: str) -> tuple[str, str, dict]:
        if cite_key in cache:
            return cache[cite_key]
        result: tuple[str, str, dict] = ("", str(cite_key or row_label), {})
        try:
            ref = lookup(cite_key)
            reference = (getattr(ref, "raw", "") or "") if ref is not None else ""
            got = None
            if ref is not None:
                for attempt in range(max(1, retries)):
                    got = resolver.resolve(str(cite_key or ""), reference or row_label)
                    if got is not None:
                        break
                    if attempt + 1 < max(1, retries):
                        time.sleep(pace_seconds * (2**attempt))
            if got is not None:
                title = getattr(got, "title", "") or ""
                abstract = getattr(got, "abstract", "") or ""
                source = getattr(got, "url", "") or str(cite_key or "")
                text = _full_text_of(got) if use_full_text else ""
                segments = split_paper(text) if whole_paper else []
                if segments:
                    composed = [
                        compose_whole_paper(title, abstract, seg, part=i + 1, of=len(segments))
                        for i, seg in enumerate(segments)
                    ]
                    result = (
                        composed[0],
                        source,
                        {"complete": True, "total_chars": len(text),
                         "segments": composed if len(composed) > 1 else []},
                    )
                else:
                    chunks: list[tuple[str, str]] = []
                    named: dict[int, bool] = {}
                    search: dict[int, dict[str, Any]] = {}
                    total = 0
                    if text and dimensions:
                        chunks, named, total, search = _full_text_chunks(
                            got, list(dimensions), chunks_per_query, chunk_chars, text=text
                        )
                    result = (
                        compose_evidence(
                            title, abstract, chunks, total_chars=total, named=named,
                            search=search,
                        ),
                        source,
                        {"named": named, "search": search, "total_chars": total,
                         "excerpts": len(chunks), "complete": False},
                    )
        except Exception:  # noqa: BLE001 — retrieval is best-effort by contract
            result = ("", str(cite_key or row_label), {})
        cache[cite_key] = result
        return result

    return evidence_for


def _full_text_of(resolved: Any) -> str:
    """The cited paper's body: :func:`citation_verifier.grounding.fulltext.full_text_of`."""
    from ..grounding.fulltext import full_text_of  # noqa: PLC0415 — lazy: keeps import network-free

    return full_text_of(resolved)


_NAMES_TAG = "names the property"

_STOP = frozenset(
    "a an the of to in on for by with and or from at as is are be that this its into "
    "under over across between without within via per".split()
)


def _stem5(word: str) -> str:
    return word[:5]


def _search_stem(word: str) -> str:
    """The stem a wording is matched by, when the word has a suffix worth stripping.

    "equivariant" -> "equivari" (so equivariance matches), "occlusion" -> "occlu" (so
    occluded matches), "features" -> "featur". A word with no such suffix keeps ALL its
    letters bar a final "y" or "e" (entity -> entit, so entities matches): truncating it
    to a fixed five would match unrelated words that merely start the same way —
    "complex" cut to "compl" matched "complete" and "compliance", and the passages it
    dragged in were counted as the paper discussing the property.
    """
    from .dimensions import _derived_stem  # noqa: PLC0415 — sibling module, no cycle

    stem = _derived_stem(word)
    if stem:
        return stem
    return word[:-1] if len(word) > 4 and word[-1] in "ye" else word


# Page chrome that reaches the text of an HTML-served paper: the site's own navigation, a
# table of contents, the author/affiliation block. Every one of them "mentions" whatever
# the section titles mention, and they were filling the recall slots meant for prose.
_CHROME_RE = re.compile(
    r"Report (?:GitHub )?Issue|Download PDF|Back to Abstract|Content selection saved"
    r"|HTML versions? of papers|Instructions for reporting errors|report(?:ing)? errors"
    r"|arXiv is now an independent|Why HTML\?|Back to arXiv|Submit without GitHub"
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"            # an e-mail address
    r"|\bAffiliation:|\bEmail:"
    r"|(?:\b\d+(?:\.\d+)*\s+[A-Z][A-Za-z-]+(?:\s+[A-Za-z-]+){0,5}\s+){3,}",  # a run of numbered headings
)


def _term_pattern(term: str) -> str | None:
    """A regex for one wording of a property: a word by its stem, a phrase by its words.

    A recall net, not a ranker, so crude is fine: "occlusion" -> "occlu" also matches
    "occluded", "rotation invariant" matches "rotation-invariant" and "invariant to
    rotations" alike (each content word by stem, in any order, within the sentence).
    Short tokens ("SE(3)", "LRF", "fast") match exactly, whole-word, since a five-letter
    stem of them is the token itself.
    """
    words = [w for w in re.findall(r"[a-z0-9()]+", (term or "").lower()) if w not in _STOP]
    if not words:
        return None
    parts = []
    for w in words:
        alpha = re.sub(r"[^a-z]", "", w)
        if len(alpha) >= 5:
            parts.append(rf"\b{re.escape(_search_stem(alpha))}[a-z]*")
        else:
            parts.append(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])")
    if len(parts) == 1:
        return parts[0]
    return "".join(f"(?=.*{p})" for p in parts)


def _header_phrase(header: str) -> str:
    """The header as ONE search wording: its substantial words, matched together.

    A column name is a phrase, and a sentence that discusses the property contains the
    phrase, not merely one of its words. Searching the words SEPARATELY made a generic
    one carry the whole term: "Screen Entity Resolution" matched every sentence of a
    coreference paper containing "resolution", the paper was recorded as mentioning the
    property fifteen times, and the judge — shown six of those fifteen and told the rest
    existed — could not conclude anything at all.

    Words shorter than five letters are dropped rather than required: they are the
    affixes and function words a header compresses into ("Anti-occlusion" is discussed as
    "occluded", never as "anti"), so requiring them would find nothing. A header left
    with a single word is that word.
    """
    return " ".join(w for w in re.findall(r"[A-Za-z]+", header or "") if len(w) >= 5)


def property_terms(dim: Any) -> list[str]:
    """Every wording to search a cited paper for: the header's words plus the glosser's
    other wordings of the property (see :attr:`Dimension.search_terms`)."""
    phrase = _header_phrase((getattr(dim, "header", "") or "").strip())
    out: list[str] = [phrase] if phrase else []
    for t in getattr(dim, "search_terms", None) or []:
        t = clean_term(t)
        if t and t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return out


def _property_name_words(header: str) -> list[str]:
    """The content words of a column header, as 5-letter stems (kept for callers)."""
    words = re.findall(r"[a-z]+", (header or "").lower())
    return sorted({_stem5(w) for w in words if len(w) >= 5})


# A line that is a heading, a table-of-contents entry or a navigation label rather than
# prose: short, no terminal punctuation, numbered or title-cased. Text served as HTML
# arrives with the site's table of contents and menus as such lines, and joined into
# "sentences" they mention every section title in the paper.
def _is_toc_line(line: str) -> bool:
    ln = line.strip()
    if not ln or len(ln) > 90 or ln[-1] in ".!?;:":
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*|[A-Z])\s+\S", ln):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z-]*", ln)
    if not words or len(words) > 10:
        return False
    capitalised = sum(1 for w in words if w[0].isupper())
    return capitalised / len(words) >= 0.6


_CONSTRUCTION_TAG = "how the work works"
# What a paper calls the section where it says what it built and what it trained on.
_CONSTRUCTION_HEADINGS = (
    "method", "methodology", "approach", "model", "architecture", "framework",
    "network", "system", "algorithm", "formulation", "implementation", "training",
    "learning", "data", "dataset", "annotation", "setup", "our ", "proposed",
    "preliminar", "problem", "overview", "pipeline", "design",
)
# Sentences that state the construction rather than motivate or evaluate it.
_CONSTRUCTION_CUE_RE = re.compile(
    r"\b(?:we\s+(?:use|train|learn|adopt|employ|build|design|construct|define|apply|"
    r"annotat\w*|label|optimi[sz]e|minimi[sz]e|feed|take|predict|estimate|represent)|"
    r"our\s+\w+\s+(?:is|are|uses|takes|predicts|consists|comprises|outputs)|"
    r"is\s+(?:trained|learned|supervised|implemented|parameteri[sz]ed|composed|based)|"
    r"are\s+(?:trained|learned|supervised|annotated|labelled|labeled|obtained)|"
    r"consists?\s+of|takes?\s+as\s+input|the\s+(?:input|output|loss|objective|"
    r"backbone|encoder|decoder|detector|network|architecture)\s+is|"
    r"requires?\s+(?:human|manual|ground[- ]truth|annotat\w*|label))",
    re.IGNORECASE,
)


def _construction_passages(sections: list[tuple[str, str]], *, limit: int,
                           max_chars: int) -> list[tuple[str, str]]:
    """Passages in which the paper says what its method IS and what it was trained on.

    A comparison-table property is often settled by the cited work's construction rather
    than by anything it says about the property: a detector trained on hand-annotated
    keypoints is not unsupervised, and a crop-and-regress CNN carries no equivariance
    guarantee — neither paper needs to use the word. Those passages are never retrieved by
    a property-driven search (the property's words are not in them), so they are collected
    separately and always sent, whatever the columns are.
    """
    def scan(only_method_sections: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for heading, body in sections:
            key = re.sub(r"[^a-z ]", " ", (heading or "").lower())
            if only_method_sections and not any(h in key for h in _CONSTRUCTION_HEADINGS):
                continue
            for sent in re.split(r"(?<=[.!?])\s+", _prose_lines(body)):
                sent = re.sub(r"\s+", " ", sent).strip()
                if len(sent) < 40 or len(sent) > 1200 or _CHROME_RE.search(sent):
                    continue
                if not _CONSTRUCTION_CUE_RE.search(sent):
                    continue
                k = re.sub(r"\W+", "", sent.lower())[:120]
                if k in seen:
                    continue
                seen.add(k)
                out.append((f"{_CONSTRUCTION_TAG} — {heading or 'body'}", sent[:max_chars]))
                if len(out) >= limit:
                    return out
        return out

    got = scan(only_method_sections=True)
    # Papers head their method section whatever they like ("Watermarking Scheme", "CATER",
    # a bare number in extracted PDF text). When the headings gave us next to nothing, the
    # construction sentences are still in there — take them from anywhere.
    return got if len(got) * 2 >= limit else scan(only_method_sections=False)


def _prose_lines(text: str) -> str:
    """``text`` with heading / table-of-contents / navigation lines removed."""
    return "\n".join(ln for ln in (text or "").split("\n") if not _is_toc_line(ln))


_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*$", re.MULTILINE
)


def _body_without_references(text: str) -> str:
    """The paper up to its reference list.

    A bibliography entry that happens to carry the word ("Efficient estimation of word
    representations …") is not the paper mentioning the property, and it was the only
    "mention" behind more than one 'the paper does discuss this' flag. The cut is taken
    at the LAST references heading in the back half of the text, so a paper whose
    related-work section is headed "References to prior art" is not truncated.
    """
    t = text or ""
    cut = None
    for m in _REFERENCES_HEADING_RE.finditer(t):
        if m.start() >= len(t) * 0.5:
            cut = m.start()
    return t[:cut] if cut else t


_ANNOTATION_RE = re.compile(r"\s*\([^()]*\)\s*$")


def clean_term(term: str) -> str:
    """A search wording without a trailing annotation: "single-rule refinement (absence)"
    -> "single-rule refinement". The glosser is asked for the phrasing of a property's
    absence and sometimes labels it; the label is not part of the wording."""
    return _ANNOTATION_RE.sub("", re.sub(r"\s+", " ", str(term or ""))).strip(" .,;:\"'")


def _passages_naming(
    text: str, terms: list[str] | str, *, limit: int, max_chars: int
) -> list[str]:
    """Sentences mentioning the property under any of ``terms``, at most ``limit``."""
    return _mentions(text, terms, limit=limit, max_chars=max_chars)[0]


def _mentions(
    text: str, terms: list[str] | str, *, limit: int, max_chars: int
) -> tuple[list[str], int]:
    """Sentences in which the paper mentions this property under any of ``terms``.

    The column's name is the CITING paper's vocabulary; the cited paper has its own. A
    detector "proven to be rotation invariant" never says "equivariant", and a method
    that "handles missing parts" never says "occlusion" — so the net is cast over every
    wording the glosser could think of, not the header alone. Cheap over-inclusion: an
    extra passage costs the judge a paragraph, a missed one cost a wrong verdict.
    """
    if isinstance(terms, str):  # a bare header: its words, matched together
        terms = [_header_phrase(terms)]
    pats = [p for p in (_term_pattern(t) for t in terms or []) if p]
    if not pats:
        return [], 0
    out: list[str] = []
    total = 0
    for sent in re.split(r"(?<=[.!?])\s+", _prose_lines(_body_without_references(text))):
        low = sent.lower()
        if len(low) > 2000 or _CHROME_RE.search(sent):
            continue  # a flattened table or page chrome — not prose
        if any(re.search(p, low) for p in pats):
            total += 1
            if len(out) < limit:
                out.append(re.sub(r"\s+", " ", sent).strip()[:max_chars])
    return out, total


def _full_text_chunks(
    resolved: Any, dimensions: list[Any], per_query: int, chunk_chars: int, *, text: str = ""
) -> tuple[list[tuple[str, str]], dict[int, bool], int, dict[int, dict[str, Any]]]:
    """Excerpts of the cited paper's body, plus what the code can vouch for about them.

    Two passes. Relevance ranking picks ``per_query`` excerpts per column — but it is
    lexical overlap with the gloss's wording, and "self-occlusion" does not match
    "occluded", so the one sentence that decides a cell can rank outside the top few.
    A second pass therefore adds every passage that names the property by (a stem of)
    its column name, so silence in the excerpts about a NAMED property never happens.

    Returns ``(chunks, named, total_chars, search)`` where ``named[col_index]`` says
    whether the paper mentions that property anywhere, and ``search[col_index]`` records
    the wordings tried, how many passages mention one of them (``mentions``) and how many
    of those are in ``chunks`` (``shown``). When every mention is shown, the judge has
    read all the paper says under those wordings and may conclude absence from them; when
    only some are, it may not.
    """
    text = text or _full_text_of(resolved)
    if not text:
        return [], {}, 0, {}
    from ..grounding.fulltext import (  # noqa: PLC0415 — lazy: keeps import network-free
        select_evidence_chunks,
        split_sections,
    )
    queries = _dimension_queries(dimensions)
    sections = split_sections(text)
    # Interleaved by rank, not grouped by column: every column's best excerpt comes before
    # any column's second-best. The evidence block has a character cap, and a run of one
    # column's excerpts at the front would push the last columns out of it entirely —
    # those cells would then read as unverifiable for a reason having nothing to do with
    # the cited paper.
    per_column = [
        select_evidence_chunks(q, sections, k=per_query, max_chars=chunk_chars)
        for q in queries
    ]
    construction = _construction_passages(
        sections, limit=_CONSTRUCTION_PASSAGES, max_chars=chunk_chars
    )
    out: list[tuple[str, str]] = []
    for rank in range(per_query):
        for picked in per_column:
            if rank < len(picked):
                out.append(picked[rank])
    # Recall net: every passage that names the property, whether or not it ranked. These
    # go FIRST — the evidence block is cut from the tail to fit its cap, and a passage the
    # paper wrote about the property by name is the last thing that should be dropped.
    named: dict[int, bool] = {}
    search: dict[int, dict[str, Any]] = {}
    front: list[tuple[str, str]] = []
    seen = {re.sub(r"\W+", "", c.lower())[:120] for _h, c in out}
    for d in dimensions or []:
        terms = property_terms(d)
        col = int(getattr(d, "col_index", 0))
        hits, total = _mentions(text, terms, limit=_RECALL_PER_PROPERTY, max_chars=chunk_chars)
        named[col] = total > 0
        search[col] = {"terms": terms, "mentions": total, "shown": len(hits)}
        for sent in hits:
            key = re.sub(r"\W+", "", sent.lower())[:120]
            if key and key not in seen:
                seen.add(key)
                front.append((f"{_NAMES_TAG} #{getattr(d, 'col_index', '?')}", sent))
    # Order: what the paper says about the properties, then how it is built, then the
    # ranked excerpts. The block is cut from the tail to fit its cap.
    built = [c for c in construction
             if re.sub(r"\W+", "", c[1].lower())[:120] not in seen]
    return front + built + out, named, len(text), search
