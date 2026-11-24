"""
pdf_margin.py — find the line-number gutter of a review-format PDF page.

Papers under review carry line numbers in the margin (the ICLR, NeurIPS, ICML and CVPR
submission templates, the ``lineno`` package). Read as text, those numbers are not
noise one can ignore: every reference-list line starts with one ("224 Avrim Blum and
Y. Mansour. …"), so each wrapped line is taken for a new numbered entry — three
references came out as five, two of them a dangling "Economic Behavior, 21:40–55, 1997"
— and a list that looks numbered makes the body's bracketed intervals "[0, 1]" read as
citations [0] and [1]. One ten-paper batch of submissions lost most of its citations
that way.

:func:`content_clip` returns the part of a page that excludes such a column, for
PyMuPDF's ``get_text(..., clip=…)``. A column is a line numbering when it sits wholly
in the margin outside the text, holds at least eight bare numbers, and counts upward
down the page. Page numbers, equation numbers "(3)", and numeric table columns (which
sit inside the text block) do not qualify.

The clip also leaves out a running header or footer — a line in the top or bottom band
that repeats on most pages ("Under review as a workshop paper at ICLR 2025"). Read with
the text it was glued to the first sentence of every page and handed to the judge as
part of the claim.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

__all__ = ["content_clip"]

_BARE_NUMBER = re.compile(r"\d{1,4}")
_MIN_NUMBERS = 8
_BAND = 60.0  # points from the top / bottom edge where running headers and footers sit
_RUNNING: dict[tuple, tuple[frozenset, frozenset]] = {}


def content_clip(page: Any) -> Any | None:
    """The page rectangle minus its margin line-number column(s), or ``None`` if none."""
    try:
        words = page.get_text("words")
    except Exception:  # noqa: BLE001 — fail soft: no clip
        return None
    numbers = [w for w in words if _BARE_NUMBER.fullmatch(w[4])]
    text = [w for w in words if not _BARE_NUMBER.fullmatch(w[4])]
    if len(text) < 20:
        return None
    # The text block's edges, robust to a stray word in the margin (a header, a URL).
    lefts = sorted(w[0] for w in text)
    rights = sorted(w[2] for w in text)
    left_edge = lefts[len(lefts) // 50]
    right_edge = rights[-1 - len(rights) // 50]
    left = [w for w in numbers if w[2] < left_edge - 2]
    right = [w for w in numbers if w[0] > right_edge + 2]
    rect = page.rect
    x0, x1 = rect.x0, rect.x1
    if _counts_lines(left):
        x0 = max(w[2] for w in left) + 0.5
    if _counts_lines(right):
        x1 = min(w[0] for w in right) - 0.5
    y0, y1 = _running_bands(page, words)
    if (x0, x1, y0, y1) == (rect.x0, rect.x1, rect.y0, rect.y1):
        return None
    return type(rect)(x0, y0, x1, y1)


def _band_text(words: list, top: bool, height: float) -> tuple[str, float]:
    """The text of the top (or bottom) band, digits masked, and its inner edge."""
    band = [w for w in words if (w[3] <= _BAND if top else w[1] >= height - _BAND)]
    if not band:
        return "", 0.0
    text = re.sub(r"\d+", "#", " ".join(w[4] for w in sorted(band, key=lambda w: (round(w[1]), w[0]))))
    edge = max(w[3] for w in band) if top else min(w[1] for w in band)
    return text, edge


def _running_bands(page: Any, words: list) -> tuple[float, float]:
    """``(y0, y1)`` of the page without a header / footer that repeats across the document."""
    rect = page.rect
    y0, y1 = rect.y0, rect.y1
    doc = getattr(page, "parent", None)
    if doc is None or len(doc) < 2:
        return y0, y1
    try:
        first = doc[0].get_text()[:200]
    except Exception:  # noqa: BLE001
        first = ""
    key = (getattr(doc, "name", ""), len(doc), rect.height, hash(first))
    if key not in _RUNNING:
        tops, bottoms = Counter(), Counter()
        for other in doc:
            try:
                ow = other.get_text("words")
            except Exception:  # noqa: BLE001
                continue
            tops[_band_text(ow, True, other.rect.height)[0]] += 1
            bottoms[_band_text(ow, False, other.rect.height)[0]] += 1
        need = max(2, len(doc) // 2)
        _RUNNING[key] = (
            frozenset(t for t, n in tops.items() if t and n >= need),
            frozenset(t for t, n in bottoms.items() if t and n >= need),
        )
    running_top, running_bottom = _RUNNING[key]
    text, edge = _band_text(words, True, rect.height)
    if text in running_top:
        y0 = edge + 0.5
    text, edge = _band_text(words, False, rect.height)
    if text in running_bottom:
        y1 = edge - 0.5
    return y0, y1


def _counts_lines(words: list) -> bool:
    """True when these numbers, read top to bottom, count upward one or two at a time."""
    if len(words) < _MIN_NUMBERS:
        return False
    values = [int(w[4]) for w in sorted(words, key=lambda w: (w[1], w[0]))]
    steps = [b - a for a, b in zip(values, values[1:], strict=False)]
    return sum(1 for s in steps if 1 <= s <= 2) >= 0.7 * len(steps)
