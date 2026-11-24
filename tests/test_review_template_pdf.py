"""PDFs in review format: line numbers down the margin, math intervals in the text.

Built with PyMuPDF so the page geometry is real. Offline.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from citation_verifier.extract.pdf import (  # noqa: E402
    _expand_num_marker,
    extract_pdf_text,
    load_pdf_references,
)
from citation_verifier.extract.pdf_margin import content_clip  # noqa: E402

_BODY = [
    "1 INTRODUCTION",
    "A forecaster announces a probability p in [0, 1] for a binary outcome and is judged",
    "along two axes. Foster & Vohra (1997) connect calibration to internal regret, and",
    "Blum & Mansour (2005) reduce internal regret to external regret for any finite set",
    "of actions. We study the gap between the two on the interval [0, 1] in detail.",
    "Our main result is a sharp separation with explicit finite-horizon constants here.",
]
_REFS = [
    "REFERENCES",
    "Avrim Blum and Y. Mansour. From external to internal regret. Journal of Machine",
    "    Learning Research, 8:1307-1324, 2005.",
    "Dean Phillips Foster and R. Vohra. Calibrated learning and correlated equilibrium. Games and",
    "    Economic Behavior, 21:40-55, 1997.",
    "Isaac Gibbs and E. Candes. Adaptive conformal inference under distribution shift. ArXiv,",
    "    abs/2106.00170, 2021.",
]


def _review_pdf(path, numbered: bool = True):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y, n = 90, 0
    for line in _BODY + [""] + _REFS:
        if line:
            x = 120 if line.startswith("    ") else 108
            page.insert_text((x, y), line.strip(), fontsize=9.5)
        if numbered:
            page.insert_text((75, y), f"{n:03d}", fontsize=8)
        y += 13
        n += 1
    doc.save(str(path))
    return path


def test_the_line_number_column_is_cut_from_the_page(tmp_path):
    with fitz.open(str(_review_pdf(tmp_path / "r.pdf"))) as doc:
        clip = content_clip(doc[0])
        assert clip is not None and 86 < clip.x0 < 108
    with fitz.open(str(_review_pdf(tmp_path / "plain.pdf", numbered=False))) as doc:
        assert content_clip(doc[0]) is None          # no gutter, nothing cut


def test_a_review_pdf_reads_like_the_camera_ready(tmp_path):
    """Read with its line numbers, each wrapped reference line was taken for a new
    numbered entry (three references came out as five)."""
    review = load_pdf_references(_review_pdf(tmp_path / "r.pdf"))
    plain = load_pdf_references(_review_pdf(tmp_path / "plain.pdf", numbered=False))
    assert len(review) == len(plain) == 3
    assert [c.year for c in review.values()] == [2005, 1997, 2021]
    text = extract_pdf_text(tmp_path / "r.pdf")
    assert "000" not in text and "Foster & Vohra (1997)" in " ".join(text.split())


def test_an_interval_is_not_a_citation():
    """"[0, 1]" counted as a citation of reference 1 — in a probability paper dozens of
    times — and the style test then declared an author-year paper numbered."""
    assert _expand_num_marker("0, 1") == []
    assert _expand_num_marker("0,100") == []
    assert _expand_num_marker("3, 5-7") == ["ref-3", "ref-5", "ref-6", "ref-7"]
    assert _expand_num_marker("10") == ["ref-10"]


def test_a_table_after_the_bibliography_is_not_a_reference():
    from citation_verifier.extract.pdf_refs import _has_words

    assert not _has_words("0.961 0") and not _has_words("k = 5 k = 10 k = 20")
    assert _has_words("Isaac Gibbs and E. Candes. Adaptive conformal inference.")


def test_a_link_belongs_to_the_word_it_covers_most():
    """A link box is a point or two taller than its line and grazes the line above; the
    first word it touched was up there, and the citation went to the previous sentence."""
    from citation_verifier.extract.pdf_links import _sentence_at_rect

    words = []
    def line(y, text, x=108.0):
        nonlocal words
        for tok in text.split():
            words.append((x, y, x + 6 * len(tok), y + 10, tok, 0, 0, 0))
            x += 6 * len(tok) + 4
    line(255, "They study a related but distinct question.")
    line(266, "Cerou et al. (2011) give non-asymptotic variance bounds.")
    rect = fitz.Rect(108, 264, 190, 277)     # over "Cerou et al." — and 1pt into the line above
    claim, _ = _sentence_at_rect(words, rect, 612)
    assert claim.startswith("Cerou et al. (2011) give")


def test_a_given_name_first_list_is_not_read_as_surname_comma_initials():
    from citation_verifier.extract.pdf import _parse_ref_entry

    c = _parse_ref_entry("Anastasios Nikolas Angelopoulos, E. Candes, and R. Tibshirani. Conformal pid "
                         "control for time series prediction. Advances in Neural Information "
                         "Processing Systems, 2023.")
    assert c.authors[0] == "Anastasios Nikolas Angelopoulos" and len(c.authors) == 3
    # …while a real "Surname, I." list still is.
    c = _parse_ref_entry("Blum, A. and Mansour, Y. From external to internal regret. JMLR, 2007.")
    assert c.authors == ["Blum, A.", "Mansour, Y."]


def test_a_link_whose_key_says_nothing_binds_by_the_names_it_shows():
    """``\\cite{none}``, rendered "(Bengio & Grandvalet)" — no year, no usable key."""
    from citation_verifier.extract.pdf import _bind_from_link_text
    from citation_verifier.schema import CitedAs

    refs = {
        "ref-1": CitedAs(raw="", authors=["Yoshua Bengio", "Yves Grandvalet"], title="No unbiased estimator"),
        "ref-2": CitedAs(raw="", authors=["Ian Goodfellow", "Yoshua Bengio"], title="Deep learning", year=2016),
    }
    assert _bind_from_link_text("Bengio & Grandvalet", refs) is refs["ref-1"]
    assert _bind_from_link_text("Goodfellow et al. (2016)", refs) is refs["ref-2"]
    assert _bind_from_link_text("Goodfellow et al. (2019)", refs) is None   # the year disagrees
    assert _bind_from_link_text("(2016)", refs) is None                     # no name, no guess


def test_a_running_header_is_cut_from_every_page(tmp_path):
    """Read with the text, the template's header was glued to the first sentence of
    each page and handed to the judge as part of the claim."""
    doc = fitz.open()
    for n in range(3):
        page = doc.new_page(width=612, height=792)
        page.insert_text((108, 40), "Under review as a workshop paper at ICLR 2025", fontsize=9)
        y = 100
        for i in range(25):
            page.insert_text((108, y), f"Body sentence {i} of page {n} says something ordinary here.", fontsize=10)
            y += 14
        page.insert_text((300, 770), str(n + 1), fontsize=9)
    path = tmp_path / "h.pdf"
    doc.save(str(path))
    with fitz.open(str(path)) as d:
        clip = content_clip(d[1])
        assert clip is not None and 40 < clip.y0 < 100 and clip.y1 < 770
    text = extract_pdf_text(path)
    assert "Under review" not in text and "Body sentence 3 of page 1" in text
