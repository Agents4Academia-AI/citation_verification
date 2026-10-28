"""The sentence a citation sits in: the shared segmenter, and each extractor's use of it.

Offline. The PDF cases build their pages with PyMuPDF so the geometry is real.
"""

from __future__ import annotations

import pytest

from citation_verifier.extract import latex as L
from citation_verifier.extract.pdf import _looks_like_table_dump, _scan_body
from citation_verifier.extract.sentences import (
    MAX_CLAIM_CHARS,
    heading_title,
    is_heading_line,
    section_at,
    sentence_span,
)
from citation_verifier.schema import CitedAs


def _sentence(text: str, probe: str, **kw) -> str:
    a, b = sentence_span(text, text.index(probe), **kw)
    return text[a:b]


# ── the segmenter ────────────────────────────────────────────────────────────
def test_abbreviations_initials_and_asides_do_not_end_a_sentence():
    text = (
        "Earlier work is limited. As shown by Smith et al. (2020) and D. P. Kingma, the loss "
        "falls (see Fig. 3. for the curve), e.g. on ImageNet [4]. The next sentence starts here."
    )
    s = _sentence(text, "[4]")
    assert s.startswith("As shown by Smith et al.") and s.endswith("ImageNet [4].")


def test_a_blank_line_is_a_boundary_and_explicit_breaks_replace_it():
    text = "The first half of a sentence that runs past a column\n\nbreak ends here [2]. Next."
    assert _sentence(text, "[2]") == "break ends here [2]."
    joined = _sentence(text, "[2]", breaks=[])
    assert joined.startswith("The first half") and joined.endswith("[2].")
    cut = text.index("break")
    assert _sentence(text, "[2]", breaks=[cut]) == "break ends here [2]."


def test_a_long_sentence_is_kept_whole_up_to_the_cap():
    clause = "method {} (Author{}, 2020), "
    text = "Prior methods include " + "".join(clause.format(i, i) for i in range(14)) + "and others [9]."
    assert 360 < len(text) < MAX_CLAIM_CHARS
    assert _sentence(text, "[9]") == text


def test_past_the_cap_the_clauses_around_the_citation_are_kept():
    filler = "word " * 150
    text = f"{filler.strip()}; the cited clause holds the claim [7]; {filler.strip()}."
    s = _sentence(text, "[7]")
    assert "the cited clause holds the claim [7]" in s and len(s) <= MAX_CLAIM_CHARS
    run_on = "token " * 400 + "cite [3] " + "token " * 400
    s = _sentence(run_on, "[3]")
    assert "[3]" in s and len(s) <= MAX_CLAIM_CHARS and not s.startswith("oken")


@pytest.mark.parametrize(
    "line",
    ["1 Introduction", "2.1 Experimental setup", "2 RELATED WORK", "A Proofs", "B.2 Additional Results",
     "Related Work", "ACKNOWLEDGMENTS", "References", "IV. EXPERIMENTS", "5 Conclusion and Future Work"],
)
def test_headings_are_recognised(line):
    assert is_heading_line(line)


@pytest.mark.parametrize(
    "line",
    ["100 Million parameters and", "A simple baseline that we use", "2 models were trained on the",
     "We report results in Table 2.", "3 Experiments on ImageNet show that the effect of",
     "the cat sat", "1 Department of Electrical Engineering, University of Singapore, 119077"],
)
def test_prose_lines_are_not_headings(line):
    assert not is_heading_line(line)


def test_heading_title_drops_the_number_and_section_at_finds_the_last_one():
    assert heading_title("2.1  Experimental setup") == "Experimental setup"
    assert heading_title("Related Work") == "Related Work"
    heads = [(0, "Introduction"), (500, "Method")]
    assert section_at(heads, 10) == "Introduction" and section_at(heads, 900) == "Method"
    assert section_at([], 5) is None


# ── PDF text path (numbered markers) ─────────────────────────────────────────
def test_a_heading_is_a_boundary_and_names_the_section():
    body = (
        "Change-point detection is unreliable under\ngradual drift.\n\n2 RELATED WORK\n\n"
        "ACI [3] tracks a quantile by an online gradient step."
    )
    scan, headings, breaks, _ = _scan_body(body)
    pos = scan.index("[3]")
    a, b = sentence_span(scan, pos, breaks=breaks)
    assert scan[a:b] == "ACI [3] tracks a quantile by an online gradient step."
    assert section_at(headings, pos) == "RELATED WORK"


def test_a_caption_between_two_halves_of_a_sentence_is_moved_out():
    body = (
        "The guidance from foundation models [6] has been used to improve training\n\n"
        "Figure 2. The model achieves the best result on ImageNet.\n\n"
        "convergence speed and quality [58] of generative models.\n"
    )
    scan, _, breaks, _ = _scan_body(body)
    pos = scan.index("[58]")
    a, b = sentence_span(scan, pos, breaks=breaks)
    claim = " ".join(scan[a:b].split())
    assert claim.startswith("The guidance from foundation models [6]")
    assert "Figure 2" not in claim and claim.endswith("of generative models.")
    assert "Figure 2. The model achieves" in scan  # still there, after the text


def test_figure_numbers_and_table_cells_are_not_prose():
    body = (
        "We split the data into training and test sets [42], and evaluate on both.\n"
        "1 2 4 8 16 32 64 128\n"
        "Number of samples k\n\n"
        "VAE [40]\nKL\n55M\n0.27\nLDM-4 [40]\n400M\n3.60\n\n"
        "The next paragraph is prose again."
    )
    scan, _, breaks, grids = _scan_body(body)
    pos = scan.index("[42]")
    a, b = sentence_span(scan, pos, breaks=breaks)
    assert " ".join(scan[a:b].split()) == (
        "We split the data into training and test sets [42], and evaluate on both."
    )
    cell = scan.index("VAE [40]")
    assert any(x <= cell < y for x, y in grids)  # a table cell, set apart as one


def test_a_sentence_listing_models_by_size_is_not_a_table_row():
    prose = (
        "We select 15 LLMs from eight families: Llama (Llama3.1-8B-Instruct, Llama3.3-70B-Instruct, "
        "Llama 4 Scout; Grattafiori et al. 2024), Gemini (2.5 Flash, 2.5 Pro; Team et al. 2023), "
        "Claude (3.7 Sonnet), Phi-4 (Abdin et al., 2024), Deepseek-R1 (Guo et al., 2025), and "
        "Command-A (Cohere et al., 2025), covering both open and closed models of all sizes."
    )
    assert not _looks_like_table_dump(prose)
    assert _looks_like_table_dump("VAE [40] KL 55M 4096 0.27 LDM-4 [40] 400M Diff. 3.60 SD-VAE 84M 1.50")


# ── PDF link path: the document read as one flow ─────────────────────────────
def test_the_flow_runs_across_pages_with_headings_apart_and_footnotes_out(tmp_path):
    fitz = pytest.importorskip("fitz")
    from citation_verifier.extract.pdf_links import _claim_at, _document_flow

    doc = fitz.open()
    p1 = doc.new_page(width=612, height=792)
    p1.insert_text((72, 90), "1 Introduction", fontsize=12, fontname="hebo")
    y = 120
    for line in ["Online methods track a quantile with an online step and are the basis of",
                 "nearly every controller in use today, and the claim in this sentence",
                 "continues onto the"]:
        p1.insert_text((72, y), line, fontsize=10)
        y += 13
    p1.insert_text((72, 740), "1 A footnote that must not land inside the sentence.", fontsize=7)
    p2 = doc.new_page(width=612, height=792)
    p2.insert_text((72, 90), "next page and ends here (Smith et al., 2020). Another sentence.", fontsize=10)
    path = tmp_path / "flow.pdf"
    doc.save(str(path))

    with fitz.open(str(path)) as d:
        flow = _document_flow(d)
    text = flow.text
    claim, _ = _claim_at(text, text.index("Smith"))
    assert claim.startswith("Online methods track a quantile")
    assert "continues onto the next page and ends here (Smith et al., 2020)." in claim
    assert "footnote" not in claim and "footnote" in text
    assert flow.headings and flow.headings[0][1] == "Introduction"
    assert not claim.startswith("1 Introduction")


# ── LaTeX ────────────────────────────────────────────────────────────────────
def test_latex_structure_is_a_boundary_and_leaves_no_residue():
    tex = (
        r"We conclude the abstract here. \end{abstract} \section{Introduction}\label{sec:intro} "
        r"Temperature scaling \citep{guo} is the simplest method. Next sentence."
    )
    claim, _ = L._sentence_around(tex, tex.index(r"\citep"))
    assert claim.startswith("Temperature scaling") and claim.endswith("simplest method.")
    assert "sec:intro" not in claim and "abstract" not in claim


def test_latex_citations_are_written_out_from_the_bibliography():
    refs = {
        "guo": CitedAs(raw="x", authors=["Chuan Guo", "Geoff Pleiss", "Yu Sun"], year=2017),
        "zhao": CitedAs(raw="x", authors=["Zhao, Shengjia", "Ermon, Stefano"], year=2021),
    }
    tex = r"Earlier. \citet{guo} observe that networks are overconfident \citep{guo,zhao}. Next."
    claim, _ = L._sentence_around(tex, tex.index(r"\citet"), render=lambda s: L.render_citations(s, refs))
    assert claim == (
        "Guo et al. (2017) observe that networks are overconfident "
        "(Guo et al., 2017; Zhao and Ermon, 2021)."
    )
    assert L.render_citations(r"see \citep{missing}", refs) == "see ([missing], n.d.)"


def test_a_bibliography_written_by_filecontents_is_read(tmp_path):
    (tmp_path / "paper.tex").write_text(
        "\\begin{filecontents}{references.bib}\n"
        "@article{guo2017,\n  title = {On Calibration of Modern Neural Networks},\n"
        "  author = {Guo, Chuan and Pleiss, Geoff},\n  year = {2017}\n}\n"
        "\\end{filecontents}\n\\documentclass{article}\\begin{document}x\\end{document}\n"
    )
    refs = L.parse_embedded_bib(tmp_path)
    assert refs["guo2017"].title == "On Calibration of Modern Neural Networks"
    assert refs["guo2017"].year == 2017
