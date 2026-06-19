"""
Offline tests for the LaTeX extractor's bibliography parsing — in particular the
inline `\\begin{thebibliography}` path (arXiv e-prints that ship NO separate
.bbl/.bib, e.g. 1706.03762), which previously left every cited_as empty.
"""

from __future__ import annotations

from citation_verifier.extract.latex import (
    _parse_bibitems,
    _strip_line_comments,
    parse_inline_bib,
)
from citation_verifier.extract.pdf import (
    _INTEXT_NUM_RE,
    _expand_num_marker,
    _normalize_pdf_text,
    _split_author_title,
)


def test_split_author_title_is_style_agnostic():
    # Vancouver "Surname A, Surname B. Title."
    a, t = _split_author_title(
        "Radford A, Wu J, Sutskever I. Language models are unsupervised. OpenAI Blog. 2019"
    )
    assert a == ["Radford A", "Wu J", "Sutskever I"]
    assert t == "Language models are unsupervised"
    # initials-first "M. Jadeja and N. Varia, Title"
    a, t = _split_author_title("M. Jadeja and N. Varia, Perspectives for evaluating AI. 2017")
    assert a == ["M. Jadeja", "N. Varia"]
    assert t == "Perspectives for evaluating AI"
    # "Surname R. Title. In: Venue"
    a, t = _split_author_title("Carpenter R. Evaluation of Cleverbot. In: Proceedings")
    assert a == ["Carpenter R"] and t == "Evaluation of Cleverbot"
    # space-less merged block -> no boundary, honest ([], None)
    assert _split_author_title("CarpenterR.Jabberwacky-acasestudy.In:Proceedings") == ([], None)


def test_normalize_pdf_text_dehyphenates_and_despaces():
    # 1) line-break hyphenation joined; real compounds preserved
    norm = _normalize_pdf_text("knowl- edge of state-of-the-art GPT-3 Lan-\nguage models")
    assert "knowledge" in norm and "Language" in norm
    assert "state-of-the-art" in norm and "GPT-3" in norm
    # 2) a char-spaced line collapses (entry number + title become readable);
    #    a normal line is left untouched.
    spaced = "1 1 .Y a n gZ ,G a nZ ,W a n gJ . An empirical study of GPT-3"
    out = _normalize_pdf_text(spaced + "\nThis normal sentence stays intact.")
    assert "11." in out and "YangZ" in out
    assert "This normal sentence stays intact." in out


def test_pdf_intext_markers_tolerate_pypdf_spacing():
    # pypdf renders markers with a space just inside the brackets ("[ 2]",
    # "[ 82–85]"); the scan must still catch them and expand ranges. Missing this
    # dropped ~half the citations on a real two-column PDF.
    def keys(s):
        out = []
        for m in _INTEXT_NUM_RE.finditer(s):
            out += _expand_num_marker(m.group(1))
        return out

    assert keys("models [ 2].") == ["ref-2"]
    assert keys("GPT models [ 82–85].") == ["ref-82", "ref-83", "ref-84", "ref-85"]
    assert keys("approaches [13, 14] and [ 40, 41]") == [
        "ref-13", "ref-14", "ref-40", "ref-41",
    ]

_SAMPLE = r"""
\begin{thebibliography}{99}
\bibitem[Vaswani et al.(2017)]{vaswani2017}
Ashish Vaswani and Noam Shazeer.
\newblock Attention is all you need.
\newblock In NeurIPS, 2017. arXiv:1706.03762
\bibitem{lstm}
Sepp Hochreiter.
\newblock Long short-term memory.
\newblock 1997.
\end{thebibliography}
"""


def test_parse_bibitems_reads_inline_block():
    refs = _parse_bibitems(_SAMPLE)
    assert set(refs) == {"vaswani2017", "lstm"}
    assert refs["vaswani2017"].arxiv_id == "1706.03762"
    assert refs["vaswani2017"].raw  # non-empty reference string


def test_parse_inline_bib_from_tex_dir(tmp_path):
    (tmp_path / "ms.tex").write_text(_SAMPLE, encoding="utf-8")
    refs = parse_inline_bib(str(tmp_path))
    assert "vaswani2017" in refs and "lstm" in refs
    assert refs["lstm"].year == 1997


def test_parse_inline_bib_no_dir_is_empty():
    assert parse_inline_bib(None) == {}


# LaTeX tie `~` is a non-breaking space; stripping it (instead of converting to a
# space) concatenated author names — e.g. real 1706.03762 entries produced
# "MoYu" / "Nogueirados Santos" / "vanden Oord" / "QuocVV Le", which lowered the
# grounding resolver's author-gate recall. The tie must become a space.
_TIES = r"""
\begin{thebibliography}{99}
\bibitem{lin2017structured}
Zhouhan Lin, Cicero Nogueira~dos Santos, Mo~Yu, and Aaron van~den Oord.
\newblock A structured self-attentive sentence embedding.
\newblock arXiv:1703.03130, 2017.
\bibitem{sutskever14}
Ilya Sutskever, Oriol Vinyals, and Quoc~VV Le.
\newblock Sequence to sequence learning with neural networks.
\newblock 2014.
\end{thebibliography}
"""


def test_latex_tie_becomes_space_not_concatenation():
    refs = _parse_bibitems(_TIES)
    raw = refs["lin2017structured"].raw
    # tie-joined name parts must be space-separated, not glued together
    assert "Mo Yu" in raw and "MoYu" not in raw
    assert "Nogueira dos Santos" in raw and "Nogueirados Santos" not in raw
    assert "van den Oord" in raw and "vanden Oord" not in raw
    # title past the tie-bearing author list is preserved
    assert "structured self-attentive sentence embedding" in raw.lower()
    # second entry: the source's own "VV" typo is faithful, but no glue
    assert "Quoc VV Le" in refs["sutskever14"].raw


def test_strip_line_comments_drops_commented_cites():
    # A commented-out \cite must not survive the citation-site scan; a real one
    # on the same kind of line (and an escaped \% ) must.
    src = (
        "Real claim \\cite{real}.\n"
        "% fully commented \\cite{ghost}\n"
        "Mid line \\cite{keep} % \\cite{ghost2}\n"
        "Escaped 50\\% done \\cite{also_keep}.\n"
    )
    out = _strip_line_comments(src)
    assert "\\cite{real}" in out
    assert "\\cite{keep}" in out
    assert "\\cite{also_keep}" in out  # escaped \% is not a comment
    assert "ghost" not in out and "ghost2" not in out
