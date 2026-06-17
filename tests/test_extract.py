"""
Offline tests for the LaTeX extractor's bibliography parsing — in particular the
inline `\\begin{thebibliography}` path (arXiv e-prints that ship NO separate
.bbl/.bib, e.g. 1706.03762), which previously left every cited_as empty.
"""

from __future__ import annotations

from citation_verifier.extract.latex import _parse_bibitems, parse_inline_bib

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
