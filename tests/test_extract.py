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
