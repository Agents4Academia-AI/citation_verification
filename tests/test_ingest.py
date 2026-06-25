"""Offline tests for ingest input routing: local LaTeX zip (#3) and generic PDF
URL (#2), plus the arXiv path still winning for arxiv.org URLs. Network is
monkeypatched; archives are built in a tmp dir."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import citation_verifier.ingest as ing
from citation_verifier.config import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(papers_dir=tmp_path / "papers")


def test_local_latex_zip_routes_to_latex_extractor(tmp_path):
    z = tmp_path / "paper.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("main.tex", r"\documentclass{article}\begin{document}\cite{x}\end{document}")
        zf.writestr("refs.bib", "@article{x, title={T}, author={A}, year={2020}}")
    src = ing.ingest(str(z), settings=_settings(tmp_path))
    assert src.tex_available is True and src.kind == "latex"
    assert src.tex_dir and (Path(src.tex_dir) / "main.tex").exists()


def test_zip_slip_member_is_skipped(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("main.tex", r"\documentclass{article}\begin{document}\end{document}")
        zf.writestr("../escape.tex", "should not land outside")  # path-traversal
    src = ing.ingest(str(z), settings=_settings(tmp_path))
    assert src.tex_available is True
    assert not (tmp_path / "escape.tex").exists()  # guard held


def test_pdf_url_downloads_and_routes_to_pdf(tmp_path, monkeypatch):
    def fake_dl(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.5\n...bytes...")
        return True

    monkeypatch.setattr(ing, "_download", fake_dl)
    src = ing.ingest("https://example.com/papers/2401.pdf", settings=_settings(tmp_path))
    assert src.kind == "pdf" and src.tex_available is False
    assert src.pdf_path and Path(src.pdf_path).read_bytes().startswith(b"%PDF")


def test_non_pdf_url_raises_actionable_error(tmp_path, monkeypatch):
    def fake_dl(url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<html>a landing/paywall page</html>")
        return True

    monkeypatch.setattr(ing, "_download", fake_dl)
    with pytest.raises(ValueError, match="did not return a PDF"):
        ing.ingest("https://example.com/landing", settings=_settings(tmp_path))


def test_arxiv_url_still_takes_the_arxiv_path(tmp_path, monkeypatch):
    # arxiv.org URLs must NOT fall into the generic-URL branch.
    monkeypatch.setattr(ing, "_download", lambda url, dest: False)  # no network
    monkeypatch.setattr(ing, "_probe_tex_available", lambda aid: False)
    src = ing.ingest("https://arxiv.org/abs/2310.06825", settings=_settings(tmp_path))
    assert src.arxiv_id == "2310.06825" and src.kind == "arxiv_pdf"
