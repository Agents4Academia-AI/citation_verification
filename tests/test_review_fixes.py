"""
Offline regression tests for defects found in a review of the reference pipeline:
reference extraction (LaTeX + PDF), resolution, and the title / author / venue / year
comparison. Each test names the failure it pins down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citation_verifier.extract import latex as L
from citation_verifier.extract import pdf as P
from citation_verifier.interfaces import Candidate, PaperSource
from citation_verifier.schema import CitationRecord, CitedAs, Claim, MatchMethod, Resolved
from citation_verifier.stages import correctness as C


# ── LaTeX: which commands cite ───────────────────────────────────────────────
def test_every_citing_command_is_seen_not_a_closed_list_of_spellings():
    """The scanner knew ``\\citep`` but not ``\\citep*``, ``\\Citet``, ``\\citeyearpar``,
    ACL's ``\\newcite``, biblatex's multicite forms, or a paper's own wrapper macro — and a
    citation the scanner does not see is a citation never checked."""
    def keys(tex: str) -> list[str]:
        return [k for _pos, ks in L.cite_sites(tex) for k in ks]

    for cmd in ("cite", "citep", "citet*", "citep*", "Citet", "citeauthor*", "citeyearpar",
                "newcite", "shortcite", "supercite", "smartcite", "parencite", "tcite"):
        assert keys(rf"\{cmd}{{smith2020}}") == ["smith2020"], cmd
    assert keys(r"\citep[see][p.~3]{a, b}") == ["a", "b"]
    assert keys(r"\cites{a}{b}") == ["a", "b"]
    assert keys(r"\parencites[see][]{a}[cf.][]{b,c}") == ["a", "b", "c"]


def test_commands_that_contain_cite_but_do_not_cite_are_ignored():
    for tex in (r"\nocite{a}", r"\bibcite{a}{1}", r"\citeform{\textbf{#1}}",
                r"\newcommand{\tcite}[1]{\citet{#1}}", r"\setcitestyle{numbers}"):
        assert L.cite_sites(tex) == [], tex


def test_a_claim_is_not_cut_at_et_al():
    r"""``Smith et al. \cite{smith} propose …`` was split after "al." — the backslash that
    lets a sentence open with a command is also what follows "et al." — so the claim read
    "propose …" with its subject left in the previous sentence."""
    tex = r"Earlier work is limited. Smith et al. \cite{smith} propose a new method. Next one."
    claim, _span = L._sentence_around(tex, tex.index(r"\cite"))
    assert claim.startswith("Smith et al.") and "propose a new method" in claim
    tex = r"Old news. As shown by e.g. \cite{x}, this holds in general. Next."
    assert L._sentence_around(tex, tex.index(r"\cite"))[0].startswith("As shown by e.g.")
    # …while a sentence that really does open with a citation still starts there.
    tex = r"We use Adam. \citet{kingma} introduced it for deep nets. Then."
    assert "We use Adam" not in L._sentence_around(tex, tex.index(r"\citet"))[0]


# ── LaTeX: the bibliography ──────────────────────────────────────────────────
BBL = r"""
\bibitem[Vaswani et~al.(2017)Vaswani, Shazeer, Parmar]{vaswani2017}
Ashish Vaswani, Noam Shazeer, and Niki Parmar.
\newblock Attention is all you need.
\newblock In \emph{Advances in Neural Information Processing Systems}, 2017.

\bibitem{he2016}
K.~He, X.~Zhang, S.~Ren, and J.~Sun, ``Deep residual learning for image recognition,'' in \emph{Proc. CVPR}, 2016, pp. 1998--2005.

\bibitem[Kingma \& Ba(2015)]{kingma}
Kingma, D.~P. and Ba, J.
\newblock Adam: A method for stochastic optimization.
\newblock In \emph{ICLR}, 2015. \url{https://arxiv.org/abs/1412.6980}

\bibitem[Liu et~al.(2022)]{liu2022}
Liu, Z., Mao, H., and Xie, S.
\newblock A ConvNet for the 2020s.
\newblock In \emph{CVPR}, pp. 11976--11986, 2022.
"""


def test_a_bbl_entry_yields_its_authors():
    """Authors were never parsed from a ``.bbl`` ("unreliable; leave empty"), so no
    ``.bbl``-sourced reference — most arXiv sources ship only that — could be checked for
    a wrong or invented author, and a same-title paper by other people passed unremarked."""
    refs = L._parse_bibitems(BBL)
    assert refs["vaswani2017"].authors == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    assert refs["kingma"].authors[0] == "Kingma, D. P." and len(refs["kingma"].authors) == 2
    # IEEE style: no \newblock at all — authors run into a quoted title.
    assert refs["he2016"].authors == ["K. He", "X. Zhang", "S. Ren", "J. Sun"]
    assert refs["he2016"].title == "Deep residual learning for image recognition"


def test_short_surnames_are_not_read_as_initials():
    assert L.split_author_block("K. He, X. Zhang, S. Ren, and J. Sun") == [
        "K. He", "X. Zhang", "S. Ren", "J. Sun"]
    assert L.split_author_block("Wu, Y. and He, K.")[0] == "Wu, Y."
    # Not names at all -> nothing, rather than a list that will be reported as a mismatch.
    assert L.split_author_block("The 2019 Conference on Something, Proceedings of 12 workshops") == []


def test_the_publication_year_is_not_the_first_four_digits():
    """Titles carry years ("A ConvNet for the 2020s"), and so do page ranges and arXiv
    ids; a year read from any of them was then reported as a year MISMATCH."""
    refs = L._parse_bibitems(BBL)
    assert refs["liu2022"].year == 2022          # not 2020 from the title
    assert refs["he2016"].year == 2016           # not 1998/2005 from the page range
    assert refs["kingma"].year == 2015 and refs["kingma"].arxiv_id == "1412.6980"

    def year(ref: str) -> int | None:
        return P._parse_ref_entry(ref).year

    assert year("J. Deng and L. Fei-Fei. ImageNet large scale visual recognition "
                "challenge 2012. IJCV, 115(3):211-252, 2015.") == 2015
    assert year("A. Author. Lessons from the 2008 financial crisis. J. Finance, 2019.") == 2019
    assert year("OpenAI. GPT-4 technical report. 2023. Accessed: 2024-11-02.") == 2023
    assert year("Kingma, D. P. and Ba, J. Adam. CoRR, abs/1412.6980, 2014.") == 2014
    # A year inside an address is not the year; one glued to its end (PDF text drops the
    # space) is — and for a repository it is the only year the reference has.
    assert L.publication_year("J. Liu and L. Zhang. Code-R1. https://github.com/ganler/code-r1,2025. "
                              "GitHub repository.", "Code-R1") == 2025
    assert L.publication_year("A. Author. A post. https://example.org/blog/2019/post, 2021.", "A post") == 2021
    assert L.publication_year("OpenAI. System card. https://openai.com/index/2024/card/", "System card") is None
    # No venue after the title, so the title parse took the year with it: the year is
    # still the reference's, and dropping the title must not drop it.
    assert L.publication_year("LeCun, Y., Cortes, C., et al.: Mnist handwritten digit database (2010)",
                              "Mnist handwritten digit database (2010)") == 2010
    assert L.publication_year("Harrison Chase. Langchain, October 2022. URL https://github.com/x/y.",
                              "Langchain, October 2022") == 2022
    # …while a year in the title of a reference that prints its own is still not it,
    assert L.publication_year("J. Togelius et al. The Mario AI championship 2009-2012. AI Magazine, "
                              "34(3):89-92, 2013.", "The Mario AI championship 2009-2012") == 2013
    # and a decade is not a year at all.
    assert L.publication_year("Z. Liu et al. A ConvNet for the 2020s.", "A ConvNet for the 2020s") is None


def test_bib_fields_are_decoded_and_the_arxiv_id_is_found_where_it_is_written(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(r"""
@article{a, author={Rothh{\"o}rl, Thomas and M{\"u}ller, Jan}, title={Schr{\"o}dinger bridges},
  journal={arXiv preprint arXiv:2301.12345}, year={2023}}
@inproceedings{b, author={Ann Alpha}, title={Second paper}, booktitle={NeurIPS}, year={2017},
  url={https://arxiv.org/abs/1706.03762}}
@article{c, author={X}, title={Third paper}, journal={Nature}, year={2020},
  url={https://doi.org/10.1038/s41586-020-1234-5}, eprint={hal-0123456}, archivePrefix={HAL}}
""")
    refs = L.parse_bib(bib)
    assert refs["a"].authors == ["Rothhörl, Thomas", "Müller, Jan"]
    assert refs["a"].title == "Schrödinger bridges" and refs["a"].arxiv_id == "2301.12345"
    assert refs["b"].arxiv_id == "1706.03762"
    assert refs["c"].arxiv_id is None and refs["c"].doi == "10.1038/s41586-020-1234-5"


def _latex_paper(tmp_path: Path, files: dict[str, str]) -> PaperSource:
    tex = tmp_path / "tex"
    for name, text in files.items():
        (tex / name).parent.mkdir(parents=True, exist_ok=True)
        (tex / name).write_text(text, encoding="utf-8")
    return PaperSource(paper_id="p", kind="latex", tex_dir=str(tex), tex_available=True,
                       work_dir=str(tmp_path))


def test_every_bibliography_file_is_read_not_the_alphabetically_first(tmp_path):
    """A paper that splits its references across two ``.bib`` files had every key of the
    second reported as "no bibliography entry"."""
    src = _latex_paper(tmp_path, {
        "main.tex": r"\documentclass{article}\begin{document}One \cite{a}. Two \cite{z}.\end{document}",
        "anthology.bib": "@article{a, author={Ann A}, title={The first paper}, year={2020}}",
        "custom.bib": "@article{z, author={Zed Z}, title={The last paper}, year={2021}}",
    })
    refs = L.load_references(src)
    assert set(refs) == {"a", "z"}
    records = L.LatexExtractor().extract(src)
    assert not any("no bibliography entry" in (r.notes or "") for r in records)


def test_citations_in_a_retired_draft_are_not_claims_of_the_paper(tmp_path):
    """An arXiv tree ships drafts beside the live files. A section the document no longer
    includes is not the paper; its citations were being reported as claims the paper
    makes."""
    src = _latex_paper(tmp_path, {
        "main.tex": "\\documentclass{article}\\begin{document}\n\\input{intro}\n"
                    "%\\input{old_intro}\n\\bibliography{refs}\\end{document}",
        "intro.tex": r"The live claim \cite{kept}.",
        "old_intro.tex": r"An abandoned claim \cite{kept} and another \cite{dropped}.",
        "main.bbl": "\\bibitem{kept}\nAnn Kept.\n\\newblock A kept paper.\n\\newblock 2020.\n",
    })
    records = L.LatexExtractor().extract(src)
    assert [r.cite_key for r in records] == ["kept"]
    assert "live claim" in records[0].claim.text


def test_the_compiled_bibliography_overrules_an_include_graph_that_loses_citations(tmp_path):
    """The include walker cannot know every way a file is pulled in. When restricting to
    the graph would lose a key the ``.bbl`` has, the graph is wrong, and every file is
    read after all — losing a real citation is worse than keeping a stale one."""
    src = _latex_paper(tmp_path, {
        "main.tex": "\\documentclass{article}\\begin{document}\n\\mysection{body}\n\\end{document}",
        "body.tex": r"A claim pulled in by a macro \cite{real}.",
        "main.bbl": "\\bibitem{real}\nAnn Real.\n\\newblock A real paper.\n\\newblock 2020.\n",
    })
    assert [r.cite_key for r in L.LatexExtractor().extract(src)] == ["real"]


def test_the_include_walker_follows_the_import_package_and_plain_tex(tmp_path):
    tex = tmp_path
    (tex / "main.tex").write_text(
        "\\documentclass{article}\n\\subimport{sections/}{intro}\n\\input method\n%\\input{dead}\n")
    (tex / "sections").mkdir()
    (tex / "sections" / "intro.tex").write_text("intro")
    (tex / "method.tex").write_text("method")
    (tex / "dead.tex").write_text("dead")
    kept = {p.name for p in L.included_sources(sorted(tex.rglob("*.tex")))}
    assert kept == {"main.tex", "intro.tex", "method.tex"}


# ── title / author / venue comparison ────────────────────────────────────────
def _resolved(title, authors, year=2016, venue=None):
    return Resolved(source="s2", match_method=MatchMethod.FUZZY_TITLE, title=title,
                    authors=authors, year=year, venue=venue)


_HE = ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"]
_RESNET = "Deep Residual Learning for Image Recognition"


def test_a_close_title_by_other_authors_is_a_different_paper():
    """Cited "Deep Residual Learning for Image Recognition" was accepted as "Deep Learning
    for Image Recognition" by other people — one content word apart, inside every looser
    threshold — and the relevance judge was handed the wrong paper."""
    cited = CitedAs(raw="", authors=_HE, title=_RESNET, year=2016)
    other = _resolved("Deep Learning for Image Recognition", ["Wei Wang", "Li Chen"], 2017)
    assert C._reject_bad_fuzzy_match(cited, other)


def test_a_blank_author_on_the_record_agrees_with_nobody():
    """An index returned ["", "A Isakov", ""] for a 1997 report on polystyrene shells
    whose title is three words of the cited one. The blank first author was read as
    "nothing to disagree with", and a citation from 2013 was resolved to it."""
    cited = CitedAs(
        raw="", authors=["Bahl L", "Ramabhadran B", "Elhadad M"], year=2013,
        title="Conversational understanding as an AI-hard problem: A progress report on the Richard-SimON project",
    )
    other = _resolved("Project progress report", ["", "A Isakov", ""], 1997)
    assert C._reject_bad_fuzzy_match(cited, other)
    assert C._reject_bad_fuzzy_match(cited, _resolved("Project progress report", ["", ""], 1997))
    # …and a blank entry is not reported as the first author either.
    issue = C._author_issue(["Kaiming He", "Xiangyu Zhang"], ["", "Kaiming He", "Xiangyu Zhang"])
    assert issue is None


def test_the_right_title_under_the_wrong_authors_is_kept_and_reported():
    """…whereas the SAME title credited to the wrong people is a finding about the
    citation, not a wrong match: it stays resolved and says so."""
    cited = CitedAs(raw="", authors=["John Fakeman", "Jane Invented"], title=_RESNET, year=2016)
    real = _resolved(_RESNET, _HE)
    assert C._reject_bad_fuzzy_match(cited, real) is None
    assert "first-author mismatch" in (C._author_issue(cited.authors, real.authors) or "")


def test_invented_coauthors_behind_a_correct_first_author_are_reported():
    """Right first author, two names that are not on the paper: the shape a fabricated
    reference most often has, and one a first-author check plus a 50% bar let through."""
    issue = C._author_issue(["Kaiming He", "Xiangyu Zhang", "Jane Invented", "Bob Madeup"], _HE)
    assert issue and "2 cited authors are not on the canonical author list" in issue
    assert "Jane Invented" in issue
    # An abbreviated but correct list is clean, and so is "et al." truncation.
    assert C._author_issue(["He, K.", "Zhang, X.", "Ren, S.", "Sun, J."], _HE) is None
    assert C._author_issue(["Kaiming He"], _HE) is None
    # The new rule needs a record at least as long as the citation: one carrying fewer
    # authors is truncated, and every name past its end would "not be on the paper".
    issue = C._author_issue(["Kaiming He", "Xiangyu Zhang", "Jane Invented", "Bob Madeup"], _HE[:3])
    assert issue is None or "not on the canonical author list" not in issue


def test_a_name_spelled_differently_is_not_an_invented_coauthor():
    """Replayed over 744 published references, the rule above named nine "invented"
    co-authors; eight were one person under two spellings, or not a name at all."""
    clean = [
        # what a PDF leaves in an author field besides the name
        (["Noam Shazeer", "*Azalia Mirhoseini", "*Krzysztof Maziarz", "and Jeff Dean. 2017"],
         ["Noam Shazeer", "Azalia Mirhoseini", "Krzysztof Maziarz", "Jeff Dean"]),
        (["Mark Chen", "Jerry Tworek", "Henrique Ponde De Oliveira Pinto", "et al"],
         ["Mark Chen", "Jerry Tworek", "Heewoo Jun", "Henrique Pondé de Oliveira Pinto"]),
        (["Ashish Vaswani", "Ł ukasz Kaiser", "and Illia Polosukhin. 2017"],
         ["Ashish Vaswani", "Lukasz Kaiser", "Illia Polosukhin"]),
        (["Devis Tuia", "Fernando Pérez- Cruz", "and Gustavo Camps-Valls"],
         ["D. Tuia", "F. Pérez-Cruz", "Gustau Camps-Valls"]),
        # transliteration, and an accent command a source failed to decode
        (["Frikha, A.", "Krompaß, D.", "Köpken, H.G.", "Tresp, V."],
         ["A. Frikha", "Denis Krompass", "Hans-Georg Koepken", "Volker Tresp"]),
        (["Henderson M", "Mrkšic N", "Wen T-H"],
         ["Matthew Henderson", "Nikola Mrkvsi'c", "Tsung-Hsien Wen"]),
        # a record that ran its names together
        (["Betker, J.", "Brooks, T.", "Lee, J", "Guo, Y"],
         ["James Betker", "†. TimBrooks", "JoyceLee", "YufeiGuo"]),
        # the same mark written two ways
        (["Lidiya Murakhovs’ka", "Philippe Laban"], ["Lidiya Murakhovs'ka", "Philippe Laban"]),
    ]
    for cited, canonical in clean:
        assert C._author_issue(cited, canonical) is None, (cited, C._author_issue(cited, canonical))
    # …and the people who really are not on the paper are still named.
    issue = C._author_issue(["Kaiming He", "Xiangyu Zhang", "Jane Invented", "and Bob Madeup. 2016"], _HE)
    assert issue and "2 cited authors are not on the canonical author list" in issue
    assert "'Bob Madeup'" in issue


def test_one_person_under_two_spellings_is_one_person():
    assert C._same_author("Zhang Wei", "Wei Zhang")              # family name first / last
    assert C._same_author("Martin Luther King Jr.", "King, M. L.")
    assert C._same_author("Müller, Jan", "Jan Muller")
    assert not C._same_author("Wei Zhang", "Wei Wang")


def test_an_acronym_and_the_name_it_stands_for_are_one_venue():
    """The commonest correct citation there is — "NeurIPS" for a venue the record spells
    out — shares no word with it, and was reported as a venue mismatch."""
    same = [("NeurIPS", "Advances in Neural Information Processing Systems"),
            ("ICML", "International Conference on Machine Learning"),
            ("TPAMI", "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
            ("EMNLP", "Conference on Empirical Methods in Natural Language Processing"),
            ("CVPR", "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)"),
            ("ICML", "Proceedings of Machine Learning Research")]   # a series, not a venue
    for cited, canonical in same:
        assert C._venues_match(cited, canonical), (cited, canonical)
    for cited, canonical in [("ICML", "International Conference on Learning Representations"),
                             ("NeurIPS", "IEEE Conference on Computer Vision and Pattern Recognition"),
                             ("ACL", "Nature")]:
        assert not C._venues_match(cited, canonical), (cited, canonical)


def test_titles_are_compared_with_diacritics_folded():
    assert C._strings_match("Schrodinger bridges", "Schrödinger Bridges")


# ── resolution ───────────────────────────────────────────────────────────────
def test_first_author_agreement_does_not_assume_the_surname_comes_first():
    """For a given-name-first bibliography the first word is a given name; the agreement
    test compared it to the candidate's surname, never passed, and every such reference
    ran the whole title tier — five lookups where one had already found the paper."""
    from citation_verifier.grounding import resolver as R

    cand = Candidate(source="s2", title="t", authors=["Ashish Vaswani"])
    for ref in ('Ashish Vaswani, Noam Shazeer "Attention is all you need" 2017',
                "A. Vaswani, N. Shazeer, and I. Polosukhin. Attention is all you need.",
                "Vaswani, A., Shazeer, N. Attention is all you need.",
                "Vaswani A, Shazeer N. Attention is all you need. 2017"):
        assert R._first_author_agrees(ref, cand) is True, ref
    other = Candidate(source="s2", title="t", authors=["Premkumar T. Devanbu"])
    assert R._first_author_agrees('Hindle, A., Barr, E. T. "On the naturalness"', other) is False
    assert R._first_author_agrees('"A title with no authors"', other) is None
    assert R._cand_first_surname(Candidate(source="x", authors=["Müller, Jan"])) == "muller"


def test_title_tier_stops_at_the_first_match_whose_author_agrees(monkeypatch):
    from citation_verifier.grounding import paper_lookup
    from citation_verifier.grounding.resolver import MultiSourceResolver

    calls: list[str] = []
    hit = {"source": "s2", "title": "Attention is all you need",
           "authors": ["Ashish Vaswani", "Noam Shazeer"], "year": 2017,
           "abstract": "We propose a new simple network architecture, the Transformer."}

    def s2_match(title, api_key=None):
        calls.append("s2")
        return [hit]

    def other(*_a, **_k):
        calls.append("other")
        return []

    monkeypatch.setattr(paper_lookup, "search_semantic_scholar_match", s2_match)
    for name in ("search_crossref_by_fields", "search_arxiv_by_title", "search_crossref",
                 "search_arxiv", "search_dblp", "search_semantic_scholar", "search_openalex"):
        monkeypatch.setattr(paper_lookup, name, other)
    got = MultiSourceResolver(validate_urls=False)._resolve_uncached(
        'Ashish Vaswani, Noam Shazeer "Attention is all you need" NeurIPS 2017')
    assert got is not None and got.title == "Attention is all you need"
    assert calls == ["s2"]


def test_an_incomplete_lookup_is_not_reported_as_a_search_that_found_nothing():
    """Every source fails soft to "no results", which made a rate-limited source read
    exactly like one that has no such paper — and "searched five sources, found nothing"
    is the strongest hint there is that a reference was invented."""
    class Quiet:
        sources = ("crossref", "arxiv")

        def resolve(self, key, reference, /):
            return None

        def failed_sources(self):
            return []

    class Throttled(Quiet):
        def failed_sources(self):
            return ["Semantic Scholar", "arXiv"]

    def note(resolver) -> str:
        rec = CitationRecord(paper_id="p", claim_id="c", cite_key="k",
                             claim=Claim(claim_id="c", text=""),
                             cited_as=CitedAs(raw="A. Author. A paper. 2020.", title="A paper"))
        return C.fill_correctness(rec, resolver=resolver).metadata_issues[0]

    assert note(Quiet()).startswith("could not retrieve a confident match — searched")
    incomplete = note(Throttled())
    assert "lookup incomplete" in incomplete and "Semantic Scholar, arXiv" in incomplete
    assert "NOT fully searched" in incomplete


def test_failures_are_recorded_per_thread_and_a_404_is_not_one(monkeypatch):
    import urllib.error

    from citation_verifier.grounding import paper_lookup as pl

    monkeypatch.setattr(pl.time, "sleep", lambda _s: None)

    def raising(code):
        def urlopen(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
        return urlopen

    pl.reset_failures()
    monkeypatch.setattr(pl.urllib.request, "urlopen", raising(404))
    try:
        pl._get("https://api.crossref.org/works/x", retries=1)
    except urllib.error.HTTPError:
        pass
    assert pl.failures() == []                    # the source ANSWERED: no such paper
    monkeypatch.setattr(pl.urllib.request, "urlopen", raising(429))
    try:
        pl._get("https://api.semanticscholar.org/graph/v1/paper/x", retries=1)
    except urllib.error.HTTPError:
        pass
    assert pl.failures() == ["api.semanticscholar.org"]


# ── a document with more than one reference list ─────────────────────────────
def _line(text, *, page=1, x0=50.0, y0=100.0, size=9.0, col=0):
    from citation_verifier.extract.pdf_refs import _Line

    return _Line(page=page, text=text, x0=x0, y0=y0, size=size, col=col)


_MAIN_LIST = [
    "[1] A. Alpha and B. Able. Fast methods for sorting large arrays. In ICML, 2019.",
    "[2] C. Beta. Robust estimation under heavy noise. In NeurIPS, 2020.",
    "[3] D. Gamma and E. Delta. Learning with fewer labels. JMLR, 2021.",
    "[4] F. Epsilon. A survey of optimisation tricks. arXiv preprint, 2021.",
]
_APPENDIX_LIST = [
    "[1] Z. Zeta and Y. Yota. Proof techniques for convergence bounds. In COLT, 2015.",
    "[2] X. Xi. Concentration inequalities revisited. Annals of Statistics, 2012.",
    "[3] W. Omega. Matrix perturbation theory in practice. SIAM Review, 2010.",
]


def test_every_reference_list_in_a_document_is_located():
    """A supplement brings its own bibliography, numbered from 1 again. Taking the list
    after the LAST "References" heading — the rule that skips a table's column header —
    kept only the supplement's."""
    from citation_verifier.extract.pdf_refs import _candidate_blocks, _select_sections

    lines = (
        [_line("Some body text of the paper [1].", y0=80)]
        + [_line("References", y0=100, size=12)]
        + [_line(t, y0=120 + 12 * i) for i, t in enumerate(_MAIN_LIST)]
        + [_line("A Proofs", page=2, y0=60, size=12)]                       # appendix heading: a stop
        + [_line("The bound follows [1].", page=2, y0=80)]
        + [_line("References", page=2, y0=100, size=12)]
        + [_line(t, page=2, y0=120 + 12 * i) for i, t in enumerate(_APPENDIX_LIST)]
    )
    sections = _select_sections(_candidate_blocks(lines))
    assert [len(s) for s in sections] == [4, 3]
    assert "Fast methods" in sections[0][0][1] and "Proof techniques" in sections[1][0][1]
    assert [k for k, _t in sections[1]] == ["ref-1", "ref-2", "ref-3"]      # keys are local


def test_a_references_heading_with_no_list_after_it_is_not_a_reference_section():
    """A comparison table with a "References" column prints that word on a line of its
    own; what follows is the rest of the paper, not a bibliography."""
    from citation_verifier.extract.pdf_refs import _candidate_blocks, _select_sections

    prose = [f"This line of body text number {i} discusses the method at length." for i in range(12)]
    lines = (
        [_line("References", y0=40, size=9)]                                  # the table's column header
        + [_line(t, y0=60 + 12 * i) for i, t in enumerate(prose)]
        + [_line("References", page=2, y0=100, size=12)]
        + [_line(t, page=2, y0=120 + 12 * i) for i, t in enumerate(_MAIN_LIST)]
    )
    sections = _select_sections(_candidate_blocks(lines))
    assert len(sections) == 1 and len(sections[0]) == 4


def test_the_only_bibliography_is_kept_even_when_it_fails_the_list_test():
    """The test tells several candidates apart; it must never discard the one list a
    document has (here: entries with no year at all)."""
    from citation_verifier.extract.pdf_refs import _candidate_blocks, _select_sections

    undated = ["[1] A. Alpha. An undated manuscript on sorting.",
               "[2] B. Beta. Another undated manuscript on noise.",
               "[3] C. Gamma. A third undated manuscript on labels."]
    lines = [_line("References", size=12)] + [_line(t, y0=120 + 12 * i) for i, t in enumerate(undated)]
    assert [len(s) for s in _select_sections(_candidate_blocks(lines))] == [3]


def _two_list_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()

    def page(items):
        pg = doc.new_page(width=612, height=792)
        y = 72.0
        for text, size in items:
            for ln in text.split("\n"):
                pg.insert_text((72, y), ln, fontsize=size)
                y += size + 5
            y += 4

    page([("A Study of Things", 16), ("1 Introduction", 12),
          ("Sorting is fast in practice [1]. Robust estimators tolerate noise [2].\n"
           "Fewer labels can suffice for learning [3], and tuning tricks matter [4].\n"
           "We build on the estimator of [2] throughout this paper.", 10),
          ("References", 12), *[(r, 9) for r in _MAIN_LIST]])
    page([("A Proofs", 12),
          ("The bound follows the technique of [1]. Concentration is standard [2].\n"
           "We also use a perturbation argument [3] in the last step.", 10),
          ("References", 12), *[(r, 9) for r in _APPENDIX_LIST]])
    doc.save(str(path))


def test_a_marker_is_read_against_its_own_reference_list(tmp_path):
    """With two lists both numbered from 1, every ``[n]`` of the main text was bound to
    the supplement's n-th entry — each claim checked against a paper it never cited — and
    the main bibliography itself was scanned as though it were body text."""
    import pytest

    pytest.importorskip("fitz")
    pdf = tmp_path / "two_lists.pdf"
    _two_list_pdf(pdf)
    records = P.PdfExtractor().extract(PaperSource(paper_id="p", pdf_path=str(pdf)))
    title = {r.cite_key: r.cited_as.title for r in records}

    assert title["ref-1"] == "Fast methods for sorting large arrays"         # not the appendix's [1]
    assert title["ref-2"] == "Robust estimation under heavy noise"
    assert "ref-4" in title                                                   # the list has four
    assert title["s2-ref-1"] == "Proof techniques for convergence bounds"
    assert title["s2-ref-3"] == "Matrix perturbation theory in practice"
    # the appendix claim is attached to the appendix reference…
    proof = next(r for r in records if "technique of" in r.claim.text)
    assert proof.cite_key == "s2-ref-1"
    # …and no reference-list line was mistaken for a sentence of the paper.
    assert not any(r.claim.text.startswith("References") or "In ICML, 2019" in r.claim.text
                   for r in records)
    assert set(P.load_pdf_references(pdf)) == {"ref-1", "ref-2", "ref-3", "ref-4",
                                               "s2-ref-1", "s2-ref-2", "s2-ref-3"}


def test_lists_that_cannot_be_placed_in_the_text_fall_back_to_the_single_list_reading(monkeypatch):
    """When the layout parse and the flat text cannot be lined up, nothing is guessed:
    one list (the last found) and the text before the last heading, exactly as before."""
    text = "Body cites [1] here.\nReferences\n[1] A. Alpha. Some paper title here. 2019.\n"
    only = {"ref-1": CitedAs(raw="Completely different text that is not in the document 2020")}
    other = {"ref-1": CitedAs(raw="Another entry that is also nowhere to be found 2021")}
    monkeypatch.setattr(P, "_sections_from_layout", lambda _p: [only, other])
    sections, bodies = P.load_pdf_reference_sections("x.pdf", text=text)
    assert sections == [other] and bodies == ["Body cites [1] here."]


def test_a_second_list_is_shown_as_such_in_the_report():
    from citation_verifier.render import _cite_marker

    assert _cite_marker("ref-6") == "[6]"
    assert _cite_marker("s2-ref-6") == "[6, list 2]"
    assert _cite_marker("smith2020") == "[smith2020]"


# ── the report flags what it cannot find; it does not call it fabricated ─────
def test_references_that_were_not_found_are_listed_for_a_person_to_check():
    """The pipeline never sets ``exists = no``: a reference missing from the indexes is
    what an invented one looks like, and equally a book, a thesis or last week's preprint.
    So they are listed — a search that completed and found nothing apart from one that
    was cut short — and the summary does not print a reassuring "fabricated: 0"."""
    from citation_verifier.render import render_summary
    from citation_verifier.schema import Exists, SupportsClaim

    def rec(key, exists, issue=None, title="A paper"):
        return CitationRecord(
            paper_id="p", claim_id=f"c-{key}", cite_key=key, claim=Claim(claim_id="c", text="x"),
            cited_as=CitedAs(raw=f"A. Author. {title}. 2024.", title=title, authors=["A. Author"], year=2024),
            exists=exists, supports_claim=SupportsClaim.INCONCLUSIVE,
            metadata_issues=[issue] if issue else [],
        )

    searched = "could not retrieve a confident match — searched Crossref, arXiv, DBLP"
    out = render_summary([
        rec("fine", Exists.YES),
        rec("ghost", Exists.UNRESOLVED, searched, "Hyper-efficient everything"),
        rec("ghost", Exists.UNRESOLVED, searched, "Hyper-efficient everything"),   # cited twice
        rec("throttled", Exists.UNRESOLVED, "lookup incomplete — arXiv did not respond", "Probably real"),
    ])
    assert "### Not found — check by hand (2)" in out
    assert "never declares a reference fabricated" in out
    found_nothing, cut_short = out.index("check these first (1)"), out.index("NOT fully searched")
    assert found_nothing < out.index("`ghost`") < cut_short < out.index("`throttled`")
    assert out.count("`ghost`") == 1
    assert "fabricated) | 0" not in out

    clean = render_summary([rec("fine", Exists.YES)])
    assert "Not found" not in clean
    # A backend that DID establish a fabrication still gets its row.
    assert "no (fabricated) | 1" in render_summary([rec("fake", Exists.NO)])


# ── one work, several records: the reprint, the repost, the later edition ────────────
def _cand(title, authors, year, source="crossref", doi=""):
    from citation_verifier.interfaces import Candidate

    return Candidate(source=source, title=title, authors=authors, year=year, doi=doi,
                     abstract="An abstract, so nothing is fetched to top it up.")


def test_among_records_of_one_work_the_one_dated_as_cited_wins():
    """Crossref holds the 1966 ELIZA article and its 1983 anniversary reprint under the
    same title and author. Which came back first decided the match, and the reprint's
    year was then reported as a year mismatch in a correct citation."""
    from citation_verifier.grounding.resolver import MultiSourceResolver

    title = "ELIZA - a computer program for the study of natural language communication between man and machine"
    ref = f'Weizenbaum J "{title}" 1966'     # the query the correctness stage builds
    reprint = _cand(title, ["Joseph Weizenbaum"], 1983, doi="10.1145/357980.357991")
    original = _cand(title, ["Joseph Weizenbaum"], 1966, doi="10.1145/365153.365168")
    r = MultiSourceResolver(validate_urls=False)
    assert r._match(ref, [reprint, original]).year == 1966
    assert r._match(ref, [original, reprint]).year == 1966
    assert r._match(ref, [reprint]).year == 1983          # alone, it is still the work


def test_a_copy_dated_away_from_the_citation_does_not_end_the_search(monkeypatch):
    """With the index that holds the original rate-limited, the first record to turn up
    for a 2017 paper was a 2025 repost under a new DOI. It is a fallback, not an answer."""
    from citation_verifier.grounding import paper_lookup
    from citation_verifier.grounding.resolver import MultiSourceResolver, _should_search_arxiv

    title = "A careful study of small and ordinary things"
    ref = f'Vaswani A, Shazeer N, Parmar N, Uszkoreit J "{title}" 2017'
    assert not _should_search_arxiv(ref)      # nothing in it routes the title tier to arXiv
    authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    repost = _cand(title, authors, 2025, doi="10.65215/r5bs2d54")
    original = _cand(title, authors, 2017, source="dblp")
    on_arxiv = _cand(title, authors, 2017, source="arxiv")
    asked = []

    def resolver(broad, arxiv=()):
        r = MultiSourceResolver(validate_urls=False)

        def arxiv_by_title(title, n, author):
            asked.append(title)
            return list(arxiv)

        monkeypatch.setattr(paper_lookup, "search_arxiv_by_title", arxiv_by_title)
        monkeypatch.setattr(r, "_id_query_steps", lambda reference: [])
        monkeypatch.setattr(r, "_title_query_steps", lambda reference, n=4: [(lambda: [repost], ())])
        monkeypatch.setattr(r, "_fetch", lambda fn, *args: fn(*args))
        monkeypatch.setattr(r, "_broad_candidates", lambda reference, max_results=4: broad)
        return r

    assert resolver([original])._resolve_uncached(ref).year == 2017
    assert resolver([])._resolve_uncached(ref).year == 2025       # nothing better: still resolved
    # arXiv is asked by title even though the routing had skipped it for this reference.
    assert asked and resolver([], arxiv=[on_arxiv])._resolve_uncached(ref).source == "arxiv"
    # A real title under invented authors: the copy dated as cited is still the better
    # record, so the finding stays "wrong authors", not "nothing found".
    wrong = f'Huang J, Poulis A, Pappas N, Weiss R "{title}" 2017'
    assert resolver([original])._resolve_uncached(wrong).year == 2017
    # A record whose year agrees ends the search at once, as before.
    r = resolver([])
    monkeypatch.setattr(r, "_title_query_steps", lambda reference, n=4: [(lambda: [original], ())])
    monkeypatch.setattr(r, "_broad_candidates", lambda *a, **k: pytest.fail("searched on after a good match"))
    assert r._resolve_uncached(ref).year == 2017


def test_an_abbreviated_conference_name_still_routes_to_arxiv():
    """House styles abbreviate the venue word by word; that spelling shares no whole word
    with the acronym or the full name, so arXiv was never asked."""
    from citation_verifier.grounding.resolver import _should_search_arxiv

    assert _should_search_arxiv("Vaswani A, Shazeer N. Attention is all you need. "
                                "Adv Neural Inf Proces Syst. 2017;30:5998–6008.")
    assert _should_search_arxiv("He K, Zhang X. Deep residual learning for image recognition. "
                                "Proc IEEE Conf Comput Vis Pattern Recognit. 2016. p. 770–8.")
    # A journal the table does not list as a conference is routed as before.
    assert not _should_search_arxiv("Nielsen M, Skriver C. The use of chatbots in healthcare: "
                                    "Systematic review. J Med Internet Res. 2021;23(2):e20531.")


# ── the right work, with nothing to read ─────────────────────────────────────────────
_LEANDOJO = "LeanDojo: Theorem Proving with Retrieval-Augmented Language Models"
_LEANDOJO_REF = f'Yang, K., Swope, A., Gu, A. "{_LEANDOJO}" NeurIPS 2023'


def _title_tier(monkeypatch, *, s2=(), crossref=(), arxiv=()):
    """A resolver whose title tier answers from the given rows; returns (resolver, calls)."""
    from citation_verifier.grounding import paper_lookup
    from citation_verifier.grounding.resolver import MultiSourceResolver

    calls: list[str] = []

    def source(name, rows):
        def fn(*_a, **_k):
            calls.append(name)
            return [dict(r) for r in rows]
        fn.__name__ = f"fake_{name}"
        return fn

    monkeypatch.setattr(paper_lookup, "search_semantic_scholar_match", source("s2", s2))
    monkeypatch.setattr(paper_lookup, "search_crossref_by_fields", source("crossref", crossref))
    monkeypatch.setattr(paper_lookup, "search_arxiv_by_title", source("arxiv", arxiv))
    for name in ("search_crossref", "search_arxiv", "search_dblp", "search_semantic_scholar",
                 "search_openalex"):
        monkeypatch.setattr(paper_lookup, name, source("broad", ()))
    # Hermetic: no abstract is fetched by DOI from a live index to "top up" a record.
    monkeypatch.setattr(MultiSourceResolver, "_abstract_top_up", lambda self, c: "")
    return MultiSourceResolver(validate_urls=False), calls


def test_a_record_with_nothing_to_read_does_not_end_the_search(monkeypatch):
    """Semantic Scholar rate-limited, Crossref answered with the proceedings record — no
    abstract, no arXiv id — and twenty of one paper's sixty citations came back "no
    abstract or introduction to judge from" while the papers sat on arXiv."""
    authors = ["Kaiyu Yang", "Aidan Swope", "Alex Gu"]
    proceedings = {"source": "crossref", "title": _LEANDOJO, "authors": authors, "year": 2023,
                   "doi": "10.52202/075280-0944"}
    preprint = {"source": "arxiv", "title": _LEANDOJO, "authors": authors, "year": 2023,
                "arxiv_id": "2306.15626", "abstract": "Large language models have shown promise…"}

    r, calls = _title_tier(monkeypatch, crossref=[proceedings], arxiv=[preprint])
    got = r._resolve_uncached(_LEANDOJO_REF)
    assert got.arxiv_id == "2306.15626" and got.abstract
    assert "arxiv" in calls
    assert calls.count("crossref") == 1          # the work was found; no second Crossref query

    # Nothing readable anywhere: the thin record is still the answer.
    r, _ = _title_tier(monkeypatch, crossref=[proceedings])
    got = r._resolve_uncached(_LEANDOJO_REF)
    assert got is not None and got.doi == "10.52202/075280-0944" and not got.abstract

    # arXiv is asked for a readable copy even when nothing in the reference routed the
    # title tier there (the routing goes by the title's vocabulary).
    from citation_verifier.grounding.resolver import _should_search_arxiv

    plain = "A careful study of small and ordinary things"
    ref = f'Yang, K., Swope, A., Gu, A. "{plain}" 2023'
    assert not _should_search_arxiv(ref)
    r, calls = _title_tier(monkeypatch, crossref=[dict(proceedings, title=plain)],
                           arxiv=[dict(preprint, title=plain)])
    assert r._resolve_uncached(ref).arxiv_id == "2306.15626" and calls.count("arxiv") == 1

    # A record that CAN be read ends the search where it always did.
    r, calls = _title_tier(monkeypatch, s2=[dict(preprint, source="s2")], crossref=[proceedings])
    assert r._resolve_uncached(_LEANDOJO_REF).source == "s2" and calls == ["s2"]


def test_a_publisher_record_borrows_the_preprint_beside_it(monkeypatch):
    """When the publisher's record still wins — the preprint is dated years earlier than
    the citation — it takes the arXiv id and the abstract of the same-title record."""
    authors = ["Kaiyu Yang", "Aidan Swope", "Alex Gu"]
    proceedings = {"source": "crossref", "title": _LEANDOJO, "authors": authors, "year": 2023,
                   "doi": "10.52202/075280-0944"}
    early = {"source": "arxiv", "title": _LEANDOJO, "authors": authors, "year": 2019,
             "arxiv_id": "2306.15626", "abstract": "Large language models have shown promise…"}
    r, _ = _title_tier(monkeypatch, crossref=[proceedings], arxiv=[early])
    got = r._resolve_uncached(_LEANDOJO_REF)
    assert got.source == "crossref" and got.year == 2023
    assert got.arxiv_id == "2306.15626" and got.abstract.startswith("Large language models")


def test_a_hit_found_during_an_outage_is_cached_only_when_it_is_a_good_one(monkeypatch):
    """A degraded hit — the abstract-less record, the repost — is the outage's doing. In
    the cache it outlives the outage: every later run is handed the same record."""
    from citation_verifier.grounding import paper_lookup, resolve_cache
    from citation_verifier.grounding.resolver import MultiSourceResolver

    written: list[str] = []
    monkeypatch.setattr(resolve_cache, "read", lambda reference, model: None)
    monkeypatch.setattr(resolve_cache, "write", lambda reference, found: written.append(found.title))
    monkeypatch.setattr(paper_lookup, "reset_failures", lambda: None)

    def resolve(found, failed):
        r = MultiSourceResolver(validate_urls=False)
        monkeypatch.setattr(r, "_resolve_uncached", lambda reference: found)
        monkeypatch.setattr(paper_lookup, "failures", lambda: list(failed))
        written.clear()
        r.resolve("k", 'A. Author "A title" 2017')
        return bool(written)

    thin = _resolved("A title", ["A. Author"], 2017)
    rich = Resolved(source="arxiv", match_method=MatchMethod.FUZZY_TITLE, title="A title",
                    authors=["A. Author"], year=2017, arxiv_id="1701.00001")
    repost = Resolved(source="crossref", match_method=MatchMethod.FUZZY_TITLE, title="A title",
                      authors=["A. Author"], year=2025, abstract="An abstract.")
    assert resolve(thin, failed=[]) and resolve(rich, failed=[]) and resolve(repost, failed=[])
    assert resolve(rich, failed=["api.semanticscholar.org"])          # as good as it gets
    assert not resolve(thin, failed=["api.semanticscholar.org"])      # nothing to read
    assert not resolve(repost, failed=["api.semanticscholar.org"])    # dated away from the citation


def test_a_citations_own_defects_are_reported_whatever_the_lookup_found():
    goodfellow = CitedAs(
        raw="Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning, "
            "volume 1. MIT Press, 2016.",
        authors=["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville", "Yoshua Bengio"],
        title="Deep learning, volume 1", year=2016,
    )
    assert C._citation_defects(goodfellow) == ["author listed twice: 'Yoshua Bengio'"]
    title = ("Serie scientifique scientific series no unbiased estimator of the variance of "
             "k-fold cross-validation no unbiased estimator of the variance of k-fold crossvalidation")
    garbled = CitedAs(raw=f"Yoshua Bengio and Yves Grandvalet. {title}.", authors=["Yoshua Bengio", "Yves Grandvalet"], title=title)
    got = C._citation_defects(garbled)
    assert "the citation gives no year" in got and any(d.startswith("the title repeats itself") for d in got)
    # Two people can share a name: initials only, neighbours, or a record that has both.
    assert C._citation_defects(CitedAs(raw="x 2022", authors=["Gu, S.", "Chen, D.", "Bao, J.", "Chen, D."], title="T", year=2022)) == []
    assert C._citation_defects(CitedAs(raw="x 2024", authors=["Yang Jin", "Kun Xu", "Kun Xu", "Liwei Chen"], title="T", year=2024)) == []
    assert C._citation_defects(CitedAs(raw="x 2024", authors=["Ann Lee", "Bo Li", "Ann Lee"], title="T", year=2024),
                               ["Ann Lee", "Bo Li", "Ann Lee"]) == []
    # A reference cut off by extraction lost its year to the cut, not to its author.
    cut = CitedAs(raw="Zhao, Y. and Qu, Y. Analysing sequence composition. In Proceedings of the 62nd Annual Meeting of",
                  authors=["Zhao, Y.", "Qu, Y."], title="Analysing sequence composition")
    assert C._citation_defects(cut) == []


def test_the_defects_reach_the_record_even_when_nothing_resolves():
    class NoResolver:
        def resolve(self, key, ref):
            return None

        def failed_sources(self):
            return []

    rec = CitationRecord(paper_id="p", claim_id="c", cite_key="g", claim=Claim(claim_id="c", text="x"),
                         cited_as=CitedAs(raw="Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua "
                                          "Bengio. Deep learning. MIT Press, 2016.",
                                          authors=["Ian Goodfellow", "Yoshua Bengio", "Aaron Courville", "Yoshua Bengio"],
                                          title="Deep learning", year=2016))
    C.fill_correctness(rec, resolver=NoResolver())
    assert "author listed twice: 'Yoshua Bengio'" in rec.metadata_issues
