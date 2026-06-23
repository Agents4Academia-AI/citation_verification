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
    _claim_scan_body,
    _expand_num_marker,
    _normalize_pdf_text,
    _reference_parse_score,
    _sentence_around,
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
    # A title prefix ending in an acronym + colon must not leak into authors.
    a, t = _split_author_title(
        "Kulkarni P, Mahabaleshwarkar A, Gadgil K, Conversational AI: "
        "An overview of methodologies. In: ICCUBEA. 2019."
    )
    assert a == ["Kulkarni P", "Mahabaleshwarkar A", "Gadgil K"]
    assert t == "Conversational AI: An overview of methodologies"
    # space-less merged block -> no boundary, honest ([], None)
    assert _split_author_title("CarpenterR.Jabberwacky-acasestudy.In:Proceedings") == ([], None)


def test_split_author_title_keeps_et_al_caps_titles_and_diacritics_out_of_authors():
    # "et al" must never leak into a name, and the title may open with a capital
    # phrase ("Building Watson:") without leaking into the author list.
    a, t = _split_author_title(
        "Ferrucci DA, Lally A, Prager JM et al. Building Watson: An overview of "
        "the DeepQA project. In: AI Magazine. 2010"
    )
    assert a == ["Ferrucci DA", "Lally A", "Prager JM"]
    assert t == "Building Watson: An overview of the DeepQA project"
    # trailing "et al" on the last author
    a, _ = _split_author_title("Ram A, Khandelwal P et al. Alexa prize: a challenge. In: ICASSP. 2018")
    assert a == ["Ram A", "Khandelwal P"]
    # a title opening with an acronym right after the author period — authors stay clean
    a, _ = _split_author_title("Adetokunbo I, Henderson P, Hudson J. GPT-3.5-turbo: Larger models. 2021")
    assert a == ["Adetokunbo I", "Henderson P", "Hudson J"]
    # an orphaned combining mark from despacing is healed by the normalizer so the
    # split surname rejoins (real PDF artifact "Yetiştiren" -> "Yeti" + space + U+0327)
    assert "Yetistiren B" in _normalize_pdf_text("Yeti \u0327stiren B, Tuzun E. Title")


def test_split_author_title_handles_author_year_comma_initials():
    # Author-year ("Surname, I., …. Title"): the comma sits INSIDE each name, so a
    # name must stay whole (not torn into surname + initial) and an acronym title
    # prefix ("Llemma:") must be kept — the bug that flat-comma splitting caused.
    a, t = _split_author_title(
        "Azerbayev, Z., Schoelkopf, H., Paster, K., et al. Llemma: An open "
        "language model for mathematics. In ICLR, 2024."
    )
    assert a == ["Azerbayev, Z.", "Schoelkopf, H.", "Paster, K."]
    assert t == "Llemma: An open language model for mathematics"
    # an "and"-joined last author and a multi-initial "G. P." stay intact
    a, t = _split_author_title(
        "Coquand, T. and Huet, G. P. The calculus of constructions. "
        "Information and Computation, 1988."
    )
    assert a == ["Coquand, T.", "Huet, G. P."]
    assert t == "The calculus of constructions"
    # hyphenated initials ("W.-D.") and a comma inside the title both survive
    a, t = _split_author_title(
        "Key, D., Li, W.-D., and Ellis, K. I Speak, You Verify: Toward "
        "trustworthy neural program synthesis. 2024."
    )
    assert a == ["Key, D.", "Li, W.-D.", "Ellis, K."]
    assert t == "I Speak, You Verify: Toward trustworthy neural program synthesis"
    # a single author-year name
    a, t = _split_author_title("Boltzmann, L. Lectures on gas theory. Univ Press, 2022.")
    assert a == ["Boltzmann, L."] and t == "Lectures on gas theory"
    # Vancouver ("Surname I", no comma after surname) must NOT match the author-year
    # tier — it falls through to the Vancouver split unchanged.
    a, t = _split_author_title("Radford A, Wu J, Child R. Language models. OpenAI. 2019.")
    assert a == ["Radford A", "Wu J", "Child R"] and t == "Language models"


def test_parse_reference_block_drops_trailing_publisher_boilerplate():
    # The last reference must not absorb the journal's trailing "Publisher's Note \u2026
    # Springer Nature remains neutral \u2026" boilerplate (observed swallowing ref-89).
    from citation_verifier.extract.pdf import parse_reference_block

    block = (
        "1. Foo A. A first reference. 2020.\n"
        "2. Bar B. A second reference. 2021.\n"
        "3. Deng Y, Lam W. Nonfactoid question answering. IEEE Trans Neural Netw. 2023. "
        "Publisher's Note Springer Nature remains neutral with regard to jurisdictional claims."
    )
    refs = parse_reference_block(block)
    assert refs["ref-3"].title == "Nonfactoid question answering"
    assert "Publisher" not in (refs["ref-3"].raw or "")


def test_parse_reference_block_handles_author_year_despite_stray_numbers():
    # An UNNUMBERED author-year bibliography can carry stray 'N.' line-starts (a
    # wrapped DOI, an appendix list). Those must NOT flip it to the numbered path
    # and collapse the whole list into a couple of junk entries (regression: a real
    # paper parsed to 5 instead of ~79). One entry per author-year reference.
    from citation_verifier.extract.pdf import parse_reference_block

    block = (
        "Azerbayev, Z., Schoelkopf, H. Llemma: an open language model. In ICLR, 2024.\n\n"
        "Boltzmann, L. Lectures on gas theory. Univ of California Press, 2022.\n\n"
        "Coquand, T. and Huet, G. The calculus of constructions. Inf. and Comp., 1988.\n\n"
        "Curry, H. B. Functionality in combinatory logic. PNAS, 1934.\n"
        "427. URL http://dx.doi.org/10.18653/v1/2024.acl-long.427.\n"  # stray wrapped-DOI number
    )
    refs = parse_reference_block(block)
    assert len(refs) == 4  # one per author-year entry, not 1-2 junk numbered chunks
    assert any((c.title or "").startswith("Lectures on gas theory") for c in refs.values())


def test_parse_reference_block_normalizes_spaces_before_punctuation():
    from citation_verifier.extract.pdf import parse_reference_block

    block = (
        "1. Foo A. A first reference. 2020.\n"
        "2. Bar B. A second reference. 2021.\n"
        "3. Cao Y , Lin Z, Xu X, Tang Y , Zhang Z, Zhang Y . Clinic: "
        "A secure peer-to-peer healthcare blockchain framework with privacy "
        "preservation. IEEE Trans Ind Inf. 2020;16(6):4384–95."
    )
    ref = parse_reference_block(block)["ref-3"]
    assert ref.authors == ["Cao Y", "Lin Z", "Xu X", "Tang Y", "Zhang Z", "Zhang Y"]
    assert ref.title == (
        "Clinic: A secure peer-to-peer healthcare blockchain framework with privacy preservation"
    )
    assert ref.year == 2020


def test_normalize_pdf_text_dehyphenates_and_despaces():
    # 1) line-break hyphenation joined; real compounds preserved
    norm = _normalize_pdf_text("knowl- edge of state-of-the-art GPT-3 Lan-\nguage models")
    assert "knowledge" in norm and "Language" in norm
    assert "state-of-the-art" in norm and "GPT-3" in norm
    # 2) a char-spaced line collapses (entry number + title become readable);
    #    a normal line is left untouched.
    spaced = "1 1 .Y a n gZ ,G a nZ ,W a n gJ . An empirical study of GPT-3"
    out = _normalize_pdf_text(spaced + "\nThis normal sentence stays intact.")
    assert "11." in out and "Yang Z, Gan Z, Wang J" in out
    assert "This normal sentence stays intact." in out


def test_reference_parse_score_prefers_complete_fields_over_glued_text():
    good = (
        "\nReferences\n"
        "15. Bird JJ, Ekárt A, Faria DR. Chatbot interaction with artificial "
        "intelligence: human data augmentation. J Ambient Intell Humaniz Comput. "
        "2023;14(4):3129–44.\n"
        "23. Carpenter R. Jabberwacky-a case study of intractable ambiguity. "
        "In: Proceedings. ACM; 1999. p. 124–30.\n"
    )
    glued = (
        "\nReferences\n"
        "15. BirdJJ,EkártA,FariaDR.Chatbotinteractionwithartificialintelligence: "
        "human data augmentation. J Ambient Intell Humaniz Comput. 2023;14(4):3129–44.\n"
        "23. CarpenterR.Jabberwacky-acasestudyofintractableambiguity.In: "
        "Proceedings. ACM; 1999. p. 124–30.\n"
    )
    assert _reference_parse_score(good) > _reference_parse_score(glued)


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


def test_pdf_claim_scan_strips_contact_and_affiliation_noise():
    body = (
        "Today researchers are working on creating AI systems known as Conversational AI "
        "that can comprehend and\n"
        "B Gaurang Bansal\n"
        "gaurang@u.nus.edu\n"
        "Vinay Chamola\n"
        "vinay.chamola@pilani.bits-pilani.ac.in\n"
        "1 Department of Electrical and Computer Engineering, National\n"
        "University of Singapore, Singapore 119077, Singapore\n"
        "respond to human language in a manner that resembles\n"
        "human-to-human conversation [1]."
    )
    clean = _claim_scan_body(body)
    sent, _ = _sentence_around(clean, clean.index("[1]"))
    assert "@" not in sent and "Department" not in sent and "University of Singapore" not in sent
    assert "comprehend and respond to human language" in sent


def test_pdf_claim_scan_strips_page_footer_before_claim_sentence():
    body = (
        "© The Author(s), under exclusive licence to Springer Science+Business Media, LLC\n"
        "2488 Cognitive Computation (2024) 16:2487-2510\n"
        "systems, typically chatbots, virtual assistants, or voice assistants, that can "
        "understand and respond to human language in a way that simulates real conversations [ 3]."
    )
    clean = _claim_scan_body(body)
    sent, _ = _sentence_around(clean, clean.index("[ 3]"))
    assert "Springer" not in sent and "Cognitive Computation" not in sent
    assert sent.startswith("systems, typically chatbots")


def test_sentence_around_caps_table_sized_claims():
    table = " ".join(f"Column{i} value" for i in range(80)) + " Important cell [25]."
    sent, _ = _sentence_around(table, table.index("[25]"))
    assert "[25]" in sent
    assert len(sent) <= 360


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


# ── layout-aware reference segmentation (extract/pdf_refs.py) ─────────────────
# These exercise the segmentation/join logic on synthetic _Line rows, so they run
# offline (no PDF / PyMuPDF). The PyMuPDF layout pass itself is covered by the
# live two-PDF check, not the unit suite.
def _line(text, x0=50.0, y0=100.0, page=1, col=0, size=9.0):
    from citation_verifier.extract.pdf_refs import _Line

    return _Line(page=page, text=text, x0=x0, y0=y0, size=size, col=col)


def test_pdf_refs_splits_numbered_list_on_markers_and_keys_them():
    from citation_verifier.extract.pdf_refs import _is_numbered, _split_numbered

    lines = [
        _line("1. Radford A, Wu J. Language models. 2019."),
        _line("Continued on a wrapped line."),
        _line("2. Brown T, Mann B. Few-shot learners. 2020."),
        _line("3. Devlin J. BERT. 2019."),
    ]
    assert _is_numbered(lines)
    refs = _split_numbered(lines)
    assert [k for k, _ in refs] == ["ref-1", "ref-2", "ref-3"]  # keyed to the [n] markers
    body1 = dict(refs)["ref-1"]
    assert body1.startswith("Radford A")            # "1. " marker stripped
    assert "Continued on a wrapped line." in body1  # continuation merged into ref-1


def test_pdf_refs_splits_author_year_on_hanging_indent():
    from citation_verifier.extract.pdf_refs import _is_numbered, _split_by_indent

    # entry-opening lines sit at the column-left margin (x0=50); continuations indent
    lines = [
        _line("Azerbayev, Z., et al. Llemma. 2024.", x0=50.0, y0=100.0),
        _line("In ICLR, 2024.", x0=68.0, y0=112.0),
        _line("Boltzmann, L. Lectures on gas theory. 2022.", x0=50.0, y0=130.0),
    ]
    assert not _is_numbered(lines)
    refs = _split_by_indent(lines)
    assert len(refs) == 2
    assert refs[0].startswith("Azerbayev") and "In ICLR" in refs[0]  # continuation merged
    assert refs[1].startswith("Boltzmann")


def test_pdf_refs_is_numbered_rejects_stray_numbers():
    from citation_verifier.extract.pdf_refs import _is_numbered

    # a couple of stray "N." line-starts in an author-year list (a wrapped DOI tail)
    # are NOT a dense numbered bibliography
    lines = [
        _line("Azerbayev, Z. Llemma. 2024."),
        _line("427. tail of a wrapped DOI"),
        _line("Boltzmann, L. Lectures. 2022."),
    ]
    assert not _is_numbered(lines)


def test_pdf_refs_join_glues_urls_and_dehyphenates():
    from citation_verifier.extract.pdf_refs import _join

    assert _join(["Proceed-", "ings of ACL"]) == "Proceedings of ACL"
    assert (
        _join(["https://aclanthology.org/", "2020.acl-main.1"])
        == "https://aclanthology.org/2020.acl-main.1"
    )


# ── table-citation detection (relevance is not assessed for these) ───────────
def test_latex_detects_cite_inside_table_environment():
    from citation_verifier.extract.latex import _in_table, _table_spans

    text = (
        r"Prose cites \cite{a} here. "
        r"\begin{table}\caption{C}\begin{tabular}{cc} A & \cite{b} \\ \end{tabular}\end{table} "
        r"Then \cite{c} back in prose."
    )
    spans = _table_spans(text)
    assert not _in_table(spans, text.index(r"\cite{a}"))  # prose before the table
    assert _in_table(spans, text.index(r"\cite{b}"))       # inside the table float
    assert not _in_table(spans, text.index(r"\cite{c}"))   # prose after the table


def test_pdf_table_caption_claim_is_flagged():
    from citation_verifier.extract.pdf import _TABLE_CAPTION_RE

    assert _TABLE_CAPTION_RE.match("Table 3 Comparison of GPT models [12]")
    assert _TABLE_CAPTION_RE.match("Tab. 2 Datasets used")
    assert not _TABLE_CAPTION_RE.match("We compare against prior work [12].")
    # a prose sentence that merely *references* a table is not a table cell
    assert not _TABLE_CAPTION_RE.match("As shown in Table 3, the method [12] wins.")


def test_pdf_refs_line_text_reinserts_spaces_from_kerning_gaps():
    # Some PDFs render inter-word spaces as positional gaps with no space glyph;
    # _line_text must re-insert them from the glyph bboxes (regression: a reference
    # came out glued as "BirdJJ,EkártA,FariaDR.Chatbot…", losing author+title).
    from citation_verifier.extract.pdf_refs import _line_text

    size = 8.0

    def ch(c, x0, x1):
        return {"c": c, "bbox": (x0, 0.0, x1, size)}

    # "AB" adjacent (gap 0), then a wide gap (1.5 > 0.1*size), then "CD" -> "AB CD"
    glued = {"spans": [{"size": size, "chars": [
        ch("A", 0.0, 4.0), ch("B", 4.0, 8.0), ch("C", 9.5, 13.5), ch("D", 13.5, 17.5),
    ]}]}
    assert _line_text(glued) == "AB CD"
    # an existing space glyph is preserved, not doubled
    spaced = {"spans": [{"size": size, "chars": [
        ch("A", 0.0, 4.0), ch(" ", 4.0, 6.0), ch("B", 6.0, 10.0),
    ]}]}
    assert _line_text(spaced) == "A B"
    # a decimal/version number must NOT be split, even though the '.' glyph leaves
    # a wide gap before the next digit ("GPT-3.5" must stay "3.5", not "3. 5")
    decimal = {"spans": [{"size": size, "chars": [
        ch("3", 0.0, 4.0), ch(".", 4.0, 5.0), ch("5", 6.6, 10.6),
    ]}]}
    assert _line_text(decimal) == "3.5"


# ── B1: heal mangled diacritics in the layout parser ─────────────────────────
def test_pdf_refs_clean_heals_mangled_diacritics():
    from citation_verifier.extract.pdf_refs import _clean

    assert _clean("Vuli´c I") == "Vulic I"            # orphan acute between letters
    assert _clean("Yeti¸stiren B") == "Yetistiren B"   # spacing cedilla
    assert _clean('Hakkani-T"ur D') == "Hakkani-Tur D"  # mangled umlaut (straight quote)
    assert _clean("Rozi`ere B") == "Roziere B"          # grave/backtick
    assert _clean("O'Brien J") == "O'Brien J"           # apostrophe is NOT a diacritic
    assert _clean("Özsoy I") == "Özsoy I"               # precomposed accent kept intact


# ── A1: detect comparison-table body dumps (skip their relevance) ────────────
def test_pdf_looks_like_table_dump_vs_prose():
    from citation_verifier.extract.pdf import _looks_like_table_dump

    # table body: markers each followed by a year or a capitalized cell
    assert _looks_like_table_dump("ELIZA [19] 1966 Chatbot SHRDLU [20] 1970 Task PARRY [21] 1972 Bot")
    assert _looks_like_table_dump("ELIZA [19] No N/A SHRDLU [20] No Specific PARRY [21] No Limited")
    # ordinary multi-citation prose (markers followed by punctuation/lowercase) must NOT fire
    assert not _looks_like_table_dump("Several works [1], [2], [3], and [4] proposed methods.")
    assert not _looks_like_table_dump("Technologies like Watson [25], Siri [26], Alexa [27] exist.")


# ── author-year PDF citation extraction (hyperlink + text) ───────────────────
def test_bind_author_year_matches_surname_year_and_disambiguates():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    refs = {
        "ref-1": CitedAs(authors=["Yang, K.", "Swope, A."], title="Leandojo: theorem proving", year=2023),
        "ref-2": CitedAs(authors=["Yang, L.", "Zhang, Z."], title="Diffusion models: a survey", year=2025),
        "ref-3": CitedAs(authors=["Martin-Lof, P."], title="An intuitionistic theory of types", year=1998),
        "ref-4": CitedAs(authors=["Muennighoff, N."], title="Scaling data-constrained models", year=2023),
        "ref-5": CitedAs(authors=["Coquand, T.", "Huet, G."], title="The calculus of constructions", year=1988),
    }

    def b(k):
        return _bind_author_year(k, refs)

    # two same-surname refs disambiguated by year (+ keyword)
    assert b("yang2023leandojo").title.startswith("Leandojo")
    assert b("yang2025diffusionmodelssurvey").title.startswith("Diffusion")
    # CamelCase key + hyphenated surname
    assert b("MartinLofTypeTheory1998").title.startswith("An intuitionistic")
    # a unique surname binds even when the cited year is off by 2 (2025 vs 2023)
    assert b("muennighoff2025scaling").authors[0] == "Muennighoff, N."
    # DBLP-style key: a distinctive surname embedded in the key
    assert b("DBLP:journals/iandc/CoquandH88").title.startswith("The calculus")
    # no surname/year match -> no guess
    assert b("nonexistent2020foo") is None


def test_extract_text_citations_finds_author_year_forms():
    from citation_verifier.extract.pdf_links import extract_text_citations

    text = (
        "We build on prior work (Yang et al., 2023) and extend it greatly. "
        "Lightman et al. (2024) introduced a learned verifier for this. "
        "Others (Smith and Jones, 2020) disagree with the approach."
    )
    sites = extract_text_citations(text)
    keys = {s["cite_key"] for s in sites}
    assert {"yang2023", "lightman2024", "smith2020"} <= keys
    assert all(s["claim"] for s in sites)


# ── identifier / title / bibkey fixes (A6, A7, A8) ───────────────────────────
def test_parse_ref_entry_extracts_arxiv_id_from_url():
    from citation_verifier.extract.pdf import _parse_ref_entry

    c = _parse_ref_entry("Yang L. Diffusion models: a survey, 2025. URL https://arxiv.org/abs/2209.00796.")
    assert c.arxiv_id == "2209.00796"
    assert _parse_ref_entry("Foo B. Bar baz. arxiv.org/pdf/2102.00182v2").arxiv_id == "2102.00182v2"


def test_title_from_strips_trailing_year_only_after_comma():
    from citation_verifier.extract.pdf import _title_from

    assert _title_from("Diffusion models: a comprehensive survey, 2025") == (
        "Diffusion models: a comprehensive survey"
    )
    # no comma before the number -> kept (not a year suffix)
    assert _title_from("A study of GPT 2 behaviour") == "A study of GPT 2 behaviour"


def test_bind_author_year_handles_digit_and_org_keys():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    refs = {
        "ref-1": CitedAs(authors=["Qwen Team"], title="Qwen3 technical report", year=2025, arxiv_id="2505.09388"),
    }
    b = _bind_author_year("qwen3-2025", refs)
    assert b is not None and b.arxiv_id == "2505.09388"


# ── claim extraction reading order + sentence boundaries (A1, A2) ─────────────
def test_order_reading_clusters_lines_despite_baseline_jitter():
    from citation_verifier.extract.pdf_links import _order_reading

    def w(x0, y0, t):
        return (x0, y0, x0 + 10, y0 + 10, t, 0, 0, 0)

    # "Expr" sits 0.6pt higher (inline code) — round(y) would bucket it as an
    # earlier line and scramble; clustering keeps the line "translate Expr into".
    words = [w(50, 100, "translate"), w(80, 99.4, "Expr"), w(110, 100, "into"), w(50, 114, "next")]
    assert [x[4] for x in _order_reading(words)] == ["translate", "Expr", "into", "next"]


def test_sentence_not_chopped_at_et_al_or_initials():
    from citation_verifier.extract.pdf_links import _sentence_around

    txt = "Prior work (Hindle et al., 2012) showed code is natural. A new sentence."
    claim, _ = _sentence_around(txt, txt.index("Hindle"))
    assert "showed code is natural" in claim  # not truncated at "et al."


def test_split_author_title_handles_spaced_hyphen_initial():
    # "K.- F." (a despacing artifact adds a space after the hyphen) must not break
    # the author-year run — else the whole list falls back to a comma-split that
    # tears "Xue, B." into two and flags a false author mismatch.
    from citation_verifier.extract.pdf import _split_author_title

    a, t = _split_author_title(
        "Xue, B., Zhu, Q., and Wong, K.- F. Reliablemath: Benchmark of reliable "
        "mathematical reasoning on large language models. 2025."
    )
    assert a == ["Xue, B.", "Zhu, Q.", "Wong, K.- F."]
    assert t == "Reliablemath: Benchmark of reliable mathematical reasoning on large language models"
