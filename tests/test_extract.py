"""
Offline tests for the LaTeX extractor's bibliography parsing — in particular the
inline `\\begin{thebibliography}` path (arXiv e-prints that ship NO separate
.bbl/.bib, e.g. 1706.03762), which previously left every cited_as empty.
"""

from __future__ import annotations

from citation_verifier.extract.latex import (
    _parse_bibitems,
    _strip_line_comments,
    decode_tex_accents,
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
from citation_verifier.grounding.resolver import _likely_titles


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


def test_split_author_title_lncs_multiword_surnames():
    """LNCS colon style with compound / nobiliary surnames ("Van Der Maaten",
    "Gontijo Lopes", "Karagol Ayan") — the author run must not eat the last author
    into the title (regression: ECCV refs split at the wrong author)."""
    a, t = _split_author_title(
        "Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q.: "
        "Densely connected convolutional networks. In: CVPR. pp. 4700-4708 (2017)"
    )
    assert a == ["Huang, G.", "Liu, Z.", "Van Der Maaten, L.", "Weinberger, K.Q."]
    assert t == "Densely connected convolutional networks"
    # multi-word surnames mid-list + trailing "et al." before the colon
    a, t = _split_author_title(
        "Saharia, C., Gontijo Lopes, R., Karagol Ayan, B., Salimans, T., et al.: "
        "Photorealistic text-to-image diffusion models. In: NeurIPS (2022)"
    )
    assert "Gontijo Lopes, R." in a and "Karagol Ayan, B." in a
    assert t == "Photorealistic text-to-image diffusion models"
    # a plain single-word-surname LNCS ref still parses (no regression)
    a, t = _split_author_title("Vaswani, A., Shazeer, N.: Attention is all you need. In: NeurIPS (2017)")
    assert a == ["Vaswani, A.", "Shazeer, N."] and t == "Attention is all you need"


def test_clean_ref_body_heals_capitalized_hyphen_split():
    """A line-break-split hyphenated name with a capitalized tail ("Ming- Hsuan")
    keeps its hyphen but loses the stray space, so a long given-name-first author
    list parses instead of leaking the venue into the title (ICCV VideoPoet)."""
    from citation_verifier.extract.pdf import _clean_ref_body

    assert _clean_ref_body("Ming- Hsuan Yang") == "Ming-Hsuan Yang"
    assert _clean_ref_body("Transformer- XL") == "Transformer-XL"
    # a lowercase tail is word-wrap, handled upstream — not re-hyphenated here
    assert _clean_ref_body("scale- invariant") == "scale- invariant"

    raw = (
        "Dan Kondratyuk, Lijun Yu, Hartwig Adam, Ming- Hsuan Yang, David A Ross, and Lu Jiang. "
        "Videopoet: A large language model for zero-shot video generation. "
        "In Proceedings of the 41st International Conference on Machine Learning, 2024"
    )
    a, t = _split_author_title(_clean_ref_body(raw))
    assert a[-1] == "Lu Jiang" and any("Ming-Hsuan" in x for x in a)
    assert t == "Videopoet: A large language model for zero-shot video generation"


def test_url_extraction_heals_pdf_injected_spaces():
    """A URL split by PDF spaces after "."/"-"/"/" is rejoined; trailing prose
    (which opens with a capital) is never swallowed (Genie 2, QwQ)."""
    from citation_verifier.extract.pdf import _parse_ref_entry

    u = _parse_ref_entry(
        "J Parker-Holder et al. Genie 2. URL: https://deepmind. google/discover/blog/"
        "genie- 2-a-large-scale-foundation-world-model/. Accessed 2024."
    ).url
    assert u == "https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/"
    assert _parse_ref_entry("Qwen Team. Qwq. URL https://qwenlm. github. io/blog/qwq-32b-preview.").url == (
        "https://qwenlm.github.io/blog/qwq-32b-preview"
    )
    # a clean URL followed by prose is untouched, and prose stays out
    assert _parse_ref_entry("X. A paper. https://arxiv.org/abs/2406.12345. In NeurIPS, 2024.").url == (
        "https://arxiv.org/abs/2406.12345"
    )
    assert _parse_ref_entry("Y. Tool. See https://github.com/org/repo for code. 2023.").url == (
        "https://github.com/org/repo"
    )


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


def test_split_author_title_handles_year_delimited_author_year():
    # AAAI semicolon style: "Surname, I.; …; and Surname, I. YEAR. Title." Names are
    # semicolon-separated and the YEAR sits between the authors and the title.
    a, t = _split_author_title(
        "Cai, H.; Zhang, P.; Dong, H.; and Yuan, T. 2023. Adversarial example-based "
        "test case generation for speech systems. Software Testing, 33(5)."
    )
    assert a == ["Cai, H.", "Zhang, P.", "Dong, H.", "Yuan, T."]
    assert t == "Adversarial example-based test case generation for speech systems"
    # a YEAR + disambiguation letter ("2025a") must not be read as the title
    a, t = _split_author_title(
        "Chen, Y.; Li, B.; and Ren, K. 2025a. Taught well learned ill: Towards "
        "distillation-conditional backdoor attack. In NeurIPS."
    )
    assert a == ["Chen, Y.", "Li, B.", "Ren, K."]
    assert t == "Taught well learned ill: Towards distillation-conditional backdoor attack"
    # ACL/EMNLP given-name-first with a year delimiter: "First Last, …, and First
    # Last. YEAR. Title." — the year must not be read as the title.
    a, t = _split_author_title(
        "Marthe Ballon, Andres Algaba, and Vincent Ginis. 2025. The relationship "
        "between reasoning and performance in large language models. arXiv:2502.01234."
    )
    assert a == ["Marthe Ballon", "Andres Algaba", "Vincent Ginis"]
    assert t == "The relationship between reasoning and performance in large language models"


def test_split_author_title_handles_lncs_colon_style():
    # Springer/LNCS: "Surname, I., …, Surname, I.: Title. In: Venue (Year)". The
    # author run ends at a COLON; the period tiers would steal the last author.
    a, t = _split_author_title(
        "Arjovsky, M., Chintala, S., Bottou, L.: Wasserstein generative adversarial "
        "networks. In: International conference on machine learning (2017)."
    )
    assert a == ["Arjovsky, M.", "Chintala, S.", "Bottou, L."]
    assert t == "Wasserstein generative adversarial networks"
    # an internal colon in the TITLE survives (only the author-run colon delimits)
    a, t = _split_author_title(
        "Changpinyo, S., Sharma, P., Ding, N., Soricut, R.: Conceptual 12m: Pushing "
        "webscale image-text pre-training. In: CVPR (2021)."
    )
    assert a == ["Changpinyo, S.", "Sharma, P.", "Ding, N.", "Soricut, R."]
    assert t == "Conceptual 12m: Pushing webscale image-text pre-training"
    # an "et al." author run terminates cleanly at the colon
    a, _ = _split_author_title(
        "Betker, J., Goh, G., Lee, J., et al.: Improving image generation. (2023)."
    )
    assert a == ["Betker, J.", "Goh, G.", "Lee, J."]
    # a title's OWN colon must NOT be mistaken for the delimiter: this author-year
    # entry (period after "et al.", colon only in "Llemma:") still parses correctly.
    a, t = _split_author_title(
        "Azerbayev, Z., Schoelkopf, H., Paster, K., et al. Llemma: An open language "
        "model for mathematics. In ICLR, 2024."
    )
    assert a == ["Azerbayev, Z.", "Schoelkopf, H.", "Paster, K."]
    assert t == "Llemma: An open language model for mathematics"


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


def test_pdf_refs_splits_flush_left_author_year_on_content():
    from citation_verifier.extract.pdf_refs import (
        _is_flush_left,
        _is_numbered,
        _split_author_year,
    )

    # AAAI style: NO hanging indent (every line flush-left at x0=54), semicolon
    # author lists, the year delimits authors from title. The indent split would
    # emit one entry per visual line; the content split keeps each entry whole.
    lines = [
        _line("Cai, H.; Zhang, P.; and Yuan, T. 2023. Adversarial example-based", x0=54.0, y0=100.0),
        _line("test case generation for speech systems. Software Testing, 33(5).", x0=54.0, y0=112.0),
        _line("Chen, Y.; Li, B.; and Ren, K. 2025a. Taught well learned ill:", x0=54.0, y0=124.0),
        _line("Towards distillation-conditional backdoor attack. In NeurIPS.", x0=54.0, y0=136.0),
        _line("Dehak, N.; Kenny, P. J.; and Ouellet, P. 2010. Front-end factor", x0=54.0, y0=148.0),
        _line("analysis for speaker verification. IEEE Trans, 19: 788–798.", x0=54.0, y0=160.0),
    ]
    assert not _is_numbered(lines)
    assert _is_flush_left(lines)  # no hanging indent → indent split would over-count
    refs = _split_author_year(lines)
    assert len(refs) == 3  # three entries, NOT six visual lines
    assert refs[0].startswith("Cai, H.") and "Software Testing" in refs[0]
    assert refs[1].startswith("Chen, Y.") and "NeurIPS" in refs[1]  # 2025a disambiguator
    assert refs[2].startswith("Dehak, N.") and "788" in refs[2]


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


def test_bind_author_year_disambiguates_by_exact_year_then_suffix():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    # Three same-surname refs across adjacent years. The EXACT year must win — a ±1
    # tolerance applied first swallowed it into an ambiguous tie (the bug that left
    # AAAI "cai2024" / EMNLP author-year citations unbound).
    refs = {
        "ref-1": CitedAs(authors=["Cai, H."], title="Adversarial example test generation", year=2023),
        "ref-2": CitedAs(authors=["Cai, H."], title="Stealthy backdoor attacks on speech", year=2024),
        "ref-3": CitedAs(authors=["Cai, H."], title="Clean-label backdoor attack", year=2025),
        # two same-surname, same-year refs in document order -> the a/b/c suffix picks by position
        "ref-4": CitedAs(authors=["Li, Y."], title="Backdoor learning fundamentals", year=2022),
        "ref-5": CitedAs(authors=["Li, Y."], title="Backdoor learning: a survey", year=2022),
    }

    def b(k):
        return _bind_author_year(k, refs)

    assert b("cai2024").year == 2024  # exact year, no longer an ambiguous ±1 tie
    assert b("cai2023").year == 2023
    assert b("cai2025").year == 2025
    assert b("li2022a").title.endswith("fundamentals")  # 1st same-year entry -> "a"
    assert b("li2022b").title.endswith("survey")  # 2nd same-year entry -> "b"


def test_bind_author_year_keeps_pm1_year_as_last_resort():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    # Two same-surname refs; the cited year matches neither exactly. A UNIQUE ±1
    # match (publication-vs-arXiv drift) still binds, but only as the last resort.
    refs = {
        "ref-1": CitedAs(authors=["Brown, T."], title="Few-shot learners", year=2020),
        "ref-2": CitedAs(authors=["Brown, T."], title="A later Brown paper", year=2024),
    }
    assert _bind_author_year("brown2021", refs).year == 2020  # 2020 is within ±1; 2024 is far


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


# ── given-name-first references (NeurIPS/arXiv style: "First Last, …, and First Last. Title") ──
def test_given_name_first_and_terminator_splits_authors_and_colon_title():
    # "…, and Jake M Hofman. Chatbench: …" must NOT read the surname "Hofman" as the
    # title (the colon-prefix rewind used to land on it) and must drop the "and".
    from citation_verifier.extract.pdf import _split_author_title

    a, t = _split_author_title(
        "Serina Chang, Ashton Anderson, and Jake M Hofman. Chatbench: From static "
        "benchmarks to human-ai evaluation. arXiv preprint arXiv:2504.07114,2025."
    )
    assert a == ["Serina Chang", "Ashton Anderson", "Jake M Hofman"]
    assert t == "Chatbench: From static benchmarks to human-ai evaluation"


def test_given_name_first_etal_strips_lowercase_venue_from_title():
    # "… et al. Mastering … knowledge. nature, 550(7676):354–359, 2017" — the venue
    # opens lowercase ("nature"), so the sentence split misses it; the volume(issue):page
    # cut must still trim it off the title.
    from citation_verifier.extract.pdf import _parse_ref_entry

    c = _parse_ref_entry(
        "David Silver, Julian Schrittwieser, Karen Simonyan, et al. Mastering the game "
        "of go without human knowledge. nature, 550(7676):354–359,2017"
    )
    assert c.title == "Mastering the game of go without human knowledge"
    assert c.authors[:1] == ["David Silver"]


def test_clean_ref_body_strips_trailing_backref_pages():
    from citation_verifier.extract.pdf import _clean_ref_body

    assert _clean_ref_body("… reasoning. arXiv preprint arXiv:2503.18892,2025.6") .endswith("2025")
    assert _clean_ref_body("… go. nature, 550(7676):354–359,2017.2,17").endswith("2017")


def test_ay_surnames_covers_person_org_and_vancouver():
    from citation_verifier.extract.pdf import _ay_surnames

    assert "brauner" in _ay_surnames("Philipp Brauner")  # given-name-first -> last token
    assert "qwen" in _ay_surnames("Qwen Team")  # org -> first token kept too
    assert _ay_surnames("Yang Z") == ["yang"]  # Vancouver -> first token (Z is initials)
    assert _ay_surnames("Martin-Lof, P.") == ["martinlof", "martin"]  # comma form + hyphen left part
    assert _ay_surnames("Jakub Konrád") == ["konrad", "jakub"]  # fold before strip: á kept as a
    assert "zamfirescu" in _ay_surnames("J Diego Zamfirescu-Pereira")  # hyphen left part


def test_bind_author_year_binds_ss_style_key_to_given_name_first_ref():
    # SS-style bibkey "Brauner2023WhatDT" (surname at front) must bind to the
    # given-name-first reference "Philipp Brauner" (surname last).
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    refs = {"ref-3": CitedAs(authors=["Philipp Brauner", "Alexander Hick"], title="What does the public think", year=2023)}
    b = _bind_author_year("Brauner2023WhatDT", refs)
    assert b is not None and b.title == "What does the public think"


def test_bind_author_year_folds_diacritics_and_hyphen_surnames():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    # diacritic: bibkey "konrad..." binds to "Jakub Konrád" (fold á->a, not strip->"konrd")
    refs = {"ref-33": CitedAs(authors=["Jakub Konrád", "Jan Pichl"], title="Alquist 4.0", year=2021)}
    assert _bind_author_year("konrad2021alquist", refs).title == "Alquist 4.0"
    # hyphenated surname: bibkey "zamfirescu..." binds to "J Diego Zamfirescu-Pereira" (left part)
    refs = {"ref-82": CitedAs(authors=["J Diego Zamfirescu-Pereira", "Richmond Y Wong"], title="Why johnny", year=2023)}
    assert _bind_author_year("zamfirescu2023johnny", refs).title == "Why johnny"


def test_apostrophe_split_surname_is_rejoined_and_binds():
    # the glyph-spacer split "Murakhovs’ka" into "Murakhovs’ ka"; the stray space made
    # "ka" look like the title start. Rejoin -> correct title + bindable left part.
    from citation_verifier.extract.pdf import _ay_surnames, _clean_ref_body, _parse_ref_entry

    assert "Murakhovs’ka" in _clean_ref_body("Lidiya Murakhovs’ ka, Caiming Xiong")
    c = _parse_ref_entry(
        _clean_ref_body(
            "Philippe Laban, Lidiya Murakhovs’ ka, Caiming Xiong, and Chien-Sheng Wu. "
            "Are you sure? challenging llms. arXiv preprint arXiv:2311.08596, 2023."
        )
    )
    assert c.title == "Are you sure? challenging llms"
    assert "Lidiya Murakhovs’ka" in c.authors
    # the apostrophe left part is a binding candidate (key "murakhovs…" -> "Murakhovs’ka")
    assert "murakhovs" in _ay_surnames("Lidiya Murakhovs’ka")


def test_bind_acl_anthology_and_title_name_keys():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    # ACL-Anthology key "surname-etal-year-word": the "etal" segment must not block
    # surname/year parsing, so the surname still binds.
    refs = {"ref-23": CitedAs(authors=["Ethan Perez", "Sam Ringer"], title="Discovering language model behaviors with model-written evaluations", year=2023)}
    assert _bind_author_year("perez-etal-2023-discovering", refs).year == 2023
    # title-name key "brokenmath2025": the leading slot is a TITLE word, not an author —
    # bind via the unique title word.
    refs = {"ref-24": CitedAs(authors=["Ivo Petrov", "Jasper Dekoninck"], title="Brokenmath: A benchmark for sycophancy in theorem proving with llms", year=2025)}
    assert _bind_author_year("brokenmath2025", refs).title.startswith("Brokenmath")


def test_given_name_first_solo_author_is_parsed():
    # a lone given-name-first author ("Harrison Chase. Langchain …") must yield the author.
    from citation_verifier.extract.pdf import _split_author_title

    a, t = _split_author_title("Harrison Chase. Langchain, October 2022. URL https://github.com/x")
    assert a == ["Harrison Chase"]
    assert t and t.startswith("Langchain")


def test_given_name_first_handles_accents_particles_mononyms_quotes():
    from citation_verifier.extract.pdf import _split_author_title

    # accents (Jörg, Loáiciga) + "and" stripped
    a, t = _split_author_title(
        "Yves Scherrer, Jörg Tiedemann, and Sharid Loáiciga. Analysing concatenation. In Proceedings"
    )
    assert a == ["Yves Scherrer", "Jörg Tiedemann", "Sharid Loáiciga"]
    assert t == "Analysing concatenation"
    # hyphen-with-lowercase ("Wen-tau") must not break the run -> correct QuAC title
    a, t = _split_author_title(
        "Eunsol Choi, Wen-tau Yih, and Luke Zettlemoyer. Quac: Question answering in context. arXiv"
    )
    assert "Wen-tau Yih" in a and t == "Quac: Question answering in context"
    # lowercase nobiliary particle ("van Berkel") + title opening with a quote
    a, t = _split_author_title(
        "Joel Wester, and Niels van Berkel. “as an ai language model”: Investigating denials. In Proc"
    )
    assert "Niels van Berkel" in a and t.startswith("“as an ai language model")
    # mononym last author ("and Vinci.")
    a, t = _split_author_title(
        "Liang Chen, Yifan Song, and Vinci. R1-v: Reinforcing super generalization. https://x"
    )
    assert a == ["Liang Chen", "Yifan Song", "Vinci"] and t == "R1-v: Reinforcing super generalization"


def test_bind_by_acronym_and_pool_title_word():
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    # keyless acronym key binds to the title whose initials spell it
    refs = {"r1": CitedAs(authors=["Fanjia Yan"], title="Berkeley Function Calling Leaderboard", year=2024)}
    assert _bind_author_year("2024bfcl", refs).title == "Berkeley Function Calling Leaderboard"
    # same-surname pool disambiguated by the keyword as a unique whole title word
    refs = {
        "a": CitedAs(authors=["Philippe Laban"], title="Are you sure? challenging llms", year=2023),
        "b": CitedAs(authors=["Philippe Laban"], title="Summary of a haystack", year=2024),
    }
    assert _bind_author_year("laban2023you", refs).title == "Are you sure? challenging llms"
    # a too-common / too-short keyword stays unbound (never guess between same-surname refs)
    refs = {
        "a": CitedAs(authors=["Jiawei Liu"], title="We are afraid", year=2023),
        "b": CitedAs(authors=["Jiawei Liu"], title="We propose a method", year=2023),
    }
    assert _bind_author_year("liu2023we", refs) is None


def test_given_name_first_runs_before_vancouver_misfire():
    # "Justin D. Weisz, …, and Werner Geyer. Title" must NOT be read by the Vancouver
    # tier as author "Justin D" + title "Weisz, …"; GNF runs first.
    from citation_verifier.extract.pdf import _split_author_title

    a, t = _split_author_title(
        "Justin D. Weisz, Jessica He, Michael Muller, and Werner Geyer. "
        "Design principles for generative ai applications. Proceedings"
    )
    assert a[0] == "Justin D. Weisz" and "Werner Geyer" in a
    assert t == "Design principles for generative ai applications"


def test_is_two_column_distinguishes_single_from_two_column():
    # PyMuPDF word tuple: (x0, y0, x1, y1, text, block, line, word_no). page_width=612.
    from citation_verifier.extract.pdf_links import _is_two_column

    def w(x0, x1, i):
        return (x0, 10.0 * i, x1, 10.0 * i + 8, "w", 0, 0, 0)

    # single column: 50 full-width lines straddling the page midline (306) -> not split
    single = [w(60, 550, i) for i in range(50)]
    assert _is_two_column(single, 612) is False
    # two columns: clean left/right groups with an empty central gutter -> split
    two = [w(60, 280, i) for i in range(25)] + [w(330, 552, i) for i in range(25)]
    assert _is_two_column(two, 612) is True
    # too little text to judge -> single (no split)
    assert _is_two_column([w(60, 550, 0)], 612) is False


def test_claim_scan_drops_venue_header_and_cjk_author_notes():
    from citation_verifier.extract.pdf import _claim_scan_body

    out = _claim_scan_body(
        "We rely on annotations [4, 5].\n"
        "sixth author Yang Yue (乐阳) share the same English name but different Chinese names.\n"
        "39th Conference on Neural Information Processing Systems (NeurIPS 2025).\n"
        "RL improves reasoning [6]."
    )
    assert "Neural Information Processing Systems" not in out  # venue header dropped
    assert "乐阳" not in out and "Chinese names" not in out  # CJK name-gloss note dropped
    assert "RL improves reasoning [6]." in out


def test_claim_scan_keeps_multilingual_claim_with_cjk_example():
    # a legitimate claim quoting a Chinese example/dataset must NOT be dropped just for
    # containing CJK — only author-name-gloss / mostly-CJK boilerplate is removed.
    from citation_verifier.extract.pdf import _claim_scan_body

    claim = "The model translates 你好 to hello and scores 95% on the CMRC dataset [12]."
    out = _claim_scan_body("Intro.\n" + claim + "\nNext sentence.")
    assert claim in out


def test_numeric_path_gating_distinguishes_math_interval_from_citations():
    # A math interval in author-year prose is NOT a numbered-citation paper; a body
    # whose [n] markers resolve to refs IS.
    from citation_verifier.extract.pdf import _has_numbered_citations
    from citation_verifier.schema import CitedAs

    refs = {f"ref-{n}": CitedAs(raw=f"r{n}") for n in range(1, 20)}
    assert _has_numbered_citations("a score Si ∈ [0, 100] from an evaluator", refs) is False
    assert _has_numbered_citations("PPO [12, 13] and methods [4, 5], plus [7].", refs) is True


def test_claim_scan_drops_section_headings_and_footnotes():
    from citation_verifier.extract.pdf import _claim_scan_body

    out = _claim_scan_body(
        "We unlock this potential.\n1 Introduction\n∗ Equal Contribution. † Project Lead.\n"
        "The development of reasoning LLMs is rapid."
    )
    assert "1 Introduction" not in out
    assert "Equal Contribution" not in out
    assert "The development of reasoning LLMs is rapid." in out


def test_bind_keyword_suffix_not_eaten_and_leading_word_tiebreak():
    """Regression: the a/b/c-suffix group must not eat a keyword's first letter
    ("xiong2025autoregressive" != suffix 'a' + 'utoregressive'), and when a keyword
    is in several same-surname/year titles, the title that OPENS with it wins."""
    from citation_verifier.extract.pdf import _BIBKEY_RE, _bind_author_year
    from citation_verifier.schema import CitedAs

    # the suffix no longer swallows the keyword's first letter
    m = _BIBKEY_RE.match("xiong2025autoregressive")
    assert (m.group(3), m.group(4)) == (None, "autoregressive")
    # …but a real a/b/c suffix and a CamelCase keyword still parse
    assert _BIBKEY_RE.match("smith2020a").group(3) == "a"
    assert _BIBKEY_RE.match("liu2022aBeyondPL").group(3) == "a"

    # two same-surname/same-year refs both contain "autoregressive"; the bibkey binds
    # to the one whose title STARTS with it (the keyword echoes the title's opening word)
    refs = {
        "a": CitedAs(authors=["Jingyi Xiong"], title="Autoregressive models in vision: A survey", year=2025),
        "b": CitedAs(authors=["Tao Xiong"], title="GigaTok: scaling tokenizers for autoregressive generation", year=2025),
    }
    assert _bind_author_year("xiong2025autoregressive", refs).title.startswith("Autoregressive models")
    # keyword that was previously mangled ("both"/"what"/"vila") now disambiguates
    refs = {
        "a": CitedAs(authors=["Bo Zhang"], title="Large multi-modal models can interpret features", year=2025),
        "b": CitedAs(authors=["Wei Zhang"], title="Both semantics and reconstruction matter", year=2025),
    }
    assert _bind_author_year("zhang2025both", refs).title.startswith("Both semantics")


def test_looks_like_table_dump_catches_author_year_metric_rows():
    """Author-year papers have no "[n]" markers, so a results-table row is detected by
    its model-size / metric-decimal cell density instead — without flagging prose."""
    from citation_verifier.extract.pdf import _looks_like_table_dump

    assert _looks_like_table_dump(
        "VAE (Rombach et al., 2022) 55M 4096 0.27 – LDM-4 (Rombach et al., 2022) "
        "400M Diff. 3.60 – SD-VAE (Ma et al., 2024) 84M 1.50 – MAR-H 943M 1.55 303.7"
    )
    # a prose claim that merely mentions a result or two is NOT a table dump
    assert not _looks_like_table_dump(
        "ConceptTok achieves a CKNNA score of 0.48, improving over the 0.41 baseline."
    )
    # the original [n]-marker table-row detection still fires
    assert _looks_like_table_dump("ELIZA [19] 1966 Chatbot SHRDLU [20] 1970 Task PARRY [21] 1972 Sim")


def test_bind_from_visible_text_handles_given_name_bibkey():
    """A hyperlink bibkey keyed on a GIVEN name ("guotao2024lg" -> Liang, G.) binds via
    the visible "Liang et al., 2024" in the claim, with the "lg" keyword breaking the tie
    against the other 2024 citations in the same parenthetical group."""
    from citation_verifier.extract.pdf import _bind_from_visible_text, _visible_author_years
    from citation_verifier.schema import CitedAs

    claim = "alignment using paired captions (Ge et al., 2024; Liang et al., 2024; Wu et al., 2025b)."
    assert ("liang", "2024") in _visible_author_years(claim)  # inner ";"-separated cite is seen
    refs = {
        "a": CitedAs(authors=["Guang Ge"], title="Making LLaMA see and draw with SEED tokenizer", year=2024),
        "b": CitedAs(authors=["Guotao Liang", "Bo Zhang"], title="LG-VQ: Language-guided codebook learning", year=2024),
    }
    assert _bind_from_visible_text("guotao2024lg", claim, refs).title.startswith("LG-VQ")
    # a single visible citation binds without needing the keyword
    assert _bind_from_visible_text("guotao2024lg", "see (Liang et al., 2024)", refs).title.startswith("LG-VQ")


def test_extract_text_citations_expands_grouped_and_multiyear():
    """A grouped parenthetical yields one site per citation AND per year — not just the
    first (AAAI-style author-year papers with no hyperlinks were under-counting)."""
    from citation_verifier.extract.pdf_links import extract_text_citations

    keys = lambda t: {s["cite_key"] for s in extract_text_citations(t)}  # noqa: E731
    # ";"-separated group -> every citation
    assert {"li2022b", "chen2025a", "tan2025"} <= keys(
        "methods (Li et al. 2022b; Chen et al. 2025a; Tan et al. 2025) help"
    )
    # one author, several years
    assert {"jung2019", "jung2020", "jung2022"} <= keys("defenses (Jung et al. 2019, 2020, 2022) apply")
    # mixed group with a trailing multi-year author
    assert {"chen2025b", "yi2025", "xu2024", "li2024", "hou2024", "hou2025"} <= keys(
        "(Chen et al. 2025b; Yi et al. 2025; Xu et al. 2024; Li et al. 2024; Hou et al. 2024, 2025)"
    )
    # narrative multi-year
    assert {"kim2024", "kim2025"} <= keys("Following Kim et al. (2024, 2025), we proceed.")
    # the original single/narrative/two-author forms still resolve
    assert {"yang2023", "lightman2024", "smith2020"} <= keys(
        "(Yang et al., 2023). Lightman et al. (2024) did X. Others (Smith and Jones, 2020) disagree."
    )


def test_ay_entry_boundary_splits_consecutive_authoryear_entries():
    """The author-year reference segmenter starts a fresh entry at "Surname, I.; …
    YEAR." even when the previous entry just ended with "In ACSAC." (regression guard:
    the AAAI Gu/Gao merge was an upstream text-extraction garble, not this boundary)."""
    from citation_verifier.extract.pdf_refs import _AY_ENTRY_BOUNDARY

    joined = (
        "Gao, Y.; Xu, C.; and Nepal, S. 2019. STRIP: a defence against trojan attacks. In ACSAC. "
        "Gu, T.; Liu, K.; Dolan-Gavitt, B.; and Garg, S. 2019. BadNets: evaluating backdooring attacks."
    )
    parts = [p.strip() for p in _AY_ENTRY_BOUNDARY.split(joined) if p.strip()]
    assert len(parts) == 2
    assert parts[1].startswith("Gu, T.")


def test_merge_baselines_does_not_fuse_across_column_gutter():
    """Two-column pages put a left-column heading and a right-column reference on the
    same baseline; merging them ("Acknowledgments Gu, T.; …") then drops the entry via
    the header filter. A gutter-width gap must split the line; a small gap still joins."""
    from citation_verifier.extract.pdf_refs import _merge_baselines

    # (text, x0, y0, x1, y1, size) — same baseline, separated by the column gutter
    across = [
        ("Acknowledgments", 126.0, 54.8, 210.0, 66.0, 12.0),
        ("Gu, T.; Liu, K.; and Garg, S. 2019. BadNets.", 320.0, 54.8, 540.0, 66.0, 10.0),
    ]
    assert [t[0] for t in _merge_baselines(across)] == [
        "Acknowledgments",
        "Gu, T.; Liu, K.; and Garg, S. 2019. BadNets.",
    ]
    # genuine same-line fragments (a PyMuPDF mid-line split, small gap) still merge
    same = [
        ("Gu, T.; Liu, K.; and", 320.0, 54.8, 410.0, 66.0, 10.0),
        ("Garg, S. 2019. BadNets.", 414.0, 54.8, 540.0, 66.0, 10.0),
    ]
    assert [t[0] for t in _merge_baselines(same)] == ["Gu, T.; Liu, K.; and Garg, S. 2019. BadNets."]


def test_split_author_title_single_hyphenated_given_name():
    """A lone given-name-first author with a hyphenated first name ("Chin-Yew Lin. 2004.
    ROUGE: …") must split — else the whole reference lands in authors[0] and the citation
    "(Lin, 2004)" binds to the wrong same-surname paper."""
    from citation_verifier.extract.pdf import _split_author_title

    a, t = _split_author_title(
        "Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summ."
    )
    assert a == ["Chin-Yew Lin"] and t == "ROUGE: A package for automatic evaluation of summaries"
    a, t = _split_author_title("Wen-tau Yih. 2011. Learning discriminative projections. In ACL.")
    assert a == ["Wen-tau Yih"] and t == "Learning discriminative projections"


def test_bind_author_year_rejects_large_year_gap_same_surname():
    """A unique same-surname ref binds across small (arXiv-vs-published) year drift but
    NOT a large gap — "lin2004" (ROUGE) must not bind the bib's only Lin, "Zeming Lin
    2022" (a different author). Unresolved beats a wrong-paper bind."""
    from citation_verifier.extract.pdf import _bind_author_year
    from citation_verifier.schema import CitedAs

    only_zeming = {"a": CitedAs(authors=["Zeming Lin", "Halil Akin"], title="Evolutionary-scale prediction", year=2022)}
    assert _bind_author_year("lin2004", only_zeming) is None
    # small drift (arXiv 2018 vs cited 2019) still binds on a unique surname
    assert _bind_author_year("devlin2019", {"a": CitedAs(authors=["Jacob Devlin"], title="BERT", year=2018)}) is not None
    # unknown year: no gap to judge -> still binds
    assert _bind_author_year("chase2023", {"a": CitedAs(authors=["Harrison Chase"], title="LangChain", year=None)}) is not None


def test_title_from_cuts_leaked_venue_after_question_mark():
    """A "?"-ending title followed by " In <Venue>" (no period) must drop the venue but
    keep the "?"; a "? <Subtitle>" (no "In") must NOT be cut."""
    from citation_verifier.extract.pdf import _title_from

    assert _title_from(
        "How attentive are graph attention networks? In International Conference on Learning Representations"
    ) == "How attentive are graph attention networks?"
    assert _title_from("Is BERT really robust? A strong baseline for language attack") == (
        "Is BERT really robust? A strong baseline for language attack"
    )


def test_split_author_title_org_and_etal_year_anchor():
    """Single-token org authors ("OpenAI. 2024. …", "xAI. 2025. …") and an "et al. YEAR."
    run split on the year anchor — else the year/venue leaks into a field."""
    from citation_verifier.extract.pdf import _split_author_title

    assert _split_author_title(
        "OpenAI. 2024. GPT-4 technical report. arXiv preprint arXiv:2303.08774."
    ) == (["OpenAI"], "GPT-4 technical report")
    a, t = _split_author_title("xAI. 2025. Grok 3 Beta — The Age of Reasoning Agents. Technical report.")
    assert a == ["xAI"] and t == "Grok 3 Beta — The Age of Reasoning Agents"
    a, t = _split_author_title(
        "Yubo Wang et al. 2024. MMLU-Pro. In Conference on Neural Information Processing Systems."
    )
    assert a == ["Yubo Wang"] and t == "MMLU-Pro"
    # multi-author GNF + venue leak: authors stay clean, title cut at the venue marker
    a, t = _split_author_title(
        "Shaked Brody, Uri Alon, and Eran Yahav. 2022. How attentive are graph attention "
        "networks? In International Conference on Learning Representations."
    )
    assert a == ["Shaked Brody", "Uri Alon", "Eran Yahav"]
    assert t == "How attentive are graph attention networks?"
    # a Vancouver / numbered ref (year at the end, no "Author. YEAR. Title") is untouched
    assert _split_author_title("Radford A, Wu J. Language models are unsupervised. OpenAI Blog. 2019")[0] == [
        "Radford A", "Wu J",
    ]

def test_initials_first_authors_do_not_swallow_the_title():
    """"Y. You, W. Liu, and C. Lu. Title." is the house style of IEEE/CVPR/ICRA.

    Without a tier of its own the given-name-first tier splits it on commas, cannot see
    where the run ends, and cuts mid-name — "and J" became an author and "Scholz" the
    title. Measured: two of eight cited rows in one comparison table resolved to a venue
    name and to an author's surname instead of the paper.
    """
    authors, title = _split_author_title(
        "Y. You, W. Liu, Y. Ze, Y.-L. Li, W. Wang, and C. Lu. Ukpgan: A general "
        "self-supervised keypoint detector. In Proceedings of the IEEE/CVF Conference "
        "on Computer Vision and Pattern Recognition, pages 17042-17051, 2022."
    )
    assert authors == ["Y. You", "W. Liu", "Y. Ze", "Y.-L. Li", "W. Wang", "C. Lu"]
    assert title == "Ukpgan: A general self-supervised keypoint detector"

    # a lowercase title is normal for system names, and this tier's boundary is
    # unambiguous without requiring a capital
    _a, t = _split_author_title(
        "L. Manuelli, W. Gao, P. Florence, and R. Tedrake. kpam: Keypoint affordances "
        "for category-level robotic manipulation. In ISRR, pages 132-157, 2019."
    )
    assert t == "kpam: Keypoint affordances for category-level robotic manipulation"


def test_latex_accents_are_decoded_before_the_reference_is_parsed():
    r"""An accent is a command whose argument is part of a word.

    Left undecoded, the backslash in ``Roth\"orl`` splits the reference mid-name and
    everything downstream shifts: one nine-author entry had "orl, Thomas, Hadsell, Raia,
    …" taken as its title, so the paper never resolved even though it was sitting in the
    arXiv results. European names make this routine, not exotic.
    """
    assert decode_tex_accents(r'Roth\"orl') == "Rothörl"
    assert decode_tex_accents(r'Sch\"{o}lkopf') == "Schölkopf"
    assert decode_tex_accents(r"Cort\'es") == "Cortés"
    assert decode_tex_accents(r"Ang\`ele") == "Angèle"
    assert decode_tex_accents(r"Mu\~noz") == "Muñoz"
    assert decode_tex_accents(r"Erd\H{o}s") == "Erdős"
    assert decode_tex_accents(r"Vre\v{c}ko") == "Vrečko"
    assert decode_tex_accents(r"\c{C}elik") == "Çelik"
    assert decode_tex_accents(r"Stra\ss e") == "Straße"
    assert decode_tex_accents(r"\o{}stergaard") == "østergaard"
    assert decode_tex_accents("plain text") == "plain text"


def test_a_reference_with_an_accented_author_still_yields_its_title():
    """The end-to-end consequence of the decoding above."""
    raw = (
        r'Vecerik, Mel, Regli, Jean-Baptiste, Roth\"orl, Thomas, Scholz, Jonathan. '
        '"S3K: Self-Supervised Semantic Keypoints for Robotic Manipulation via '
        'Multi-View Consistency". Conference on Robot Learning. 2021'
    )
    assert _likely_titles(decode_tex_accents(raw))[0].startswith("S3K: Self-Supervised")
