"""
Offline tests for TABLE-LEVEL verification (citation_verifier.tables).

No network, no SDK: grid parsing and dimension resolution are pure text work, and both
model seams (glosser / cell judge) are injected as stubs.

The LaTeX fixture is the real shape found in the wild (ATU, arXiv:2206.04335): a
``wraptable`` holding a ``\\resizebox``-wrapped tabular whose every cell is wrapped in
``\\multicolumn{1}{c}{\\multirow{1}{*}{…}}``, marks drawn with ``\\ding{51}``/``\\ding{55}``,
an all-empty spacer row after the header, and an "ours" row with no ``\\cite``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from citation_verifier.tables import (
    CellMark,
    CellVerdict,
    ComparisonTable,
    Dimension,
    DimensionKind,
    GlossSource,
    TableCell,
    TableRow,
    _own_name,
    asymmetry_summary,
    choose_extraction,
    compose_evidence,
    derive_cell_severity,
    extraction_quality,
    find_definition_snippets,
    looks_like_comparison_table,
    normalize_mark,
    resolve_dimensions,
    strip_tex,
    tables_from_latex,
    verify_table,
)
from citation_verifier.tables.dimensions import (
    _term_regex,
    grade_from_meaning,
    header_variants,
)
from citation_verifier.tables.evidence import _full_text_of
from citation_verifier.tables.latex_grid import included_sources
from citation_verifier.tables.pdf_grid import (
    mark_from_pdf_cell,
    repair_dingbats,
    tables_from_pdf,
)

# ── fixtures ─────────────────────────────────────────────────────────
ATU_TABLE = r"""
\begin{wraptable}[6]{r}{8.5cm}
\centering
\caption{\small Summary of existing task augmentation strategies.}
\label{tab:related_work}
    \resizebox{74mm}{11.2mm}{
        \begin{tabular}{c|c|c|c}
            \toprule
            \multicolumn{1}{l}{\multirow{2}{*}{\textbf{Method}}} &
            \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Task-aware}}} &
            \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Task-imaginary}}} &
            \multicolumn{1}{c}{\multirow{2}{*}{\textbf{Model-adaptive}}}  \\
            \multicolumn{1}{l}{\multirow{2}{*}{}} &
            \multicolumn{1}{c}{\multirow{2}{*}{}} &
            \multicolumn{1}{l}{\multirow{2}{*}{}} &
            \multicolumn{1}{c}{\multirow{2}{*}{}} \\
            \midrule
            \multicolumn{1}{l}{\multirow{1}{*}{MetaAug~\cite{rajendran2020meta}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{51}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{55}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{55}}} \\
            \multicolumn{1}{l}{\multirow{1}{*}{MLTI~\cite{yao2021meta}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{55}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{51}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{51}}} \\
            \multicolumn{1}{l}{\multirow{1}{*}{ATU}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{51}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{51}}} &
            \multicolumn{1}{c}{\multirow{1}{*}{\ding{51}}} \\
            \bottomrule
        \end{tabular}
        }
\end{wraptable}
"""

# "Task-aware" is defined in the body (a definition environment, as in the real paper);
# "Task-imaginary" only in prose; "Model-adaptive" is never defined anywhere.
ATU_BODY = r"""
\section{Analysis}
\begin{definition}[Task-aware Up-sampling]
The up-sampled task $T_{up}$ is defined to be task-aware, if and only if
$\theta_u=g(\theta_1,\cdots,\theta_{N_u})$ and $Y_u = f_{\theta_u}(X_u)$.
\end{definition}
A method is task-imaginary when it refers to generating tasks beyond the observed
task distribution rather than perturbing existing ones.
We evaluate on five datasets and report the mean accuracy.
"""

RESULTS_TABLE = r"""
\begin{table}[t]
\caption{Accuracy on five benchmarks.}
\begin{tabular}{lccc}
\toprule
Method & miniImagenet & ISIC & DermNet \\
\midrule
MAML~\cite{finn2017model} & 37.88 & 58.79 & 42.07 \\
ATU & 41.02 & 61.35 & 45.910 \\
\bottomrule
\end{tabular}
\end{table}
"""


def _table():
    return tables_from_latex(ATU_TABLE, paper_id="2206.04335", method_names={"ATU"})[0]


# ── grid extraction ──────────────────────────────────────────────────
def test_latex_grid_unwraps_multicolumn_multirow_and_reads_ding_marks():
    t = _table()
    assert t.table_id == "tab:related_work"
    assert t.caption == "Summary of existing task augmentation strategies"
    assert [d.header for d in t.dimensions] == ["Task-aware", "Task-imaginary", "Model-adaptive"]
    assert [r.label for r in t.rows] == ["MetaAug", "MLTI", "ATU"]  # spacer row dropped
    # \ding{51} -> ✓, \ding{55} -> ✗
    assert t.cell(0, 1).mark == CellMark.YES.value
    assert t.cell(0, 2).mark == CellMark.NO.value
    assert t.cell(1, 1).mark == CellMark.NO.value


def test_rows_bind_to_cite_keys_and_the_ours_row_is_detected():
    t = _table()
    assert t.rows[0].cite_keys == ["rajendran2020meta"]
    assert t.rows[1].cite_keys == ["yao2021meta"]
    # The authors' own row carries no \cite — that is how it is recognized.
    assert t.rows[2].cite_keys == [] and t.rows[2].is_self is True
    assert t.rows[0].is_self is False


def test_results_tables_are_not_treated_as_comparison_tables():
    """A numbers table must not be audited as a capability matrix."""
    assert tables_from_latex(RESULTS_TABLE, paper_id="p") == []
    assert looks_like_comparison_table([["M", "acc"], ["A", "37.88"], ["B", "41.02"]]) is False


def test_normalize_mark_covers_the_common_macro_and_word_spellings():
    for yes in (r"\ding{51}", r"\cmark", r"\checkmark", "✓", "Yes"):
        assert normalize_mark(yes) == CellMark.YES.value
    for no in (r"\ding{55}", r"\xmark", r"$\times$", "✗", "No"):
        assert normalize_mark(no) == CellMark.NO.value
    assert normalize_mark("") == CellMark.EMPTY.value
    assert normalize_mark("gradient-based") == CellMark.VALUE.value


# ── dimension meaning ────────────────────────────────────────────────
def test_definition_is_recovered_from_the_body_not_the_caption():
    """The caption defines nothing; the meaning lives in a definition env in the body."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    aware = t.dimensions[0]
    assert aware.gloss_source == "body"
    assert "defined to be task-aware" in aware.gloss_quote


def test_a_column_the_paper_never_defines_is_left_undefined_not_guessed():
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    adaptive = t.dimensions[2]
    assert adaptive.header == "Model-adaptive"
    assert adaptive.gloss_source == "header_only"
    assert adaptive.gloss == ""  # never invented


def test_glosser_may_not_invent_meaning_for_an_undefined_column():
    t = _table()

    def greedy_glosser(columns):  # a model that answers confidently for everything
        return [{"gloss": "INVENTED", "test_question": "q?"} for _ in columns]

    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption, glosser=greedy_glosser)
    assert t.dimensions[0].gloss == "INVENTED"       # had real supporting text
    assert t.dimensions[2].gloss == ""               # undefined column stays undefined


def test_find_definition_snippets_ranks_definitional_phrasing_first():
    snips = find_definition_snippets("Task-aware", ATU_BODY, caption="Summary of strategies")
    assert snips and "defined to be task-aware" in snips[0][1]


def test_enumerated_desiderata_are_recognized_as_definitions():
    """Papers usually define comparison columns by enumerating desiderata —
    `(3) \\textit{model-adaptive}: <definition>` — not by saying "is defined as".
    Missing this shape reported well-defined columns as undefined, i.e. accused the
    paper of something it had not done."""
    body = r"""
    A qualified augmentation should be: (1) \textit{task-aware}: the tasks stay close to
    the sampled ones; (3) \textit{model-adaptive}: the augmented tasks are timely in
    improving the current meta-knowledge, to which the meta-knowledge before augmentation
    struggles to generalize.
    """
    snips = find_definition_snippets("Model-adaptive", body)
    assert snips, "the enumerated definition must be found"
    assert "timely in improving the current meta-knowledge" in snips[0][1]
    assert snips[0][0] == "body"


def test_latex_comments_and_the_table_itself_do_not_count_as_a_definition():
    """A term only appearing in a reviewer comment or inside the table float is NOT
    the paper defining it — otherwise every header would look defined."""
    body = (
        "%%%% YING: model-adaptive is an adjective, fix the wording\n"
        r"\begin{table}\begin{tabular}{cc} Model-adaptive & \ding{51} \\ "
        r"\end{tabular}\end{table}" "\n"
        "We evaluate on five datasets.\n"
    )
    assert find_definition_snippets("Model-adaptive", body) == []


def test_a_columns_definition_is_never_the_tables_own_header_row():
    """Extracted PDF text CONTAINS the flattened table, so the header row itself matches
    every column and was handed to the judge as all five columns' definition (measured on
    PaCoST: every column's "definition" was `Method TDA Free CT Free TDL Free SP T Free
    String-ma…`, and the judge then reasoned about a truncated table instead of the cited
    abstract). A passage reciting two or more sibling headers is the table, not a
    definition."""
    siblings = ["TDA Free", "CT Free", "TDL Free", "SP", "T Free"]
    body = (
        "Method TDA Free CT Free TDL Free SP T Free String-match Min-k% Prob. "
        "Training Data Access Free (TDA Free) refers to a method that needs no access "
        "to the training corpus of the evaluated model."
    )
    best = find_definition_snippets("TDA Free", body, siblings=siblings)
    assert best, "the real definition must survive"
    assert "needs no access" in best[0][1]
    assert "Min-k%" not in best[0][1]  # the table dump is rejected

    # One sibling mention is legitimate — ATU defines Task-imaginary partly by contrast.
    contrast = "Task-imaginary: the tasks embrace diversity which task-awareness cannot guarantee."
    kept = find_definition_snippets(
        "Task-imaginary", contrast, siblings=["Task-aware", "Model-adaptive"]
    )
    assert kept


def test_short_abbreviated_headers_are_still_resolvable():
    """Real columns are abbreviated hard ("SP", "CT Free" in PaCoST). Dropping short
    variants made them look undefined; matching is word-boundary anchored so keeping
    them does not introduce substring false positives."""
    from citation_verifier.tables.dimensions import header_variants

    assert header_variants("SP") == ["sp"]
    assert find_definition_snippets("SP", "SP means the method needs no supervision.")
    # "SPP" must NOT match inside "SPPRIN"
    assert find_definition_snippets("SPP", "The SPPRIN network is defined as a student.") == []


def test_a_merely_mentioned_column_is_checkable_but_not_called_undefined():
    """Discussed-but-not-defined is weaker evidence, not an accusation: only a term that
    occurs nowhere outside the table is reported as an undefined column."""
    from citation_verifier.tables.dimensions import dimension_is_checkable

    t = _table()
    body = "Model-adaptive augmentation has been studied for meta-learners on small task sets."
    resolve_dimensions(t.dimensions, body, caption=t.caption)
    adaptive = t.dimensions[2]
    assert adaptive.gloss_source == "mention"
    assert dimension_is_checkable(adaptive) is True


# ── verification ─────────────────────────────────────────────────────
def _evidence(_key, _label):
    return ("MetaAug generates tasks by perturbing existing ones in the input space.", "s2:abc")


def test_a_refuted_cross_on_prior_work_is_high_severity_and_flagged():
    """The headline finding: the table says ✗ but the cited work does have the property,
    so prior work is understated and this paper's novelty is inflated."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)

    def judge(payload):
        # Report 'has' for Task-imaginary — the table marked MetaAug ✗ there.
        return [
            {"col_index": p["col_index"], "answer": "has", "quote": "generates new tasks",
             "justification": "the abstract describes task generation", "confidence": 0.8}
            for p in payload["properties"]
        ]

    report = verify_table(t, evidence_for=_evidence, judge=judge)
    metaaug = [f for f in report.findings if f.row_label == "MetaAug"]
    imaginary = next(f for f in metaaug if f.dimension == "Task-imaginary")
    assert imaginary.claimed == CellMark.NO.value
    assert imaginary.verdict == CellVerdict.CONTRADICTED.value
    assert imaginary.understates_prior_work is True
    assert imaginary.severity == "high"
    # The ✓ cell in the same row is merely supported, not a finding.
    aware = next(f for f in metaaug if f.dimension == "Task-aware")
    assert aware.verdict == CellVerdict.SUPPORTED.value and aware.severity == "ok"


def test_unclear_evidence_is_unverifiable_never_a_refutation():
    """Absence of mention must not become a contradiction (the prose pipeline's rule)."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)

    def judge(payload):
        return [{"col_index": p["col_index"], "answer": "unclear"} for p in payload["properties"]]

    report = verify_table(t, evidence_for=_evidence, judge=judge)
    judged = [f for f in report.findings if f.dimension != "Model-adaptive" and f.cite_key is not None]
    assert judged and all(f.verdict == CellVerdict.UNVERIFIABLE.value for f in judged)
    assert all(f.severity == "low" for f in judged)


def test_cells_of_an_undefined_column_are_reported_undefined():
    """Cells of a column the paper never defines are `undefined` — for the CITED rows.
    The authors' own row is skipped first (nothing external to check either way), and
    the column defect is still recorded once at table level."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    report = verify_table(t, evidence_for=_evidence, judge=lambda p: [])
    cited = [
        f for f in report.findings if f.dimension == "Model-adaptive" and f.cite_key is not None
    ]
    assert cited and all(f.verdict == CellVerdict.UNDEFINED.value for f in cited)
    assert all(f.severity == "medium" for f in cited)
    own = [f for f in report.findings if f.dimension == "Model-adaptive" and f.cite_key is None]
    assert own and all(f.verdict == CellVerdict.SKIPPED.value for f in own)
    assert any("never defined" in n for n in report.notes)


def test_self_row_is_skipped_and_no_evidence_means_unverifiable():
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    report = verify_table(t, evidence_for=lambda _key, _label: ("", ""), judge=None)
    own = [f for f in report.findings if f.row_label == "ATU"]
    assert own and all(f.verdict == CellVerdict.SKIPPED.value for f in own)
    cited = [f for f in report.findings if f.row_label == "MetaAug" and f.dimension == "Task-aware"]
    assert cited[0].verdict == CellVerdict.UNVERIFIABLE.value


def test_judge_is_never_shown_the_claimed_mark():
    """The judge must report what the evidence shows, not confirm the table."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    seen: list[dict] = []

    def judge(payload):
        seen.append(payload)
        return [{"col_index": p["col_index"], "answer": "unclear"} for p in payload["properties"]]

    verify_table(t, evidence_for=_evidence, judge=judge)
    assert seen
    blob = str(seen)
    assert "claimed" not in blob and "✓" not in blob and "✗" not in blob


def test_asymmetry_summary_reports_the_all_check_own_row_and_undefined_columns():
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)

    def judge(payload):
        return [{"col_index": p["col_index"], "answer": "has"} for p in payload["properties"]]

    summary = asymmetry_summary(verify_table(t, evidence_for=_evidence, judge=judge))
    assert summary["self_all_yes"] is True          # the authors' row is ✓ everywhere
    assert summary["understated_prior_work"] >= 1   # at least one refuted ✗
    assert summary["undefined_columns"] == ["Model-adaptive"]


# ── derived severity ─────────────────────────────────────────────────
def test_pdf_row_without_a_parsed_marker_is_not_assumed_to_be_ours():
    """In LaTeX "no \\cite" reliably means the authors' own row. In a PDF it usually means
    marker extraction failed — assuming "ours" there would silently drop real prior work
    from verification."""
    from citation_verifier.tables.latex_grid import _is_self_row

    # LaTeX: absence of a citation is a real signal.
    assert _is_self_row("ATU", [], method_names=set()) is True
    # PDF (strict): absence proves nothing; only positive evidence counts.
    assert _is_self_row("LLM-Pruner, Wanda, OWL", [], method_names=set(), strict=True) is False
    assert _is_self_row("STUN (ours)", [], method_names=set(), strict=True) is True
    assert _is_self_row("PaCoST", [], method_names={"PaCoST"}, strict=True) is True


def test_cell_prompt_forbids_relaxing_a_formal_definition():
    """Measured on the real ATU table: the judge turned "task-aware iff θ_u = g(θ_1…θ_N)"
    — a formal criterion about parameter space — into "derived from existing tasks", and
    flagged MLTI (which interpolates raw features/labels) as a refuted ✗. A false
    high-severity "understates prior work" is the worst output this tool can produce, so
    the prompt must forbid loosening a precise criterion and require the judge to name
    which part of the definition it matched."""
    from citation_verifier.tables.llm import CELL_SYSTEM as s

    assert "FORMAL" in s and "QUALITATIVE" in s
    assert "A different mechanism that achieves a similar end does not qualify" in s
    assert "name the exact part of the definition" in s
    assert "not actually about the named work" in s      # garbage evidence -> unclear
    assert "absence is not refutation" in s
    # …and the opposite failure: over-caution on a plain-words property hides real errors.
    assert "Do NOT demand a formal proof it never asked for" in s
    # The rule must be stated in the abstract — no example lifted from a test paper,
    # or the judge is tuned to one domain instead of the principle.
    for leak in ("task parameters", "meta-learning", "MetaAug", "MLTI", "keypoint"):
        assert leak not in s, f"CELL_SYSTEM leaks a corpus-specific example: {leak!r}"


def test_densest_block_splits_only_at_the_page_gutter():
    """A two-column page holds two unrelated tables that must not be spliced; a wide
    `table*` spans the page and its own column pitch can exceed any fixed width, so
    splitting on gap size alone shredded it into single columns and lost the table."""
    from citation_verifier.tables.pdf_grid import _densest_block

    def mark(x, y):  # (x0, y0, x1, y1, text, …)
        return (x, y, x + 8, y + 8, "✓", 0, 0, 0)

    page_w = 612.0
    # Wide table: 4 columns at a 150pt pitch, all on the left-to-right span. Must survive.
    wide = [mark(x, y) for y in (300, 315, 330) for x in (60, 210, 360, 510)]
    assert len(_densest_block(wide, page_w)) == len(wide)

    # Two-column page: two blocks either side of the gutter. Must pick one, not both.
    left = [mark(x, y) for y in (300, 315, 330) for x in (60, 120, 180)]
    right = [mark(x, y) for y in (300, 315) for x in (400, 460)]
    picked = _densest_block(left + right, page_w)
    assert len(picked) == len(left)


def test_caption_above_the_table_does_not_become_the_column_headers():
    """IEEE/ACM/ACL put captions above tables. Clipping to the mark block before testing
    for a caption removed the words 'Table 1:' that identify it, so caption prose was
    bucketed into headers and every column then read as never-defined."""
    from citation_verifier.tables.pdf_grid import _CAPTION_RE

    assert _CAPTION_RE.search("Table 1: Comparison of related methods on four axes.")
    assert _CAPTION_RE.search("TABLE I. Capabilities of open models")
    assert not _CAPTION_RE.search("Task-aware Scalable Tuning-free")


def test_grid_issues_flags_the_damage_that_should_trigger_ocr():
    """Text extraction fails visibly — blank headers, merged row labels, ragged mark
    counts. Those are the signals that a vision pass is worth its cost."""
    from citation_verifier.tables.pdf_grid import grid_issues

    clean = [["Method", "A", "B"], ["X [1]", "✓", "✗"], ["Y [2]", "✗", "✓"]]
    assert grid_issues(clean) == []

    damaged = [["Method", "", "B"], ["", "✓", "✗"], ["LLM-Pruner (1), Wanda (2)", "✗", ""]]
    issues = " ".join(grid_issues(damaged))
    assert "header" in issues and "row label" in issues and "merged" in issues


def test_column_spec_is_not_parsed_as_a_table_row():
    """`\\begin{tabular}{l c c c}` leaves the spec as the body's first token. Measured on
    Wanda's Table 1 (which opens with a blank spacer row): the spec became the entire
    header row, so every column read as one the paper never defined."""
    src = r"""\begin{table}\caption{C}
    \begin{tabular}{l c c c}
        & & & \\
    \toprule
    Method & Weight Update & Calibration Data & Complexity \\
    \hline
    Magnitude & \xmark & \xmark & $O(1)$ \\
    SparseGPT & \cmark & \cmark & $O(d^3)$ \\
    \bottomrule
    \end{tabular}\end{table}"""
    t = tables_from_latex(src, paper_id="p", require_comparison=False)[0]
    assert [d.header for d in t.dimensions] == [
        "Weight Update", "Calibration Data", "Complexity",
    ]
    assert [r.label for r in t.rows] == ["Magnitude", "SparseGPT"]


def test_extraction_warnings_are_not_mined_as_paper_text():
    """`legend` is fed to resolve_dimensions as page content, so tool diagnostics must
    live in their own field or the parser's own words become a column definition."""
    from citation_verifier.tables.model import ComparisonTable

    t = ComparisonTable(table_id="t", warnings=["1 row label(s) look like several merged rows"])
    assert t.legend == []
    assert t.warnings


def test_word_marks_never_match_inside_ordinary_cell_text():
    """"No-Reference"/"Full fine-tuning" are categorical values, not ✗/✓. A substring
    search turned such a column binary and then produced fabricated accusations, so word
    forms are matched only against the WHOLE cell."""
    assert normalize_mark("No-Reference") == CellMark.VALUE.value
    assert normalize_mark("Full fine-tuning") == CellMark.VALUE.value
    assert normalize_mark("limited-memory BFGS") == CellMark.VALUE.value
    # …while the genuine word cells still work
    assert normalize_mark("No") == CellMark.NO.value
    assert normalize_mark("Yes") == CellMark.YES.value
    assert normalize_mark(r"\ding{51}") == CellMark.YES.value


def test_enumerator_header_ignores_the_papers_own_contributions_list():
    """A column headed "(i)" must not be named after "(i) We propose …": that adopts this
    paper's contribution as the property and then accuses every cited work of lacking it."""
    from citation_verifier.tables.dimensions import enumerator_name

    body = (
        "Our contributions are threefold: (i) We propose a new sampling scheme, "
        "(ii) We release code. Later: \\item (i) \\textit{Anti-occlusion.} "
        "The keypoints must survive occlusion."
    )
    name, _quote = enumerator_name("(i)", body)
    assert name == "Anti-occlusion"
    # A bare "1"/"A" header is a real column, not a pointer into a list.
    assert enumerator_name("1", body) == ("", "")


def test_no_citation_means_ours_only_when_every_other_row_cites():
    """With 2 of 6 rows bound, the 4 unbound rows are competitors we failed to parse —
    not four "ours" rows to skip."""
    src = r"""\begin{table}\caption{C}\begin{tabular}{lcc}\toprule M&A&B\\ \midrule
    Wanda & \ding{51}&\ding{55}\\ LLM-Pruner & \ding{55}&\ding{51}\\ SliceGPT & \ding{51}&\ding{51}\\
    X~\cite{x} & \ding{51}&\ding{55}\\ Y~\cite{y} & \ding{55}&\ding{51}\\ Ours & \ding{51}&\ding{51}\\
    \bottomrule\end{tabular}\end{table}"""
    t = tables_from_latex(src, paper_id="p")[0]
    assert [r.label for r in t.rows if r.is_self] == ["Ours"]


def test_two_level_header_is_kept_checkable_and_not_classified_numeric():
    """`\\ding{51}` strips to the digits "51", so counting the header row as data made
    every column look numeric and the table was dropped as descriptive. And the merged
    header "Efficiency — Fast" must still resolve against a body that defines "Fast"."""
    src = r"""\begin{table}\caption{C}\begin{tabular}{lcccc}\toprule
    \multirow{2}{*}{Method} & \multicolumn{2}{c}{Efficiency} & \multicolumn{2}{c}{Quality}\\
     & Fast & Cheap & Exact & Robust \\ \midrule
    A~\cite{a} & \ding{51} & \ding{55} & \ding{51} & \ding{55}\\
    B~\cite{b} & \ding{55} & \ding{51} & \ding{55} & \ding{51}\\
    \bottomrule\end{tabular}\end{table}"""
    tabs = tables_from_latex(src, paper_id="p")
    assert tabs, "a two-level-header table must not be dropped"
    assert all(d.kind == "binary" for d in tabs[0].dimensions)
    resolve_dimensions(tabs[0].dimensions, "We call a system Fast if it runs in real time.")
    assert tabs[0].dimensions[0].gloss_source != "header_only"


def test_evidence_for_returning_a_bare_string_is_not_sliced_to_one_character():
    """`str` is indexable, so `got[0]` silently became "A" and was judged as evidence."""
    src = r"""\begin{table}\caption{C}\begin{tabular}{lcc}\toprule M&A&B\\ \midrule
    X~\cite{x} & \ding{51}&\ding{55}\\ Y~\cite{y} & \ding{55}&\ding{51}\\
    \bottomrule\end{tabular}\end{table}"""
    t = tables_from_latex(src, paper_id="p")[0]
    for d in t.dimensions:
        d.gloss_source, d.gloss = "body", "g"
    rep = verify_table(
        t,
        evidence_for=lambda _k, _l: "A is a fast open model that needs no retraining",
        judge=lambda p: [{"col_index": x["col_index"], "answer": "lacks"} for x in p["properties"]],
    )
    assert any(f.verdict != CellVerdict.UNVERIFIABLE.value for f in rep.findings)


def test_derive_cell_severity_ranks_understated_prior_work_highest():
    high = derive_cell_severity(CellMark.NO.value, CellVerdict.CONTRADICTED.value)
    assert high == "high"
    # a refuted ✓ on a competitor is a miscredit, not novelty inflation
    assert derive_cell_severity(CellMark.YES.value, CellVerdict.CONTRADICTED.value) == "medium"
    # an unearned claim about the authors' OWN method is also high
    assert derive_cell_severity(CellMark.YES.value, CellVerdict.CONTRADICTED.value, is_self=True) == "high"
    assert derive_cell_severity(CellMark.NO.value, CellVerdict.UNVERIFIABLE.value) == "low"
    assert derive_cell_severity(CellMark.YES.value, CellVerdict.SUPPORTED.value) == "ok"
    assert derive_cell_severity(CellMark.YES.value, CellVerdict.UNDEFINED.value) == "medium"


# ── real-corpus regressions (15 comparison-table papers) ─────────────
def test_multirow_nested_in_textbf_is_unwrapped():
    """LLM-FE writes `\\textbf{\\multirow{1}{*}{Domain}}`. Matching only a LEADING
    \\multirow left the macro in place and the header degraded to "1*Domain"."""
    from citation_verifier.tables.latex_grid import _unwrap_cell

    text, span = _unwrap_cell(r"\textbf{\multirow{1}{*}{Domain}}")
    assert strip_tex(text) == "Domain" and span == 1


def test_colour_macros_do_not_become_the_row_label():
    """`\\cellcolor{blue!15} \\bf {\\AlgName}` must not be read as a row called "blue!15",
    and stripping the colour must not leave the row nameless (it is the "ours" row)."""
    assert strip_tex(r"\cellcolor{blue!15} \textbf{Ours}") == "Ours"
    assert "blue" not in strip_tex(r"\rowcolor{blue!15} Method")


def test_newcommand_alias_names_the_authors_own_row():
    """Papers alias their own method (`\\newcommand*{\\AlgName}{\\text{LLM-FE}\\xspace}`) and
    write the macro in the table; without expansion the row is called "AlgName" — or
    disappears once colour macros are stripped."""
    from citation_verifier.tables.latex_grid import collect_macro_names

    macros = collect_macro_names(r"\newcommand*{\AlgName}{\text{LLM-FE}\xspace}")
    assert macros.get("AlgName") == "LLM-FE"
    src = r"""\begin{table}\caption{C}\begin{tabular}{lcc}\toprule M & A & B\\ \midrule
    X~\cite{x} & \cmark & \xmark \\
    \cellcolor{blue!15} \bf {\AlgName} & \cmark & \cmark \\
    \bottomrule\end{tabular}\end{table}"""
    t = tables_from_latex(src, paper_id="p", macros=macros)[0]
    assert [r.label for r in t.rows] == ["X", "LLM-FE"]
    assert t.rows[1].is_self


def test_graded_medium_mark_macro_is_partial():
    """RaCoT's legend is `\\cmark: strong, \\tmark: medium, \\xmark: weak` — the medium
    macro must not read as an empty cell, or the graded column collapses to binary."""
    assert normalize_mark(r"\tmark") == CellMark.PARTIAL.value
    assert normalize_mark(r"\cmark") == CellMark.YES.value
    assert normalize_mark(r"\xmark") == CellMark.NO.value


def test_a_row_named_only_by_its_citation_keeps_the_key():
    """MARRS writes `\\citet{bohnet2023coreference}` as the whole row label; stripping the
    citation leaves nothing, and the macro fallback used to name every row "citet"."""
    from citation_verifier.tables.latex_grid import _row_label

    label, keys = _row_label([r"\citet{bohnet2023coreference}", r"\cmark"], 1)
    assert keys == ["bohnet2023coreference"] and label == "bohnet2023coreference"


def test_vision_table_reader_parses_a_grid_and_the_papers_symbol_legend():
    """Text extraction loses symbol-font glyphs, column membership, and which text is a
    row label — so a damaged parse can escalate to reading the rendered page. The reader
    must also carry back the paper's OWN legend, because a ▲ only means what the caption
    says it means. (Transport stubbed: no network, no API key.)"""
    from citation_verifier.tables.llm import build_table_ocr

    class _Block:
        type = "text"
        text = (
            'Here you go:\n```json\n{"header": ["Method", "Black-box", "No Training"],'
            ' "rows": [["HotFlip [12]", "no", "no"], ["DIGA (ours)", "yes", "yes"]],'
            ' "legend": {"\\u25b2": "medium"}, "transposed": true}\n```'
        )

    class _Judge:
        model = "m"

        def _anthropic_client(self):
            class _C:
                class messages:
                    @staticmethod
                    def create(**_kw):
                        return type("R", (), {"content": [_Block()]})()
            return _C()

    ocr = build_table_ocr(_Judge())
    grid = ocr(b"\x89PNG-not-real", "Table 1: ▲ denotes medium.")
    assert grid[0] == ["Method", "Black-box", "No Training"]
    assert grid[1][0] == "HotFlip [12]"
    assert ocr.last_legend == {"▲": "medium"}


def test_symbol_legend_from_the_caption_overrides_the_builtin_glyph_table():
    """RaCoT draws its middle grade with `\\tmark`; the caption is what says it is
    "medium". Reading the legend beats any fixed glyph vocabulary."""
    from citation_verifier.tables.dimensions import grade_from_meaning, parse_symbol_legend

    legend = parse_symbol_legend(r"Comparison (\cmark: strong, \tmark: medium, \xmark: weak).")
    assert legend[r"\tmark"] == "medium"
    assert grade_from_meaning(legend[r"\tmark"]) == "partial"
    assert normalize_mark(r"\tmark", legend) == CellMark.PARTIAL.value
    # A symbol the built-in table does not know at all, defined only by the paper:
    jb = parse_symbol_legend("`-' indicates the method lacks that capability.")
    assert grade_from_meaning(jb["-"]) == "no"


def test_label_columns_follow_the_citations_not_column_zero():
    """A grouping column pushes the method names right — "Categories | Jailbreaks
    \\cite{..} | Extra Assist | …". Reading only column 0 loses every key and the whole
    table becomes unverifiable."""
    src = r"""\begin{table*}\caption{Summary.}\begin{tabular}{lllc}\toprule
    \textbf{Categories} & \textbf{Jailbreaks} & \textbf{Extra Assist} & \textbf{I/O-Based}\\ \midrule
    Manually-designed & IJP\cite{jailbreak2024ccs} & Human & Input \\
    Optimization      & GCG\cite{GCG2023arxiv}     & LLM   & Input \\
    Optimization      & SAA\cite{saa2025iclr}      & LLM   & Input \\
    Template-based    & MasterKey\cite{masterkey2024ndss} & LLM & Output \\
    Template-based    & LLMFuzzer\cite{llmfuzzer2024}     & LLM & Output \\
    Template-based    & AutoDAN\cite{autodan2024iclr}     & Human & Output \\
    \bottomrule\end{tabular}\end{table*}"""
    t = tables_from_latex(src, paper_id="p")[0]
    assert [r.label for r in t.rows][:3] == ["IJP", "GCG", "SAA"]
    assert [r.cite_keys[0] for r in t.rows][:3] == [
        "jailbreak2024ccs", "GCG2023arxiv", "saa2025iclr",
    ]
    assert [d.header for d in t.dimensions] == ["Extra Assist", "I/O-Based"]


def test_word_valued_capability_matrix_is_accepted_but_a_results_table_is_not():
    """Some comparison tables carry no ✓ at all — their cells are a small closed
    vocabulary ("Human"/"LLM"/"-", "Input"/"Output"). Requiring ✓/✗ dropped them outright.
    A results table must still be rejected: one repeated column ("Sparsity: 50%") beside
    a row of per-benchmark numbers is not a capability matrix."""
    capability = [
        ["Method", "Extra Assist", "I/O-Based", "Box"],
        ["IJP", "Human", "Input", "black"],
        ["GCG", "-", "Input", "white"],
        ["MasterKey", "LLM", "Output", "black"],
    ]
    assert looks_like_comparison_table(capability) is True

    results = [
        ["Method", "Weight Update", "Sparsity", "125m", "350m", "1.3B", "2.7B"],
        ["Magnitude", "-", "50%", "27.66", "22.00", "14.62", "12.47"],
        ["SparseGPT", "-", "50%", "26.11", "21.20", "13.98", "12.01"],
    ]
    assert looks_like_comparison_table(results) is False


def test_a_sparse_scatter_of_marks_is_not_a_capability_matrix():
    """DocsRay's results table holds three ✗ among 24 cells; without a density floor it
    was audited as a comparison table and its data rows became "methods"."""
    sparse = [["Item", "Base", "Lite"]] + [[f"row{i}", "", ""] for i in range(9)] + [
        ["row9", "", "✗"], ["row10", "", "✗"], ["row11", "", "✗"],
    ]
    assert looks_like_comparison_table(sparse) is False


def test_property_text_drops_cross_references_but_keeps_the_definition():
    """A gloss quoted from the source carries pointers the judge cannot follow
    ("(task F in Figure~\\ref{fig:intro}c)"). They cost context and invite reasoning about
    an unseen figure. The wording — including any equation — must survive untouched,
    because the judge is required to match a formal definition at its own precision."""
    from citation_verifier.tables.verify import clean_property_text

    got = clean_property_text(
        r"the augmented tasks are timely in improving the current meta-knowledge "
        r"(task F in Figure~\ref{fig:intro}c)"
    )
    assert got == "the augmented tasks are timely in improving the current meta-knowledge"

    # an equation is part of the claim and is preserved verbatim
    formal = r"$T_{up}$ is task-aware iff $\theta_u=g(\theta_1)$ and $Y_u=f(X_u)$"
    assert clean_property_text(formal) == formal

    # a reference removed mid-clause must not leave the clause dangling
    assert clean_property_text(
        r"needs no gradient updates, as reported in Table~\ref{tab:x}"
    ) == "needs no gradient updates"


def test_a_templated_question_is_not_sent_to_the_judge():
    """Without a glosser, `test_question` is a template that only restates the header, so
    sending it adds a noise line to every property."""
    from citation_verifier.tables.model import Dimension
    from citation_verifier.tables.verify import _property_entry

    templated = Dimension(
        col_index=1, header="Task-aware", gloss="the tasks stay close to the sampled ones",
        test_question="Does the cited work satisfy 'Task-aware'?",
    )
    assert "question" not in _property_entry(templated)

    real = templated.model_copy(update={"test_question": "Are new tasks built in parameter space?"})
    assert _property_entry(real)["question"] == "Are new tasks built in parameter space?"


# ── evidence retrieval ───────────────────────────────────────────────
def test_evidence_is_queried_by_the_COLUMN_not_the_row():
    """Excerpts must be selected against what the property means. Searching the cited
    paper for its own name finds nothing useful, which is how a first run produced
    honest-but-useless `unverifiable` for most cells."""
    from citation_verifier.tables.evidence import _dimension_queries
    from citation_verifier.tables.model import Dimension

    dims = [
        Dimension(col_index=1, header="Retrain-free", gloss="needs no gradient updates"),
        Dimension(col_index=2, header="Scalable"),          # no gloss recovered
    ]
    qs = _dimension_queries(dims)
    assert qs == ["Retrain-free. needs no gradient updates", "Scalable"]


def test_evidence_provider_retries_a_rate_limited_lookup():
    """The metadata APIs answer a burst of lookups with nothing, which is
    indistinguishable from "this paper does not exist" — measured: the same key failed
    back-to-back and resolved once spaced out. Without a retry a fifth of the rows
    silently become unverifiable."""
    from citation_verifier.tables import build_evidence_provider

    class _Ref:
        raw = "Ni et al. Data augmentation for meta-learning. ICML 2021"

    class _Resolved:
        title, abstract, arxiv_id, url = "Data Augmentation for Meta-Learning", "abs…", None, "u"

    class _FlakyResolver:
        def __init__(self): self.calls = 0
        def resolve(self, _key, _ref):
            self.calls += 1
            return _Resolved() if self.calls >= 3 else None   # rate-limited twice

    r = _FlakyResolver()
    ev = build_evidence_provider(lookup=lambda _k: _Ref(), resolver=r, dimensions=[],
                                 use_full_text=False, retries=3, pace_seconds=0)
    text, source = ev("ni2021data", "Meta-Maxup")
    assert r.calls == 3
    assert "Data Augmentation for Meta-Learning" in text and source == "u"


def test_evidence_provider_degrades_instead_of_raising():
    """Nothing retrieved must yield empty evidence — the cells then read `unverifiable`,
    which is honest. A retrieval failure must never propagate."""
    from citation_verifier.tables import build_evidence_provider

    class _Boom:
        def resolve(self, *_a): raise RuntimeError("network down")

    ev = build_evidence_provider(lookup=lambda _k: object(), resolver=_Boom(), dimensions=[])
    assert ev("k", "row") == ("", "k")
    # an unknown key is not an error either
    ev2 = build_evidence_provider(lookup=lambda _k: None, resolver=_Boom(), dimensions=[])
    assert ev2("missing", "row") == ("", "missing")


def test_arxiv_search_is_not_gated_away_for_conference_references():
    """A reference citing proceedings ("… ICML. 2021") mentions neither "arxiv" nor a
    topic keyword, so arXiv was never queried for it — 41 of one paper's 59 references.
    That is invisible while Semantic Scholar answers and collapses resolution the moment
    S2 rate-limits."""
    from citation_verifier.grounding.resolver import _should_search_arxiv

    for ref in (
        'Ni, Renkun. "Data augmentation for meta-learning". ICML. 2021',
        'Yao, Huaxiu. "Meta-learning with fewer tasks through task interpolation". ICLR. 2022',
        'Rajendran. "Meta-learning requires meta-augmentation". NeurIPS. 2020',
        'Liu et al. "A pruning approach". USENIX Security. 2024',
    ):
        assert _should_search_arxiv(ref), ref


def test_a_columns_definition_is_never_the_papers_own_self_description():
    """Measured cause of most `unverifiable` cells: the gloss was a sentence about the
    CITING paper's own method ("In contrast, \\AlgName supports all four aspects …"), or
    the gap motivating the column ("Many conventional methods target the first type …").
    Neither says what earns a ✓, so the judge is asked whether a competitor implements
    this paper's design — unanswerable, so every cell reads unclear."""
    own = "Model-adaptive: the augmented tasks track the current model's weaknesses."

    def glossable(snips):
        """Entries offered as the column's meaning. `self-context` rides along for the
        glosser — which is told not to mistake a self-description for a definition — and
        can never become the gloss itself."""
        return [t for t in snips if t[0] != "self-context"]

    self_desc = "In contrast, AlgName supports model-adaptive augmentation by leveraging an LLM."
    assert glossable(
        find_definition_snippets("Model-adaptive", self_desc, own_names=["AlgName"])
    ) == []
    # …also caught by the first-person phrasing alone, without knowing the method name
    assert glossable(find_definition_snippets(
        "Model-adaptive", "Our approach is model-adaptive and we evaluate it on five datasets."
    )) == []

    problem = "Most conventional methods are not model-adaptive and presume a fixed schedule."
    assert glossable(find_definition_snippets("Model-adaptive", problem)) == []

    # the real definition still survives, and wins when mixed with the noise above
    both = f"{self_desc} {problem} {own}"
    snips = find_definition_snippets("Model-adaptive", both, own_names=["AlgName"])
    assert snips and "track the current model's weaknesses" in snips[0][1]


def test_two_author_reference_is_not_mistaken_for_a_title():
    """ACL/ICLR style writes "Given Surname and Given Surname. 2023a. Real title…". With
    no comma, no initials and one segment, the author-clause test missed it and the
    AUTHORS were extracted as the title, leaving the reference unresolvable — measured:
    both of one paper's two-author citations returned no evidence at all."""
    from citation_verifier.grounding.resolver import _likely_titles, _looks_like_author_clause

    assert _looks_like_author_clause("Shahriar Golchin and Mihai Surdeanu") is True
    assert _looks_like_author_clause("Jacob Devlin and Ming-Wei Chang") is True
    # …without swallowing a genuine short title that happens to contain "and"
    assert _looks_like_author_clause("Vision and Language") is False
    assert _looks_like_author_clause("Attention Is All You Need") is False
    assert _looks_like_author_clause("Deep Residual Learning for Image Recognition") is False

    title = _likely_titles(
        "Shahriar Golchin and Mihai Surdeanu. 2023a. Data contamination quiz: "
        "A tool to detect and estimate contamination."
    )[0]
    assert title.lower().startswith("data contamination quiz")


def test_group_header_covers_every_column_of_its_span():
    r"""A ``\multicolumn{2}`` group header must reach BOTH children, not just the first.

    Measured on HLS-Packet's Table 1 (``\multicolumn{2}{c}{Retargetability}`` over
    ``Instruction Sets & Resource Constraints``): the second child came out as a bare
    "Resource Constraints", which then matched a background sentence about resource
    constraints in general and produced two high-severity false accusations.
    """
    tex = r"""
    \begin{table}
    \begin{tabular}{lccc}
      \textbf{Project} & \textbf{Rewriting} & \multicolumn{2}{c}{\textbf{Retargetability}} \\
      & & \textbf{Instruction Sets} & \textbf{Resource Constraints} \\
      \midrule
      Domino~\cite{domino} & \cmark & \cmark & \xmark \\
      Lyra~\cite{lyra} & \cmark & \xmark & \cmark \\
    \end{tabular}
    \caption{Prior work.}
    \end{table}
    """
    tables = tables_from_latex(tex, paper_id="p", require_comparison=False)
    headers = [d.header for d in tables[0].dimensions]
    assert "Retargetability — Instruction Sets" in headers
    assert "Retargetability — Resource Constraints" in headers


def test_a_caption_boasting_about_the_citing_papers_method_is_not_a_column_definition():
    r"""The table's punchline is not a definition of its columns.

    HLS-Packet's caption reads "CaT unifies prior work on program rewriting, code
    generation, resource allocation … without needing a new DSL". Adopted as the gloss for
    "Program Rewriting" it silently redefines the column as "avoids a new DSL", which every
    language-based baseline fails — two more false accusations.
    """
    dims = [Dimension(col_index=1, header="Program Rewriting")]
    resolve_dimensions(
        dims,
        "Program rewriting restructures a program into an equivalent one.",
        caption=(
            "CaT unifies prior work on program rewriting, code generation and resource "
            "allocation, without needing a new DSL."
        ),
        own_names={"CaT"},
    )
    assert "unifies prior work" not in (dims[0].gloss or "")


def test_own_method_name_is_recovered_from_the_ours_row_label():
    """Nothing upstream knows the paper's method name; the "ours" row does.

    ``\\sysname (this work)`` must resolve to "CaT" — the macro expanded, the marker gone —
    or the self-description filter above has nothing to match on.
    """
    assert _own_name("CaT (this work)") == "CaT"
    assert _own_name("LLM-FE (Ours) [1]") == "LLM-FE"
    assert _own_name("Ours") == ""  # a marker alone is not a method name


def test_a_leaf_gloss_that_ignores_its_group_header_is_demoted():
    r"""``Retargetability — Resource Constraints`` is not "resource constraints".

    The paper defines the leaf nowhere in its group context, so the best the body offers is
    a guess; grading it ``body`` let a guess license a contradiction.
    """
    dims = [Dimension(col_index=1, header="Retargetability — Resource Constraints")]
    resolve_dimensions(
        dims,
        "Failing to meet any of the resource constraints means the program cannot run.",
    )
    assert dims[0].gloss_source == GlossSource.MENTION.value


def test_an_undefined_column_cannot_produce_a_contradiction():
    """A contradiction presupposes knowing what the column asserts.

    Measured: of 15 contradictions on real papers, 12 were false, and every one of them
    rested on a column the paper never actually defined. Such a cell is *uncheckable*, not
    refuted, and must not be reported as the paper understating prior work. Filed as
    UNDEFINED rather than UNVERIFIABLE: the gap is in the citing paper, not in our reach.
    """
    table = ComparisonTable(
        paper_id="p",
        table_id="t1",
        dimensions=[Dimension(col_index=1, header="T Free", gloss="threshold",
                              gloss_source=GlossSource.MENTION.value)],
        rows=[TableRow(row_index=0, label="M", cite_keys=["m2020"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1, raw="\\xmark",
                         mark=CellMark.NO.value)],
    )
    report = verify_table(
        table,
        evidence_for=lambda key, label: ("The method needs no threshold.", "abstract"),
        judge=lambda payload: [{"col_index": 1, "answer": "has", "justification": "it does"}],
    )
    finding = report.findings[0]
    assert finding.verdict == CellVerdict.UNDEFINED.value
    assert not finding.understates_prior_work
    assert "never states what earns a mark" in finding.justification


def test_a_gloss_that_only_names_the_column_is_not_a_definition():
    """A caption that expands the abbreviation says what the column is CALLED, not what
    earns a mark in it.

    BESC's caption reads "… example composition (Composition), and example order in the
    prompt (Arrangement)". Taken as a definition it invited free association — "Arrangement"
    became "does the method model example ORDER", and a paper that merely *studied* order
    effects was reported as contradicting the table. Two false accusations.
    """
    caption = (
        "Comparison of selection methods on query dependence (Dynamic), example "
        "composition (Composition), and example order in the prompt (Arrangement)."
    )
    dims = [Dimension(col_index=i + 1, header=h)
            for i, h in enumerate(["Composition", "Arrangement"])]
    resolve_dimensions(dims, "Body text about in-context learning.", caption=caption)
    assert all(d.gloss_source == GlossSource.MENTION.value for d in dims)


def test_a_real_definition_keeps_its_grade():
    """The guard above must not demote actual definitions — they license the true findings.

    Every phrasing here comes from a paper in the measured corpus and each backs a verdict
    that human review confirmed.
    """
    for header, body in [
        ("Fidelity", "Fidelity: The watermarked content quality shall not be compromised."),
        ("Model-adaptive",
         "Model-adaptive: the augmented tasks are timely in improving the current "
         "meta-knowledge."),
        ("Anti-occlusion",
         "Anti-occlusion: The keypoints should be repeatable under self-occlusion."),
        ("Code Generation",
         "Code Generation: maps a programmer's computations on to the restricted space "
         "of transformations supported by a device."),
    ]:
        dims = [Dimension(col_index=1, header=header)]
        resolve_dimensions(dims, body)
        assert dims[0].gloss_source == GlossSource.BODY.value, header


def test_an_empty_gloss_from_the_glosser_overrides_the_keyword_guess():
    """"The passages do not pin this term down" is an answer, not a non-answer.

    Measured on LLM-FE, whose four columns the keyword search "defined" with ablation
    results — "Without domain knowledge, the performance significantly drops to 0" — and
    the judge then applied those to twenty cells of other people's methods. Keeping the
    guess when the glosser declines is how a guess becomes authoritative.
    """
    dims = [Dimension(col_index=1, header="Domain Knowledge")]
    resolve_dimensions(
        dims,
        "Without domain knowledge, the performance significantly drops to 0.",
        glosser=lambda cols: [{"gloss": "", "test_question": ""} for _ in cols],
    )
    assert dims[0].gloss == ""
    assert dims[0].gloss_source == GlossSource.HEADER_ONLY.value


def test_the_glosser_may_recover_a_column_the_keyword_search_could_not_reach():
    """A meaning carried by the caption or a symbol legend is still the paper's meaning.

    OutputConstr's columns are defined only by a legend ("'-' indicates the method does not
    use the listed resource"), and BESC abbreviates "sequence length" to "Seq. Len." — the
    keyword search reaches neither, and eighteen cells were written off as undefined.
    Graded `recovered` — a grade of its own, so a reader is not told the paper "merely
    mentions" a column it in fact defines in its caption, while the verdict gate still
    refuses to let it accuse the authors.
    """
    dims = [Dimension(col_index=1, header="Seq. Len.")]
    resolve_dimensions(
        dims,
        "Body text.",
        caption="Methods differ in the sequence length of the selected examples.",
        glosser=lambda cols: [
            {"gloss": "The method controls the sequence length of the example prompt.",
             "test_question": "Does it control example sequence length?"} for _ in cols
        ],
    )
    assert dims[0].gloss.startswith("The method controls")
    assert dims[0].gloss_source == GlossSource.RECOVERED.value


def test_a_gloss_sharing_no_vocabulary_with_its_sources_is_rejected():
    """The glosser is told to work only from the supplied material; this checks it did.

    A gloss that reuses none of the caption, legend or passages was written from the
    model's own knowledge of what the term usually means — the one thing the deterministic
    path exists to prevent.
    """
    dims = [Dimension(col_index=1, header="Retargetability")]
    resolve_dimensions(
        dims,
        "Body text with no mention.",
        caption="Comparison of prior compilers.",
        glosser=lambda cols: [
            {"gloss": "Supports quantum annealing on photonic substrates."} for _ in cols
        ],
    )
    assert dims[0].gloss == ""


def test_a_cross_in_a_technique_column_is_not_a_capability_claim():
    """In a column of technique names, ✗ means "none reported", not "cannot".

    HLS-Packet's Resource Allocation column holds "ILP", "SMT", "Heuristics", "Table
    Merging, PHV Sharing" — and ✗ for three compilers. The judge was asked the binary
    question ("does this compiler allocate computations to units?"), answered yes, and
    three high-severity accusations followed against a claim the table never made.
    """
    table = ComparisonTable(
        paper_id="p",
        table_id="t1",
        dimensions=[Dimension(col_index=1, header="Resource Allocation",
                              kind=DimensionKind.CATEGORICAL.value,
                              gloss="The method allocates every computation to a unit.",
                              gloss_source=GlossSource.BODY.value)],
        rows=[TableRow(row_index=0, label="Domino", cite_keys=["domino"]),
              TableRow(row_index=1, label="Lyra", cite_keys=["lyra"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                         raw="\\xmark", mark=CellMark.NO.value),
               TableCell(cell_id="t1.r1.c1", row_index=1, col_index=1,
                         raw="SMT", mark=CellMark.VALUE.value)],
    )
    report = verify_table(
        table,
        evidence_for=lambda key, label: ("The compiler allocates computations to units.", "abstract"),
        judge=lambda payload: [{"col_index": 1, "answer": "has", "justification": "it does"}],
    )
    finding = next(f for f in report.findings if f.row_label == "Domino")
    assert finding.verdict == CellVerdict.UNDEFINED.value
    assert not finding.understates_prior_work


def test_a_binary_column_still_reports_contradictions():
    """The guard above is scoped to technique columns — it must not mute real findings.

    Every confirmed finding in the measured corpus sits in a binary ✓/✗ column (ATU's
    Model-adaptive, USEEK's keypoint desiderata, REMARK-LLM's watermark properties).
    """
    table = ComparisonTable(
        paper_id="p",
        table_id="t1",
        dimensions=[Dimension(col_index=1, header="Model-adaptive",
                              kind=DimensionKind.BINARY.value,
                              gloss="The augmented tasks target what the current model fails on.",
                              gloss_source=GlossSource.BODY.value)],
        rows=[TableRow(row_index=0, label="Meta-Maxup", cite_keys=["ni2021data"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                         raw="\\xmark", mark=CellMark.NO.value)],
    )
    report = verify_table(
        table,
        evidence_for=lambda key, label: ("We choose the augmented task that maximizes loss.", "body"),
        judge=lambda payload: [{"col_index": 1, "answer": "has", "justification": "maximises loss"}],
    )
    assert report.findings[0].verdict == CellVerdict.CONTRADICTED.value
    assert report.findings[0].understates_prior_work


def test_a_paragraph_heading_names_the_term_it_defines():
    r"""``\paragraph{Anaphora Resolution}`` IS the definiendum, and the passage under it
    the definition — often a worked example rather than prose.

    Measured on MARRS, which defines five columns this way. Searching raw LaTeX returned
    ``} \label{fig:x} \end{figure} \paragraph{Anaphora Resolution} \begin{verbatim} User:``
    — the glosser rightly reported that the passages pin nothing down, and all five
    columns came back undefined across twelve cells.
    """
    body = (
        r"\begin{figure}\label{fig:flow}\end{figure}"
        r"\paragraph{Anaphora Resolution}"
        r"\begin{verbatim}" "\n"
        r"User: What is Ohio's capital?" "\n"
        r"Agent: Columbus" "\n"
        r"User: What is its population?" "\n"
        r"\end{verbatim}"
    )
    snips = find_definition_snippets("Anaphora", body)
    assert snips, "the heading's passage must be reachable"
    quote = snips[0][1]
    assert snips[0][0] == GlossSource.BODY.value
    assert "\\label" not in quote and "\\begin" not in quote


def test_the_authors_own_design_choice_is_not_part_of_the_criterion():
    """"Thus, we prefer X to Y" says what the authors did, not what others must do.

    USEEK states "The keypoints should be repeatable in the face of self-occlusion. Thus,
    we prefer raw 3D inputs (i.e., point clouds) to multi-view images." Carried into the
    gloss, the second sentence turned the column into "uses point clouds", and a
    multi-view detector was marked wrong for its input format alone.
    """
    dims = [Dimension(col_index=1, header="Anti-occlusion")]
    resolve_dimensions(
        dims,
        "Anti-occlusion: The keypoints should be repeatable in the face of self-occlusion. "
        "Thus, we prefer raw 3D inputs (i.e., point clouds) to multi-view images.",
    )
    assert "repeatable in the face of self-occlusion" in dims[0].gloss
    assert "multi-view" not in dims[0].gloss


def test_a_contrastive_sentence_keeps_the_half_about_prior_work():
    """"Prior methods do X. In contrast, ours does Y" defines the column in clause one.

    LLM-FE's only statement of what its columns mean is such a sentence; dropping it whole
    because the second clause names the authors' method left four columns to be "defined"
    by ablation results instead.
    """
    body = (
        "Existing methods generate simple features or refine only a single rule, however "
        "AlgName supports all four aspects."
    )
    snips = find_definition_snippets("simple features", body, own_names=["AlgName"])
    ctx = [t for t in snips if t[0] == "self-context"]
    # Reaches the glosser, which can read the column's subject off it — but never as the
    # gloss itself: "existing methods do X" states a problem, not what earns a ✓.
    assert ctx and "AlgName" not in ctx[0][1]
    assert "simple features" in ctx[0][1]
    assert [t for t in snips if t[0] != "self-context"] == []


def test_a_passage_still_carrying_markup_scores_below_clean_prose():
    """Residual LaTeX means the passage straddles a float or macro, so what survived is a
    fragment — and a fragment read as a definition sends the judge after the wrong
    property. Measured on LLM-FE, where `\\input{...}` leaked into a column's gloss."""
    clean = "Robustness: the watermark survives paraphrasing attacks on the text."
    dirty = r"Robustness: \input{figs/ablation_fig} Figure~ shows the watermark under attack."
    snips = find_definition_snippets("Robustness", f"{dirty} {clean}")
    assert snips and "input{" not in snips[0][1]


def test_citation_commands_are_not_read_as_a_damaged_passage():
    r"""``\cite`` is pervasive in ordinary prose and says nothing about passage quality.

    Penalising every residual backslash command cost four real column definitions in one
    run — HLS-Packet writes ``\para{...} The reference P4 compiler~\cite{p4c}, which …``
    and the whole passage scored below zero. Only structural residue (a float, an include,
    an environment boundary) means the passage straddles a structure.
    """
    body = (
        r"Program rewriting~\cite{p4c} restructures a program into an equivalent one "
        r"that fits the pipeline~\citep{domino, chipmunk}."
    )
    snips = find_definition_snippets("Program rewriting", body)
    real = [t for t in snips if t[0] != "self-context"]
    assert real, "citation noise must not sink an otherwise clean sentence"
    assert "\\cite" not in real[0][1]


def test_a_dingbat_that_lost_its_unicode_mapping_is_still_a_mark():
    r"""``\ding{51}`` reaches the text layer as 0x13 when the PDF ships no usable CMap.

    The marks ARE the grid's anchors, so losing them does not degrade the table — it
    deletes it. Measured: two of seven papers whose headers and row labels extracted
    perfectly produced no table at all, because every ✓ came through as a control code.
    """
    assert repair_dingbats("\x13") == "✓"
    assert repair_dingbats("\x17") == "✗"
    assert mark_from_pdf_cell("\x13") == CellMark.YES.value
    assert mark_from_pdf_cell("\x17") == CellMark.NO.value
    # printable text is untouched — no character maps into the mark set under this offset
    assert repair_dingbats("Domino") == "Domino"
    assert repair_dingbats("✓ yes") == "✓ yes"


def test_a_dash_is_the_paper_saying_no():
    """A dash is the ordinary typographic alternative to ✗ in a capability matrix.

    OutputConstr's caption says so outright — "'-' indicates the method does not use the
    listed resource or lacks that capability" — and reading it as an abstention excluded
    that table's 72 cells from checking entirely. "N/A" and "?" are different: those are
    the paper declining to say, and reading them as ✗ would manufacture an accusation.
    """
    for dash in ("-", "–", "—", "--"):
        assert normalize_mark(dash) == CellMark.NO.value, dash
        assert mark_from_pdf_cell(dash) == CellMark.NO.value, dash
    for abstain in ("n/a", "N.A.", "?", "unknown"):
        assert normalize_mark(abstain) == CellMark.EMPTY.value, abstain
        assert mark_from_pdf_cell(abstain) == CellMark.EMPTY.value, abstain


def test_the_papers_own_legend_still_outranks_the_dash_default():
    """A paper that redefines the dash wins over the default above."""
    # OutputConstr spells the dash out, and it agrees with the default
    assert normalize_mark(
        "-", legend={"-": "the method does not use the listed resource"}
    ) == CellMark.NO.value
    # a paper that uses the dash for "not measured" is abstaining, not asserting
    assert grade_from_meaning("not applicable in this setting") is None
    assert grade_from_meaning("does not support this") == "no"


def test_row_labels_do_not_bleed_into_the_neighbouring_row():
    """The mark tolerance is a clustering parameter, not a row height.

    A dense table puts its rows 7.6pt apart, so a ±6pt label window reaches into the row
    above. Measured consequence: every cited row inherited the previous row's citation
    marker, and five of six rows across two papers were verified against the WRONG paper
    while looking perfectly resolved. Each word must belong to exactly one row.
    """
    row_y = [100.0, 107.6, 115.2]

    def owner(word_centre: float) -> float:
        pitch = min(b - a for a, b in zip(row_y, row_y[1:], strict=False))
        reach = max(2.0, min(6.0, pitch / 2))
        near = min(row_y, key=lambda y: abs(word_centre - y))
        return near if abs(word_centre - near) <= reach else -1.0

    assert owner(100.4) == 100.0
    assert owner(107.2) == 107.6      # would have matched 100.0 too under a ±6pt window
    assert owner(115.0) == 115.2
    assert owner(140.0) == -1.0       # far below the table: belongs to no row


def _mini_table(paper_id, table_id, labels, keys, headers, grid):
    return ComparisonTable(
        paper_id=paper_id, table_id=table_id,
        dimensions=[Dimension(col_index=j + 1, header=h) for j, h in enumerate(headers)],
        rows=[TableRow(row_index=i, label=label, cite_keys=list(k))
              for i, (label, k) in enumerate(zip(labels, keys, strict=False))],
        cells=[TableCell(cell_id=f"{table_id}.r{i}.c{j+1}", row_index=i, col_index=j + 1,
                         raw=m, mark=m)
               for i, row in enumerate(grid) for j, m in enumerate(row)],
    )


def test_the_better_of_the_two_extractions_wins():
    r"""Neither source wins outright, so the choice has to be made per table.

    Measured: one paper's LaTeX tree shipped a stale five-column draft the document never
    ``\input``s, while its PDF carried the published six-column table. Scoring on what
    verification needs — a resolvable citation, a named column, a label that is a method
    name — picks the published one.
    """
    good = _mini_table("p", "t", ["Domino", "Lyra"], [["domino"], ["lyra"]],
                       ["Program Rewriting", "Code Generation"], [["yes", "no"], ["no", "yes"]])
    stale = _mini_table("p", "t", ["adaptability and deeper context of the", "Lyra"],
                        [[], ["lyra"]], ["Rewriting", ""], [["yes", "no"], ["no", "yes"]])
    assert extraction_quality(good) > extraction_quality(stale)
    kept = choose_extraction([stale], [good])
    assert kept[0] is good
    assert any("kept pdf" in w for w in kept[0].warnings)


def test_the_two_extractions_cross_check_each_other():
    """Two independent readings of one printed table are the cheapest check available.

    A cell the paths read differently is exactly the cell a human should look at, so the
    disagreement is recorded on the table that is kept rather than silently discarded.
    """
    a = _mini_table("p", "t", ["Domino", "Lyra"], [["domino"], ["lyra"]],
                    ["Rewriting", "Codegen"], [["yes", "no"], ["no", "yes"]])
    b = _mini_table("p", "t", ["Domino", "Lyra"], [["domino"], ["lyra"]],
                    ["Rewriting", "Codegen"], [["yes", "no"], ["no", "no"]])
    kept = choose_extraction([a], [b])
    assert any("extractions disagree on 'Lyra' × 'Codegen'" in w for w in kept[0].warnings)


def test_a_table_only_one_path_found_is_kept_as_is():
    """The PDF path cannot read a table drawn with ○/● instead of ✓/✗ — that is not a
    reason to drop what LaTeX read perfectly well."""
    only = _mini_table("p", "t", ["GCG"], [["gcg"]], ["Extra Assist"], [["no"]])
    assert choose_extraction([only], []) == [only]
    assert choose_extraction([], [only]) == [only]


def test_the_pdf_path_reads_value_cells_not_only_marks():
    r"""A cell with no ✓/✗ may still hold a technique name, and the type matters.

    HLS-Packet's Resource Allocation column holds "ILP", "SMT", "Heuristics" and three ✗.
    Reading only the marks empties the value cells, the column then looks like a plain
    ✓/✗ column, and the guard that stops a ✗ there from being read as "cannot do this"
    never fires — measured, three high-severity accusations came back.
    """
    pytest.importorskip("pymupdf")
    pdf = Path("/tmp/cv_papers/cmp_2211.06475.pdf")
    if not pdf.exists():
        pytest.skip("corpus PDF not present")
    for table in tables_from_pdf(str(pdf), paper_id="hls"):
        col = next((d for d in table.dimensions if "Resource Allocation" in d.header), None)
        if col is None:
            continue
        values = [c.raw for c in table.cells
                  if c.col_index == col.col_index and c.mark == CellMark.VALUE.value]
        assert values, "technique names must survive extraction"
        assert col.kind == DimensionKind.CATEGORICAL.value
        return
    pytest.fail("Resource Allocation column not found")


def test_only_the_files_the_document_includes_are_parsed(tmp_path):
    r"""An arXiv tree ships drafts and old submissions beside the live files.

    Measured: one tree held a five-column draft of a table under the same ``\label`` as
    the published six-column one, in a file ``main.tex`` never ``\input``s — and the draft
    is what got verified, against columns the paper does not have.
    """
    (tmp_path / "main.tex").write_text(
        r"\documentclass{article}\begin{document}\input{background}\end{document}"
    )
    (tmp_path / "background.tex").write_text("live")
    (tmp_path / "background_old.tex").write_text("stale draft")
    # a standalone snippet carries its own \documentclass; the real document is the one
    # that pulls in the most, not merely the first one found
    (tmp_path / "snippet.tex").write_text(r"\documentclass{standalone}\begin{document}x")
    kept = {p.name for p in included_sources(sorted(tmp_path.glob("*.tex")))}
    assert kept == {"main.tex", "background.tex"}


def test_a_tree_with_no_root_keeps_every_file():
    """Following the include graph must never lose a table, only stale duplicates."""
    assert included_sources([]) == []


def test_silence_across_the_full_text_is_read_differently_from_a_missing_body():
    """"The paper never claims this" and "we only saw the abstract" are different answers.

    Collapsing them into one abstention reported "could not check" for cells where the
    cited paper had been read end to end — 46 of 63 unverifiable cells in one run. With
    the full text in hand, a ✗ the paper never contradicts is consistent with the mark,
    while a ✓ the paper never claims is the citing authors crediting prior work with
    something its own paper does not assert.
    """
    def table(mark):
        return ComparisonTable(
            paper_id="p", table_id="t1",
            dimensions=[Dimension(col_index=1, header="Fidelity", kind=DimensionKind.BINARY.value,
                                  gloss="The watermark does not degrade text quality.",
                                  gloss_source=GlossSource.BODY.value)],
            rows=[TableRow(row_index=0, label="KGW", cite_keys=["kgw"])],
            cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                             raw="m", mark=mark)],
        )

    def run(mark, answer):
        return verify_table(
            table(mark),
            evidence_for=lambda key, label: (
                "COVERAGE: full text\nTITLE: A watermark", "url"),
            judge=lambda payload: [{"col_index": 1, "answer": answer, "justification": "n/a"}],
        ).findings[0]

    assert run(CellMark.NO.value, "absent").verdict == CellVerdict.SUPPORTED.value
    flagged = run(CellMark.YES.value, "absent")
    assert flagged.verdict == CellVerdict.MAY_NOT_SUPPORT.value
    assert flagged.severity == "medium"
    # an abstention still means we could not check, whatever the mark
    assert run(CellMark.YES.value, "unclear").verdict == CellVerdict.UNVERIFIABLE.value
    assert run(CellMark.NO.value, "unclear").verdict == CellVerdict.UNVERIFIABLE.value


def test_the_evidence_block_declares_how_much_of_the_paper_it_covers():
    """The judge cannot weigh silence without knowing whether it read the whole paper."""
    assert "title and abstract only" in compose_evidence("T", "A", [])
    assert "full text" in compose_evidence("T", "A", [("intro", "a passage from the body")])


def test_a_term_keeps_the_punctuation_that_is_part_of_it():
    r"""``SE(3)`` is a group name — the parentheses are the term, not stray punctuation.

    Normalising every non-word character turned the header into "se 3 -equivariant", which
    matches nothing. Measured on USEEK, which defines that column with an equation while
    eight of its cells were reported as a column the paper never defines.
    """
    variants = header_variants("SE(3)-equivariant")
    body = "The function is SE(3)-equivariant if for any point cloud P and any transform"
    assert any(_term_regex(v).search(body) for v in variants)


def test_classical_plurals_match_their_singular():
    """A header reads "Ellipses" while the section defining it is headed "Ellipsis
    Resolution" — the regular -s/-es rule reaches neither. These endings are pervasive in
    academic prose, and four MARRS cells were lost to this one."""
    assert _term_regex("ellipses").search("Ellipsis Resolution: the user omits a word")
    assert _term_regex("analyses").search("the analysis is defined as")
    assert _term_regex("matrices").search("the matrix is defined as")
    assert _term_regex("indices").search("the index is defined as")
    assert _term_regex("criteria").search("the criterion is defined as")
    # and not a coincidental prefix
    assert not _term_regex("ellipses").search("elliptical orbits are described")


def test_a_quantified_if_reads_as_a_formal_definition():
    """"X is Y if for any Z…" is how maths and CS state a property; the bare "if" means
    "iff". An ordinary conditional must not qualify."""
    dims = [Dimension(col_index=1, header="SE(3)-equivariant")]
    resolve_dimensions(
        dims,
        "The function is SE(3)-equivariant if for any point cloud and any rigid body "
        "transformation, the following equation holds.",
    )
    assert dims[0].gloss_source == GlossSource.BODY.value


def test_retrieval_landing_on_the_wrong_paper_is_marked_as_such():
    """"We read the right paper and it did not settle it" and "we read a different paper"
    are not the same outcome.

    Measured: one cell's retrieval returned an unrelated work, the judge said so in its
    reasoning, and the cell was still filed as ordinary "not enough evidence" — reading as
    though the cited paper had been consulted and come up short, when it never was.
    """
    table = ComparisonTable(
        paper_id="p", table_id="t1",
        dimensions=[Dimension(col_index=1, header="Anti-occlusion",
                              kind=DimensionKind.BINARY.value,
                              gloss="Keypoints stay repeatable under self-occlusion.",
                              gloss_source=GlossSource.BODY.value)],
        rows=[TableRow(row_index=0, label="Keypoints into the Future", cite_keys=["k2021"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                         raw="\\xmark", mark=CellMark.NO.value)],
    )
    finding = verify_table(
        table,
        evidence_for=lambda key, label: ("COVERAGE: full text\nTITLE: MoSS: meta-RL", "url"),
        judge=lambda payload: [{"col_index": 1, "answer": "wrong_paper",
                                "justification": "the excerpts are about MoSS"}],
    ).findings[0]
    assert finding.verdict == CellVerdict.UNVERIFIABLE.value
    assert "retrieval failure" in finding.justification
    assert not finding.understates_prior_work


def test_the_glosser_cannot_veto_a_definition_the_paper_states_outright():
    r"""One nondeterministic call must not erase a fact about the paper.

    The empty-gloss veto exists because the keyword search will otherwise "define" a
    column with an ablation result. But an explicit definition — a ``\paragraph{Term}``
    heading, an "X: …" list item, a definition environment — is not a guess. Measured: a
    run where the glosser returned nothing for two columns MARRS defines under such
    headings moved twelve cells to "the paper never defines this column", while the
    previous run glossed the same two columns without trouble.
    """
    def silent(cols):
        return [{"gloss": "", "test_question": ""} for _ in cols]

    stated = [Dimension(col_index=1, header="Anaphora")]
    resolve_dimensions(
        stated,
        "Anaphora: the user refers back to an entity named earlier in the conversation.",
        glosser=silent,
    )
    assert stated[0].gloss_source == GlossSource.BODY.value
    assert stated[0].gloss

    # …while a column the search only ever saw mentioned is still overturned
    guessed = [Dimension(col_index=1, header="Domain Knowledge")]
    resolve_dimensions(
        guessed,
        "Without domain knowledge, the performance significantly drops to 0.",
        glosser=silent,
    )
    assert guessed[0].gloss_source == GlossSource.HEADER_ONLY.value
    assert guessed[0].gloss == ""


def test_the_full_text_cascade_matches_the_prose_stage():
    """Stopping after arXiv and the resolved URL reduces a whole class of papers to their
    abstract: an ICCV or ICRA reference resolves to a `doi.org/10.1109/…` link that is a
    landing page, not a PDF, while an open-access copy exists. The table path now walks the
    same four steps the prose stage does — arXiv, resolved URL, the DOI's OA copy, then a
    title search."""
    import citation_verifier.grounding.fulltext as ft
    import citation_verifier.grounding.oa_fulltext as oa

    calls = []

    class R:
        arxiv_id = None
        url = ""
        doi = "10.1109/iccv.2019.00045"
        title = "USIP"
        year = 2019

    def fake_oa(doi, **kw):
        calls.append(("doi", doi))
        return type("F", (), {"text": "the body", "source": "unpaywall", "url": "u"})()

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(oa, "fulltext_by_doi_with_source", fake_oa)
        monkey.setattr(ft, "fetch_full_text_from_url", lambda u, **kw: "")
        assert _full_text_of(R()) == "the body"
        assert calls == [("doi", "10.1109/iccv.2019.00045")]
    finally:
        monkey.undo()
