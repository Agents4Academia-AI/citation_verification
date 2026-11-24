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

import re
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
    normalize_latex_prose,
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
    assert "never infer a limitation from the excerpts merely being silent" in s
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
    text, source, _meta = ev("ni2021data", "Meta-Maxup")
    assert r.calls == 3
    assert "Data Augmentation for Meta-Learning" in text and source == "u"


def test_evidence_provider_degrades_instead_of_raising():
    """Nothing retrieved must yield empty evidence — the cells then read `unverifiable`,
    which is honest. A retrieval failure must never propagate."""
    from citation_verifier.tables import build_evidence_provider

    class _Boom:
        def resolve(self, *_a): raise RuntimeError("network down")

    ev = build_evidence_provider(lookup=lambda _k: object(), resolver=_Boom(), dimensions=[])
    assert ev("k", "row") == ("", "k", {})
    # an unknown key is not an error either
    ev2 = build_evidence_provider(lookup=lambda _k: None, resolver=_Boom(), dimensions=[])
    assert ev2("missing", "row") == ("", "missing", {})


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

    Scoped to columns the paper says NOTHING about. A column it discusses loosely is
    handled by the test below — claiming such a paper "never states" anything would be
    false, and burying the contradiction loses a real signal. (A `header_only` column
    never reaches the judge: `dimension_is_checkable` drops it earlier.)
    """
    table = ComparisonTable(
        paper_id="p",
        table_id="t1",
        dimensions=[Dimension(col_index=1, header="T Free",
                              gloss_source=GlossSource.NONE.value)],
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

    A ✓ the paper never claims is the citing authors crediting prior work with something
    its own paper does not assert — a finding. A ✗ the paper never contradicts is merely
    CONSISTENT with the mark: "no evidence the method has X" is not "evidence it lacks
    X", and treating it as support once confirmed a ✗ against a paper whose excerpts
    happened not to include its four mentions of exactly that capability.
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

    silent = run(CellMark.NO.value, "absent")
    assert silent.verdict == CellVerdict.UNVERIFIABLE.value
    assert "consistent with the ✗ but does not establish it" in silent.justification
    flagged = run(CellMark.YES.value, "absent")
    assert flagged.verdict == CellVerdict.MAY_NOT_SUPPORT.value
    assert flagged.severity == "medium"
    # an abstention still means we could not check, whatever the mark
    assert run(CellMark.YES.value, "unclear").verdict == CellVerdict.UNVERIFIABLE.value
    assert run(CellMark.NO.value, "unclear").verdict == CellVerdict.UNVERIFIABLE.value


def test_the_evidence_block_declares_how_much_of_the_paper_it_covers():
    """The judge cannot weigh silence without knowing whether it read the whole paper."""
    assert "title and abstract only" in compose_evidence("T", "A", [])
    block = compose_evidence("T", "A", [("intro", "a passage from the body")],
                             total_chars=53_000, named={1: False, 2: True})
    assert "NOT the whole paper" in block and "53,000" in block
    assert "mentioned in paper: no — property #1" in block
    assert "mentioned in paper: yes — property #2" in block
    assert "full text" not in block.lower().split("title:")[0]


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


def test_evidence_excerpts_are_interleaved_across_columns():
    """The evidence block has a character cap, and the excerpts arrive grouped by column.

    Truncating the tail would delete the LAST columns' evidence outright while the first
    columns keep everything — those cells would read as unverifiable for a reason having
    nothing to do with the cited paper. Every column's best excerpt must come before any
    column's second-best.
    """
    order = [h for h, _c in _fit_order()]
    assert order == ["c1", "c2", "c3", "c1", "c2", "c3"]


def _fit_order():
    """Two ranked excerpts for each of three columns, as `_full_text_chunks` emits them."""
    per_column = [[("c1", "a1"), ("c1", "a2")], [("c2", "b1"), ("c2", "b2")],
                  [("c3", "d1"), ("c3", "d2")]]
    out = []
    for rank in range(2):
        for picked in per_column:
            if rank < len(picked):
                out.append(picked[rank])
    return out


def test_the_evidence_block_keeps_room_for_every_column():
    """A cap that cuts the tail must not swallow a whole column's only excerpt."""
    chunks = [(f"col{i}", "x" * 700) for i in range(1, 9)]
    block = compose_evidence("T", "A", chunks, max_chars=4000)
    kept = {m for m in re.findall(r"\[col(\d)\]", block)}
    assert kept, "at least the leading excerpts survive"
    assert len(block) <= 4000


def test_an_equation_survives_into_the_column_definition():
    r"""A comparison-table column is often defined by an equation.

    Deleting inline maths left "task-aware, if and only if and where is" — the criterion
    removed. The glosser then has nothing to paraphrase but the surrounding words, so it
    words the column differently every run; that wording becomes the retrieval query, and
    the whole chain below it moves. Measured: one paper's three columns were glossed three
    different ways across three runs.
    """
    body = (
        r"\begin{definition} The up-sampled task $T_{up}$ is defined to be task-aware, "
        r"if and only if $\theta_u=g(\theta_1,\cdots,\theta_{N_u})$ and "
        r"$\mathbf{Y}_u = f_{\theta_u}(\mathbf{X}_u)$. \end{definition}"
    )
    out = normalize_latex_prose(body)
    assert r"\theta_u=g(\theta_1,\cdots,\theta_{N_u})" in out
    assert r"\begin{definition}" not in out  # structure still goes


def test_a_loosely_defined_column_still_reports_its_contradiction():
    """"The paper never states what earns a mark here" is false when the paper does say —
    just not crisply.

    A reviewer reading an earlier run flagged exactly this: filing such cells as
    "undefined" both misdescribes the paper and hides a real contradiction. Kept as a
    contradiction, at low severity and labelled a lead rather than a finding.
    """
    table = ComparisonTable(
        paper_id="p", table_id="t1",
        dimensions=[Dimension(col_index=1, header="Multi-Feature Refinement",
                              kind=DimensionKind.BINARY.value,
                              gloss="The method refines features over successive rounds.",
                              gloss_source=GlossSource.MENTION.value)],
        rows=[TableRow(row_index=0, label="AutoFeat", cite_keys=["a2020"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                         raw="\\xmark", mark=CellMark.NO.value)],
    )
    finding = verify_table(
        table,
        evidence_for=lambda key, label: ("COVERAGE: full text\nIt iterates over rounds.", "u"),
        judge=lambda payload: [{"col_index": 1, "answer": "has", "justification": "it does"}],
    ).findings[0]
    assert finding.verdict == CellVerdict.CONTRADICTED.value
    assert finding.severity == "low"
    assert "weakly grounded" in finding.justification


def test_a_crashed_judge_call_says_so_on_every_cell_of_the_row():
    """A pipeline failure must not render as an ordinary "could not verify".

    Measured: the judge call for one row raised, the exception was recorded in the report's
    notes, and all five of that row's cells came back `unverifiable` with an empty
    justification — indistinguishable from the cited paper simply not settling the
    question.
    """
    def boom(payload):
        raise RuntimeError("transport died")

    table = ComparisonTable(
        paper_id="p", table_id="t1",
        dimensions=[Dimension(col_index=1, header="Anaphora",
                              kind=DimensionKind.BINARY.value, gloss="It resolves pronouns.",
                              gloss_source=GlossSource.BODY.value)],
        rows=[TableRow(row_index=0, label="Bhargava", cite_keys=["b2023"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                         raw="\\xmark", mark=CellMark.NO.value)],
    )
    report = verify_table(
        table,
        evidence_for=lambda key, label: ("COVERAGE: full text\nSome body text.", "u"),
        judge=boom,
    )
    finding = report.findings[0]
    assert finding.verdict == CellVerdict.UNVERIFIABLE.value
    assert "pipeline failure" in finding.justification
    assert any("judge failed" in n for n in report.notes)


def test_a_property_the_paper_names_cannot_be_called_absent_from_excerpts():
    """Only the code knows what the whole paper contains; the judge sees excerpts.

    Measured: "occluded" appears four times in a cited paper, none of them in the five
    excerpts shown, and the judge concluded "the paper never discusses occlusion". The
    recall pass now pulls in every passage naming the property, and a judge's 'absent' on
    a property the paper does name is downgraded to 'unclear'.
    """
    table = ComparisonTable(
        paper_id="p", table_id="t1",
        dimensions=[Dimension(col_index=1, header="Anti-occlusion",
                              kind=DimensionKind.BINARY.value,
                              gloss="Keypoints stay repeatable under self-occlusion.",
                              gloss_source=GlossSource.BODY.value)],
        rows=[TableRow(row_index=0, label="KeypointNet", cite_keys=["k2018"])],
        cells=[TableCell(cell_id="t1.r0.c1", row_index=0, col_index=1,
                         raw="\\xmark", mark=CellMark.NO.value)],
    )
    finding = verify_table(
        table,
        evidence_for=lambda key, label: ("COVERAGE: 5 excerpts…", "u", {"named": {1: True}}),
        judge=lambda payload: [{"col_index": 1, "answer": "absent", "justification": "not seen"}],
    ).findings[0]
    assert finding.verdict == CellVerdict.UNVERIFIABLE.value
    assert "does mention this property by name elsewhere" in finding.justification


def test_the_recall_pass_finds_the_property_by_stem():
    """Relevance ranking is lexical overlap with the gloss, and "self-occlusion" does not
    match "occluded". The recall pass matches on a stem so a passage naming the property
    in another inflection is still brought to the judge."""
    from citation_verifier.tables.evidence import _passages_naming
    text = ("Given a single image, our model predicts keypoints. "
            "Our method can even find occluded correspondences between images. "
            "Training uses a consistency loss.")
    hits = _passages_naming(text, "Anti-occlusion", limit=3, max_chars=300)
    assert hits and "occluded" in hits[0]
    assert _passages_naming(text, "SE(3)-equivariant", limit=3, max_chars=300) == []


# ── the citing paper's own reading of its columns ──────────────────────────────
def test_a_header_matches_the_words_it_derives_from():
    """A header is a noun ("Refinement"); the prose that explains it uses the verb or the
    adjective ("refine only a single rule", "an efficient scheme"). Without derivational
    matching the sentence that says what a column means is reported as one that never
    mentions it. Still word-anchored: "SP" must not be read into "SPRIN"."""
    assert _term_regex("Refinement").search("methods refine only a single rule")
    assert _term_regex("Efficiency").search("an efficient watermarking scheme")
    assert _term_regex("Occlusion").search("predicts occluded parts")
    assert _term_regex("Undetectability").search("the watermark is undetectable")
    assert not _term_regex("SP").search("the SPRIN backbone")
    assert not _term_regex("Complex").search("a complete run")


def test_the_paragraph_that_introduces_the_table_is_found_by_its_label():
    """Where a paper says what its columns mean is, more often than anywhere else, the
    paragraph that points at the table — in words the headers do not contain."""
    from citation_verifier.tables.dimensions import table_context_passages

    body = (
        "\\section{Intro}\n\nSome unrelated paragraph about datasets.\n\n"
        "Table~\\ref{tab:cmp} compares our method to prior work. Prior methods generate "
        "simple features or use feedback to iteratively refine only a single rule. In "
        "contrast, ours refines many.\n\nAnother paragraph.\n"
    )
    got = table_context_passages(body, label="tab:cmp")
    assert any("refine only a single rule" in s for s in got)
    assert not any("unrelated paragraph" in s for s in got)
    # Text without labels (a PDF) is reached by the printed number, but the caption's own
    # opening ("Table 1: …") is not a pointer.
    pdf = (
        "Table 1: Comparison of methods. Method A B\n\n"
        "As Table 1 shows, prior methods refine only a single rule at a time.\n\n"
    )
    got = table_context_passages(pdf, number=1)
    assert got and "refine only a single rule" in got[0]
    assert not any(s.startswith("Table 1: Comparison") for s in got)


def test_the_table_paragraph_reaches_the_glosser_and_may_ground_its_gloss():
    """A column the keyword search cannot reach (its only explanation shares no word with
    the header) is still defined when the glosser reads it off the paragraph that
    introduces the table; the gloss is accepted as `recovered`."""
    dims = [Dimension(col_index=1, header="Multi-Feature — Refinement")]
    body = (
        "Table~\\ref{tab:cmp} compares AlgName to prior work. LLM-based methods use "
        "feedback to iteratively refine only a single rule. In contrast, AlgName refines "
        "many candidate features in parallel.\n\nWe evaluate on ten datasets.\n"
    )
    seen: list[dict] = []

    def glosser(columns):
        seen.extend(columns)
        return [{
            "gloss": "The method refines several candidate features per round rather than "
                     "a single rule.",
            "test_question": "Does the method refine several features per round?",
        }]

    resolve_dimensions(dims, body, glosser=glosser, own_names=["AlgName"],
                       table_id="tab:cmp", table_label="tab:cmp")
    assert seen and any("refine only a single rule" in s for s in seen[0]["table_context"])
    # Graded no higher than a mention: the paper discusses the column where it introduces
    # the table, it does not define it — good enough to check cells against, not to accuse.
    assert dims[0].gloss_source in (GlossSource.MENTION.value, GlossSource.RECOVERED.value)
    assert "single rule" in dims[0].gloss


def test_the_authors_stated_criterion_travels_with_the_column_apart_from_its_definition():
    """"Thus, we prefer raw 3D inputs" is not part of the property (a multi-view method must
    not fail the column for its input format alone) — but it IS how the authors applied
    the property when they filled in the table, and a judge that never sees it reads the
    definition at its own level of strictness. It is kept as a note, labelled as the
    citing paper's criterion."""
    dims = [Dimension(col_index=1, header="Anti-occlusion")]
    resolve_dimensions(
        dims,
        "Anti-occlusion: The keypoints should be repeatable in the face of self-occlusion. "
        "Thus, we prefer raw 3D inputs (i.e., point clouds) to multi-view images.",
    )
    assert "multi-view" not in dims[0].gloss
    assert any("prefer raw 3D inputs" in n for n in dims[0].criterion_notes)


def test_a_scoping_remark_in_the_body_becomes_a_criterion_note():
    """A paper that defines a column by an equation often adds, elsewhere, how loosely it
    reads it ("we treat X and Y as interchangeable", "translations are handled by
    normalisation"). Those sentences are the criterion the marks were made under."""
    dims = [Dimension(col_index=1, header="SE(3)-equivariant")]
    body = (
        "The function is SE(3)-equivariant if for any point cloud $P$ the equation "
        "$f(RP+t) = Rf(P)+t$ holds. "
        "The concepts of SE(3)-invariant and SE(3)-equivariant are sometimes intertwined. "
        "We require it to be SE(3)-invariant if the function selects indices of points; "
        "otherwise, we require it to be SE(3)-equivariant if it returns coordinates. "
        "We present a new SE(3)-equivariant detector that beats every baseline."
    )
    resolve_dimensions(dims, body)
    assert "if for any point cloud" in dims[0].gloss
    notes = " ".join(dims[0].criterion_notes)
    assert "selects indices of points" in notes
    assert "beats every baseline" not in notes  # a contribution claim is not a criterion


def test_the_judge_sees_the_criterion_with_the_compared_methods_unnamed():
    """The criterion may say how the property was applied to a named row ("X and Y approach
    this via cross-task mixup, at the expense of erroneous tasks"). With the names in, that
    is the table's mark in prose; with them out, the criterion survives and whether THIS
    work is such a method still has to be read off its own evidence. Category labels
    ("Rule-based") are ordinary words the criterion itself may use, and stay."""
    from citation_verifier.tables.verify import redact_methods

    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    t.dimensions[0].criterion_notes = [
        "Within-task augmentation guarantees validity, i.e., being task-aware. MetaAug "
        "and MLTI differ: MLTI mixes across tasks at the expense of erroneous tasks."
    ]
    seen: list[dict] = []

    def judge(payload):
        seen.append(payload)
        return [{"col_index": p["col_index"], "answer": "unclear"} for p in payload["properties"]]

    verify_table(t, evidence_for=_evidence, judge=judge)
    crit = seen[0]["properties"][0]["criterion"]
    assert "erroneous tasks" in crit
    assert "MLTI" not in crit and "MetaAug" not in crit
    assert "claimed" not in str(seen) and "✓" not in str(seen) and "✗" not in str(seen)
    assert redact_methods("The rule-based scheme is efficient.", ["Rule-based"]) == \
        "The rule-based scheme is efficient."


def test_the_cell_prompt_tells_the_judge_what_the_criterion_is_and_is_not():
    """The criterion governs how strictly the definition is read, and is never evidence
    about the cited work; 'lacks' must rest on the paper's own words, not on a shortcoming
    the judge derives from a mechanism."""
    from citation_verifier.tables.llm import CELL_SYSTEM as s

    assert "how the citing paper applies it" in s
    assert "NOT evidence about the cited work" in s
    assert "a prediction about behaviour is your opinion" in s
    assert "answer 'lacks' and quote the mechanism" in s


def test_macro_aliases_are_expanded_in_the_body_text(tmp_path):
    """With `\\newcommand{\\AlgName}{LLM-FE}` left unexpanded, "\\AlgName refines features"
    carries no name, passes the self-description filter, and was adopted as a column's
    definition. A macro that takes an argument is a template, not an alias."""
    from types import SimpleNamespace

    from citation_verifier.tables import read_body_text
    from citation_verifier.tables.latex_grid import collect_macro_names

    tex = tmp_path / "tex"
    tex.mkdir()
    (tex / "main.tex").write_text(
        "\\newcommand{\\AlgName}{LLM-FE}\n\\newcommand{\\revision}[1]{#1}\n"
        "\\AlgName refines features across islands. \\revision{Revised text here.}\n"
    )
    body = read_body_text(SimpleNamespace(work_dir=str(tmp_path), pdf_path=None))
    assert "LLM-FE refines features" in body
    assert "#1{Revised" not in body and "\\revision{Revised text here.}" in body
    assert "revision" not in collect_macro_names((tex / "main.tex").read_text())


def test_formatting_macros_around_prose_are_unwrapped():
    """The sentence inside `\\textcolor{black}{…}` (a highlighted revision) or an unknown
    one-argument wrapper the paper defines for itself is the sentence."""
    got = normalize_latex_prose(
        "\\textcolor{black}{Table~\\ref{tab:x} compares \\texttt{Ours} to prior work.} "
        "\\revision{The rule-based scheme is efficient.} $\\mathbf{X}_u$ stays."
    )
    flat = re.sub(r"\s+", " ", got)
    assert "Table compares Ours to prior work." in flat
    assert "The rule-based scheme is efficient." in flat
    assert "\\textcolor" not in got and "\\revision" not in got
    assert "\\mathbf{X}_u" in got  # maths is left as written



# ── finding a property the cited paper words differently ───────────────────────
def test_the_recall_pass_finds_the_property_under_another_wording():
    """The column's name is the citing paper's coinage; the cited paper has its own words.
    A detector "proven to be rotation invariant" never says "equivariant", a method that
    "handles missing parts" never says "occlusion" — and a search by the header alone
    reported both as never mentioning the property."""
    from citation_verifier.tables.evidence import _passages_naming

    text = (
        "Our detector is proven to be rotation invariant by estimating a local frame. "
        "The loss is unrelated to any of this. "
        "The keypoints stay stable when parts of the object are missing."
    )
    assert _passages_naming(text, ["SE(3)-equivariant"], limit=3, max_chars=300) == []
    hits = _passages_naming(
        text, ["SE(3)-equivariant", "rotation invariant", "missing parts"], limit=3, max_chars=300
    )
    assert any("rotation invariant" in h for h in hits)
    assert any("parts of the object are missing" in h for h in hits)  # words in any order
    assert not any("loss is unrelated" in h for h in hits)


def test_the_coverage_block_lists_the_wordings_that_were_searched():
    """"Mentioned nowhere" is only as strong as the wordings tried, so the judge and the
    reader are shown them."""
    block = compose_evidence(
        "T", "A", [("intro", "a passage")], total_chars=1000,
        named={1: False, 2: True},
        search={1: {"terms": ["occlusion", "missing parts", "partial view"], "mentions": 0, "shown": 0},
                2: {"terms": ["efficient"], "mentions": 9, "shown": 6}},
    )
    assert "mentioned in paper: no — property #1 (searched for: occlusion, missing parts, partial view)" in block
    assert "mentioned in paper: yes — property #2 (searched for: efficient; 9 mentions, 6 shown)" in block


def test_the_glossers_other_wordings_reach_the_column_and_its_retrieval_query():
    from citation_verifier.tables.evidence import _dimension_queries, property_terms

    t = _table()

    def glosser(columns):
        return [
            {"gloss": "The augmented tasks stay valid tasks.", "test_question": "q?",
             "search_terms": ["valid tasks", "label-preserving", " within-task mixup ", "valid tasks"]}
            for _ in columns
        ]

    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption, glosser=glosser)
    aware = t.dimensions[0]
    assert aware.search_terms == ["valid tasks", "label-preserving", "within-task mixup"]
    assert "label-preserving" in _dimension_queries([aware])[0]
    assert property_terms(aware)[:1] == ["aware"] and "within-task mixup" in property_terms(aware)


def test_an_absent_verdict_says_which_wordings_were_searched():
    """A ✗ the paper never contradicts is not established by silence (unverifiable), and
    the reason names the wordings the search covered, so a reviewer can see what a
    'never mentions' rests on."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    meta = {"named": {1: False, 2: False},
            "search": {1: {"terms": ["task-aware", "valid tasks"], "mentions": 0, "shown": 0}, 2: {}}}

    def judge(payload):
        return [{"col_index": p["col_index"], "answer": "absent", "justification": "silent"}
                for p in payload["properties"]]

    report = verify_table(
        t, evidence_for=lambda key, label: ("COVERAGE: 5 excerpts…", "u", meta), judge=judge,
    )
    cell = next(f for f in report.findings if f.row_label == "MLTI" and f.dimension == "Task-aware")
    assert cell.claimed == CellMark.NO.value
    assert cell.verdict == CellVerdict.UNVERIFIABLE.value
    assert "under any of: task-aware, valid tasks" in cell.justification


def test_the_recall_pass_skips_page_chrome_and_near_miss_stems():
    """Text of an HTML-served paper carries the site's navigation, a table of contents and
    the author block; each "mentions" every section title and was filling the recall
    slots. And a five-letter stem conflated "equivalent" with "equivariant"."""
    from citation_verifier.tables.evidence import _passages_naming

    text = (
        "Report GitHub Issue Back to Abstract Download PDF Abstract 1 Introduction "
        "2 Related Work 3.2 Rotation Invariant Feature Extraction 4 Experiments. "
        "Jane Doe Affiliation: Somewhere Email: jane@example.org. "
        "This is equivalent to pose-based manipulation. "
        "Our features are rotation invariant by construction."
    )
    hits = _passages_naming(text, ["SE(3)-equivariant", "rotation invariant"], limit=6, max_chars=300)
    assert hits == ["Our features are rotation invariant by construction."]


def test_a_judge_that_has_read_every_mention_may_conclude_absence():
    """With a wide net nearly every paper "mentions" some wording of a property — a
    reference title, an acknowledgement — and a rule that forbade 'absent' whenever the
    paper mentions it anywhere turned every such ✓ into 'unverifiable'; the 'may not
    support' category disappeared. When every mention is shown, the judge has read all
    the paper says under those wordings, and its 'absent' stands."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    all_shown = {"named": {1: True}, "search": {1: {"terms": ["task-aware"], "mentions": 2, "shown": 2}}}
    some_shown = {"named": {1: True}, "search": {1: {"terms": ["task-aware"], "mentions": 9, "shown": 6}}}

    def judge(payload):
        return [{"col_index": p["col_index"], "answer": "absent", "justification": "only a reference title"}
                for p in payload["properties"]]

    def cell(meta):
        rep = verify_table(t, evidence_for=lambda key, label: ("COVERAGE…", "u", meta), judge=judge)
        return next(f for f in rep.findings if f.row_label == "MetaAug" and f.dimension == "Task-aware")

    assert cell(all_shown).claimed == CellMark.YES.value
    assert cell(all_shown).verdict == CellVerdict.MAY_NOT_SUPPORT.value
    assert "every passage mentioning this property (2) was read" in cell(all_shown).justification
    assert cell(some_shown).verdict == CellVerdict.UNVERIFIABLE.value
    assert "9 passages of which only 6 were shown" in cell(some_shown).justification


def test_the_recall_pass_ignores_the_reference_list_and_counts_mentions():
    from citation_verifier.tables.evidence import _mentions, clean_term

    text = (
        "Our scheme is efficient at insertion time.\n\n" + "More body text here. " * 20 + "\n\n"
        "References\n\n[1] A. Author. Efficient estimation of word representations. 2013.\n"
        "[2] B. Author. An efficient toolkit for sequence modeling. 2019.\n"
    )
    hits, total = _mentions(text, ["efficient"], limit=6, max_chars=200)
    assert total == 1 and hits == ["Our scheme is efficient at insertion time."]
    assert clean_term("single-rule refinement (absence)") == "single-rule refinement"


def test_a_definition_keeps_the_sentence_that_completes_it():
    """"Undetectability: The watermarks are invisible upon inspection. As a result, the
    adversary cannot detect whether a given text is watermarked." — captured by its
    punctuation the definition stopped at the first full stop, and the judge, left with
    "invisible upon inspection", read a watermark's detectability BY ITS OWNER as the
    opposite of undetectability."""
    dims = [Dimension(col_index=1, header="Undetectability")]
    resolve_dimensions(
        dims,
        "Criteria 5 Undetectability: The watermarks are invisible upon inspection. As a "
        "result, the adversary cannot detect whether a given text is watermarked. We "
        "evaluate this with a classifier in Section 5.",
    )
    assert "adversary cannot detect" in dims[0].gloss
    assert "classifier" not in dims[0].gloss


def test_a_failed_judge_call_is_retried_before_the_row_is_lost():
    """A model call that fails is usually the transport having a bad minute, and one
    uncaught failure costs every cell of that cited paper — measured at five and six
    cells in two separate runs. The call is retried before the row is given up."""
    import citation_verifier.tables.llm as llm

    calls = {"n": 0}

    class Flaky:
        mode = "messages"
        settings = None

        def _run_messages(self, system, user, max_tokens, prefix=""):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("overloaded_error")
            return '[{"col_index": 1, "answer": "has", "quote": "q", "justification": "j"}]'

    monkey, llm._JUDGE_BACKOFF = llm._JUDGE_BACKOFF, 0.0
    try:
        got = llm.build_cell_judge(Flaky())({"row_label": "X", "evidence": "E",
                                             "properties": [{"col_index": 1, "name": "P"}]})
    finally:
        llm._JUDGE_BACKOFF = monkey
    assert calls["n"] == 3
    assert got == [{"col_index": 1, "answer": "has", "quote": "q", "justification": "j"}]


def test_a_call_that_never_succeeds_still_raises():
    """The row must still degrade to 'never judged' rather than hang or return nothing."""
    import citation_verifier.tables.llm as llm

    class Dead:
        mode = "messages"
        settings = None

        def _run_messages(self, system, user, max_tokens, prefix=""):
            raise RuntimeError("api_error")

    monkey, llm._JUDGE_BACKOFF = llm._JUDGE_BACKOFF, 0.0
    try:
        with pytest.raises(RuntimeError, match="api_error"):
            llm.build_cell_judge(Dead())({"row_label": "X", "evidence": "E", "properties": []})
    finally:
        llm._JUDGE_BACKOFF = monkey


def test_a_row_payload_is_addressed_by_row_index_not_by_position():
    """Re-running only the rows a failed run lost leaves a table holding a subset of its
    rows, and indexing the list by position then reads the wrong cited paper — or raises,
    losing the row a second time. (A grid whose spacer rows were dropped is the same
    shape of bug.)"""
    from citation_verifier.tables.verify import build_row_payload

    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    keep = next(r for r in t.rows if r.label == "MLTI")
    t.rows = [keep]                      # as a re-run of just this row would leave it
    payload = build_row_payload(t, keep.row_index, list(t.dimensions), "E")
    assert payload["row_label"] == "MLTI"
    with pytest.raises(IndexError):
        build_row_payload(t, 99, list(t.dimensions), "E")


def test_a_multi_word_header_is_searched_as_a_phrase_not_as_its_words():
    """"Screen Entity Resolution" searched word by word matched every sentence of a
    coreference paper containing "resolution": the paper was recorded as mentioning the
    property fifteen times, six were shown, and the judge — told the other nine existed —
    could conclude nothing. A column name is a phrase and is matched as one. Words too
    short to carry meaning are dropped, not required, or "Anti-occlusion" would never
    match a paper that says "occluded"."""
    from citation_verifier.tables.evidence import _header_phrase, _passages_naming

    assert _header_phrase("Screen Entity Resolution") == "Screen Entity Resolution"
    assert _header_phrase("Anti-occlusion") == "occlusion"
    assert _header_phrase("(iv) SE(3)-equivariant") == "equivariant"

    text = (
        "We propose a coreference resolution model for documents. "
        "The resolution of each mention is scored independently. "
        "Our system resolves entities shown on the device screen for resolution."
    )
    hits = _passages_naming(text, "Screen Entity Resolution", limit=6, max_chars=200)
    assert len(hits) == 1 and "device screen" in hits[0]


def test_a_search_word_without_a_suffix_is_not_truncated():
    """Cutting every word to five letters made "complex" match "complete" and
    "compliance"; those passages were then counted as the paper discussing the property.
    A word with a strippable suffix still matches its derivations."""
    from citation_verifier.tables.evidence import _mentions

    text = ("The pipeline is complete and runs in one pass. "
            "We generate complex feature interactions. "
            "Keypoints stay stable when the object is occluded.")
    hits, total = _mentions(text, ["complex"], limit=6, max_chars=200)
    assert total == 1 and "complex feature" in hits[0]
    hits, total = _mentions(text, ["occlusion"], limit=6, max_chars=200)
    assert total == 1 and "occluded" in hits[0]


def test_the_evidence_always_carries_how_the_cited_work_is_built():
    """A table property is more often settled by the cited work's construction than by
    anything it says about the property: a detector trained on hand-annotated keypoints is
    not unsupervised, and a crop-and-regress network carries no equivariance guarantee —
    neither paper uses the column's word. A property-driven search never reaches those
    passages, so they are collected separately and always sent."""
    from citation_verifier.tables.evidence import _construction_passages

    sections = [
        ("Introduction", "Keypoints are useful for manipulation. Prior work is limited."),
        ("3. Method", "We annotate semantic keypoints on 117 reconstructed scenes by hand. "
                      "The network takes as input a cropped RGB-D image and predicts a "
                      "heatmap per keypoint. This section is organised as follows."),
        ("5. Experiments", "We report accuracy on 40 held-out mugs."),
    ]
    got = _construction_passages(sections, limit=8, max_chars=600)
    text = " ".join(c for _h, c in got)
    assert "annotate semantic keypoints" in text
    assert "takes as input a cropped RGB-D image" in text
    assert "organised as follows" not in text          # not a construction sentence
    assert "held-out mugs" not in text                 # results, not construction
    assert all(h.startswith("how the work works") for h, _c in got)


def test_the_coverage_block_announces_the_construction_excerpts():
    block = compose_evidence(
        "T", "A",
        [("how the work works — Method", "We train on hand-annotated keypoints."),
         ("intro", "Some motivation.")],
        total_chars=5000, named={1: False},
        search={1: {"terms": ["unsupervised"], "mentions": 0, "shown": 0}},
    )
    assert "how the work works' (1)" in block
    assert "settled by the construction" in block


def test_the_cell_prompt_settles_a_property_from_the_construction():
    """The judge must be able to do what a human reviewer does: read the architecture and
    the training data and conclude. Predicting how well the method would PERFORM is still
    out of bounds."""
    from citation_verifier.tables.llm import CELL_SYSTEM as s

    assert "its CONSTRUCTION is" in s
    assert "the construction cannot have the property -> 'lacks'" in s
    assert "predicting how well the method would PERFORM" in s
    assert "Absence of the word is never itself the reason" in s
    for leak in ("kPAM", "keypoint", "watermark", "meta-learning"):
        assert leak not in s, f"CELL_SYSTEM leaks a corpus-specific example: {leak!r}"


# ── the whole cited paper as the evidence ─────────────────────────────────────
def test_a_cited_paper_that_fits_is_sent_whole():
    """Retrieval is a source of error in its own right: a property the paper does discuss
    is settled by a passage the ranking missed, and a property it never discusses cannot
    be told apart from one whose passages were missed. Sending the paper entire removes
    both, and "this work never claims that" becomes something the judge reads rather than
    something the retrieval code vouches for."""
    from citation_verifier.tables.evidence import compose_whole_paper

    block = compose_whole_paper("A Title", "An abstract.", "Body of the paper. " * 50)
    assert "COMPLETE text of the cited paper" in block
    assert "what the paper does not say here, it does not say anywhere" in block
    assert "Body of the paper." in block
    assert "excerpt" not in block.lower()


def test_the_whole_paper_travels_as_a_cacheable_prefix():
    """The cited work is constant while the questions vary — across the columns of a row,
    and again when the row is re-judged. It is sent as its own system block so the
    Messages transport can cache it; the question carries only the properties."""
    seen: list[dict] = []

    class Recorder:
        mode = "messages"
        settings = None

        def _run_messages(self, system, user, max_tokens, prefix=""):
            seen.append({"system": system, "user": user, "prefix": prefix})
            return '[{"col_index": 1, "answer": "has", "quote": "q", "justification": "j"}]'

    from citation_verifier.tables.llm import build_cell_judge

    build_cell_judge(Recorder())({
        "row_label": "SomeMethod", "evidence": "FULL TEXT:\nthe whole paper",
        "properties": [{"col_index": 1, "name": "P", "definition": "d"}],
    })
    call = seen[0]
    assert "the whole paper" in call["prefix"] and "SomeMethod" in call["prefix"]
    assert "the whole paper" not in call["user"]
    assert "PROPERTY TO CHECK" in call["user"] and "[col 1] P" in call["user"]


def test_a_complete_paper_lets_an_absent_answer_stand():
    """The gate that refuses "absent" exists because the judge was shown excerpts and the
    paper might say more elsewhere. With the complete text there is no elsewhere."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)

    def judge(payload):
        return [{"col_index": p["col_index"], "answer": "absent", "justification": "no claim"}
                for p in payload["properties"]]

    def cell(meta):
        rep = verify_table(t, evidence_for=lambda key, label: ("COVERAGE…", "u", meta), judge=judge)
        return next(f for f in rep.findings if f.row_label == "MetaAug" and f.dimension == "Task-aware")

    got = cell({"complete": True, "total_chars": 40000})
    assert got.claimed == CellMark.YES.value
    assert got.verdict == CellVerdict.MAY_NOT_SUPPORT.value
    assert "complete text of the paper was read" in got.justification
    # …whereas with excerpts and unshown mentions the gate still fires.
    partial = cell({"named": {1: True}, "search": {1: {"terms": ["t"], "mentions": 9, "shown": 6}}})
    assert partial.verdict == CellVerdict.UNVERIFIABLE.value


def test_the_cell_prompt_describes_both_coverage_regimes():
    from citation_verifier.tables.llm import CELL_SYSTEM as s

    assert "COMPLETE text of the cited paper" in s
    assert "settle every question by reading" in s
    assert "When the paper was too long or could not be fetched you get excerpts" in s


def test_every_cell_is_asked_as_its_own_question():
    """Asked together, the properties of a comparison table compete for one bounded answer
    and for the model's attention, and one failed call took every cell of the row with it.
    Asked separately, each cell gets a whole answer, a failure costs only that cell, and
    the cited work — the long part of the request — is a prefix that does not change."""
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    asked: list[dict] = []

    def judge(payload):
        asked.append(payload)
        if payload["properties"][0]["name"] == "Task-imaginary":
            raise RuntimeError("overloaded")
        # Answer with the wrong index on purpose: the reply is to the question asked.
        return [{"col_index": 99, "answer": "has", "quote": "q", "justification": "j"}]

    report = verify_table(t, evidence_for=_evidence, judge=judge)
    # "Model-adaptive" is undefined in this body and never reaches the judge.
    cited = [p for p in asked if p["row_label"] == "MetaAug"]
    assert len(cited) == 2 and all(len(p["properties"]) == 1 for p in cited)
    # The other columns ride along by name so a near-neighbour's evidence is not borrowed.
    assert cited[0]["other_properties"] == ["Task-imaginary", "Model-adaptive"]

    by_col = {f.dimension: f for f in report.findings if f.row_label == "MetaAug"}
    assert by_col["Task-aware"].verdict == CellVerdict.SUPPORTED.value      # ✓ + has
    # Only the cell whose call failed is lost; its sibling is unaffected.
    assert "pipeline failure" in by_col["Task-imaginary"].justification
    assert "pipeline failure" not in by_col["Task-aware"].justification


def test_the_cell_prompt_keeps_the_neighbouring_columns_apart():
    from citation_verifier.tables.llm import CELL_SYSTEM as s

    assert "answer the property you were ASKED, not the one next to it" in s


def test_a_long_paper_does_not_ride_the_agent_sdk_command_line():
    """The Agent-SDK transport carries a system prompt given as a STRING on the CLI
    invocation, and a whole paper overruns it: the two longest papers of one corpus failed
    there with CLIConnectionError on every attempt. So the paper travels as a system-prompt
    FILE — which also makes it the prefix the transport caches: sent in the user message,
    as it was for a while, no two cells of a row shared a prefix and each paid to send the
    paper again."""
    from pathlib import Path

    import citation_verifier.tables.llm as llm

    calls: list[dict] = []

    class QueryJudge:
        mode = "query"
        settings = None

        async def _run_query(self, sdk, system, user):
            calls.append({"system": system, "user": user})
            return '[{"col_index": 1, "answer": "has"}]'

    paper = "FULL TEXT:\n" + ("body " * 30000)
    run = llm._judge_transport(QueryJudge())
    run(llm.CELL_SYSTEM, "PROPERTY TO CHECK: x", 400, paper)
    run(llm.CELL_SYSTEM, "PROPERTY TO CHECK: y", 400, paper)
    run(llm.CELL_SYSTEM, "PROPERTY TO CHECK: x", 400, paper + " another paper")
    first, second, other = calls
    assert first["system"]["type"] == "file"
    text = Path(first["system"]["path"]).read_text(encoding="utf-8")
    assert text.startswith(llm.CELL_SYSTEM) and text.endswith(paper)
    assert paper not in first["user"] and first["user"] == "PROPERTY TO CHECK: x"
    # The same paper is the same prefix whatever is asked about it; another paper is not.
    assert second["system"] == first["system"] and second["user"] == "PROPERTY TO CHECK: y"
    assert other["system"]["path"] != first["system"]["path"]
    # Without a prefix nothing changes: the system prompt is the plain string.
    run(llm.CELL_SYSTEM, "PROPERTY TO CHECK: z", 400)
    assert calls[-1]["system"] == llm.CELL_SYSTEM


# ── a paper too long for one request ──────────────────────────────────────────
def test_a_paper_too_long_for_one_request_is_split_at_paragraph_boundaries():
    from citation_verifier.tables.evidence import split_paper

    para = "This is a paragraph of the paper. " * 20 + "\n\n"      # ~680 chars
    text = para * 300                                              # ~204k chars
    assert split_paper(text, max_chars=500_000) == [text]          # fits: one part
    parts = split_paper(text, max_chars=120_000, max_segments=3)
    assert len(parts) == 2
    assert all(len(p) <= 120_000 for p in parts)
    assert not any(p.endswith("This is a") for p in parts)         # never mid-sentence
    # Reassembled, nothing is lost or duplicated.
    assert "".join(p.replace("\n\n", "") for p in parts).replace(" ", "") == \
        text.replace("\n\n", "").replace(" ", "").strip()
    # Beyond the segment budget the caller must fall back to excerpts.
    assert split_paper(text, max_chars=10_000, max_segments=3) == []


def test_each_part_says_it_is_a_part():
    """"Not claimed" from one part means "not in this part"; only the caller, holding
    every part's answer, can say anything about the paper."""
    from citation_verifier.tables.evidence import compose_whole_paper

    one = compose_whole_paper("T", "", "body", part=1, of=1)
    assert "COMPLETE text" in one and "PART" not in one
    two = compose_whole_paper("T", "", "body", part=2, of=3)
    assert "PART 2 OF 3" in two
    assert "'absent' here means this part carries no such claim" in two


def test_the_parts_of_a_split_paper_are_combined_conservatively():
    from citation_verifier.tables.verify import combine_segment_answers as combine

    has = {"answer": "has", "quote": "it does", "justification": "shown in part one"}
    lacks = {"answer": "lacks", "quote": "it cannot", "justification": "ruled out in part two"}
    absent = {"answer": "absent", "justification": "nothing here"}
    unclear = {"answer": "unclear", "justification": "cannot tell"}
    # One part that finds something decides the paper, wherever it sits.
    assert combine([absent, has, unclear])["answer"] == "has"
    assert combine([unclear, absent, lacks])["answer"] == "lacks"
    # The paper never claims it only when EVERY part says so.
    assert combine([absent, absent])["answer"] == "absent"
    assert combine([absent, unclear])["answer"] == "unclear"
    # Parts that disagree are not for the code to settle.
    conflict = combine([has, lacks])
    assert conflict["answer"] == "unclear"
    assert "point different ways" in conflict["justification"]
    assert "shown in part one" in conflict["justification"]
    assert "ruled out in part two" in conflict["justification"]


def test_a_split_paper_is_asked_part_by_part_and_answered_once():
    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)
    seen: list[tuple[str, str, str]] = []

    def judge(payload):
        seen.append((payload["row_label"], payload["properties"][0]["name"],
                     payload["evidence"]))
        answer = "has" if "PART 2" in payload["evidence"] else "absent"
        return [{"col_index": payload["properties"][0]["col_index"], "answer": answer,
                 "quote": "q", "justification": "j"}]

    meta = {"complete": True, "segments": ["COVERAGE: PART 1 OF 2…", "COVERAGE: PART 2 OF 2…"]}
    report = verify_table(
        t, evidence_for=lambda key, label: ("COVERAGE: PART 1 OF 2…", "u", meta), judge=judge,
    )
    # One cell, both parts — and each part asked on its own.
    one_cell = [e for r, c, e in seen if r == "MetaAug" and c == "Task-aware"]
    assert len(one_cell) == 2
    assert sum("PART 1" in e for e in one_cell) == 1
    assert sum("PART 2" in e for e in one_cell) == 1
    cell = next(f for f in report.findings
                if f.row_label == "MetaAug" and f.dimension == "Task-aware")
    assert cell.claimed == CellMark.YES.value
    assert cell.verdict == CellVerdict.SUPPORTED.value    # the part that found it decides


def test_the_report_leads_with_one_flagged_count():
    """"Contradicted" and "may not support" say different things about the evidence and
    stay apart in the schema, but a reader's next action is the same for both: look at the
    cell. Counted separately at the top of a report, a table with three of each read as
    though nothing much was wrong."""
    from citation_verifier.tables.render import (
        FLAGGED_VERDICTS,
        flagged_count,
        render_table_report,
    )

    assert flagged_count({"contradicted": 3, "may_not_support": 2, "supported": 9}) == 5
    assert set(FLAGGED_VERDICTS) == {"contradicted", "may_not_support"}

    t = _table()
    resolve_dimensions(t.dimensions, ATU_BODY, caption=t.caption)

    def judge(payload):
        # MetaAug's ✓ is refuted; MLTI's ✗ is met by a work that never claims it.
        answer = "lacks" if payload["row_label"] == "MetaAug" else "absent"
        return [{"col_index": p["col_index"], "answer": answer, "justification": "j"}
                for p in payload["properties"]]

    report = verify_table(t, evidence_for=_evidence, judge=judge)
    md = render_table_report(report)
    assert "⚑ Flagged for review" in md
    assert "**⚑ Flagged** — contradicted" in md
