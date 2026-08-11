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

from citation_verifier.tables import (
    CellMark,
    CellVerdict,
    asymmetry_summary,
    derive_cell_severity,
    find_definition_snippets,
    looks_like_comparison_table,
    normalize_mark,
    resolve_dimensions,
    tables_from_latex,
    verify_table,
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
    body = "Our approach is model-adaptive in spirit and we discuss model-adaptive behaviour."
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

    assert "FORMAL definition" in s and "QUALITATIVE definition" in s
    assert "analogous-sounding mechanism does not qualify" in s
    assert "interpolating raw features and labels" in s   # the concrete measured failure
    assert "name the exact part of the definition" in s
    assert "not actually about the named work" in s      # garbage evidence -> unclear
    assert "absence is not refutation" in s
    # …and the opposite failure: over-caution on a plain-words property hides real errors.
    assert "Do NOT demand a formal proof it never asked for" in s


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
