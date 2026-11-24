"""
Offline tests for the WHOLE-PAPER table pass: ``tables.verify_paper_tables`` and its
wiring into ``orchestrator.run_verification`` / the CLI / the rendered report.

No network and no SDK. The paper is a small LaTeX tree written to ``tmp_path``; the
resolver, the judge and (for orchestrator runs) the backend are stubs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citation_verifier.interfaces import PaperSource, RunUsage, VerificationResult
from citation_verifier.schema import (
    CitationRecord,
    CitedAs,
    Claim,
    Exists,
    MatchMethod,
    Resolved,
    SupportsClaim,
)
from citation_verifier.tables import (
    CellVerdict,
    annotate_table_citations,
    is_related_work_table,
    tables_from_latex,
    verify_paper_tables,
)

MAIN_TEX = r"""
\documentclass{article}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
\begin{document}
\section{Introduction}
Prior work is fast \cite{alpha2020} but not robust. We require two properties.
Robust: the method remains correct when its inputs are perturbed.
Training-free: the method needs no additional training.
Table~\ref{tab:related} compares \textsc{Ours} with prior methods.

\begin{table}
\caption{Comparison with prior methods.}
\label{tab:related}
\begin{tabular}{lcc}
\toprule
Method & Robust & Training-free \\
\midrule
Alpha~\cite{alpha2020} & \xmark & \cmark \\
Beta~\cite{beta2021} & \cmark & \xmark \\
Ours & \cmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}

\section{Experiments}
\begin{table}
\caption{Accuracy on three benchmarks.}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
Method & A & B & C \\
\midrule
Alpha~\cite{alpha2020} & 71.2 & 80.1 & 64.3 \\
Ours & 75.9 & 83.4 & 70.0 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}
\caption{Ablation of our components.}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
Variant & Module A & Module B \\
\midrule
Ours w/o A & \xmark & \cmark \\
Ours w/o B & \cmark & \xmark \\
Ours full & \cmark & \cmark \\
\bottomrule
\end{tabular}
\end{table}
\bibliography{refs}
\end{document}
"""

REFS_BIB = r"""
@inproceedings{alpha2020, author={Ann Alpha and Bob Able}, title={Alpha: a fast method}, booktitle={ICML}, year={2020}}
@inproceedings{beta2021, author={Carl Beta}, title={Beta: a robust method}, booktitle={NeurIPS}, year={2021}}
"""


def _paper(tmp_path: Path) -> PaperSource:
    tex = tmp_path / "tex"
    tex.mkdir(parents=True)
    (tex / "main.tex").write_text(MAIN_TEX, encoding="utf-8")
    (tex / "refs.bib").write_text(REFS_BIB, encoding="utf-8")
    return PaperSource(
        paper_id="synthetic", kind="latex", tex_dir=str(tex), tex_available=True,
        work_dir=str(tmp_path),
    )


def _resolved(title: str) -> Resolved:
    return Resolved(source="stub", match_method=MatchMethod.FUZZY_TITLE, match_score=1.0,
                    title=title, authors=["A. Author"], year=2020, abstract=f"{title}. An abstract.")


class _StubResolver:
    """Resolves by title lookup in a fixed map; counts calls."""

    sources = ("stub",)

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, cite_key: str, reference: str, /):
        self.calls.append(cite_key)
        for title in ("Alpha: a fast method", "Beta: a robust method"):
            if title.lower() in reference.lower():
                return _resolved(title)
        return None

    def candidates(self, reference: str, /, max_results: int = 4):
        return []


def _judge(answers: dict[tuple[str, str], str]):
    asked: list[tuple[str, str]] = []

    def judge(payload: dict) -> list[dict]:
        prop = payload["properties"][0]
        asked.append((payload["row_label"], prop["name"]))
        answer = answers.get((payload["row_label"], prop["name"]), "unclear")
        return [{"col_index": prop["col_index"], "answer": answer, "quote": "q", "justification": "j"}]

    judge.asked = asked  # type: ignore[attr-defined]
    return judge


# ── scope: which tables are verified ─────────────────────────────────────────
def test_only_tables_that_position_cited_work_are_verified(tmp_path):
    """A results table is dropped at extraction; an ablation grid has ✓/✗ but no cited
    row, so there is no other paper to hold it against and it is dropped here."""
    judge = _judge({})
    run = verify_paper_tables(
        _paper(tmp_path), resolver=_StubResolver(), judge=judge, use_full_text=False
    )
    assert [r.table.table_id for r in run.reports] == ["tab:related"]
    assert {row for row, _col in judge.asked} == {"Alpha", "Beta"}  # never "Ours"

    ablation = next(
        t for t in tables_from_latex(MAIN_TEX, paper_id="p") if t.label == "tab:ablation"
    )
    assert not is_related_work_table(ablation)


def test_cells_are_judged_against_the_cited_papers(tmp_path):
    judge = _judge({
        ("Alpha", "Robust"): "has",            # table says ✗ -> contradicted
        ("Alpha", "Training-free"): "has",     # table says ✓ -> supported
        ("Beta", "Robust"): "has",             # ✓ -> supported
        ("Beta", "Training-free"): "lacks",    # ✗ -> supported
    })
    run = verify_paper_tables(
        _paper(tmp_path), resolver=_StubResolver(), judge=judge, use_full_text=False
    )
    got = {(f.row_label, f.dimension): f.verdict for f in run.reports[0].findings}
    assert got[("Alpha", "Robust")] == CellVerdict.CONTRADICTED.value
    assert got[("Alpha", "Training-free")] == CellVerdict.SUPPORTED.value
    assert got[("Beta", "Training-free")] == CellVerdict.SUPPORTED.value
    assert got[("Ours", "Robust")] == CellVerdict.SKIPPED.value
    assert run.errors == []


def test_a_citation_is_the_same_paper_in_the_table_as_in_the_prose(tmp_path):
    """Resolutions from the prose pass are reused: the table path must not resolve a
    citation a second time, from a differently-built reference string, and risk landing
    on a different paper than the one the prose verdict is about."""
    resolver = _StubResolver()
    prose = {"alpha2020": _resolved("Alpha: a fast method")}
    run = verify_paper_tables(
        _paper(tmp_path), resolver=resolver, resolved=prose,
        judge=_judge({}), use_full_text=False,
    )
    assert run.reports
    assert resolver.calls == ["beta2021"]          # alpha came from the prose pass


def test_an_unresolvable_row_is_unverified_not_a_finding(tmp_path):
    class Nothing(_StubResolver):
        def resolve(self, cite_key, reference, /):
            return None

    run = verify_paper_tables(
        _paper(tmp_path), resolver=Nothing(), judge=_judge({}), use_full_text=False
    )
    verdicts = {f.verdict for f in run.reports[0].findings if f.row_label != "Ours"}
    assert verdicts == {CellVerdict.UNVERIFIABLE.value}


def test_without_a_model_the_tables_are_still_extracted_and_defined(tmp_path):
    """The offline floor: no judge, no network — the grid and the column meanings are
    reported and every cell is honestly unverifiable."""
    run = verify_paper_tables(_paper(tmp_path), resolver=_StubResolver(), use_model=False)
    report = run.reports[0]
    assert [d.header for d in report.table.dimensions] == ["Robust", "Training-free"]
    assert "perturbed" in report.table.dimensions[0].gloss
    assert all(
        f.verdict in (CellVerdict.UNVERIFIABLE.value, CellVerdict.SKIPPED.value)
        for f in report.findings
    )


def test_a_table_past_the_cell_budget_is_reported_but_not_judged(tmp_path):
    judge = _judge({})
    run = verify_paper_tables(
        _paper(tmp_path), resolver=_StubResolver(), judge=judge, use_full_text=False,
        max_cells=3,                                # the table has 4 checkable cells
    )
    assert judge.asked == []
    assert any("budget of 3 table cells" in e for e in run.errors)
    assert all("not checked" in f.justification for f in run.reports[0].findings
               if f.verdict == CellVerdict.UNVERIFIABLE.value)


# ── the prose records that sit inside a verified table ───────────────────────
def test_table_citation_records_point_at_the_table_findings(tmp_path):
    judge = _judge({("Alpha", "Robust"): "has"})   # ✗ contradicted -> flagged
    run = verify_paper_tables(
        _paper(tmp_path), resolver=_StubResolver(), judge=judge, use_full_text=False
    )

    def rec(key: str, *, in_table: bool) -> CitationRecord:
        return CitationRecord(
            paper_id="synthetic", claim_id=f"c-{key}-{in_table}", cite_key=key,
            claim=Claim(claim_id="c", text="x"), cited_as=CitedAs(raw="r"), in_table=in_table,
            notes="cited inside a table — relevance not assessed" if in_table else None,
        )

    records = [rec("alpha2020", in_table=True), rec("alpha2020", in_table=False),
               rec("gamma1999", in_table=True)]
    assert annotate_table_citations(records, run.reports) == 1
    assert "verified cell by cell" in records[0].notes
    assert "1 flagged for review" in records[0].notes
    assert "relevance not assessed" not in records[0].notes
    assert records[1].notes is None                       # a prose citation: untouched
    assert "relevance not assessed" in records[2].notes   # in a table we did not verify


# ── orchestrator / CLI / report ──────────────────────────────────────────────
class _FakeBackend:
    name = "fake"

    def verify(self, source, stubs):
        for s in stubs:
            s.exists = Exists.YES
            s.resolved = _resolved(s.cited_as.title or "t")
            if not s.in_table:
                s.supports_claim = SupportsClaim.SUPPORTS
        return VerificationResult(paper_id=source.paper_id, backend="fake",
                                  records=list(stubs), usage=RunUsage(backend="fake"))


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """``run_verification`` over the synthetic paper, with no network and no model."""
    import citation_verifier.ingest as ingest
    import citation_verifier.orchestrator as orch
    import citation_verifier.tables.pipeline as pipeline
    from citation_verifier.config import Settings

    source = _paper(tmp_path)
    monkeypatch.setattr(ingest, "ingest", lambda src, settings=None: source)
    monkeypatch.setattr(orch, "_get_backend", lambda name, settings=None: _FakeBackend())
    monkeypatch.setattr(pipeline, "_build_llm", lambda settings: None)
    settings = Settings(papers_dir=tmp_path, cost_ceiling_usd=0.0)
    return orch, settings, tmp_path


def test_a_normal_run_verifies_the_related_work_table_too(wired):
    from citation_verifier.render import render_report

    orch, settings, work = wired
    result = orch.run_verification("synthetic", settings=settings, resume=False)

    assert [t.table.table_id for t in result.tables] == ["tab:related"]
    in_table = [r for r in result.records if r.in_table and r.cite_key == "beta2021"]
    assert in_table and "verified cell by cell" in in_table[0].notes
    assert SupportsClaim(in_table[0].supports_claim) is SupportsClaim.SKIPPED

    saved = json.loads((work / "tables.json").read_text())
    assert saved["tables"][0]["table"]["table_id"] == "tab:related"
    md = render_report(result)
    assert "## Comparison tables" in md and "## Comparison tables" in (work / "report.md").read_text()


def test_tables_off_leaves_the_report_as_it_was(wired):
    from citation_verifier.render import render_report

    orch, settings, work = wired
    result = orch.run_verification("synthetic", settings=settings, resume=False, tables="off")
    assert result.tables == []
    assert not (work / "tables.json").exists()
    assert "## Comparison tables" not in render_report(result)
    assert any("relevance not assessed" in (r.notes or "") for r in result.records if r.in_table)


def test_tables_only_runs_nothing_else_and_touches_no_report(wired):
    orch, settings, work = wired
    (work / "report.json").write_text('{"sentinel": true}')
    result = orch.run_verification("synthetic", settings=settings, resume=False, tables="only")
    assert result.records == []
    assert [t.table.table_id for t in result.tables] == ["tab:related"]
    assert json.loads((work / "report.json").read_text()) == {"sentinel": True}
    assert (work / "tables.json").exists() and "Comparison tables" in (work / "tables.md").read_text()
    # …and a second run is served from tables.json.
    again = orch.run_verification("synthetic", settings=settings, resume=True, tables="only")
    assert [t.table.table_id for t in again.tables] == ["tab:related"]


def test_a_resumed_run_gets_its_tables_back(wired):
    orch, settings, _work = wired
    orch.run_verification("synthetic", settings=settings, resume=False)
    again = orch.run_verification("synthetic", settings=settings, resume=True)
    assert [t.table.table_id for t in again.tables] == ["tab:related"]


def test_an_unknown_table_mode_is_an_input_error(wired):
    orch, settings, _work = wired
    with pytest.raises(ValueError, match="tables must be one of"):
        orch.run_verification("synthetic", settings=settings, tables="sometimes")


def test_the_cli_exposes_the_table_modes():
    from citation_verifier.cli import build_parser

    assert build_parser().parse_args(["1706.03762"]).tables is None
    assert build_parser().parse_args(["1706.03762", "--tables", "only"]).tables == "only"


# ── scoring against human cell labels (evals/table_eval.py) ──────────────────
def test_table_eval_scores_flagged_cells_against_human_labels(tmp_path):
    """``contradicted`` and ``may_not_support`` are one category — "a person should look"
    — on both sides; a cell we leave open is a miss when the person flags it and is
    otherwise neither a clearance nor a false alarm."""
    from evals.table_eval import load_findings, load_labels, score

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "HUMAN VERDICT (correct / incorrect / indeterminate),Table,Cited method (row),Property column\n"
        "Contradicted,tab:x,A,P\n"                    # we flag            -> TP
        "Supported / May Not Support,tab:x,B,P\n"     # undecided = flagged; we flag -> TP
        "Supported,tab:x,C,P\n"                       # we flag            -> FP
        "May Not Support,tab:x,D,P\n"                 # we clear           -> FN
        "May not.,tab:x,E,P\n"                        # we leave open      -> FN (missed_open)
        "Supported,tab:x,F,P\n"                       # we clear           -> TN
        "Supported,tab:x,G,P\n"                       # we leave open      -> neither
        "Undetermined,tab:x,H,P\n"                    # column undefined   -> apart
        "Supported,tab:x,Ours,P\n",                   # skipped            -> ignored
        encoding="utf-8",
    )
    verdicts = {"A": "contradicted", "B": "may_not_support", "C": "contradicted", "D": "supported",
                "E": "unverifiable", "F": "supported", "G": "unverifiable", "H": "contradicted",
                "Ours": "skipped"}
    findings = tmp_path / "tables.json"
    findings.write_text(json.dumps({"tables": [{
        "table": {"table_id": "tab:x"},
        "findings": [{"row_label": r, "dimension": "P", "verdict": v} for r, v in verdicts.items()],
    }]}))

    got = score(load_findings([findings]), load_labels(labels))
    assert got["flagged"] == {"tp": 2, "fp": 1, "fn": 2, "tn": 1,
                              "precision": 0.6667, "recall": 0.5, "f1": 0.5714}
    assert got["left_open"] == {"flag": 1, "clear": 1}
    assert got["on_human_undetermined"] == {"flag": 1}
    assert got["cells_joined"] == 9 and got["unjoined"] == []


def test_table_eval_stays_on_its_side_of_the_boundary():
    """The scorer reads the agent's output from disk and imports nothing from it."""
    import evals.table_eval as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import citation_verifier" not in source and "from citation_verifier" not in source
