"""
tables/pipeline.py — one entry point for table-level verification of a whole paper.

:func:`verify_paper_tables` is what both callers run:

  * the orchestrator, after the prose pass, so a normal ``cverify`` run checks the
    related-work tables the prose pipeline has always skipped; and
  * a tables-only run (``cverify <paper> --tables only``), which is how the reported
    table results are reproduced — the same function, with nothing else around it.

It is deliberately the ONLY place the four stages (grid -> meaning -> evidence -> verdict)
are wired to real references, a real resolver and a real model. Experiments used to wire
them by hand in throwaway scripts; a result that only a script outside the repository can
produce is not reproducible, and that script no longer exists.

Two things here are shared with the prose pipeline on purpose:

  * **the reference resolution.** A table row is bound to the same ``CitedAs`` entry a
    prose citation of that key is, and resolved by the same correctness stage
    (:func:`citation_verifier.stages.fill_correctness`) — reusing the prose pass's result
    when there is one. Resolving it separately, from a differently-built reference string,
    let the two halves of one report disagree about which paper a citation is, and left
    the table path without the wrong-paper veto the prose path has.
  * **the scope.** Only tables that position CITED work are verified: a row needs a
    citation to check against. Results tables are dropped at extraction, and an ablation
    grid of ✓/✗ with no cited rows is dropped here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..interfaces import PaperSource, RunUsage
from ..schema import CitationRecord, CitedAs, Claim, Exists, Resolved
from .dimensions import dimension_is_checkable, resolve_dimensions
from .evidence import build_evidence_provider
from .model import CellMark, CellVerdict, ComparisonTable, TableReport
from .verify import verify_table

__all__ = ["TableRun", "verify_paper_tables", "is_related_work_table", "annotate_table_citations"]

# A paper-level bound on judged cells. One cell is one model call carrying the whole cited
# paper, so a survey's 40-row matrix is hundreds of long requests; a default-on feature in
# a deployed bot needs a ceiling that does not depend on what gets uploaded.
DEFAULT_MAX_CELLS = 150

_VERIFIABLE = (CellMark.YES.value, CellMark.NO.value, CellMark.PARTIAL.value)


@dataclass
class TableRun:
    """What a table pass produced: the reports, what the model cost, and what degraded."""

    reports: list[TableReport] = field(default_factory=list)
    usage: RunUsage = field(default_factory=RunUsage)
    errors: list[str] = field(default_factory=list)


def is_related_work_table(table: ComparisonTable) -> bool:
    """True when the table makes claims about CITED work that can be checked.

    Needs at least one row that carries a citation and is not the authors' own, and at
    least one ✓/✗-style cell in such a row. An ablation table (variants of the authors'
    method against components) has marks but no cited rows: there is no other paper to
    hold it against.
    """
    cited_rows = {r.row_index for r in table.rows if r.cite_keys and not r.is_self}
    if not cited_rows:
        return False
    return any(c.row_index in cited_rows and c.mark in _VERIFIABLE for c in table.cells)


def _checkable_cells(table: ComparisonTable) -> int:
    """How many cells of ``table`` would be sent to the judge."""
    rows = {r.row_index for r in table.rows if r.cite_keys and not r.is_self}
    cols = {d.col_index for d in table.dimensions if dimension_is_checkable(d)}
    return sum(
        1 for c in table.cells
        if c.row_index in rows and c.col_index in cols and c.mark in _VERIFIABLE
    )


class _Grounder:
    """Binds a table row's citation to a reference, and the reference to a paper.

    ``lookup`` and ``resolve`` are the two callables
    :func:`~citation_verifier.tables.evidence.build_evidence_provider` needs. A row's
    citation is a ``\\cite`` key in a LaTeX table and a printed marker in a PDF one
    (``"21"``, ``"Rajendran 2020"``); both are mapped onto the paper's reference list.
    """

    def __init__(
        self,
        references: dict[str, CitedAs],
        resolved: dict[str, Resolved] | None,
        resolver: Any,
        errors: list[str],
        web: Any = None,
    ) -> None:
        self.references = references
        self.resolved = dict(resolved or {})
        self.resolver = resolver
        self.errors = errors
        self.web = web
        self._tried: set[str] = set()

    def key_for(self, marker: str | None) -> str | None:
        """The reference-list key a row's citation refers to, or ``None``."""
        m = (marker or "").strip()
        if not m:
            return None
        if m in self.references:
            return m
        if m.isdigit() and f"ref-{m}" in self.references:
            return f"ref-{m}"
        parts = m.split()
        if len(parts) == 2 and parts[1][:4].isdigit():  # "Rajendran 2020"
            from ..extract.pdf import _bind_author_year  # noqa: PLC0415 — lazy, optional path

            hit = _bind_author_year(f"{parts[0]}{parts[1]}", self.references)
            if hit is not None:
                return next((k for k, v in self.references.items() if v is hit), None)
        return None

    def lookup(self, marker: str | None) -> CitedAs | None:
        key = self.key_for(marker)
        return self.references.get(key) if key else None

    def resolve(self, marker: str, _reference: str, /) -> Resolved | None:
        """The paper this citation resolves to — the prose pass's answer when it has one."""
        key = self.key_for(marker)
        if key is None:
            return None
        if key in self.resolved:
            return self.resolved[key]
        if key in self._tried or self.resolver is None:
            return None
        self._tried.add(key)
        # Same stage, same veto, same reference string — and the same last resort, the
        # web — as a prose citation of this key.
        from ..stages import fill_correctness  # noqa: PLC0415 — keeps the import network-free

        stub = CitationRecord(
            paper_id="", claim_id=f"table:{key}", cite_key=key,
            claim=Claim(claim_id=f"table:{key}", text=""), cited_as=self.references[key],
        )
        try:
            if self.web is not None:
                fill_correctness(stub, resolver=self.resolver, web=self.web)
            else:
                fill_correctness(stub, resolver=self.resolver)
        except Exception as exc:  # noqa: BLE001 — degrade-not-crash
            self.errors.append(f"table reference {key!r}: resolution failed: {exc!r}")
            return None
        if getattr(stub.exists, "value", stub.exists) == Exists.YES.value and stub.resolved:
            self.resolved[key] = stub.resolved
            return stub.resolved
        return None

    def coverage(self, table: ComparisonTable) -> float:
        """Share of the table's cited rows whose citation maps onto the reference list."""
        rows = [r for r in table.rows if r.cite_keys and not r.is_self]
        if not rows:
            return 0.0
        return sum(1 for r in rows if self.key_for(r.cite_keys[0])) / len(rows)


def _build_llm(settings: Any | None) -> Any | None:
    """The model transport for the glosser and the cell judge, or ``None`` (offline floor)."""
    if settings is not None and not getattr(settings, "enable_relevance_judge", True):
        return None
    try:
        from ..backends.relevance_judge import build_relevance_judge  # noqa: PLC0415

        return build_relevance_judge(settings)
    except Exception:  # noqa: BLE001 — no SDK / no key: the deterministic floor still runs
        return None


def verify_paper_tables(
    source: PaperSource,
    *,
    settings: Any | None = None,
    references: dict[str, CitedAs] | None = None,
    resolved: dict[str, Resolved] | None = None,
    resolver: Any | None = None,
    judge: Callable[[dict], list[dict]] | None = None,
    glosser: Callable[[list[dict]], list[dict]] | None = None,
    use_model: bool = True,
    use_full_text: bool = True,
    max_cells: int | None = None,
    budget_usd: float | None = None,
    progress: Callable[[str], None] | None = None,
    web: Any | None = None,
) -> TableRun:
    """Verify every related-work comparison table of one paper.

    Args:
        source: the ingested paper. Its LaTeX e-print is fetched if it is not on disk.
        settings: resolved settings (model routing, source keys); optional.
        references: the paper's bibliography, ``{cite_key: CitedAs}``. Loaded by the same
            route as the prose extractor when omitted.
        resolved: citations the prose pass already resolved, ``{cite_key: Resolved}``;
            reused so a citation is the same paper in the table as in the prose.
        resolver: a :class:`~citation_verifier.interfaces.Resolver` for rows the prose
            pass did not cover. Built from ``settings`` when omitted.
        judge, glosser: the two model seams, for tests and custom transports. Built from
            ``settings`` only when BOTH are omitted and ``use_model`` is true; supplying
            either one means no model is constructed here.
        use_model: ``False`` runs the deterministic floor — tables extracted, columns
            defined from the paper's own text, every cell ``unverifiable``.
        use_full_text: fetch each cited paper's body (the default, and what the reported
            results use). ``False`` judges from title and abstract only — no retrieval.
        max_cells: ceiling on judged cells for the whole paper (:data:`DEFAULT_MAX_CELLS`;
            0 = no cap). Tables beyond it are extracted and reported, not judged, and say so.
        budget_usd: what this pass may still spend. Checked before each table — a table
            is not started once the spend so far has reached it. ``None`` = no cap.
        progress: optional callback for one-line status updates.
        web: a :class:`~citation_verifier.grounding.web_lookup.WebLocator`, the last resort
            for a row's citation no index places. Built from ``settings`` when omitted —
            only when the model is (a web search is a model call), and only if
            ``enable_web_search`` is on.

    Returns:
        A :class:`TableRun`. Never raises: a failure is recorded in ``errors`` and the
        tables that could be handled still are.
    """
    from . import extract_tables, read_body_text  # noqa: PLC0415 — package-level convenience

    run = TableRun(usage=RunUsage(backend="tables"))
    say = progress or (lambda _msg: None)

    try:
        from ..extract.latex import ensure_eprint  # noqa: PLC0415

        ensure_eprint(source)  # no-op when already on disk / offline / not an arXiv paper
        tables = [t for t in extract_tables(source) if is_related_work_table(t)]
    except Exception as exc:  # noqa: BLE001 — degrade-not-crash
        run.errors.append(f"table extraction failed: {exc!r}")
        return run
    if not tables:
        return run

    if references is None:
        try:
            from ..extract import load_references  # noqa: PLC0415

            references = load_references(source)
        except Exception as exc:  # noqa: BLE001
            run.errors.append(f"could not load the bibliography for table rows: {exc!r}")
            references = {}

    if resolver is None:
        try:
            from ..grounding import MultiSourceResolver  # noqa: PLC0415

            resolver = MultiSourceResolver(settings=settings)
        except Exception as exc:  # noqa: BLE001
            run.errors.append(f"no resolver available for table rows: {exc!r}")

    # A model is built only when NEITHER seam was supplied. A caller that injects its own
    # judge has decided what answers the questions; silently adding a real model beside
    # it to write the column definitions would make a stubbed run spend money.
    llm = None
    if use_model and judge is None and glosser is None:
        llm = _build_llm(settings)
        if llm is not None:
            from .llm import build_cell_judge, build_glosser  # noqa: PLC0415

            judge = build_cell_judge(llm)
            glosser = build_glosser(llm)
        else:
            off = settings is not None and not getattr(settings, "enable_relevance_judge", True)
            why = (
                "the model judge is switched off (ENABLE_RELEVANCE_JUDGE=false)" if off
                else "no model transport is available (no API key and no Agent SDK)"
            )
            run.errors.append(
                f"{why}: comparison tables were extracted and their columns defined, "
                f"but no cell was judged"
            )

    searcher = None
    if web is None and llm is not None:
        web, searcher = _build_web(settings, resolver)
    # A cited work whose PDF is a scan is read by a model — in a run that built one.
    reader = _build_scan_reader(settings) if llm is not None else None

    body = read_body_text(source)
    grounder = _Grounder(references or {}, resolved, resolver, run.errors, web)
    pdf_grounder: _Grounder | None = None
    budget = DEFAULT_MAX_CELLS if max_cells is None else max_cells
    spent = 0

    from ..grounding.fulltext import reading_scans  # noqa: PLC0415 — lazy, network-free

    with reading_scans(reader):
        for table in tables:
            own = {n for r in table.rows if r.is_self for n in [_own_name(r.label)] if n}
            try:
                resolve_dimensions(
                    table.dimensions, body, caption=table.caption, legend=table.legend,
                    glosser=glosser, own_names=own, paper_id=table.paper_id,
                    table_id=table.table_id, table_label=table.label or "",
                )
            except Exception as exc:  # noqa: BLE001
                run.errors.append(f"{table.table_id}: column definitions failed: {exc!r}")

            # A table read from the PDF cites by printed marker; when those do not map onto
            # the LaTeX keys the bibliography was loaded under, bind them to the PDF's own
            # reference list instead.
            bound = grounder
            if grounder.coverage(table) < 0.5 and source.pdf_path:
                if pdf_grounder is None:
                    try:
                        from ..extract.pdf import load_pdf_references  # noqa: PLC0415

                        pdf_grounder = _Grounder(
                            load_pdf_references(source.pdf_path), None, resolver, run.errors, web
                        )
                    except Exception as exc:  # noqa: BLE001
                        run.errors.append(f"could not read the PDF reference list: {exc!r}")
                if pdf_grounder is not None and pdf_grounder.coverage(table) > grounder.coverage(table):
                    bound = pdf_grounder

            cells = _checkable_cells(table)
            cost_so_far = float(getattr(getattr(llm, "usage", None), "cost_usd", 0.0) or 0.0)
            over_cells = budget > 0 and spent + cells > budget
            over_cost = budget_usd is not None and cost_so_far >= budget_usd
            over_budget = bool(judge) and (over_cells or over_cost)
            say(f"table {table.table_id}: {cells} cell(s) to check" + (" — over budget, skipped" if over_budget else ""))
            evidence_for = build_evidence_provider(
                lookup=bound.lookup, resolver=bound, dimensions=list(table.dimensions),
                use_full_text=use_full_text,
            )
            report = verify_table(table, evidence_for=evidence_for, judge=None if over_budget else judge)
            if over_budget:
                if over_cost and not over_cells:
                    note = (
                        "not checked — the run's cost ceiling was reached before this table "
                        "(raise COST_CEILING_USD, or run the tables on their own)"
                    )
                else:
                    note = (
                        f"not checked — this table's {cells} cells would take the run past "
                        f"its budget of {budget} table cells (set TABLE_MAX_CELLS to raise it)"
                    )
                for f in report.findings:
                    if f.verdict == CellVerdict.UNVERIFIABLE.value:
                        f.justification = note
                report.notes.append(note)
                run.errors.append(f"{table.table_id}: {note}")
            elif judge is not None:
                spent += cells
            run.reports.append(report)

    down = getattr(llm, "unavailable", "") if llm is not None else ""
    if down:
        from ..errors import unavailable_message  # noqa: PLC0415

        run.errors.append(unavailable_message(down, "comparison-table cells were NOT judged"))
    if llm is not None and getattr(llm, "usage", None) is not None:
        run.usage.add(llm.usage)
        run.usage.model = getattr(llm, "model", "") or run.usage.model
    for helper in (searcher, reader):
        drain = getattr(helper, "drain_usage", None)
        if callable(drain):
            run.usage.add(drain())
    if getattr(reader, "unavailable", ""):
        run.errors.append(
            f"scan reading unavailable ({reader.unavailable}); cited works whose PDF is a scan "
            "were not read"
        )
    if getattr(web, "unavailable", ""):
        run.errors.append(
            f"web search unavailable ({web.unavailable}); table citations no index found were "
            "not searched on the web"
        )
    return run


def _build_scan_reader(settings: Any | None) -> Any:
    """The scan reader for the table pass, or ``None`` when scan reading is off or cannot
    be built. ``settings["scan_reader"]`` injects one (tests)."""
    def setting(name: str, default: Any) -> Any:
        if isinstance(settings, dict):
            return settings.get(name, default)
        return getattr(settings, name, default) if settings is not None else default

    if not setting("read_scans", True):
        return None
    reader = setting("scan_reader", None)
    if reader is not None:
        return reader
    try:
        from ..backends.scan_reader import ScanReader  # noqa: PLC0415

        return ScanReader(settings)
    except Exception:  # noqa: BLE001 — scan wiring never breaks the table pass
        return None


def _build_web(settings: Any | None, resolver: Any) -> tuple[Any, Any]:
    """``(locator, searcher)`` for the table pass, or ``(None, None)`` when web search is
    off or cannot be built. ``settings["web_searcher"]`` injects the finder (tests)."""
    def setting(name: str, default: Any) -> Any:
        if isinstance(settings, dict):
            return settings.get(name, default)
        return getattr(settings, name, default) if settings is not None else default

    if resolver is None or not setting("enable_web_search", True):
        return None, None
    try:
        from ..grounding.web_lookup import locator_for  # noqa: PLC0415

        searcher = setting("web_searcher", None)
        if searcher is None:
            from ..backends.web_search import WebSearcher  # noqa: PLC0415

            searcher = WebSearcher(settings)
        return locator_for(searcher, resolver, settings), searcher
    except Exception:  # noqa: BLE001 — search wiring never breaks the table pass
        return None, None


def _own_name(label: str) -> str:
    from . import _own_name as impl  # noqa: PLC0415 — defined beside extract_tables

    return impl(label)


def annotate_table_citations(records: list[CitationRecord], reports: list[TableReport]) -> int:
    """Point each table-cell citation record at the table findings that now cover it.

    The prose pipeline emits a record for a ``\\cite`` inside a table and marks it
    ``skipped`` ("cited inside a table — relevance not assessed"): a cell is not a
    sentence. Once the table is verified that note is out of date — the citation WAS
    assessed, cell by cell — so it is replaced with where to look and what was found.
    Returns how many records were annotated. ``supports_claim`` stays ``skipped``: the
    verdicts live on the cells, and one row has several.
    """
    by_key: dict[str, list[tuple[str, dict[str, int]]]] = {}
    for rep in reports:
        counts: dict[str, dict[str, int]] = {}
        for f in rep.findings:
            if f.cite_key and f.verdict != CellVerdict.SKIPPED.value:
                tally = counts.setdefault(f.cite_key, {})
                tally[f.verdict] = tally.get(f.verdict, 0) + 1
        for key, tally in counts.items():
            by_key.setdefault(key, []).append((rep.table.table_id, tally))

    flagged = (CellVerdict.CONTRADICTED.value, CellVerdict.MAY_NOT_SUPPORT.value)
    done = 0
    for rec in records:
        if not rec.in_table or rec.cite_key not in by_key:
            continue
        bits = []
        for table_id, tally in by_key[rec.cite_key]:
            n = sum(tally.values())
            bad = sum(tally.get(v, 0) for v in flagged)
            bits.append(
                f"`{table_id}`: {n} cell(s) checked, "
                + (f"{bad} flagged for review" if bad else "none flagged")
            )
        note = "cited in a comparison table — verified cell by cell (" + "; ".join(bits) + "); see Comparison tables"
        kept = [
            part for part in (rec.notes or "").split("; ")
            if part and "relevance not assessed" not in part
        ]
        rec.notes = "; ".join([*kept, note])
        done += 1
    return done
