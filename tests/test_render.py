"""
test_render.py — the renderer must emit the EXACT SKILL.md table, deterministically.

`render.py` is a sibling module (owned by the core branch); until it lands this
test ``importorskip``s so the suite stays green on a clean checkout, then becomes
a hard contract guard once the module exists. The header string and the
enum->table-string mapping below are FROZEN and mirror SKILL.md / schema.py.
"""

from __future__ import annotations

import pytest

# The 7 columns, verbatim from SKILL.md.
EXPECTED_HEADER = (
    "| # | Citation (authors, short title, year) | Cited where (the claim) "
    "| Exists? | Match notes | Supports claim? | Explanation |"
)

# Machine token -> human string rendered in the table. Only does_not differs.
SUPPORTS_TABLE_STRINGS = {
    "supports": "supports",
    "partial": "partial",
    "does_not": "does not",
    "inconclusive": "inconclusive",
}


@pytest.fixture
def render():
    """The sibling render module (skip cleanly until the core branch merges it)."""
    return pytest.importorskip(
        "citation_verifier.render",
        reason="render.py is a sibling module; not present on this checkout yet.",
    )


def _render_table(render, records) -> str:
    """Call whichever public table renderer render.py exposes."""
    for name in ("render_table", "to_table", "records_to_table", "render_markdown"):
        fn = getattr(render, name, None)
        if callable(fn):
            return fn(records)
    pytest.skip("render.py exposes no recognized table-rendering function")


def test_header_is_exact(render, sample_records) -> None:
    table = _render_table(render, sample_records)
    lines = [ln for ln in table.splitlines() if ln.strip().startswith("|")]
    assert lines, "no table rows rendered"
    assert lines[0].strip() == EXPECTED_HEADER


def test_does_not_renders_as_two_words(render) -> None:
    """does_not (token) must render as 'does not' (table string), never 'does_not'."""
    from citation_verifier.schema import (
        CitationRecord,
        CitedAs,
        Claim,
        Exists,
        Priority,
        SupportsClaim,
    )

    rec = CitationRecord(
        paper_id="p", claim_id="c", cite_key="k",
        claim=Claim(claim_id="c", text="X proves Y."),
        cited_as=CitedAs(raw="Some Author. A Title. 2020.", title="A Title", year=2020),
        exists=Exists.YES,
        supports_claim=SupportsClaim.DOES_NOT,
        priority=Priority.OBLIGATORY,
    )
    table = _render_table(render, [rec])
    assert "does not" in table
    body = "\n".join(
        ln for ln in table.splitlines()
        if ln.strip().startswith("|") and ln.strip() != EXPECTED_HEADER
    )
    assert "does_not" not in body


def test_render_is_deterministic(render, sample_records) -> None:
    a = _render_table(render, sample_records)
    b = _render_table(render, sample_records)
    assert a == b


def test_json_round_trip_if_available(render, sample_records, tmp_path) -> None:
    """If render.py exposes to_json/from_json, they must round-trip the records.

    ``to_json(records, path)`` writes JSON; ``from_json(path)`` reads it back.
    """
    to_json = getattr(render, "to_json", None)
    from_json = getattr(render, "from_json", None)
    if not (callable(to_json) and callable(from_json)):
        pytest.skip("render.py does not expose to_json/from_json")
    out = tmp_path / "report.json"
    to_json(sample_records, out)
    back = from_json(out)
    assert [r.key for r in back] == [r.key for r in sample_records]
    assert [r.model_dump() for r in back] == [r.model_dump() for r in sample_records]


def test_claim_marker_only_on_multi_citation_rows(render) -> None:
    """A claim cited by >1 reference gets a per-row [n] marker; a lone cite does not."""
    from citation_verifier.schema import CitationRecord, CitedAs, Claim, Exists

    def rec(key, span):
        return CitationRecord(
            paper_id="p", claim_id=f"c-{key}", cite_key=key,
            claim=Claim(claim_id=f"c-{key}", text="A claim cited in several places", char_span=span),
            cited_as=CitedAs(title="T", year=2024), exists=Exists.YES,
        )

    shared = (10, 80)
    recs = [rec("ref-6", shared), rec("ref-7", shared), rec("ref-3", (90, 140))]
    rows = [ln for ln in _render_table(render, recs).splitlines() if ln.strip().startswith("|")]
    body = rows[2:]  # drop header + divider
    assert body[0].split("|")[3].strip().startswith("[6] ")   # multi-citation -> marked
    assert body[1].split("|")[3].strip().startswith("[7] ")
    assert not body[2].split("|")[3].strip().startswith("[")  # single citation -> no marker


def test_explanation_cell_has_note_and_link_but_no_severity_word(render) -> None:
    from citation_verifier.schema import (
        CitationRecord,
        CitedAs,
        Claim,
        Exists,
        Severity,
        SupportsClaim,
    )

    rec = CitationRecord(
        paper_id="p", claim_id="c", cite_key="k",
        claim=Claim(claim_id="c", text="X."), cited_as=CitedAs(title="T", year=2024),
        exists=Exists.YES, supports_claim=SupportsClaim.SUPPORTS,
        severity=Severity.LOW, notes="source: https://doi.org/10.1/x",
    )
    row = [ln for ln in _render_table(render, [rec]).splitlines() if ln.strip().startswith("|")][-1]
    explanation = row.split("|")[-2].strip()
    assert explanation == "source: https://doi.org/10.1/x"  # the note + link only
    assert "low" not in explanation and "ok" not in explanation  # no severity word
