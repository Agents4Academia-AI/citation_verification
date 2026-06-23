"""
Offline tests for the orchestrator's per-record finalize. Focus: a resolved
citation (``exists == yes``) gets its source link appended to the explanation
(``notes``) so the verdict is auditable; everything else is left untouched.
"""

from __future__ import annotations

from citation_verifier.orchestrator import _append_source_links
from citation_verifier.schema import CitationRecord, CitedAs, Claim, Exists, Resolved


def _rec(exists, *, url="", doi="", notes=None):
    return CitationRecord(
        paper_id="p",
        claim_id="c",
        cite_key="ref-1",
        claim=Claim(claim_id="c", text="x"),
        cited_as=CitedAs(raw="r"),
        exists=exists,
        resolved=Resolved(source="s2", url=url, doi=doi) if (url or doi) else None,
        notes=notes,
    )


def test_append_source_links_adds_link_for_existing_citations():
    rec = _rec(Exists.YES, url="https://s2.org/abc", notes="Evidence supports the claim.")
    _append_source_links([rec])
    assert rec.notes == "Evidence supports the claim. · source: https://s2.org/abc"


def test_append_source_links_is_idempotent():
    rec = _rec(Exists.YES, url="https://s2.org/abc", notes="ok")
    _append_source_links([rec])
    _append_source_links([rec])
    assert rec.notes.count("source: https://s2.org/abc") == 1


def test_append_source_links_falls_back_to_doi_and_skips_unresolved():
    doi_rec = _rec(Exists.YES, doi="10.1/abc")
    unresolved = _rec(Exists.UNRESOLVED, notes="no match")
    _append_source_links([doi_rec, unresolved])
    assert doi_rec.notes == "source: https://doi.org/10.1/abc"
    assert unresolved.notes == "no match"  # untouched
