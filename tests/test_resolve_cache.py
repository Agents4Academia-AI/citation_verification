"""On-disk caches for reference resolution and retrieved full text."""

from __future__ import annotations

import pytest

from citation_verifier.grounding import fulltext_cache, resolve_cache
from citation_verifier.schema import MatchMethod, Resolved


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", str(tmp_path))
    return tmp_path


REF = "J. Doe and A. Smith. A study of things. In NeurIPS, 2020."
REC = Resolved(source="arxiv", match_method=MatchMethod.FUZZY_TITLE, title="A study of things",
               authors=["J. Doe"], year=2020, arxiv_id="2001.00001")


def test_a_resolution_survives_the_process(cache):
    """Resolving one reference costs 2-19s and bibliographic records do not change, so the
    same forty-odd references should not be re-resolved on every run."""
    assert resolve_cache.read(REF, Resolved) is None
    resolve_cache.write(REF, REC)
    got = resolve_cache.read(REF, Resolved)
    assert got is not None
    assert got.title == "A study of things"
    assert got.arxiv_id == "2001.00001"


def test_the_same_reference_reached_two_ways_hits_one_entry(cache):
    """A reference read from a LaTeX .bbl and from a PDF's reference list differs only in
    whitespace and case; both must reuse the one resolution."""
    resolve_cache.write(REF, REC)
    noisy = "  j. doe and a. smith.   A STUDY of things. In NeurIPS,  2020. "
    assert resolve_cache.read(noisy, Resolved) is not None


def test_a_failed_resolution_is_not_cached(cache):
    """A miss is very often a rate limit rather than a fact about the world — the same key
    failed back-to-back and resolved once spaced out. Caching it would freeze a transient
    failure into a permanent one."""
    resolve_cache.write(REF, None)
    assert resolve_cache.read(REF, Resolved) is None
    assert list(cache.rglob("*.json")) == []


def test_a_corrupt_entry_reads_as_a_miss(cache):
    """An unwritable or damaged cache degrades to no cache, never to an error."""
    resolve_cache.write(REF, REC)
    entry = next(cache.rglob("*.json"))
    entry.write_text("{not json", encoding="utf-8")
    assert resolve_cache.read(REF, Resolved) is None


def test_the_cache_can_be_turned_off(monkeypatch):
    """Setting the variable to empty disables it — the suite relies on this to stay
    hermetic, and a deployment may want no on-disk state at all."""
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", "")
    assert resolve_cache.cache_dir() is None
    resolve_cache.write(REF, REC)
    assert resolve_cache.read(REF, Resolved) is None


def test_clear_empties_it(cache):
    resolve_cache.write(REF, REC)
    assert resolve_cache.clear() == 1
    assert resolve_cache.read(REF, Resolved) is None


# ── full-text cache ───────────────────────────────────────────────────


@pytest.fixture
def ft_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", str(tmp_path))
    return tmp_path


def test_a_retrieved_full_text_survives_the_process(ft_cache):
    """Fetching one paper's body walks arXiv HTML -> LaTeX e-print -> PDF and downloads up
    to 200 KB. The content does not change, so it should be fetched once, not once per
    run — and re-fetching is itself a source of run-to-run variance."""
    assert fulltext_cache.read("arxiv:2106.02695") is None
    fulltext_cache.write("arxiv:2106.02695", "the body text", "arxiv_html", "u")
    assert fulltext_cache.read("arxiv:2106.02695") == ("the body text", "arxiv_html", "u")


def test_an_empty_retrieval_is_not_cached(ft_cache):
    """An empty fetch is usually a timeout or a transient 403; caching it would freeze a
    passing failure into a permanent one."""
    fulltext_cache.write("arxiv:1", "", "arxiv_html", "u")
    assert fulltext_cache.read("arxiv:1") is None


def test_the_full_text_cache_shares_the_disable_switch(monkeypatch):
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", "")
    assert fulltext_cache.cache_dir() is None
    fulltext_cache.write("arxiv:1", "text")
    assert fulltext_cache.read("arxiv:1") is None


def test_the_two_caches_do_not_collide(ft_cache):
    """Resolutions and full texts live side by side under one root."""
    resolve_cache.write(REF, REC)
    fulltext_cache.write("arxiv:1", "body")
    assert resolve_cache.read(REF, Resolved) is not None
    assert fulltext_cache.read("arxiv:1") is not None
    assert resolve_cache.clear() == 1
    assert fulltext_cache.read("arxiv:1") is not None
