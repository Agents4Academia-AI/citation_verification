"""
Offline tests for the Stage-2 evidence core (`grounding/fulltext.py`): splitting a
fetched paper into sections and retrieving the few claim-relevant chunks the
relevance judge needs — no network, pure functions.
"""

from __future__ import annotations

from citation_verifier.grounding import fulltext as ft
from citation_verifier.grounding.fulltext import (
    fetch_full_text_from_url,
    fetch_full_text_via_search,
    select_evidence_chunks,
    split_sections,
)

_LATEX = r"""
\section{Introduction}
Conversational AI lets machines understand and respond to human language.
\section{Method}
We fine-tune a transformer on the WebText corpus with 1.5B parameters.
\section{Results}
On the VQA benchmark our model reaches 81.2 F1, outperforming the baseline.
\section{Conclusion}
The approach generalizes across dialogue tasks.
"""

_PLAIN = """Introduction

Conversational AI lets machines understand human language.

Experiments

We evaluate on the SQuAD dataset and report exact-match accuracy.
"""


def test_split_sections_latex_and_plain():
    secs = dict(split_sections(_LATEX))
    assert set(secs) == {"Introduction", "Method", "Results", "Conclusion"}
    assert "WebText" in secs["Method"]

    heads = [h for h, _ in split_sections(_PLAIN)]
    assert "Introduction" in heads and "Experiments" in heads


def test_split_sections_no_structure_is_single_block():
    assert split_sections("Just one blob of prose with no headings at all.") == [
        ("", "Just one blob of prose with no headings at all.")
    ]


def test_select_chunks_prefers_claim_overlap():
    sections = split_sections(_LATEX)
    # a generic (non-experimental) claim -> stays in default sections
    hits = select_evidence_chunks("Conversational AI understands human language", sections, k=2)
    assert hits and "understand" in hits[0][1].lower()


def test_select_chunks_pulls_experimental_section_only_when_claim_warrants_it():
    sections = split_sections(_LATEX)
    # claim about a metric/benchmark -> the Results section becomes in-scope and wins
    exp = select_evidence_chunks("the model achieves 81.2 F1 on the VQA benchmark", sections, k=1)
    assert exp and exp[0][0] == "Results"
    # a non-experimental claim must NOT surface the Results/Method chunks
    generic = select_evidence_chunks("conversational AI understands language", sections, k=4)
    assert all(h not in ("Results", "Method") for h, _ in generic)


# ── fetch_full_text_from_url: off-arXiv OA PDF (S2 openAccessPdf) ─────
def test_fetch_full_text_from_url_extracts_pdf(monkeypatch):
    # %PDF magic -> the bytes go to the PDF extractor and its text is returned.
    monkeypatch.setattr(ft, "_http_get_bytes", lambda url, timeout: b"%PDF-1.5 ...bytes...")
    monkeypatch.setattr(ft, "_pdf_bytes_to_text", lambda data: "FULL TEXT OF THE PAPER")
    assert fetch_full_text_from_url("https://dl.acm.org/doi/pdf/10.1145/x") == "FULL TEXT OF THE PAPER"


def test_fetch_full_text_from_url_skips_non_pdf(monkeypatch):
    # An HTML landing page / paywall interstitial must fail soft, never parsed.
    monkeypatch.setattr(ft, "_http_get_bytes", lambda url, timeout: b"<!DOCTYPE html><html>...")
    monkeypatch.setattr(
        ft, "_pdf_bytes_to_text",
        lambda data: (_ for _ in ()).throw(AssertionError("must not parse non-PDF bytes")),
    )
    assert fetch_full_text_from_url("https://example.com/article") == ""


def test_fetch_full_text_from_url_empty_url_no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("empty url must not hit the network")

    monkeypatch.setattr(ft, "_http_get_bytes", boom)
    assert fetch_full_text_from_url("") == ""
    assert fetch_full_text_from_url(None) == ""


# ── fetch_full_text_via_search: title-verified web-search fallback ────
_GPT2_TITLE = "Language Models are Unsupervised Multitask Learners"


def test_via_search_returns_verified_pdf(monkeypatch):
    # search yields a URL; the fetched text contains the cited title -> accepted.
    monkeypatch.setattr(ft, "fetch_full_text_from_url", lambda u, **k: (
        "Language Models are Unsupervised Multitask Learners. Alec Radford et al. "
        "We train on WebText..."
    ))
    out = fetch_full_text_via_search(_GPT2_TITLE, search=lambda q, **k: ["http://x/p.pdf"])
    assert out.startswith("Language Models are Unsupervised")


def test_via_search_rejects_namesake(monkeypatch):
    # The 2024 namesake says "Supervised" (lacks the cited token "unsupervised")
    # -> the title gate rejects it and we fall through to "".
    monkeypatch.setattr(ft, "fetch_full_text_from_url", lambda u, **k: (
        "Instruction Pre-Training: Language Models are Supervised Multitask Learners. "
        "We propose instruction pre-training..."
    ))
    assert fetch_full_text_via_search(_GPT2_TITLE, search=lambda q, **k: ["http://x/p.pdf"]) == ""


def test_default_search_falls_back_to_ddg_without_google_key(monkeypatch):
    # No Google key -> _google_cse_search returns [] -> DDG backend is used.
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")  # opt in to the open-web fallback
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    monkeypatch.setattr(ft, "_ddg_search", lambda q, **k: ["http://x/p.pdf"])
    monkeypatch.setattr(ft, "fetch_full_text_from_url", lambda u, **k:
                        f"{_GPT2_TITLE}. Alec Radford et al.")
    assert fetch_full_text_via_search(_GPT2_TITLE).startswith("Language Models")


def test_via_search_no_hits_no_fetch(monkeypatch):
    # Both backends dry -> never fetch, return "".
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")  # opt in; both backends still dry
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    monkeypatch.setattr(ft, "_ddg_search", lambda q, **k: [])
    monkeypatch.setattr(ft, "fetch_full_text_from_url", lambda u, **k: (_ for _ in ()).throw(
        AssertionError("must not fetch when search returns nothing")))
    assert fetch_full_text_via_search(_GPT2_TITLE) == ""


def test_default_search_gated_off_by_default(monkeypatch):
    # ENABLE_WEB_SEARCH unset -> the open-web fallback never runs (no CSE, no DDG).
    monkeypatch.delenv("ENABLE_WEB_SEARCH", raising=False)
    monkeypatch.setattr(ft, "_google_cse_search", lambda q, **k: (_ for _ in ()).throw(
        AssertionError("gated: must not hit Google CSE")))
    monkeypatch.setattr(ft, "_ddg_search", lambda q, **k: (_ for _ in ()).throw(
        AssertionError("gated: must not hit DuckDuckGo")))
    assert ft._default_web_search("anything") == []
    # and the public fetcher returns nothing without ever searching
    assert fetch_full_text_via_search(_GPT2_TITLE) == ""


def test_parse_ddg_results_unwraps_and_dedupes():
    html = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fceur-ws.org'
        '%2FVol-2563%2Faics_12.pdf&rut=abc">Conv AI</a>'
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fceur-ws.org'
        '%2FVol-2563%2Faics_12.pdf">dup</a>'  # same target -> deduped
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com'
        '%2Fpaper.pdf">Other</a>'
    )
    assert ft._parse_ddg_results(html, max_results=5) == [
        "https://ceur-ws.org/Vol-2563/aics_12.pdf",
        "https://example.com/paper.pdf",
    ]


def test_via_search_short_title_skips(monkeypatch):
    monkeypatch.setattr(ft, "_google_cse_search", lambda q, **k: (_ for _ in ()).throw(
        AssertionError("too-short title must not search")))
    assert fetch_full_text_via_search("GPT") == ""


def test_text_matches_paper_gate():
    body = "Conversational AI: Social and Ethical Considerations. They are used in..."
    assert ft._text_matches_paper(body, "Conversational AI: Social and Ethical Considerations")
    # a different paper sharing only some tokens is rejected.
    assert not ft._text_matches_paper(body, "Conversational AI for Healthcare Diagnosis")
