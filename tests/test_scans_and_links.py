"""
Full text behind a web page and inside a scan.

A record's link is often a page, not a PDF — a publisher's landing page, an institution's
list of its reports — and some cited PDFs are scans with no text layer. The full-text
layer follows the page to the work's PDF (accepting it only when it carries the title),
hands a scan to the reader the run installed, and, when nothing gives text, says what was
tried. All offline: fetches are monkeypatched and the PDFs are built in memory.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from citation_verifier.grounding import fulltext as ft
from citation_verifier.grounding import oa_fulltext as oa
from citation_verifier.schema import CitationRecord, CitedAs, Claim, Resolved, SupportsClaim

fitz = pytest.importorskip("fitz")

TITLE = "A note on importance sampling using standardized weights"
LIST_URL = "https://stat.example.edu/library/technical-reports/"
PDF_URL = "https://files.example.net/tech-rpts/tr348.pdf"


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    """Each test gets its own full-text cache (never the user's)."""
    monkeypatch.setenv("CITATION_VERIFIER_CACHE", str(tmp_path / "cache"))


def _text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(54, 54, 540, 780), text, fontsize=10)
    return doc.tobytes()


def _scan_pdf(pages: int = 2) -> bytes:
    """A PDF whose pages are images only — what a scanner produces."""
    pix = fitz.Pixmap(fitz.csGRAY, fitz.IRect(0, 0, 60, 80), False)
    pix.set_rect(pix.irect, (200,))
    png = pix.tobytes("png")
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, stream=png)
    return doc.tobytes()


def _listing(rows: str) -> bytes:
    return f"<html><head><title>Technical reports</title></head><body><table>{rows}</table></body></html>".encode()


def _serve(monkeypatch, pages: dict[str, bytes]) -> list[str]:
    fetched: list[str] = []

    def get(url, timeout):
        fetched.append(url)
        return pages.get(url, b"")

    monkeypatch.setattr(ft, "_http_get_bytes", get)
    return fetched


# ── finding the work's PDF on a page ────────────────────────────────
def test_a_list_page_links_the_title_to_its_file():
    page = _listing(
        f'<tr><td>Kong, A.</td><td><a href="{PDF_URL}?mtime=1">{TITLE}</a></td></tr>'
        '<tr><td>Other, B.</td><td><a href="/other.pdf">Another report</a></td></tr>'
    ).decode()
    assert ft._pdf_links_for(page, LIST_URL, TITLE) == [f"{PDF_URL}?mtime=1"]


def test_a_pdf_link_counts_only_beside_the_title_in_its_own_entry():
    # The cited entry has no file; the next row's PDF is a few hundred characters away.
    page = _listing(
        f"<tr><td>Kong, A.</td><td>{TITLE}. Journal, 1994.</td></tr>"
        '<tr><td>Mykland, P.</td><td><a href="https://x.org/bartlett.pdf">Bartlett identities</a></td></tr>'
    ).decode()
    assert ft._pdf_links_for(page, LIST_URL, TITLE) == []
    beside = _listing(f'<tr><td>{TITLE}.</td><td><a href="tr348.pdf">[PDF]</a></td></tr>').decode()
    assert ft._pdf_links_for(beside, LIST_URL, TITLE) == [LIST_URL + "tr348.pdf"]


def test_the_works_own_page_declares_its_pdf_and_relative_links_use_its_address():
    page = (
        f'<html><head><title>{TITLE} | Publisher</title>'
        '<link rel="canonical" href="https://publisher.example.com/chapter/42"/>'
        '<meta name="citation_pdf_url" content="https://publisher.example.com/pdf/42.pdf"/></head>'
        '<body><h1>Chapter</h1><a href="download/42.pdf">Download PDF</a></body></html>'
    )
    links = ft._pdf_links_for(page, "https://doi.org/10.1/abc", TITLE)
    assert links == [
        "https://publisher.example.com/pdf/42.pdf",  # the declared PDF first
        "https://publisher.example.com/chapter/download/42.pdf",  # relative to the page itself
    ]


def test_a_page_is_followed_to_the_pdf_that_carries_the_title(monkeypatch):
    _serve(monkeypatch, {
        LIST_URL: _listing(f'<tr><td><a href="{PDF_URL}">{TITLE}</a></td></tr>'),
        PDF_URL: _text_pdf(f"{TITLE}\nAugustine Kong\nThe rule of thumb is to consider m/(1+Var) " * 3),
    })
    report: dict = {}
    text = ft.fetch_full_text_from_url(LIST_URL, title=TITLE, report=report)
    assert "rule of thumb" in text
    assert report == {"url": PDF_URL, "source": "linked_pdf"}


def test_a_linked_pdf_of_another_work_is_not_taken(monkeypatch):
    _serve(monkeypatch, {
        LIST_URL: _listing(f'<tr><td><a href="{PDF_URL}">{TITLE}</a></td></tr>'),
        PDF_URL: _text_pdf("Bartlett identities for martingales. " * 10),
    })
    report: dict = {}
    assert ft.fetch_full_text_from_url(LIST_URL, title=TITLE, report=report) == ""
    assert "another work" in report["note"]


def test_one_list_page_gives_each_work_its_own_pdf(monkeypatch):
    # The page is cached per work: a list links many.
    other = "Sequential imputations and Bayesian missing data problems"
    _serve(monkeypatch, {
        LIST_URL: _listing(
            f'<tr><td><a href="{PDF_URL}">{TITLE}</a></td></tr>'
            f'<tr><td><a href="https://files.example.net/415.pdf">{other}</a></td></tr>'
        ),
        PDF_URL: _text_pdf(f"{TITLE}. First report. " * 5),
        "https://files.example.net/415.pdf": _text_pdf(f"{other}. Second report. " * 5),
    })
    assert "First report" in ft.fetch_full_text_from_url(LIST_URL, title=TITLE)
    assert "Second report" in ft.fetch_full_text_from_url(LIST_URL, title=other)


def test_a_page_without_a_title_to_look_for_is_not_followed(monkeypatch):
    fetched = _serve(monkeypatch, {LIST_URL: _listing(f'<tr><td><a href="{PDF_URL}">{TITLE}</a></td></tr>')})
    report: dict = {}
    assert ft.fetch_full_text_from_url(LIST_URL, report=report) == ""
    assert fetched == [LIST_URL] and report["note"].endswith("a web page, not a PDF")


# ── scans ───────────────────────────────────────────────────────────
def test_a_scan_is_told_apart_from_a_text_pdf():
    scan = _scan_pdf()
    assert ft._looks_scanned(scan, ft._pdf_bytes_to_text(scan))
    text = _text_pdf("Plenty of words on this page. " * 20)
    assert not ft._looks_scanned(text, ft._pdf_bytes_to_text(text))


def test_a_scan_with_no_reader_says_so(monkeypatch):
    _serve(monkeypatch, {PDF_URL: _scan_pdf()})
    report: dict = {}
    assert ft.fetch_full_text_from_url(PDF_URL, report=report) == ""
    assert "scan with no text layer" in report["note"]


def test_a_scan_is_read_by_the_installed_reader_and_labelled(monkeypatch):
    _serve(monkeypatch, {
        LIST_URL: _listing(f'<tr><td><a href="{PDF_URL}">{TITLE}</a></td></tr>'),
        PDF_URL: _scan_pdf(),
    })
    seen: list[bytes] = []

    def reader(data, title=""):
        seen.append(data)
        return f"[page 1]\n{TITLE}\nAugustine Kong\n[page 2]\nthe rule of thumb"

    report: dict = {}
    with ft.reading_scans(reader):
        text = ft.fetch_full_text_from_url(LIST_URL, title=TITLE, report=report)
    assert "rule of thumb" in text and len(seen) == 1
    assert report["source"] == "linked_pdf" + ft.SCAN_LABEL
    assert ft._SCAN_READER is None  # the block put things back
    # Read once: the transcription is cached, so a later run without a reader has it.
    assert "rule of thumb" in ft.fetch_full_text_from_url(LIST_URL, title=TITLE)


def test_a_reader_that_fails_leaves_a_reason(monkeypatch):
    _serve(monkeypatch, {PDF_URL: _scan_pdf()})

    def reader(data, title=""):
        raise RuntimeError("model said no")

    report: dict = {}
    with ft.reading_scans(reader):
        assert ft.fetch_full_text_from_url(PDF_URL, report=report) == ""
    assert "reading it failed (model said no)" in report["note"]


def test_reading_scans_with_none_keeps_the_outer_reader():
    outer = object()
    with ft.reading_scans(outer), ft.reading_scans(None):
        assert ft._SCAN_READER is outer
    assert ft._SCAN_READER is None


# ── what was tried ──────────────────────────────────────────────────
def test_with_no_text_the_result_says_what_was_tried(monkeypatch):
    _serve(monkeypatch, {"https://doi.org/10.1017/x.008": b"<html><title>Chapter</title><body>Buy</body></html>"})
    monkeypatch.setattr(oa, "_oa_pdf_sources", lambda doi, timeout: [])
    got = ft.full_text_with_trail(
        Resolved(source="crossref", title="On the method of bounded differences",
                 doi="10.1017/x.008", url="https://doi.org/10.1017/x.008")
    )
    assert got.text == ""
    assert got.tried == (
        "doi.org: a web page that links no PDF of the work",
        "DOI 10.1017/x.008: no open-access copy listed (OpenAlex, Unpaywall, publisher page)",
    )
    assert ft.full_text_with_trail(Resolved(source="web")).tried == ("no arXiv id, link or DOI to fetch it by",)


def test_the_doi_copy_that_is_a_landing_page_is_followed(monkeypatch):
    landing = "https://repo.example.org/item/7"
    _serve(monkeypatch, {
        landing: f'<html><head><title>{TITLE}</title></head><a href="/bitstream/7.pdf">PDF</a></html>'.encode(),
        "https://repo.example.org/bitstream/7.pdf": _text_pdf(f"{TITLE}. Body text. " * 5),
    })
    monkeypatch.setattr(oa, "_oa_pdf_sources", lambda doi, timeout: [(landing, "unpaywall")])
    got = oa.fulltext_by_doi_with_source("10.1/abc", title=TITLE)
    assert "Body text" in got.text
    assert (got.source, got.url) == ("unpaywall", "https://repo.example.org/bitstream/7.pdf")


# ── the judge ───────────────────────────────────────────────────────
def _judge():
    from citation_verifier.backends.relevance_judge import LLMRelevanceJudge

    return LLMRelevanceJudge(settings={"escalate_full_text": False}, mode="messages")


def test_a_text_with_no_introduction_is_judged_from_its_opening(monkeypatch):
    judge = _judge()
    body = "We consider importance sampling with standardized weights. " * 10
    monkeypatch.setattr(ft, "full_text_with_trail", lambda r: ft.FullTextResult(body, "linked_pdf" + ft.SCAN_LABEL, PDF_URL))
    seen = []

    def chunk(rows):
        seen.extend(grp["evidence"] for _ck, grp in rows)
        return {ck: {cid: _verdict() for _i, cid, _c in grp["members"]} for ck, grp in rows}

    monkeypatch.setattr(judge, "_judge_citation_chunk", chunk)
    out = judge.judge_batch([{"cite_key": "kong", "claim_id": "c1", "claim": "ESS rule of thumb",
                              "abstract": "", "resolved": Resolved(source="web", title=TITLE, url=LIST_URL)}])
    assert seen and seen[0].startswith("[Opening of the text]\nWe consider importance sampling")
    assert out[0].basis == "the opening of the cited work's text" + ft.SCAN_LABEL


def test_with_nothing_to_read_the_judge_says_what_was_tried(monkeypatch):
    judge = _judge()
    tried = ("doi.org: a web page that links no PDF of the work", "DOI 10.1017/x: no open-access copy listed")
    monkeypatch.setattr(ft, "full_text_with_trail", lambda r: ft.FullTextResult("", tried=tried))
    monkeypatch.setattr(judge, "_judge_citation_chunk", lambda rows: pytest.fail("nothing to judge"))
    out = judge.judge_batch([{"cite_key": "mcd", "claim_id": "c1", "claim": "a bound", "abstract": "",
                              "resolved": Resolved(source="crossref", title="On bounded differences", doi="10.1017/x")}])
    assert out[0].supports_claim == SupportsClaim.INCONCLUSIVE
    assert out[0].justification == (
        "no text of the cited work to judge from: the record has no abstract; full text not found — "
        "doi.org: a web page that links no PDF of the work; DOI 10.1017/x: no open-access copy listed"
    )


def _verdict():
    from citation_verifier.stages.relevance import RelevanceVerdict

    return RelevanceVerdict(SupportsClaim.SUPPORTS, justification="ok")


def test_a_full_text_verdict_on_a_scan_says_so(monkeypatch):
    from citation_verifier.stages.relevance import RelevanceVerdict, fill_relevance_batch

    body = f"{TITLE}. The rule of thumb is to consider m/(1+Var) as the effective sample size."
    monkeypatch.setattr(ft, "full_text_with_trail", lambda r: ft.FullTextResult(body, "linked_pdf" + ft.SCAN_LABEL, PDF_URL))

    def judge_batch(items):
        return [
            RelevanceVerdict(SupportsClaim.SUPPORTS, justification="stated")
            if "complete text" in it["abstract"]
            else RelevanceVerdict(SupportsClaim.INCONCLUSIVE, justification="no abstract")
            for it in items
        ]

    rec = CitationRecord(
        paper_id="p", claim_id="c1", cite_key="kong", claim=Claim(claim_id="c1", text="ESS rule of thumb"),
        cited_as=CitedAs(raw="Kong 1992"), resolved=Resolved(source="web", title=TITLE, url=LIST_URL),
    )
    fill_relevance_batch([rec], resolver=None, judge_batch=judge_batch)
    assert rec.supports_claim == SupportsClaim.SUPPORTS
    assert "(based on full text" in rec.notes and ft.SCAN_LABEL in rec.notes


# ── the reader ──────────────────────────────────────────────────────
def test_the_reader_sends_page_images_and_notes_unread_pages(monkeypatch):
    from citation_verifier.backends import scan_reader as sr

    requests = []

    class Messages:
        def create(self, **kw):
            requests.append(kw)
            n = sum(1 for b in kw["messages"][0]["content"] if b["type"] == "image")
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="text", text=f"[{n} pages read]")],
                usage=types.SimpleNamespace(input_tokens=1000, output_tokens=200,
                                            cache_creation_input_tokens=0, cache_read_input_tokens=0),
            )

    reader = sr.ScanReader({"anthropic_api_key": "k", "scan_max_pages": 7}, model="claude-haiku-4-5")
    reader._client = types.SimpleNamespace(messages=Messages())
    text = reader(_scan_pdf(9), title=TITLE)
    assert [sum(b["type"] == "image" for b in r["messages"][0]["content"]) for r in requests] == [5, 2]
    assert all(TITLE not in str(r) for r in requests)  # the model is not told what it reads
    assert text == "[5 pages read]\n\n[2 pages read]\n\n[pages 8–9 of the scan were not read]"
    assert reader.drain_usage().input_tokens == 2000


def test_the_subscription_reader_hands_the_images_to_a_session_without_tools(monkeypatch):
    from citation_verifier.backends import scan_reader as sr

    sent, opts = [], []

    class Options:
        def __init__(self, **kw):
            opts.append(kw)

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, text):
            self.content, self.error = [TextBlock(text)], None

    class ResultMessage:
        is_error, total_cost_usd, usage, num_turns = False, 0.01, {}, 1

    async def query(prompt, options):
        async for message in prompt:
            sent.append(message)
        yield AssistantMessage("[page 1]\ntext")
        yield ResultMessage()

    fake = types.SimpleNamespace(query=query, ClaudeAgentOptions=Options, AssistantMessage=AssistantMessage,
                                 TextBlock=TextBlock, ResultMessage=ResultMessage)
    reader = sr.ScanReader({}, model="claude-haiku-4-5")
    out = asyncio.run(reader._run_query(fake, [{"type": "image", "source": {}}, {"type": "text", "text": "go"}]))
    assert out == "[page 1]\ntext"
    assert sent[0]["message"]["content"][0]["type"] == "image"
    assert opts[0]["tools"] == [] and opts[0]["setting_sources"] == []
    assert opts[0]["thinking"] == {"type": "disabled"}  # copying text needs no thinking


def test_a_reader_whose_model_is_unreachable_stops_asking(monkeypatch):
    from citation_verifier.backends import scan_reader as sr
    from citation_verifier.errors import ModelUnavailable

    reader = sr.ScanReader({"anthropic_api_key": "k"})

    class AuthenticationError(Exception):
        pass

    calls = []

    def create(**kw):
        calls.append(1)
        raise AuthenticationError("bad key")

    reader._client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    for _ in range(2):
        with pytest.raises(ModelUnavailable):
            reader(_scan_pdf(1))
    assert calls == [1]


# ── wiring ──────────────────────────────────────────────────────────
def test_the_agentic_run_installs_its_reader_and_counts_what_it_cost(monkeypatch):
    from citation_verifier.backends.agentic import AgenticBackend
    from citation_verifier.interfaces import PaperSource, RunUsage

    seen = []

    class Reader:
        unavailable = ""

        def __call__(self, data, title=""):
            return ""

        def drain_usage(self):
            return RunUsage(backend="agentic", model="m", input_tokens=900, cost_usd=0.03)

    class Judge:
        usage = None

        def judge_batch(self, items):
            seen.append(ft._SCAN_READER)
            return [_verdict() for _ in items]

    reader = Reader()
    backend = AgenticBackend(settings={"relevance_judge": Judge(), "scan_reader": reader})

    def correctness(rec, resolver=None, **kw):
        rec.exists = "yes"
        rec.resolved = Resolved(source="crossref", title="T", abstract="An abstract about the work. " * 12)

    monkeypatch.setattr(backend, "_load_pipeline", lambda: (correctness, None, None))
    stub = CitationRecord(paper_id="p", claim_id="c1", cite_key="k", claim=Claim(claim_id="c1", text="a claim"),
                          cited_as=CitedAs(raw="A. Author. T. 2020."))
    result = backend.verify(PaperSource(paper_id="p"), [stub])
    assert seen and seen[0] is reader  # the judge ran with the reader installed
    assert ft._SCAN_READER is None  # and only during the run
    assert result.usage.by_tier["bulk"].cost_usd == pytest.approx(0.03, abs=0.001)  # + the lookups' estimate


def test_a_run_without_a_model_reads_no_scans():
    from citation_verifier.backends.agentic import AgenticBackend
    from citation_verifier.tables.pipeline import _build_scan_reader

    assert AgenticBackend(settings={}).scan_reader is None
    assert _build_scan_reader({"read_scans": False, "scan_reader": object()}) is None
    injected = object()
    assert _build_scan_reader({"scan_reader": injected}) is injected


# ── a publisher's preview is not the work ───────────────────────────
CHAPTER = "From external to internal regret"
CHAPTER_PAGE = "https://link.example.com/chapter/10.1007/11503415_42"
FULL_PDF = "https://link.example.com/content/pdf/10.1007/11503415_42.pdf"
PREVIEW_PDF = "https://page-one.example.com/pdf/preview/10.1007/11503415_42"


def _chapter_page() -> bytes:
    return (
        f'<html><head><title>{CHAPTER} | SpringerLink</title>'
        f'<meta name="citation_pdf_url" content="{FULL_PDF}"/></head><body>'
        f'<a href="{PREVIEW_PDF}">Preview</a> <a href="{FULL_PDF}">Download PDF</a></body></html>'
    ).encode()


def test_a_preview_link_is_tried_last_and_says_what_it_is():
    links = ft._pdf_link_candidates(_chapter_page().decode(), CHAPTER_PAGE, CHAPTER)
    assert links == [(FULL_PDF, False), (PREVIEW_PDF, True)]


def test_a_preview_is_read_but_labelled_as_the_first_pages(monkeypatch):
    _serve(monkeypatch, {
        CHAPTER_PAGE: _chapter_page(),
        FULL_PDF: b"<html><title>Log in</title></html>",  # the paywall
        PREVIEW_PDF: _text_pdf(f"{CHAPTER}. Avrim Blum and Yishay Mansour. Abstract. We show a reduction. " * 3),
    })
    report: dict = {}
    assert "reduction" in ft.fetch_full_text_from_url(CHAPTER_PAGE, title=CHAPTER, report=report)
    assert report["source"] == "linked_pdf" + ft.PREVIEW_LABEL


def test_the_rejudge_does_not_call_a_preview_the_complete_text(monkeypatch):
    from citation_verifier.stages.relevance import RelevanceVerdict, fill_relevance_batch

    monkeypatch.setattr(ft, "full_text_with_trail", lambda r: ft.FullTextResult(
        f"{CHAPTER}. We show a reduction.", "linked_pdf" + ft.PREVIEW_LABEL, PREVIEW_PDF))
    seen = []

    def judge_batch(items):
        seen.extend(it["abstract"] for it in items)
        return [RelevanceVerdict(SupportsClaim.PARTIAL, justification="p") for _ in items]

    rec = CitationRecord(
        paper_id="p", claim_id="c1", cite_key="blum", claim=Claim(claim_id="c1", text="a swap-regret bound"),
        cited_as=CitedAs(raw="Blum 2005"),
        resolved=Resolved(source="crossref", title=CHAPTER, url=CHAPTER_PAGE, abstract="A reduction. " * 30),
    )
    fill_relevance_batch([rec], resolver=None, judge_batch=judge_batch)
    rejudge = seen[1]
    assert rejudge.startswith("[the first pages of the cited work only") and "complete text" not in rejudge
    assert "(based on the cited work's first pages" in rec.notes and ft.PREVIEW_LABEL in rec.notes


def test_a_table_cell_is_not_judged_on_a_preview(monkeypatch):
    from citation_verifier.tables.evidence import _full_text_of

    monkeypatch.setattr(ft, "full_text_with_trail", lambda r: ft.FullTextResult(
        "first page", "linked_pdf" + ft.PREVIEW_LABEL, PREVIEW_PDF))
    assert _full_text_of(Resolved(source="crossref", title=CHAPTER)) == ""


# ── the reason reads once, not once per page ────────────────────────
def test_a_chain_of_pages_is_told_by_its_first_page_and_its_cause(monkeypatch):
    doi = "https://doi.org/10.1017/x.008"
    chapter = "https://www.cambridge.example.org/core/books/surveys/chapter"
    pdf = "https://www.cambridge.example.org/core/services/content/x.pdf"
    _serve(monkeypatch, {
        doi: (f'<html><head><title>On the method of bounded differences</title>'
              f'<link rel="canonical" href="{chapter}"/>'
              f'<meta name="citation_pdf_url" content="{pdf}"/></head></html>').encode(),
        pdf: b"<html><title>Log in</title><body>Access through your institution</body></html>",
    })
    report: dict = {}
    assert ft.fetch_full_text_from_url(doi, title="On the method of bounded differences", report=report) == ""
    assert report["note"] == (
        "cambridge.example.org links the work's PDF, but the link returns a web page, not the PDF"
    )


def test_a_list_page_reports_the_innermost_cause(monkeypatch):
    detail = "https://reports.example.edu/tr348"
    _serve(monkeypatch, {
        LIST_URL: _listing(f'<tr><td><a href="{detail}">{TITLE}</a></td></tr>'),
        detail: f'<html><head><title>{TITLE}</title></head><a href="https://cdn.example.net/tr348.pdf">PDF</a></html>'.encode(),
        "https://cdn.example.net/tr348.pdf": _scan_pdf(),
    })
    report: dict = {}
    assert ft.fetch_full_text_from_url(LIST_URL, title=TITLE, report=report) == ""
    assert report["note"] == (
        "stat.example.edu links the work's PDF, but cdn.example.net: the PDF is a scan with no text "
        "layer, and no model in this run reads scans"
    )
