"""
Offline tests for the web front-end (citation_verifier.web). The heavy
``run_verification`` call is mocked, so the upload → job → SSE-progress → report
flow is exercised without any network, SDK, or real verification. Skipped
entirely when the optional ``web`` extra (FastAPI) is not installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, for form/file uploads

from fastapi.testclient import TestClient  # noqa: E402

import citation_verifier  # noqa: E402
from citation_verifier.web.app import create_app  # noqa: E402


def _fake_result():
    rec = SimpleNamespace(cite_key="ref-1", exists="yes", supports_claim="supports", severity="ok")
    # a pydantic-like .model_dump so the JSON download path is exercised
    rec.model_dump = lambda mode="json": {
        "cite_key": "ref-1", "exists": "yes", "supports_claim": "supports", "severity": "ok",
    }
    usage = SimpleNamespace(cost_usd=0.12, wall_seconds=3.4)
    return SimpleNamespace(
        paper_id="2310.06825", backend="agentic", records=[rec], usage=usage, errors=[]
    )


@pytest.fixture
def client(monkeypatch):
    # Mock the heavy pipeline: emit a couple of progress events, return a result.
    def fake_run(source, *, backend, settings, resume, progress_cb=None):
        if progress_cb:
            progress_cb({"stage": "ingesting"})
            progress_cb({"stage": "verifying", "count": 2})
        return _fake_result()

    monkeypatch.setattr(citation_verifier, "run_verification", fake_run)
    monkeypatch.setattr("citation_verifier.render.render_report", lambda r: "# Report\n\nall good")
    return TestClient(create_app())


def test_page_serves(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "RefWarden" in r.text  # product display name
    assert "/events/" in r.text  # the SSE wiring is in the page


def test_verify_requires_input(client):
    assert client.post("/verify", data={}).status_code == 400


def test_arxiv_flow_streams_progress_then_report(client):
    r = client.post("/verify", data={"arxiv": "https://arxiv.org/abs/2310.06825"})
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"]
    assert body["label"] == "2310.06825"  # arXiv id parsed from the URL

    stream = client.get(f"/events/{body['job_id']}").text
    assert '"stage": "ingesting"' in stream
    assert '"stage": "verifying"' in stream
    assert '"stage": "report"' in stream
    assert "Report" in stream  # rendered report HTML relayed to the page


def test_pdf_upload_flow(client):
    files = {"file": ("paper.pdf", b"%PDF-1.4 fake bytes", "application/pdf")}
    r = client.post("/verify", files=files)
    assert r.status_code == 200
    assert r.json()["label"] == "paper.pdf"
    stream = client.get(f"/events/{r.json()['job_id']}").text
    assert '"stage": "report"' in stream


def test_unknown_job_is_404(client):
    assert client.get("/events/nope").status_code == 404


def test_download_md_and_json_after_report(client):
    jid = client.post("/verify", data={"arxiv": "https://arxiv.org/abs/2310.06825"}).json()["job_id"]
    assert client.get(f"/events/{jid}").status_code == 200  # drain the SSE stream; worker finishes

    md = client.get(f"/download/{jid}?fmt=md")
    assert md.status_code == 200
    assert 'attachment; filename="2310.06825-report.md"' in md.headers["content-disposition"]
    assert "Report" in md.text  # the rendered markdown

    js = client.get(f"/download/{jid}?fmt=json")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("application/json")
    assert '"records"' in js.text and '"ref-1"' in js.text


def test_download_unknown_job_is_404(client):
    assert client.get("/download/nope").status_code == 404


def test_status_unknown_job_is_404(client):
    r = client.get("/status/nope")
    assert r.status_code == 404
    assert r.json() == {"known": False}


def test_refresh_reconnect_replays_progress_and_report(client):
    """The bug: refreshing mid-run lost everything. Now the event log is replayable —
    reopening /events for the same job (as a refreshed page does) replays the whole
    history (progress + report), and /status reports the job as known + done."""
    jid = client.post("/verify", data={"arxiv": "https://arxiv.org/abs/2310.06825"}).json()["job_id"]

    first = client.get(f"/events/{jid}").text          # original stream → runs the job to completion
    assert '"stage": "report"' in first

    again = client.get(f"/events/{jid}").text           # a "refreshed page" reconnects → full replay
    assert '"stage": "ingesting"' in again
    assert '"stage": "verifying"' in again
    assert '"stage": "report"' in again
    assert "id: 0" in again                              # events carry ids (Last-Event-ID resume support)

    st = client.get(f"/status/{jid}").json()
    assert st["known"] is True and st["done"] is True and st["label"] == "2310.06825"
    assert isinstance(st["elapsed"], (int, float))  # drives an accurate resumed timer


def test_last_event_id_resumes_past_seen_events(client):
    """An auto-reconnect that sends Last-Event-ID gets only newer events (no full replay)."""
    jid = client.post("/verify", data={"arxiv": "https://arxiv.org/abs/2310.06825"}).json()["job_id"]
    client.get(f"/events/{jid}")  # let the job finish so the full log exists
    # Pretend we already saw event 0 (ingesting); resume should skip it but still reach the report.
    resumed = client.get(f"/events/{jid}", headers={"Last-Event-ID": "0"}).text
    assert '"stage": "ingesting"' not in resumed
    assert '"stage": "report"' in resumed
