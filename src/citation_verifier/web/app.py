"""
web/app.py — FastAPI front-end: upload a PDF or paste an arXiv link, watch a
progress bar, read the rendered report in the browser.

The whole verification is the public ``run_verification`` call, run in a worker
thread (it is blocking and takes minutes). Its ``progress_cb`` events are appended
to a per-job, append-only event log and relayed to the browser as Server-Sent
Events, so the page shows real stages ("found N citations", "verifying …") plus a
live timer. Because the log is replayable (not a consume-once queue) and the browser
remembers its ``job_id``, a page refreshed mid-run reconnects and rebuilds its UI
from the full history instead of losing the run. Nothing here re-implements
verification — it only sequences the existing seams.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from ..config import apply_auth, load_settings
from ..ingest import parse_arxiv_id

# ── job registry ─────────────────────────────────────────────────────────────


@dataclass
class _Job:
    job_id: str
    label: str  # what's being verified (arXiv id or filename), for the UI
    started: float = field(default_factory=time.monotonic)
    events: list[dict] = field(default_factory=list)  # append-only SSE log, replayable on reconnect
    done: bool = False
    report_md: str | None = None    # rendered Markdown report, kept for download
    report_json: str | None = None  # canonical report.json text, kept for download


_JOBS: dict[str, _Job] = {}
_JOBS_CAP = 64  # bound memory; oldest jobs are pruned (their reports stop being downloadable)


def _emit(job: _Job, event: dict) -> None:
    """Append an event to the job's replayable log (the SSE stream reads from it).

    Unlike a consume-once queue, the log lets a browser that refreshed mid-run
    reopen ``/events/{job_id}`` and replay the whole history — progress events and
    the final report — to rebuild its UI from scratch.
    """
    job.events.append(event)


def _records_json(result: Any) -> str | None:
    """Serialize the canonical ``report.json`` (matches ``orchestrator._write_report``).

    Best-effort: returns ``None`` if the records can't be dumped (e.g. a test stub
    without ``model_dump``), so the Markdown download still works.
    """
    try:
        payload = {
            "paper_id": getattr(result, "paper_id", None),
            "backend": getattr(result, "backend", None),
            "records": [r.model_dump(mode="json") for r in result.records],
            "errors": list(getattr(result, "errors", []) or []),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001 — download is best-effort; Markdown still available
        return None


def _safe_name(label: str) -> str:
    """A filesystem-safe stem for the download filename (from the job label)."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", label or "report").strip("-._")
    return stem or "report"


# ── rendering helpers ────────────────────────────────────────────────────────
def _md_to_html(md: str) -> str:
    """Markdown → HTML. Uses the optional ``markdown`` lib; falls back to <pre>."""
    try:
        import markdown  # part of the 'web' extra

        return markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    except Exception:
        import html as _html

        return f"<pre class='raw-md'>{_html.escape(md)}</pre>"


def _summary(records: list) -> dict[str, Any]:
    """Small at-a-glance tally for the result header (per the schema enums)."""
    def val(x: Any) -> str:
        return getattr(x, "value", x)

    exists_by_ref: dict[str, str] = {}
    for r in records:
        exists_by_ref.setdefault(r.cite_key, val(r.exists))
    ex = Counter(exists_by_ref.values())
    sc = Counter(val(r.supports_claim) for r in records)
    high = sum(1 for r in records if val(r.severity) == "high")
    return {
        "refs": len(exists_by_ref),
        "pairs": len(records),
        "exists": dict(ex),
        "supports": dict(sc),
        "high": high,
    }


# ── the worker: one full verification, emitting progress onto the job's event log ──
def _run_job(job: _Job, source: str, settings: Any) -> None:
    from .. import run_verification  # lazy: keeps web import light
    from ..render import render_report

    try:
        result = run_verification(
            source,
            backend="agentic",
            settings=settings,
            resume=False,
            progress_cb=lambda ev: _emit(job, ev),
        )
        md = render_report(result)
        # Keep the report artifacts so the page can offer file downloads.
        job.report_md = md
        job.report_json = _records_json(result)
        _emit(
            job,
            {
                "stage": "report",
                "html": _md_to_html(md),
                "summary": _summary(result.records),
                "cost_usd": round(result.usage.cost_usd, 4),
                "wall_s": round(result.usage.wall_seconds, 1),
                "errors": result.errors,
                "downloads": {"md": True, "json": job.report_json is not None},
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure to the page
        _emit(job, {"stage": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        job.done = True
        _emit(job, {"stage": "__end__"})


# ── app factory ──────────────────────────────────────────────────────────────
def create_app(settings: Any | None = None):
    """Build the FastAPI app. ``settings`` defaults to ``load_settings()``."""
    settings = settings or load_settings()
    auth = apply_auth(settings)  # subscription unless ANTHROPIC_API_KEY is set
    uploads = Path(settings.papers_dir) / "_uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="RefWarden")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML.replace("{{AUTH}}", auth)

    @app.post("/verify")
    async def verify(
        arxiv: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
    ):
        # Resolve the input to a `source` string run_verification understands.
        if file is not None and file.filename:
            data = await file.read()
            if not data:
                return JSONResponse({"error": "uploaded file is empty"}, status_code=400)
            safe = Path(file.filename).name or "upload.pdf"
            dest = uploads / f"{uuid.uuid4().hex[:8]}_{safe}"
            dest.write_bytes(data)
            source, label = str(dest), safe
        elif arxiv and arxiv.strip():
            raw = arxiv.strip()
            source = raw
            label = parse_arxiv_id(raw) or raw
        else:
            return JSONResponse({"error": "provide an arXiv link or a PDF file"}, status_code=400)

        job = _Job(job_id=uuid.uuid4().hex[:12], label=label)
        _JOBS[job.job_id] = job
        # Bound memory: keep only the most recent jobs (dict preserves insertion order).
        while len(_JOBS) > _JOBS_CAP:
            _JOBS.pop(next(iter(_JOBS)), None)
        threading.Thread(target=_run_job, args=(job, source, settings), daemon=True).start()
        return {"job_id": job.job_id, "label": label}

    @app.get("/download/{job_id}")
    def download(job_id: str, fmt: str = "md"):
        """Download a finished job's report as a file (``fmt=md`` | ``fmt=json``)."""
        job = _JOBS.get(job_id)
        if job is None or job.report_md is None:
            return JSONResponse({"error": "no result available for this job"}, status_code=404)
        if fmt.lower() == "json":
            if job.report_json is None:
                return JSONResponse({"error": "JSON report unavailable"}, status_code=404)
            body, media, ext = job.report_json, "application/json; charset=utf-8", "json"
        else:
            body, media, ext = job.report_md, "text/markdown; charset=utf-8", "md"
        filename = f"{_safe_name(job.label)}-report.{ext}"
        return Response(
            content=body,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/status/{job_id}")
    def status(job_id: str):
        """Lightweight existence/done check so a refreshed page can decide whether to
        reconnect (the page then reopens ``/events`` to replay progress + report)."""
        job = _JOBS.get(job_id)
        if job is None:
            return JSONResponse({"known": False}, status_code=404)
        return {
            "known": True,
            "done": job.done,
            "label": job.label,
            "elapsed": round(time.monotonic() - job.started, 1),  # so a resumed timer continues, not restarts
        }

    @app.get("/events/{job_id}")
    async def events(job_id: str, last_event_id: str | None = Header(default=None)):
        job = _JOBS.get(job_id)
        if job is None:
            # Unknown job — e.g. the server restarted and dropped its in-memory jobs, and
            # an old tab's EventSource (or a resumed page) reconnects to a now-gone id.
            # Return a *valid* SSE stream that emits one `gone` event and ends, NOT a 404:
            # a 404 makes EventSource treat the stream as failed and reconnect — some
            # browsers (Safari/macOS) reconnect aggressively, hammering the endpoint and
            # freezing the tab. A clean `gone` lets the page clear its saved id and reset.
            async def _gone():
                yield "event: gone\ndata: {}\n\n"

            return StreamingResponse(_gone(), media_type="text/event-stream")
        # Replay from the start by default; on an EventSource auto-reconnect the browser
        # sends Last-Event-ID, so we resume just past what it already received.
        try:
            start_i = int(last_event_id) + 1 if last_event_id is not None else 0
        except ValueError:
            start_i = 0

        async def stream():
            i = max(0, start_i)
            last = time.monotonic()
            while True:
                if i < len(job.events):
                    ev = job.events[i]
                    if ev.get("stage") == "__end__":
                        break
                    yield f"id: {i}\ndata: {json.dumps(ev)}\n\n"
                    i += 1
                    last = time.monotonic()
                    continue
                if job.done:
                    break
                # keep proxies/connections alive during the long verify phase
                if time.monotonic() - last > 12:
                    last = time.monotonic()
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    """Run the dev server: ``cverify-web`` (host/port via CVERIFY_WEB_HOST/PORT)."""
    import os

    import uvicorn

    host = os.environ.get("CVERIFY_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("CVERIFY_WEB_PORT", "8000"))
    print(f"cverify-web → http://{host}:{port}  (auth: {apply_auth(load_settings())})")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


# ── the single-page UI (no external assets; fully offline) ───────────────────
_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RefWarden</title>
<style>
  :root {  /* light (default) */
    --bg:#f5f6f8; --card:#ffffff; --line:#e4e7ec; --line2:#d3d8e0; --fg:#111827;
    --mut:#6b7280; --soft:#f3f4f6; --field:#ffffff; --zebra:#fafbfc;
    --accent-soft:#eef4ff; --ring:rgba(37,99,235,.18);
    --acc:#2563eb; --acc-h:#1d4ed8; --ok:#059669; --warn:#b45309; --bad:#dc2626; }
  :root[data-theme="dark"] {  /* dark (softer than pure black) */
    --bg:#0f1318; --card:#181d25; --line:#272e38; --line2:#39414d; --fg:#e7e9ee;
    --mut:#9aa4b2; --soft:#1e242d; --field:#10151c; --zebra:#161c24;
    --accent-soft:#172234; --ring:rgba(91,156,255,.28);
    --acc:#5b9cff; --acc-h:#7db3ff; --ok:#3ddc97; --warn:#e2b340; --bad:#f0616d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:16px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:880px; margin:0 auto; padding:44px 20px 80px; position:relative; }
  .theme-toggle { position:absolute; top:20px; right:20px; height:36px; padding:0 14px;
    display:flex; align-items:center; gap:7px; border:1px solid var(--line2); background:var(--card);
    color:var(--mut); border-radius:99px; font-size:13.5px; cursor:pointer; transition:.15s; }
  .theme-toggle:hover { border-color:var(--acc); color:var(--fg); }
  .title { font-size:44px; font-weight:700; text-align:center; letter-spacing:-.025em; margin:4px 0 12px;
           background:linear-gradient(100deg,#0ea5e9,#2563eb 45%,#7c3aed);
           -webkit-background-clip:text; background-clip:text;
           color:transparent; -webkit-text-fill-color:transparent; }
  .sub { color:var(--mut); text-align:center; margin:0 auto 30px; font-size:15.5px; max-width:62ch; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:24px; box-shadow:0 1px 3px rgba(16,24,40,.05); }
  .row { display:flex; gap:10px; align-items:stretch; }
  input[type=text] { flex:1; min-width:0; background:var(--field); border:1px solid var(--line2); color:var(--fg);
                     border-radius:10px; padding:0 15px; height:48px; font-size:15px; }
  input[type=text]::placeholder { color:var(--mut); opacity:.8; }
  input[type=text]:focus { outline:none; border-color:var(--acc); box-shadow:0 0 0 3px var(--ring); }
  button { background:var(--acc); color:#fff; border:0; border-radius:10px; padding:0 24px; height:48px;
           font-weight:600; cursor:pointer; font-size:15px; white-space:nowrap; transition:background .15s; }
  button:hover:not(:disabled) { background:var(--acc-h); }
  button:disabled { opacity:.55; cursor:default; }
  .or { color:var(--mut); text-align:center; font-size:12px; margin:15px 0; letter-spacing:.12em; opacity:.8; }
  .drop { display:block; width:100%; border:1.5px dashed var(--line2); border-radius:11px;
          padding:30px; text-align:center; color:var(--mut); font-size:15px; cursor:pointer;
          background:var(--soft); transition:.15s; }
  .drop:hover, .drop.hot { border-color:var(--acc); color:var(--fg); background:var(--accent-soft); }
  .hidden { display:none; }
  /* progress */
  #prog { margin-top:24px; }
  .bar { height:9px; background:var(--soft); border-radius:99px; overflow:hidden; }
  .fill { height:100%; width:0; background:var(--acc); border-radius:99px; transition:width .4s ease; }
  .meta { display:flex; justify-content:space-between; gap:12px; margin-top:10px; font-size:14px; color:var(--mut); }
  .stage { color:var(--fg); font-weight:600; }
  /* result */
  #out { margin-top:28px; }
  /* Results break OUT of the 880px form column to use the page width for the wide
     report table — centered on the viewport, capped so it stays readable. The form
     and intro keep their comfortable 880px measure. */
  #out:not(:empty) { width:min(95vw,1400px); margin-left:calc(50% - min(47.5vw,700px)); }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
  .chip { background:var(--soft); border:1px solid var(--line); border-radius:99px;
          padding:6px 13px; font-size:13px; color:var(--mut); }
  .chip b { color:var(--fg); font-weight:600; }
  .dl { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
  .dlbtn { display:inline-flex; align-items:center; gap:6px; height:36px; padding:0 14px;
           border:1px solid var(--line2); background:var(--card); color:var(--fg);
           border-radius:8px; font-size:13.5px; font-weight:600; text-decoration:none;
           cursor:pointer; transition:.15s; }
  .dlbtn:hover { border-color:var(--acc); color:var(--acc); }
  .report { background:var(--card); border:1px solid var(--line); border-radius:14px;
            padding:4px 24px 22px; overflow-x:auto; box-shadow:0 1px 3px rgba(16,24,40,.05); }
  .report table { border-collapse:collapse; width:100%; font-size:13px; margin:12px 0; }
  .report th, .report td { border:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
  .report th { background:var(--soft); position:sticky; top:0; font-weight:600; }
  .report tr:nth-child(even) td { background:var(--zebra); }
  .report h1 { font-size:21px; font-weight:600; }
  .report h2 { font-size:16px; border-top:1px solid var(--line); padding-top:16px; margin-top:18px; }
  .report code { background:var(--soft); padding:1px 6px; border-radius:5px; font-size:12.5px; }
  .report a { color:var(--acc); text-decoration:none; }
  .report a:hover { text-decoration:underline; }
  .err { color:var(--bad); }
  .raw-md { white-space:pre-wrap; }
</style>
<script>
  // Set the theme before first paint (no flash): saved choice, else system preference.
  (function () {
    try {
      var t = localStorage.getItem("cv-theme");
      if (t !== "light" && t !== "dark") t = "light";  // default light; the toggle persists a choice
      document.documentElement.setAttribute("data-theme", t);
    } catch (e) {}
  })();
</script>
</head>
<body>
<div class="wrap">
  <button id="theme" class="theme-toggle" aria-label="Toggle light/dark theme" title="Toggle light/dark"></button>
  <h1 class="title">RefWarden</h1>
  <p class="sub">Upload a PDF or a LaTeX source archive (.zip / .tar.gz), or paste an arXiv link / PDF URL — checks every reference exists, has correct metadata, and actually supports the claim it's attached to. <span style="opacity:.7">auth: {{AUTH}}</span></p>

  <div class="card" id="form">
    <div class="row">
      <input id="arxiv" type="text" placeholder="arXiv link/id, or a PDF URL  —  e.g. https://arxiv.org/abs/2504.13837" />
      <button id="go">Verify</button>
    </div>
    <div class="or">— OR —</div>
    <label class="drop" id="drop">
      <input id="file" type="file" accept="application/pdf,.pdf,.zip,.gz,.tgz,.tar" class="hidden"/>
      <span id="droptext">Drop a PDF or a LaTeX source archive (.zip / .tar.gz) here, or click to choose a file</span>
    </label>
  </div>

  <div id="prog" class="hidden">
    <div class="bar"><div class="fill" id="fill"></div></div>
    <div class="meta"><span class="stage" id="stage">Starting…</span><span id="time">0:00</span></div>
  </div>

  <div id="out"></div>
</div>

<script>
const $ = s => document.querySelector(s);
let es, t0, timer, etaS = 0, picked = null, curJob = null;

// Theme toggle (top-right): label shows the mode you'll switch TO; choice persists.
const root = document.documentElement;
function paintTheme(){ $('#theme').textContent = root.getAttribute('data-theme') === 'dark' ? '☀ Light' : '☾ Dark'; }
$('#theme').addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  root.setAttribute('data-theme', next);
  try { localStorage.setItem('cv-theme', next); } catch (e) {}
  paintTheme();
});
paintTheme();

const drop = $('#drop'), fileIn = $('#file');
// NOTE: #drop is a <label> wrapping #file, so a click already opens the picker
// natively — do NOT also call fileIn.click() here (that double-fires and the
// dialog reopens after you pick a file).
fileIn.addEventListener('change', () => { picked = fileIn.files[0] || null;
  $('#droptext').textContent = picked ? ('📄 ' + picked.name) : 'Drop a PDF or a LaTeX source archive (.zip / .tar.gz) here, or click to choose a file'; });
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hot'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hot'); }));
drop.addEventListener('drop', ev => { picked = ev.dataTransfer.files[0] || null;
  if (picked) $('#droptext').textContent = '📄 ' + picked.name; });

$('#go').addEventListener('click', start);
$('#arxiv').addEventListener('keydown', e => { if (e.key === 'Enter') start(); });

function fmt(s){ const m = Math.floor(s/60), x = Math.floor(s%60); return m + ':' + String(x).padStart(2,'0'); }

// POST /verify via XHR so we get real UPLOAD progress (fetch can't report it).
function postVerify(fd, onUp){
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/verify');
    xhr.upload.addEventListener('progress', e => { if (e.lengthComputable) onUp(e.loaded / e.total); });
    xhr.addEventListener('load', () => {
      try { resolve(JSON.parse(xhr.responseText)); } catch (e) { reject(new Error('bad response')); }
    });
    xhr.addEventListener('error', () => reject(new Error('network error')));
    xhr.send(fd);
  });
}

async function start(){
  const arxiv = $('#arxiv').value.trim();
  if (!arxiv && !picked){ alert('Paste an arXiv/PDF link or choose a file.'); return; }
  $('#go').disabled = true; $('#out').innerHTML = ''; $('#prog').classList.remove('hidden');
  setBar(2, picked ? 'Uploading…' : 'Submitting…');

  const fd = new FormData();
  if (picked) fd.append('file', picked); else fd.append('arxiv', arxiv);
  let r;
  try {
    r = await postVerify(fd, frac => {
      const pct = Math.round(frac * 100);                       // real upload % for a file
      if (picked) setBar(pct, 'Uploading ' + (picked.name || 'file') + '… ' + pct + '%');
    });
  } catch(e){ return fail('upload failed: ' + e.message); }
  if (r.error) return fail(r.error);

  setBar(6, 'Starting verification…');   // upload done; the long verify phase begins
  t0 = Date.now();
  timer = setInterval(() => {
    const el = (Date.now()-t0)/1000; $('#time').textContent = fmt(el);
    if (etaS) setWidth(Math.min(92, 12 + 80*el/etaS));   // soft moving bar during verify
  }, 250);

  curJob = r.job_id;
  try { localStorage.setItem('cv-job', curJob); } catch(e) {}  // survive a page refresh
  listen(curJob);
}

// Open (or re-open) the SSE stream for a job. Reopening replays the whole history,
// so the handlers rebuild the UI idempotently — that's what makes refresh-resume work.
function listen(jobId){
  if (es) { try { es.close(); } catch(e) {} }
  es = new EventSource('/events/' + jobId);
  es.onmessage = ev => onEvent(JSON.parse(ev.data));
  es.addEventListener('end', () => { es.close(); clearInterval(timer); $('#go').disabled = false; });
  es.addEventListener('gone', () => resetToIdle());   // server: this job no longer exists (restart) → stop, don't reconnect
  es.onerror = () => {
    // A dropped stream is usually transient — the browser auto-reconnects with backoff.
    // If it's terminally CLOSED, stop and reset rather than let anything spin.
    if (es && es.readyState === EventSource.CLOSED) resetToIdle();
  };
}

// Tear down a run's UI and forget its id — return to the idle form. Used when the job
// is gone (e.g. the server restarted), so a reload starts clean instead of reconnecting.
function resetToIdle(){
  if (es) { try { es.close(); } catch(e) {} es = null; }
  clearInterval(timer);
  curJob = null;
  try { localStorage.removeItem('cv-job'); } catch(e) {}
  $('#go').disabled = false;
  $('#prog').classList.add('hidden');
  $('#stage').textContent = 'Starting…';
}

// Recover an in-flight (or just-finished) job after a page refresh: the job id is
// remembered in localStorage, /status says whether the server still has it, and
// reopening the stream replays progress + the final report.
(function resume(){
  let jid = null; try { jid = localStorage.getItem('cv-job'); } catch(e) {}
  if (!jid) return;
  fetch('/status/' + jid).then(r => r.ok ? r.json() : null).then(s => {
    if (!s || !s.known) { try { localStorage.removeItem('cv-job'); } catch(e) {} return; }
    curJob = jid;
    $('#prog').classList.remove('hidden');
    $('#go').disabled = true;
    $('#stage').textContent = 'Reconnecting…';
    t0 = Date.now() - Math.max(0, s.elapsed || 0) * 1000;   // continue real elapsed time, not from 0
    timer = setInterval(() => {
      const el = (Date.now()-t0)/1000; $('#time').textContent = fmt(el);
      if (etaS) setWidth(Math.min(92, 12 + 80*el/etaS));
    }, 250);
    listen(jid);
  }).catch(() => {});
})();

function onEvent(ev){
  switch(ev.stage){
    case 'ingesting':  setBar(8,  'Fetching the paper…'); break;
    case 'extracting': setBar(16, 'Extracting references & claims…'); break;
    case 'verifying':
      etaS = Math.max(20, (ev.count||20) * 5);  // rough ETA → a bar that moves
      setBar(20, `Verifying ${ev.count} citations… (existence, metadata, relevance)`); break;
    case 'report':  return showReport(ev);
    case 'error':   return fail(ev.message);
  }
}

function showReport(ev){
  setWidth(100); $('#stage').textContent = 'Done';
  const s = ev.summary || {}, ex = s.exists||{}, sc = s.supports||{};
  const chip = (label,v,cls)=>`<span class="chip ${cls||''}">${label} <b>${v}</b></span>`;
  let chips = '';
  chips += chip('references', s.refs||0);
  chips += chip('claim·cite pairs', s.pairs||0);
  if (ex['no']) chips += chip('fabricated', ex['no']);
  if (ex['unresolved']) chips += chip('unresolved', ex['unresolved']);
  if (sc['does_not']) chips += chip('irrelevant', sc['does_not']);
  if (sc['partial']) chips += chip('partly relevant', sc['partial']);
  if (sc['supports']) chips += chip('relevant', sc['supports']);
  if (s.high) chips += chip('high-severity', s.high);
  chips += chip('cost', '$'+(ev.cost_usd??0));
  chips += chip('time', fmt(ev.wall_s||0));
  let errs = (ev.errors&&ev.errors.length) ? `<p class="err">⚠ ${ev.errors.length} note(s): ${ev.errors.join(' · ')}</p>` : '';
  const avail = ev.downloads ? Object.keys(ev.downloads).filter(k => ev.downloads[k]) : ['md','json'];
  const dlbar = curJob ? `<div class="dl">` + avail.map(f =>
    `<a class="dlbtn" href="/download/${curJob}?fmt=${f}" download>⬇ Download ${f.toUpperCase()}</a>`).join('') + `</div>` : '';
  $('#out').innerHTML = `<div class="chips">${chips}</div>${dlbar}${errs}<div class="report">${ev.html}</div>`;
}

function fail(msg){ clearInterval(timer); $('#go').disabled = false;
  $('#stage').textContent = 'Failed'; $('#stage').className = 'stage err';
  setWidth(100); $('#fill').style.background = 'var(--bad)';
  $('#out').innerHTML = `<p class="err">✗ ${msg}</p>`; if (es) es.close(); }

function setBar(w, stage){ setWidth(w); $('#stage').textContent = stage; }
function setWidth(w){ $('#fill').style.width = w + '%'; }
</script>
</body>
</html>"""
