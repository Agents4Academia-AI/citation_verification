"""
web/app.py — FastAPI front-end: upload a PDF or paste an arXiv link, watch a
progress bar, read the rendered report in the browser.

The whole verification is the public ``run_verification`` call, run in a worker
thread (it is blocking and takes minutes). Its ``progress_cb`` events are pushed
onto a per-job queue and relayed to the browser as Server-Sent Events, so the
page shows real stages ("found N citations", "verifying …") plus a live timer.
Nothing here re-implements verification — it only sequences the existing seams.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..config import apply_auth, load_settings
from ..ingest import parse_arxiv_id

# ── job registry ─────────────────────────────────────────────────────────────


@dataclass
class _Job:
    job_id: str
    label: str  # what's being verified (arXiv id or filename), for the UI
    started: float = field(default_factory=time.monotonic)
    q: queue.Queue[dict] = field(default_factory=queue.Queue)
    done: bool = False


_JOBS: dict[str, _Job] = {}


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


# ── the worker: one full verification, emitting progress onto the job queue ──
def _run_job(job: _Job, source: str, settings: Any) -> None:
    from .. import run_verification  # lazy: keeps web import light
    from ..render import render_report

    try:
        result = run_verification(
            source,
            backend="agentic",
            settings=settings,
            resume=False,
            progress_cb=lambda ev: job.q.put(ev),
        )
        job.q.put(
            {
                "stage": "report",
                "html": _md_to_html(render_report(result)),
                "summary": _summary(result.records),
                "cost_usd": round(result.usage.cost_usd, 4),
                "wall_s": round(result.usage.wall_seconds, 1),
                "errors": result.errors,
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface any failure to the page
        job.q.put({"stage": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        job.done = True
        job.q.put({"stage": "__end__"})


# ── app factory ──────────────────────────────────────────────────────────────
def create_app(settings: Any | None = None):
    """Build the FastAPI app. ``settings`` defaults to ``load_settings()``."""
    settings = settings or load_settings()
    auth = apply_auth(settings)  # subscription unless ANTHROPIC_API_KEY is set
    uploads = Path(settings.papers_dir) / "_uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Citation Verifier")

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
        threading.Thread(target=_run_job, args=(job, source, settings), daemon=True).start()
        return {"job_id": job.job_id, "label": label}

    @app.get("/events/{job_id}")
    async def events(job_id: str):
        job = _JOBS.get(job_id)
        if job is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)

        async def stream():
            last = time.monotonic()
            while True:
                try:
                    ev = job.q.get_nowait()
                except queue.Empty:
                    if job.done:
                        break
                    # keep proxies/connections alive during the long verify phase
                    if time.monotonic() - last > 12:
                        last = time.monotonic()
                        yield ": heartbeat\n\n"
                    await asyncio.sleep(0.4)
                    continue
                if ev.get("stage") == "__end__":
                    break
                last = time.monotonic()
                yield f"data: {json.dumps(ev)}\n\n"
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
<title>Citation Verifier</title>
<style>
  :root { --bg:#0f1115; --card:#171a21; --line:#2a2f3a; --fg:#e6e8ee; --mut:#9aa3b2;
          --acc:#5b9cff; --ok:#36b37e; --warn:#e2b340; --bad:#f0616d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  .wrap { max-width:920px; margin:0 auto; padding:32px 20px 80px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--mut); margin:0 0 24px; font-size:13px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; }
  .row { display:flex; gap:10px; align-items:center; }
  input[type=text] { flex:1; background:#0e1015; border:1px solid var(--line); color:var(--fg);
                     border-radius:8px; padding:11px 12px; font-size:14px; }
  input[type=text]:focus { outline:none; border-color:var(--acc); }
  button { background:var(--acc); color:#fff; border:0; border-radius:8px; padding:11px 18px;
           font-weight:600; cursor:pointer; font-size:14px; }
  button:disabled { opacity:.5; cursor:default; }
  .or { color:var(--mut); text-align:center; font-size:12px; margin:12px 0; letter-spacing:.08em; }
  .drop { border:1.5px dashed var(--line); border-radius:10px; padding:22px; text-align:center;
          color:var(--mut); cursor:pointer; transition:.15s; }
  .drop.hot { border-color:var(--acc); color:var(--fg); background:#0e1320; }
  .hidden { display:none; }
  /* progress */
  #prog { margin-top:22px; }
  .bar { height:10px; background:#0e1015; border:1px solid var(--line); border-radius:99px; overflow:hidden; }
  .fill { height:100%; width:0; background:linear-gradient(90deg,var(--acc),#7db3ff);
          transition:width .4s ease; }
  .meta { display:flex; justify-content:space-between; margin-top:8px; font-size:13px; color:var(--mut); }
  .stage { color:var(--fg); font-weight:600; }
  /* result */
  #out { margin-top:26px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
  .chip { background:#0e1015; border:1px solid var(--line); border-radius:99px;
          padding:5px 11px; font-size:12px; color:var(--mut); }
  .chip b { color:var(--fg); }
  .report { background:var(--card); border:1px solid var(--line); border-radius:12px;
            padding:6px 22px 18px; overflow-x:auto; }
  .report table { border-collapse:collapse; width:100%; font-size:12.5px; margin:10px 0; }
  .report th, .report td { border:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
  .report th { background:#0e1015; position:sticky; top:0; }
  .report h1 { font-size:18px; } .report h2 { font-size:15px; border-top:1px solid var(--line); padding-top:14px; }
  .report code { background:#0e1015; padding:1px 5px; border-radius:5px; }
  .report a { color:var(--acc); }
  .err { color:var(--bad); }
  .raw-md { white-space:pre-wrap; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Citation Verifier</h1>
  <p class="sub">Upload a paper PDF or paste an arXiv link — checks every reference exists, has correct metadata, and actually supports the claim it's attached to. <span style="opacity:.7">auth: {{AUTH}}</span></p>

  <div class="card" id="form">
    <div class="row">
      <input id="arxiv" type="text" placeholder="arXiv link or id  —  e.g. https://arxiv.org/abs/2310.06825 or 2310.06825" />
      <button id="go">Verify</button>
    </div>
    <div class="or">— OR —</div>
    <label class="drop" id="drop">
      <input id="file" type="file" accept="application/pdf,.pdf" class="hidden"/>
      <span id="droptext">Drop a PDF here, or click to choose a file</span>
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
let es, t0, timer, etaS = 0, picked = null;

const drop = $('#drop'), fileIn = $('#file');
drop.addEventListener('click', () => fileIn.click());
fileIn.addEventListener('change', () => { picked = fileIn.files[0] || null;
  $('#droptext').textContent = picked ? ('📄 ' + picked.name) : 'Drop a PDF here, or click to choose a file'; });
['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('hot'); }));
['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('hot'); }));
drop.addEventListener('drop', ev => { picked = ev.dataTransfer.files[0] || null;
  if (picked) $('#droptext').textContent = '📄 ' + picked.name; });

$('#go').addEventListener('click', start);
$('#arxiv').addEventListener('keydown', e => { if (e.key === 'Enter') start(); });

function fmt(s){ const m = Math.floor(s/60), x = Math.floor(s%60); return m + ':' + String(x).padStart(2,'0'); }

async function start(){
  const arxiv = $('#arxiv').value.trim();
  if (!arxiv && !picked){ alert('Paste an arXiv link or choose a PDF.'); return; }
  $('#go').disabled = true; $('#out').innerHTML = ''; $('#prog').classList.remove('hidden');
  setBar(4, 'Uploading…');

  const fd = new FormData();
  if (picked) fd.append('file', picked); else fd.append('arxiv', arxiv);
  let r;
  try { r = await (await fetch('/verify', {method:'POST', body:fd})).json(); }
  catch(e){ return fail('upload failed: ' + e); }
  if (r.error) return fail(r.error);

  t0 = Date.now();
  timer = setInterval(() => {
    const el = (Date.now()-t0)/1000; $('#time').textContent = fmt(el);
    if (etaS) setWidth(Math.min(92, 12 + 80*el/etaS));   // soft moving bar during verify
  }, 250);

  es = new EventSource('/events/' + r.job_id);
  es.onmessage = ev => onEvent(JSON.parse(ev.data));
  es.addEventListener('end', () => { es.close(); clearInterval(timer); $('#go').disabled = false; });
  es.onerror = () => { /* keep waiting; server sends heartbeats */ };
}

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
  if (sc['does_not']) chips += chip('does not support', sc['does_not']);
  if (sc['supports']) chips += chip('supports', sc['supports']);
  if (s.high) chips += chip('high-severity', s.high);
  chips += chip('cost', '$'+(ev.cost_usd??0));
  chips += chip('time', fmt(ev.wall_s||0));
  let errs = (ev.errors&&ev.errors.length) ? `<p class="err">⚠ ${ev.errors.length} note(s): ${ev.errors.join(' · ')}</p>` : '';
  $('#out').innerHTML = `<div class="chips">${chips}</div>${errs}<div class="report">${ev.html}</div>`;
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
