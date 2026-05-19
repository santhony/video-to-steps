# video-to-steps Implementation Plan — Phase 6: Server + result page

**Goal:** A browser flow from URL submission to a result page that visibly
works. FastAPI serves the form, validates URLs, spawns a background
pipeline task, polls status via HTMX, and renders the final illustrated
steps.

**Architecture:** Single FastAPI app. POST `/process` allocates a job id,
writes a `queued` manifest, schedules `run_job` via `BackgroundTasks`, and
redirects to `/job/{id}`. The job page contains an HTMX fragment that polls
`/job/{id}/status` every 2 seconds. When the manifest reaches `status=done`
the status fragment returns a "see results" stub fragment without polling
attributes (HTMX stops polling automatically). Result page reads
`steps.json` + `meta.json` and renders ordered steps + thumbnail grid.

**Tech Stack:** FastAPI, uvicorn, Jinja2, python-multipart (form parsing),
HTMX (vendored single JS file).

**Scope:** 6 of 7 phases.

**Codebase verified:** 2026-05-18. Phase 5's `run_job` + `Manifest`
expected; this phase adds the surface around them.

**External dependency findings:**
- ✓ FastAPI `BackgroundTasks` is the canonical async-post-303 pattern;
  `background_tasks.add_task(async_coro, *args)` after a `RedirectResponse`
  fires the coroutine on FastAPI's event loop after the response is sent.
- ✓ HTMX polling: `<div hx-get="..." hx-trigger="load, every 2s"
  hx-swap="outerHTML">...</div>`. To stop polling, return a fragment whose
  outer element does NOT include `hx-trigger="every 2s"` (the swap
  replaces the polling element with a non-polling one). HTTP 286 is an
  alternative; we use the fragment-replacement approach for clarity.
- ✓ YouTube URL regex covers `watch?v=`, `youtu.be/`, `shorts/`, m./www.
  prefixes. The canonical 11-char `[A-Za-z0-9_-]{11}` video ID shape
  validates the meaningful part. We use:
  `re.search(r"(?:youtu\.be/|youtube\.com/(?:embed/|v/|shorts/|watch\?(?:[^#]*&)?v=))([A-Za-z0-9_-]{11})", url)`
  and accept the URL iff a match is found.
- ✓ `uvicorn.run("server:app", host, port)` programmatic invocation
  doesn't read `UVICORN_HOST/PORT` env vars; the `__main__` block reads
  `APP_HOST`/`APP_PORT` and passes them explicitly.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### vts-v1.AC8: Server and result page
- **vts-v1.AC8.1 Success:** `GET /` renders the URL input form.
- **vts-v1.AC8.2 Success:** `POST /process` with a valid YouTube URL returns a 303 redirect to `/job/{id}` and spawns a pipeline task.
- **vts-v1.AC8.3 Failure:** `POST /process` with a non-YouTube URL returns 400 with a human-readable message.
- **vts-v1.AC8.4 Success:** `/job/{id}/status` returns an HTMX fragment showing current `status`, `progress`, and running `cost.total_usd`.
- **vts-v1.AC8.5 Success:** `/job/{id}/result` renders ordered steps each with caption-alt-texted thumbnails; small print shows `mode`, `cost.total_usd`, embedder name, vision model, and LLM model.

---

<!-- START_TASK_1 -->
### Task 1: Add server-tier requirements

**Files:**
- Modify: `requirements.txt` (append)

**Implementation:**

```
fastapi>=0.110
uvicorn[standard]>=0.30
jinja2>=3.1
python-multipart>=0.0.9
```

`python-multipart` is needed for FastAPI form parsing on `POST /process`.

**Verification:**

```bash
source venv/bin/activate
uv pip install -r requirements-dev.txt
python -c "import fastapi, uvicorn, jinja2; print(fastapi.__version__)"
```
Expected: version string.

**Commit:**

```bash
git add requirements.txt
git commit -m "chore(vts-v1): add fastapi + uvicorn + jinja2 + python-multipart"
```
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-4) -->

<!-- START_TASK_2 -->
### Task 2: Templates + static assets

**Files:**
- Create: `templates/base.html`
- Create: `templates/index.html`
- Create: `templates/job.html`
- Create: `templates/status_fragment.html`
- Create: `templates/status_done_fragment.html`
- Create: `templates/result.html`
- Create: `static/css/main.css`
- Create: `static/js/htmx.min.js` (vendored — see fetch instruction)

**Implementation:**

Use the HTMX 2.0+ single-file build (≈14 KB minified). Fetch it once during
setup:

```bash
curl -fsSL https://unpkg.com/htmx.org@2/dist/htmx.min.js -o static/js/htmx.min.js
```

`templates/base.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}video-to-steps{% endblock %}</title>
  <link rel="stylesheet" href="/static/css/main.css">
  <script src="/static/js/htmx.min.js" defer></script>
</head>
<body>
  <header><a href="/" class="brand">video-to-steps</a></header>
  <main>{% block main %}{% endblock %}</main>
</body>
</html>
```

`templates/index.html`:

```html
{% extends "base.html" %}
{% block main %}
<form method="post" action="/process">
  <label for="url">YouTube URL</label>
  <input type="url" id="url" name="url" required
         placeholder="https://www.youtube.com/watch?v=..." autofocus>
  <button type="submit">Make steps</button>
</form>
<p class="hint">Paste the URL of a short instructional video. We download
captions and frames, then build illustrated step-by-step instructions.</p>
{% endblock %}
```

`templates/job.html`:

```html
{% extends "base.html" %}
{% block main %}
<h2>Job {{ job_id }}</h2>
<div id="status"
     hx-get="/job/{{ job_id }}/status"
     hx-trigger="load, every 2s"
     hx-swap="outerHTML">
  <p>Loading…</p>
</div>
{% endblock %}
```

`templates/status_fragment.html` (returned while status ∈ {queued,running}):

```html
<div id="status"
     hx-get="/job/{{ manifest.job_id }}/status"
     hx-trigger="every 2s"
     hx-swap="outerHTML">
  <p class="status status-{{ manifest.status }}">
    <strong>Status:</strong> {{ manifest.status }}
    {% if manifest.progress %} — {{ manifest.progress }}{% endif %}
  </p>
  <p class="cost">Running cost: ${{ '%.4f' % manifest.cost.total_usd }}</p>
</div>
```

`templates/status_done_fragment.html` (returned when status ∈ {done,error};
NOTE: no `hx-trigger="every 2s"`, so polling stops):

```html
<div id="status">
  {% if manifest.status == "done" %}
    <p class="status status-done"><strong>Done.</strong></p>
    <p><a href="/job/{{ manifest.job_id }}/result">View result →</a></p>
    <p class="cost">Final cost: ${{ '%.4f' % manifest.cost.total_usd }}</p>
  {% else %}
    <p class="status status-error"><strong>Error.</strong></p>
    <pre class="error">{{ manifest.error }}</pre>
  {% endif %}
</div>
```

`templates/result.html`:

```html
{% extends "base.html" %}
{% block title %}Result — {{ manifest.job_id }}{% endblock %}
{% block main %}
<section class="meta">
  <small>
    Mode: {{ manifest.mode }} · Cost: ${{ '%.4f' % manifest.cost.total_usd }}
    · Embedder: {{ manifest.config_snapshot.embed_backend }}
    · LLM: {{ manifest.config_snapshot.llm_model }}
    · Vision: {{ manifest.config_snapshot.vision_model }}
  </small>
</section>

<ol class="steps">
{% for step in steps %}
  <li>
    <p class="instruction">{{ step.instruction }}</p>
    <div class="frame-grid">
      {% for frame in step.frames %}
        <a href="/job/{{ manifest.job_id }}/frame/{{ '%04d' % (frame.index + 1) }}.jpg">
          <img src="/job/{{ manifest.job_id }}/frame/{{ '%04d' % (frame.index + 1) }}.jpg"
               alt="{{ frame_captions.get(frame.index|string, '') or step.instruction }}"
               loading="lazy">
        </a>
      {% endfor %}
    </div>
  </li>
{% endfor %}
</ol>
{% endblock %}
```

`static/css/main.css` (terse; not styled to perfection, just usable):

```css
:root { --fg: #1a1a1a; --bg: #fafafa; --accent: #2a6fdb; --muted: #666; }
body {
  font-family: system-ui, -apple-system, sans-serif;
  max-width: 880px; margin: 1em auto; padding: 0 1em;
  color: var(--fg); background: var(--bg); line-height: 1.45;
}
header .brand { font-weight: 700; color: var(--fg); text-decoration: none; }
form { display: flex; flex-direction: column; gap: 0.5em; margin-top: 2em; }
input[type=url] { padding: 0.6em; font-size: 1rem; }
button { padding: 0.6em 1.2em; font-size: 1rem; background: var(--accent); color: white; border: 0; border-radius: 4px; cursor: pointer; }
.hint { color: var(--muted); margin-top: 1em; }
.status-running { color: var(--accent); }
.status-done    { color: #1a7a3a; }
.status-error   { color: #b00020; }
.error          { white-space: pre-wrap; background: #fff0f0; padding: 1em; border-radius: 4px; }
.cost           { color: var(--muted); font-size: 0.9em; }
.meta small     { color: var(--muted); }
.steps          { padding-left: 1.4em; }
.steps li       { margin: 1.2em 0; }
.instruction    { font-size: 1.05em; }
.frame-grid     { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 6px; margin-top: 0.4em; }
.frame-grid img { width: 100%; height: auto; border-radius: 4px; }
```

**Verification:**

```bash
ls templates/*.html static/css/main.css static/js/htmx.min.js
test -s static/js/htmx.min.js && echo "htmx OK"
```
Expected: all listed; htmx.min.js non-empty.

**Commit:**

```bash
git add templates/ static/
git commit -m "feat(vts-v1): server templates + static assets (HTMX vendored)"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `server.py` — FastAPI app and routes

**Verifies:** vts-v1.AC8.1, vts-v1.AC8.2, vts-v1.AC8.3, vts-v1.AC8.4, vts-v1.AC8.5

**Files:**
- Create: `server.py`

**Implementation:**

```python
"""FastAPI app for video-to-steps.

Routes:
  GET  /                        — URL input form.
  POST /process                 — validate URL, allocate job, spawn pipeline, 303.
  GET  /job/{id}                — job page with HTMX polling fragment.
  GET  /job/{id}/status         — HTMX status fragment (polling) or final fragment.
  GET  /job/{id}/result         — final result page.
  GET  /job/{id}/frame/{n}.jpg  — frame image.

All disk reads go through pipeline.storage helpers so atomic writes
guarantee parseable JSON.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import get_settings
from pipeline.pipeline import run_job
from pipeline.storage import ensure_job_dir, read_json, write_json_atomic
from pipeline.types import Manifest


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


_YT_RE = re.compile(
    r"(?:youtu\.be/|"
    r"(?:www\.|m\.)?youtube\.com/(?:embed/|v/|shorts/|watch\?(?:[^#]*&)?v=))"
    r"([A-Za-z0-9_-]{11})"
)


def _is_valid_youtube_url(url: str) -> bool:
    return _YT_RE.search(url) is not None


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def _load_manifest_dict(jobs_root: Path, job_id: str) -> dict | None:
    """Returns the manifest as a plain dict (templates use attribute access via dict-like wrapper)."""
    p = jobs_root / job_id / "meta.json"
    if not p.exists():
        return None
    return read_json(p)


class _AttrDict(dict):
    """Dict subclass enabling attribute access for Jinja2 templates.

    Nested dicts AND lists-of-dicts are wrapped lazily on each attribute
    access, so templates can use chained attribute syntax like
    `step.frames[0].index` without an eager-wrap pass.
    """
    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(v, dict):
            return _AttrDict(v)
        if isinstance(v, list):
            return [_AttrDict(x) if isinstance(x, dict) else x for x in v]
        return v


app = FastAPI(title="video-to-steps")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # vts-v1.AC8.1
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/process")
async def process(
    request: Request,
    background_tasks: BackgroundTasks,
    url: str = Form(...),
) -> RedirectResponse:
    if not _is_valid_youtube_url(url):
        # vts-v1.AC8.3
        raise HTTPException(
            status_code=400,
            detail="Not a recognized YouTube URL. Provide a youtube.com/watch?v=… or youtu.be/… link.",
        )

    settings = get_settings()
    jobs_root = Path(settings.jobs_root)
    job_id = _new_job_id()
    ensure_job_dir(jobs_root, job_id)

    initial = Manifest(job_id=job_id, url=url, status="queued", progress="queued")
    write_json_atomic(jobs_root / job_id / "meta.json", initial)

    # vts-v1.AC8.2 — spawn pipeline task.
    background_tasks.add_task(run_job, job_id, url, settings, jobs_root)

    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str) -> HTMLResponse:
    settings = get_settings()
    m = _load_manifest_dict(Path(settings.jobs_root), job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    return templates.TemplateResponse(request, "job.html", {"job_id": job_id})


@app.get("/job/{job_id}/status", response_class=HTMLResponse)
async def job_status(request: Request, job_id: str) -> HTMLResponse:
    settings = get_settings()
    m = _load_manifest_dict(Path(settings.jobs_root), job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    manifest = _AttrDict(m)
    template = (
        "status_done_fragment.html"
        if m.get("status") in ("done", "error")
        else "status_fragment.html"
    )
    # vts-v1.AC8.4
    return templates.TemplateResponse(request, template, {"manifest": manifest})


@app.get("/job/{job_id}/result", response_class=HTMLResponse)
async def job_result(request: Request, job_id: str) -> HTMLResponse:
    settings = get_settings()
    jobs_root = Path(settings.jobs_root)
    m = _load_manifest_dict(jobs_root, job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    if m.get("status") != "done":
        # Send the user back to the job page until done.
        return RedirectResponse(f"/job/{job_id}", status_code=303)  # type: ignore[return-value]

    steps = read_json(jobs_root / job_id / "steps.json")
    captions_path = jobs_root / job_id / "frame_captions.json"
    frame_captions = read_json(captions_path) if captions_path.exists() else {}
    # vts-v1.AC8.5
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "manifest": _AttrDict(m),
            "steps": [_AttrDict(s) for s in steps],
            "frame_captions": frame_captions,
        },
    )


@app.get("/job/{job_id}/frame/{name}.jpg")
async def job_frame(job_id: str, name: str) -> FileResponse:
    settings = get_settings()
    # name is %04d; guard against traversal.
    if not name.isdigit() or len(name) != 4:
        raise HTTPException(400, "Bad frame name.")
    p = Path(settings.jobs_root) / job_id / "frames" / f"{name}.jpg"
    if not p.exists():
        raise HTTPException(404, "Frame not found.")
    return FileResponse(p, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8090"))
    uvicorn.run("server:app", host=host, port=port)
```

**Verification:**

```bash
source venv/bin/activate
# Syntax + import check only — full route tests are in Task 4.
python -c "import server; print(server.app.title)"
```
Expected: `video-to-steps`.

**Commit:**

```bash
git add server.py
git commit -m "feat(vts-v1): FastAPI server (form, process, status, result, frame)"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_4 -->
### Task 4: Server tests (unit + smoke)

**Verifies:** vts-v1.AC8.1, vts-v1.AC8.2, vts-v1.AC8.3, vts-v1.AC8.4, vts-v1.AC8.5

**Files:**
- Create: `tests/test_server.py` (unit)

**Implementation:**

Use `httpx.AsyncClient(app=server.app)` against the FastAPI app
directly — no live socket. Monkeypatch `run_job` to a no-op coroutine so
the test doesn't hit the real pipeline; we test the routing/UX surface
only. AC8.4 and AC8.5 use hand-built manifest + steps.json files written
directly into a tmp_path jobs root.

```python
"""Tests for server.py — routes, validation, fragments."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

import server
from pipeline.storage import ensure_job_dir, write_json_atomic
from pipeline.types import CostBreakdown, Manifest


@pytest.fixture
def jobs_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("JOBS_ROOT", str(tmp_path / "jobs"))
    # config.get_settings is called freshly per request — it picks up env.
    return tmp_path / "jobs"


@pytest.fixture
def stub_run_job(monkeypatch):
    """Replace run_job with a no-op so POST /process doesn't actually run the pipeline."""
    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr(server, "run_job", _noop)
    return _noop


async def _client():
    transport = httpx.ASGITransport(app=server.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_index_renders_form(jobs_root):
    # vts-v1.AC8.1
    async with await _client() as c:
        r = await c.get("/")
    assert r.status_code == 200
    assert 'name="url"' in r.text
    assert 'method="post"' in r.text


@pytest.mark.asyncio
async def test_process_redirects_303_and_writes_manifest(jobs_root, stub_run_job):
    # vts-v1.AC8.2
    async with await _client() as c:
        r = await c.post("/process",
                         data={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                         follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/job/")
    job_id = r.headers["location"].rsplit("/", 1)[-1]
    assert (jobs_root / job_id / "meta.json").exists()


@pytest.mark.asyncio
async def test_process_rejects_non_youtube_400(jobs_root, stub_run_job):
    # vts-v1.AC8.3
    async with await _client() as c:
        r = await c.post("/process",
                         data={"url": "https://example.com/some-video"},
                         follow_redirects=False)
    assert r.status_code == 400
    assert "YouTube" in r.text or "youtube" in r.text


@pytest.mark.asyncio
@pytest.mark.parametrize("good", [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/shorts/dQw4w9WgXcQ",
    "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
])
async def test_process_accepts_known_youtube_shapes(jobs_root, stub_run_job, good):
    async with await _client() as c:
        r = await c.post("/process", data={"url": good}, follow_redirects=False)
    assert r.status_code == 303


@pytest.mark.asyncio
async def test_status_fragment_running(jobs_root):
    # vts-v1.AC8.4
    job_id = "abc123abc123"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(job_id=job_id, url="https://youtu.be/x", status="running",
                 progress="embedding frames", cost=CostBreakdown(total_usd=0.0123))
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/status")
    assert r.status_code == 200
    assert "running" in r.text
    assert "embedding frames" in r.text
    assert "0.0123" in r.text
    # Polling attribute MUST be present for in-progress states.
    assert 'hx-trigger="every 2s"' in r.text


@pytest.mark.asyncio
async def test_status_fragment_done_stops_polling(jobs_root):
    job_id = "donedonedone"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(job_id=job_id, url="https://youtu.be/x", status="done",
                 cost=CostBreakdown(chat_usd=0.01, vision_usd=0.02, embed_usd=0.005, total_usd=0.035))
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/status")
    assert r.status_code == 200
    assert "Done" in r.text
    # Polling must STOP — the fragment must NOT include the every-2s trigger.
    assert 'hx-trigger="every 2s"' not in r.text


@pytest.mark.asyncio
async def test_result_page_renders_steps_and_meta(jobs_root):
    # vts-v1.AC8.5
    job_id = "resultjob123"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(
        job_id=job_id, url="https://youtu.be/x",
        status="done", mode="cloud",
        config_snapshot={
            "embed_backend": "jina_v4", "llm_model": "deepseek-chat",
            "vision_model": "gpt-4o-mini",
        },
        cost=CostBreakdown(total_usd=0.0789),
    )
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    write_json_atomic(
        jobs_root / job_id / "steps.json",
        [
            {"index": 0, "start": 0.0, "end": 10.0,
             "instruction": "Heat the pan.",
             "frames": [{"index": 4, "timestamp": 4.0, "path": "/x/0005.jpg"}]},
            {"index": 1, "start": 10.0, "end": 20.0,
             "instruction": "Slice the onion. Add to pan.",
             "frames": [{"index": 14, "timestamp": 14.0, "path": "/x/0015.jpg"}]},
            {"index": 2, "start": 20.0, "end": 30.0,
             "instruction": "Stir. Serve.",
             "frames": [{"index": 24, "timestamp": 24.0, "path": "/x/0025.jpg"}]},
        ],
    )
    write_json_atomic(jobs_root / job_id / "frame_captions.json",
                       {"4": "hands over a pan", "14": "knife slicing onion", "24": "spoon in pan"})

    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/result")
    assert r.status_code == 200
    assert "Heat the pan." in r.text
    assert "Slice the onion." in r.text
    assert "Stir." in r.text
    # AC8.5 small-print elements present.
    assert "cloud" in r.text
    assert "0.0789" in r.text
    assert "jina_v4" in r.text
    assert "gpt-4o-mini" in r.text
    assert "deepseek-chat" in r.text
    # Alt text uses caption when present.
    assert 'alt="hands over a pan"' in r.text


@pytest.mark.asyncio
async def test_result_redirects_when_not_done(jobs_root):
    job_id = "stillrunning"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(job_id=job_id, url="https://youtu.be/x", status="running")
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/result", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/job/{job_id}"


@pytest.mark.asyncio
async def test_unknown_job_id_404(jobs_root):
    async with await _client() as c:
        r = await c.get("/job/doesnotexist00/status")
    assert r.status_code == 404
```

**Verification:**

```bash
source venv/bin/activate
pytest tests/test_server.py -v
```
Expected: all 11 tests pass.

**Smoke (manual, optional):**

```bash
# In one terminal:
./start.sh
# In another:
curl -s http://127.0.0.1:8090/ | grep -c '<form'   # should print 1
./stop.sh
```

**Commit:**

```bash
git add tests/test_server.py
git commit -m "test(vts-v1): server routes — AC8.1..AC8.5"
```

**Done when:** `pytest tests/test_server.py` passes; a manual browser
walkthrough of `./start.sh → GET / → submit a YouTube URL → poll → result`
succeeds against a real cloud `.env`.
<!-- END_TASK_4 -->
