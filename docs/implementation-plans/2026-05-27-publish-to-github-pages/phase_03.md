# Publish to GitHub Pages — Phase 3: Server Routes + UI Wiring

**Goal:** Wire two HTMX-driven routes (`POST /job/{id}/publish` and `POST /job/{id}/unpublish`) into `server.py`, add the `_publish_controls_fragment.html` partial, render it on the result page, and add integration tests that mock `build_publish_repo`.

**Architecture:** The server is the second sanctioned manifest writer (the first is `pipeline.pipeline._update`; the second carve-out was already the initial `queued` write in `POST /process`). Publish/unpublish are HTMX endpoints returning a fragment that swaps the controls in place.

**Tech Stack:** FastAPI routes, Jinja2 fragments, HTMX `hx-post` / `hx-target` / `hx-indicator`, pytest + httpx AsyncClient (existing pattern in `tests/test_server.py`).

**Scope:** Phase 3 of 4 from `docs/design-plans/2026-05-27-publish-to-github-pages.md`.

**Codebase verified:** 2026-05-27

---

## Reference: verified codebase state

- Route patterns and validators in `server.py:43-301`. `_guard_job_id(job_id)` (line 58) validates job_id; `_load_manifest_dict(jobs_root, job_id)` (line 117) loads manifest as dict; `_AttrDict` (line 125) wraps dicts for template attribute access.
- Existing manifest writer in `server.py:220` (initial `queued` write). The publish/unpublish routes are the third allowed writer (CLAUDE.md will be updated in Phase 4).
- Templates use the `templates.TemplateResponse(request, "name.html", {context})` shape (e.g. line 258).
- HTMX fragment style: `templates/status_fragment.html` — root `<div id="...">` carrying HTMX attributes, content uses `{{ manifest.field }}`.
- `result.html` has the meta section at lines 10-17; the new publish-controls div goes between the meta `<section>` and the `<ol class="steps">`.
- Integration tests use `httpx.AsyncClient(app=server.app)` with `monkeypatch` fixtures (see `tests/test_server.py`).

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Add `_publish_controls_fragment.html` and render it in `result.html`

**Files:**
- Create: `templates/_publish_controls_fragment.html`
- Modify: `templates/result.html` (insert `{% include %}` between the meta `<section>` and `<ol class="steps">`)

**Step 1: Create the fragment**

`templates/_publish_controls_fragment.html`:

```jinja
<div id="publish-controls" class="publish-controls">
  {% if manifest.published_url %}
    <p class="published">
      Published at
      <a href="{{ manifest.published_url }}" target="_blank" rel="noopener noreferrer">{{ manifest.published_url }}</a>
    </p>
    <button type="button"
            hx-post="/job/{{ manifest.job_id }}/unpublish"
            hx-target="#publish-controls"
            hx-swap="outerHTML"
            hx-indicator="#publish-spinner">
      Unpublish
    </button>
    <span id="publish-spinner" class="htmx-indicator">…</span>
  {% else %}
    <button type="button"
            hx-post="/job/{{ manifest.job_id }}/publish"
            hx-target="#publish-controls"
            hx-swap="outerHTML"
            hx-indicator="#publish-spinner">
      Publish to GitHub Pages
    </button>
    <span id="publish-spinner" class="htmx-indicator">publishing…</span>
  {% endif %}
</div>
```

**Step 2: Modify `templates/result.html` to include it**

Insert after the `</section>` closing tag of the meta block (current line 17) and before `<ol class="steps">` (current line 19). In `static_mode` we skip rendering the controls entirely so the snapshot has no publish UI.

Replace the blank line at result.html line 18 with:

```jinja
{% if not static_mode and publish_enabled %}
{% include "_publish_controls_fragment.html" %}
{% endif %}
```

This means the controls only render when (a) we are in live mode and (b) the operator has set `PUBLISH_ENABLED=1`. The server route handler (Task 2) passes `publish_enabled=settings.publish_enabled` in the context.

**Step 3: Verify rendering without route changes yet**

We can't run an integration test yet because the route doesn't exist. Just verify the template parses:

Run: `python -c "from jinja2 import Environment, FileSystemLoader; e = Environment(loader=FileSystemLoader('templates')); e.get_template('_publish_controls_fragment.html'); print('ok')"`
Expected: `ok`

Run: `python -m pytest`
Expected: existing tests still pass (we haven't broken `result.html`).

**Step 4: Commit**

```bash
git add templates/_publish_controls_fragment.html templates/result.html
git commit -m "Add publish controls fragment and include it in result.html"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Pass `publish_enabled` into the result-page context

**Files:**
- Modify: `server.py` (the `job_result` handler, lines 261-288)

**Step 1: Write the failing test**

Append to `tests/test_server.py` (or a new `tests/test_server_publish.py`):

```python
"""Integration tests for the publish UI + routes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

import server
from pipeline.types import StaticBundle


@pytest.fixture
def jobs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def done_job(jobs_root):
    """Lay out a minimal 'done' job on disk so /job/{id}/result renders."""
    job_id = "dQw4w9WgXcQ_a1b2c3"
    (jobs_root / job_id).mkdir()
    (jobs_root / job_id / "frames").mkdir()
    meta = {
        "job_id": job_id,
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Test",
        "status": "done",
        "progress": "",
        "error": "",
        "mode": "cloud",
        "config_snapshot": {"embed_backend": "jina_v4", "llm_model": "x", "vision_model": "y"},
        "cost": {"chat_usd": 0.0, "vision_usd": 0.0, "embed_usd": 0.0, "total_usd": 0.0},
        "published_url": None,
        "published_at": None,
    }
    (jobs_root / job_id / "meta.json").write_text(json.dumps(meta))
    (jobs_root / job_id / "steps.json").write_text(json.dumps([]))
    return job_id


@pytest.mark.asyncio
async def test_result_page_hides_publish_button_when_disabled(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "false")
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get(f"/job/{done_job}/result")
    assert r.status_code == 200
    assert "Publish to GitHub Pages" not in r.text


@pytest.mark.asyncio
async def test_result_page_shows_publish_button_when_enabled(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.get(f"/job/{done_job}/result")
    assert r.status_code == 200
    assert "Publish to GitHub Pages" in r.text
    assert f'hx-post="/job/{done_job}/publish"' in r.text
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_server_publish.py -v` (or `tests/test_server.py` if you appended).
Expected: `test_result_page_shows_publish_button_when_enabled` FAILs — `publish_enabled` is not passed into the template, so the controls never render.

**Step 3: Modify `server.py`**

In `job_result` (current lines 261-288), add `"publish_enabled": settings.publish_enabled` to the context dict passed to `templates.TemplateResponse`. The handler becomes:

```python
@app.get("/job/{job_id}/result", response_class=HTMLResponse)
async def job_result(request: Request, job_id: str) -> HTMLResponse:
    _guard_job_id(job_id)
    settings = get_settings()
    jobs_root = Path(settings.jobs_root)
    m = _load_manifest_dict(jobs_root, job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    if m.get("status") != "done":
        return RedirectResponse(f"/job/{job_id}", status_code=303)  # type: ignore[return-value]

    steps = read_json(jobs_root / job_id / "steps.json")
    captions_path = jobs_root / job_id / "frame_captions.json"
    frame_captions = read_json(captions_path) if captions_path.exists() else {}
    step_links = [_video_deep_link(m.get("url", ""), s.get("start", 0)) for s in steps]
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "manifest": _AttrDict(m),
            "steps": [_AttrDict(s) for s in steps],
            "frame_captions": frame_captions,
            "step_links": step_links,
            "publish_enabled": settings.publish_enabled,
        },
    )
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_server_publish.py -v`
Expected: Both tests pass.

**Step 5: Commit**

```bash
git add server.py tests/test_server_publish.py
git commit -m "Pass publish_enabled into result-page context"
```
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->

<!-- START_TASK_3 -->
### Task 3: Implement `POST /job/{id}/publish` and `POST /job/{id}/unpublish`

**Files:**
- Modify: `server.py` (add imports + two new routes + a small manifest-merge helper)

**Step 1: Write the failing tests**

Append to `tests/test_server_publish.py`:

```python
class FakePublishRepo:
    """Stand-in for PublishRepo that records calls instead of touching gh/git."""

    def __init__(self, *, url: str = "https://example.github.io/r/abc/"):
        self.published: list[tuple[str, StaticBundle]] = []
        self.unpublished: list[str] = []
        self._url = url

    async def publish_job(self, job_id: str, bundle):
        self.published.append((job_id, bundle))
        return f"https://example.github.io/r/{job_id}/"

    async def unpublish_job(self, job_id: str):
        self.unpublished.append(job_id)


@pytest.mark.asyncio
async def test_publish_route_calls_repo_and_updates_manifest(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    fake = FakePublishRepo()
    monkeypatch.setattr(server, "build_publish_repo", lambda settings: fake)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post(f"/job/{done_job}/publish")

    assert r.status_code == 200
    assert len(fake.published) == 1
    assert fake.published[0][0] == done_job
    # Returned fragment shows the "Unpublish" button
    assert "Unpublish" in r.text
    assert f"https://example.github.io/r/{done_job}/" in r.text
    # Manifest now has published_url + published_at
    meta = json.loads((jobs_root / done_job / "meta.json").read_text())
    assert meta["published_url"] == f"https://example.github.io/r/{done_job}/"
    assert isinstance(meta["published_at"], str)
    # ISO-8601-ish: YYYY-MM-DDTHH:MM:SS[.fff][+00:00]
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", meta["published_at"])


@pytest.mark.asyncio
async def test_publish_route_400_when_disabled(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "false")
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post(f"/job/{done_job}/publish")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_publish_route_400_when_job_not_done(jobs_root, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    job_id = "dQw4w9WgXcQ_a1b2c3"
    (jobs_root / job_id).mkdir()
    (jobs_root / job_id / "frames").mkdir()
    meta = {
        "job_id": job_id, "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "T", "status": "running", "progress": "", "error": "", "mode": "",
        "config_snapshot": {}, "cost": {"chat_usd": 0, "vision_usd": 0, "embed_usd": 0, "total_usd": 0},
        "published_url": None, "published_at": None,
    }
    (jobs_root / job_id / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(server, "build_publish_repo", lambda settings: FakePublishRepo())

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post(f"/job/{job_id}/publish")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_unpublish_route_clears_manifest(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    # Pre-populate the manifest as if previously published
    meta = json.loads((jobs_root / done_job / "meta.json").read_text())
    meta["published_url"] = f"https://example.github.io/r/{done_job}/"
    meta["published_at"] = "2026-05-27T12:00:00+00:00"
    (jobs_root / done_job / "meta.json").write_text(json.dumps(meta))

    fake = FakePublishRepo()
    monkeypatch.setattr(server, "build_publish_repo", lambda settings: fake)

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post(f"/job/{done_job}/unpublish")

    assert r.status_code == 200
    assert fake.unpublished == [done_job]
    assert "Publish to GitHub Pages" in r.text  # fragment now shows the Publish button
    meta_after = json.loads((jobs_root / done_job / "meta.json").read_text())
    assert meta_after["published_url"] is None
    assert meta_after["published_at"] is None


@pytest.mark.asyncio
async def test_publish_route_500_on_publish_error(jobs_root, done_job, monkeypatch):
    """If PublishRepo raises PublishError, the route returns 500 and does NOT update the manifest."""
    from pipeline.types import PublishError

    monkeypatch.setenv("PUBLISH_ENABLED", "true")

    class FailingRepo(FakePublishRepo):
        async def publish_job(self, job_id, bundle):
            raise PublishError("git push failed: rejected")

    monkeypatch.setattr(server, "build_publish_repo", lambda settings: FailingRepo())

    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        r = await c.post(f"/job/{done_job}/publish")

    assert r.status_code == 500
    # Manifest unchanged — both publish fields stay null
    meta = json.loads((jobs_root / done_job / "meta.json").read_text())
    assert meta["published_url"] is None
    assert meta["published_at"] is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_server_publish.py -v`
Expected: All 5 new tests FAIL — routes don't exist (404 on POST).

**Step 3: Add the routes to `server.py`**

Add imports near line 33-36:

```python
from datetime import datetime, timezone

from jinja2 import Environment, FileSystemLoader

from pipeline.publish import build_static_bundle
from pipeline.publish_repo import build_publish_repo
from pipeline.types import Manifest, PublishError
```

Add a module-level Jinja Environment instance near the existing `templates =` line (around line 40). We need a `jinja2.Environment` directly because `Jinja2Templates` wraps it; the pipeline's `build_static_bundle` takes the raw Environment:

```python
# Jinja2 Environment used by pipeline.publish for static bundles.
# Lives alongside the FastAPI Jinja2Templates wrapper so they share a loader root.
_publish_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=True,
)
```

Add the two route handlers after the existing `job_frame` handler (after current line 301):

```python
def _merge_manifest(jobs_root: Path, job_id: str, **updates: Any) -> dict:
    """Load the manifest dict, apply `updates`, write atomically. Returns the merged dict."""
    m = _load_manifest_dict(jobs_root, job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    m.update(updates)
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    return m


@app.post("/job/{job_id}/publish", response_class=HTMLResponse)
async def job_publish(request: Request, job_id: str) -> HTMLResponse:
    _guard_job_id(job_id)
    settings = get_settings()
    if not settings.publish_enabled:
        raise HTTPException(400, "Publishing is disabled (PUBLISH_ENABLED=false).")

    jobs_root = Path(settings.jobs_root)
    m = _load_manifest_dict(jobs_root, job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    if m.get("status") != "done":
        raise HTTPException(400, "Only completed jobs can be published.")

    # Reconstruct the data the snapshot needs (same shape as job_result).
    steps_raw = read_json(jobs_root / job_id / "steps.json")
    captions_path = jobs_root / job_id / "frame_captions.json"
    frame_captions = read_json(captions_path) if captions_path.exists() else {}
    step_links = [_video_deep_link(m.get("url", ""), s.get("start", 0)) for s in steps_raw]

    # Rebuild dataclasses from on-disk dicts for the pure bundler.
    manifest = _manifest_from_dict(m)
    steps = [_step_from_dict(s) for s in steps_raw]

    bundle = build_static_bundle(
        manifest=manifest,
        steps=steps,
        frame_captions=frame_captions,
        step_links=step_links,
        frames_dir=jobs_root / job_id / "frames",
        css_path=BASE_DIR / "static" / "css" / "main.css",
        templates_env=_publish_env,
    )

    repo = build_publish_repo(settings)
    try:
        url = await repo.publish_job(job_id, bundle)
    except PublishError as e:
        raise HTTPException(500, str(e))

    merged = _merge_manifest(
        jobs_root, job_id,
        published_url=url,
        published_at=datetime.now(timezone.utc).isoformat(),
    )
    return templates.TemplateResponse(
        request,
        "_publish_controls_fragment.html",
        {"manifest": _AttrDict(merged)},
    )


@app.post("/job/{job_id}/unpublish", response_class=HTMLResponse)
async def job_unpublish(request: Request, job_id: str) -> HTMLResponse:
    _guard_job_id(job_id)
    settings = get_settings()
    if not settings.publish_enabled:
        raise HTTPException(400, "Publishing is disabled (PUBLISH_ENABLED=false).")

    jobs_root = Path(settings.jobs_root)
    m = _load_manifest_dict(jobs_root, job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")

    repo = build_publish_repo(settings)
    try:
        await repo.unpublish_job(job_id)
    except PublishError as e:
        raise HTTPException(500, str(e))

    merged = _merge_manifest(jobs_root, job_id, published_url=None, published_at=None)
    return templates.TemplateResponse(
        request,
        "_publish_controls_fragment.html",
        {"manifest": _AttrDict(merged)},
    )
```

Add two small helpers (near the other `_*` helpers around line 117). These rebuild the `Manifest` and `Step` dataclasses from the on-disk dicts so `build_static_bundle` (which is typed against the dataclasses) can use them:

```python
def _manifest_from_dict(m: dict) -> Manifest:
    """Rebuild a Manifest dataclass from an on-disk meta.json dict.

    Drops unknown keys defensively — older job files predating
    `published_*` already work without these fields.
    """
    from pipeline.types import CostBreakdown
    cost_d = m.get("cost") or {}
    cost = CostBreakdown(
        chat_usd=cost_d.get("chat_usd", 0.0),
        vision_usd=cost_d.get("vision_usd", 0.0),
        embed_usd=cost_d.get("embed_usd", 0.0),
        total_usd=cost_d.get("total_usd", 0.0),
    )
    return Manifest(
        job_id=m["job_id"],
        url=m["url"],
        title=m.get("title", ""),
        status=m.get("status", ""),
        progress=m.get("progress", ""),
        error=m.get("error", ""),
        mode=m.get("mode", ""),
        config_snapshot=m.get("config_snapshot") or {},
        cost=cost,
        # published_* are not needed for rendering the snapshot
    )


def _step_from_dict(s: dict):
    """Rebuild a Step (and its Frames) from an on-disk steps.json entry."""
    from pipeline.types import Frame, Step
    frames = [
        Frame(index=f["index"], timestamp=f["timestamp"], path=Path(f["path"]))
        for f in s.get("frames") or []
    ]
    return Step(
        index=s["index"],
        start=s["start"],
        end=s["end"],
        instruction=s["instruction"],
        frames=frames,
    )
```

Also extend the imports at the top of `server.py` to include `Any`:

```python
from typing import Any
```

(if not already imported).

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_server_publish.py -v`
Expected: All 5 tests pass.

Run full suite: `python -m pytest`
Expected: All offline tests pass.

**Step 5: Commit**

```bash
git add server.py tests/test_server_publish.py
git commit -m "Add /job/{id}/publish and /job/{id}/unpublish routes"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: CSS for `.publish-controls` and the spinner

**Files:**
- Modify: `static/css/main.css` (append publish-controls styles)

**Step 1: Append CSS**

Add to the end of `static/css/main.css`:

```css

/* Publish to GitHub Pages controls */
.publish-controls {
  margin: 1rem 0 1.5rem;
  padding: 0.75rem 1rem;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fafafa;
}
.publish-controls .published {
  margin: 0 0 0.5rem;
  word-break: break-all;
}
.publish-controls button {
  font: inherit;
  padding: 0.4rem 0.9rem;
  cursor: pointer;
}
.publish-controls .htmx-indicator {
  display: none;
  margin-left: 0.75rem;
  font-style: italic;
  color: #666;
}
/* HTMX adds .htmx-request to the originating element while the request is in flight.
   The indicator becomes visible only during that window. */
.publish-controls .htmx-request .htmx-indicator,
.publish-controls .htmx-indicator.htmx-request {
  display: inline;
}
```

**Step 2: Verify visually (operator step)**

Run: `./start.sh`

In a browser:
1. Set `PUBLISH_ENABLED=true` in `.env` first, then restart.
2. Open `/job/<existing-done-job>/result`.
3. Confirm the "Publish to GitHub Pages" button renders below the meta line and above the steps.
4. (Don't click it yet — clicking would actually hit gh/git. The smoke script from Phase 2 Task 5 is the right way to exercise the full path.)

Run: `./stop.sh`

**Step 3: Commit**

```bash
git add static/css/main.css
git commit -m "Style the publish-controls fragment + htmx spinner"
```
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_TASK_5 -->
### Task 5: Verify Phase 3 end-to-end

**Files:** none (verification only).

**Step 1: Full test suite**

Run: `python -m pytest`
Expected: All offline tests pass. Specifically:
- `tests/pipeline/test_publish.py` (Phase 1 tests)
- `tests/pipeline/test_publish_types.py` (Phase 1/2 tests)
- `tests/pipeline/test_storage.py` (Phase 2 tests)
- `tests/pipeline/test_publish_repo.py` (Phase 2 tests)
- `tests/test_config.py` (Phase 2 tests)
- `tests/test_server_publish.py` (Phase 3 tests)
- All pre-existing tests

**Step 2: Live UI check (operator)**

With `PUBLISH_ENABLED=true` and a real `gh auth login`, run the smoke script from Phase 2 Task 5 instead of clicking buttons in the UI — it's deterministic and easier to clean up.

**Step 3: Confirm no regression in the non-publish flow**

With `PUBLISH_ENABLED=false`, open an existing job result page. Verify:
- No publish controls render.
- Existing meta + steps + frame grid render as before.
<!-- END_TASK_5 -->
