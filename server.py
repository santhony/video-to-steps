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

Pattern: Imperative Shell
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
