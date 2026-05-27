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

pattern: Imperative Shell
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import base64
import secrets

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

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

# Two accepted shapes:
#   1. `<11-char video id>_<6 hex>` — new format (since v1.2.1) so the URL
#      reflects the source video without forcing operators to dig into
#      the manifest to know which video a job pertains to.
#   2. `<12 hex>` — legacy format from v1.0/1.1; preserved so jobs already
#      on disk and their URLs remain valid after this change.
_JOB_ID_RE = re.compile(r"^(?:[A-Za-z0-9_-]{11}_[a-f0-9]{6}|[a-f0-9]{12})$")


def _guard_job_id(job_id: str) -> None:
    """Validate job_id format. Raises HTTPException(400) if invalid."""
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(400, "Bad job id.")


def _is_valid_youtube_url(url: str) -> bool:
    return _YT_RE.search(url) is not None


def _normalize_url(raw: str) -> str:
    """Trim whitespace and prepend `https://` if the user pasted a URL
    without a scheme (e.g. `youtube.com/watch?v=…` or `youtu.be/…`).

    Leaves explicit `http://` and `https://` URLs alone, including the
    rare case of `http://` (some old YouTube embeds). Anything with a
    different scheme (`ftp://`, `file://`, etc.) is not modified — it
    will fail _YT_RE downstream and surface a 400 from /process.
    """
    s = raw.strip()
    if not s:
        return s
    lower = s.lower()
    if lower.startswith(("http://", "https://")):
        return s
    # Heuristic: if the user pasted something with a different scheme
    # we leave it alone (it won't be a YouTube URL anyway). The check
    # is "://" appears in the first 10 chars" — schemes are short.
    if "://" in s[:10]:
        return s
    return "https://" + s


def _video_deep_link(manifest_url: str, t_sec: float) -> str | None:
    """Build a canonical YouTube deep link to the given timestamp.

    Returns None when the manifest URL doesn't contain a recognizable
    11-char video id (defensive against malformed or non-YouTube inputs
    that somehow bypassed `_is_valid_youtube_url`).
    """
    m = _YT_RE.search(manifest_url)
    if m is None:
        return None
    video_id = m.group(1)
    # YouTube's `t=` accepts integer seconds; decimals are silently
    # truncated. Negative times are clamped to 0.
    t = max(0, int(t_sec))
    return f"https://www.youtube.com/watch?v={video_id}&t={t}s"


def _new_job_id(video_id: str | None = None) -> str:
    """Allocate a job id. When `video_id` is the 11-char YouTube id, the
    returned id is `<video_id>_<6 hex>` so the URL reflects the source
    video. Falls back to plain 12-hex when no video id is supplied."""
    if video_id:
        return f"{video_id}_{uuid.uuid4().hex[:6]}"
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


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """HTTP Basic auth gate for the whole app.

    Active only when both APP_BASIC_AUTH_USER and APP_BASIC_AUTH_PASS are set
    in the environment — the local-dev default (both empty) makes this a
    no-op. Intended for cloud deployments (Hugging Face Spaces) where the
    public URL would otherwise expose a paid OpenAI key via /process.
    Credentials are compared with secrets.compare_digest to avoid leaking
    timing information; failures return 401 with WWW-Authenticate so the
    browser shows its native prompt.
    """

    def __init__(self, app, username: str, password: str) -> None:
        super().__init__(app)
        self._user = username
        self._pass = password

    async def dispatch(self, request: Request, call_next):
        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8", errors="replace")
                user, _, pw = decoded.partition(":")
                if secrets.compare_digest(user, self._user) and secrets.compare_digest(pw, self._pass):
                    return await call_next(request)
            except Exception:
                pass
        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="video-to-steps"'},
        )


_auth_user = (os.getenv("APP_BASIC_AUTH_USER") or "").strip()
_auth_pass = os.getenv("APP_BASIC_AUTH_PASS") or ""
if _auth_user and _auth_pass:
    app.add_middleware(BasicAuthMiddleware, username=_auth_user, password=_auth_pass)


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    # vts-v1.AC8.1
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
) -> RedirectResponse:
    # Normalize first so "youtu.be/ABC..." and "  www.youtube.com/watch?v=…  "
    # pasted without a scheme reach the validator with `https://` prepended.
    url = _normalize_url(url)
    if not _is_valid_youtube_url(url):
        # vts-v1.AC8.3
        raise HTTPException(
            status_code=400,
            detail="Not a recognized YouTube URL. Provide a youtube.com/watch?v=… or youtu.be/… link.",
        )

    settings = get_settings()
    jobs_root = Path(settings.jobs_root)
    # Extract the YouTube video id (we already know URL matches _YT_RE)
    # so the job id — and therefore the URL — surfaces it.
    yt_match = _YT_RE.search(url)
    video_id = yt_match.group(1) if yt_match else None
    job_id = _new_job_id(video_id=video_id)
    ensure_job_dir(jobs_root, job_id)

    initial = Manifest(job_id=job_id, url=url, status="queued", progress="queued")
    write_json_atomic(jobs_root / job_id / "meta.json", initial)

    # vts-v1.AC8.2 — spawn pipeline task.
    background_tasks.add_task(run_job, job_id, url, settings, jobs_root)

    return RedirectResponse(url=f"/job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str) -> HTMLResponse:
    _guard_job_id(job_id)
    settings = get_settings()
    m = _load_manifest_dict(Path(settings.jobs_root), job_id)
    if m is None:
        raise HTTPException(404, "Unknown job id.")
    # Pass the title (if any) so the page header shows the human-readable
    # video name even before the first 2s HTMX poll fires.
    return templates.TemplateResponse(
        request,
        "job.html",
        {"job_id": job_id, "title": m.get("title", "") or ""},
    )


@app.get("/job/{job_id}/status", response_class=HTMLResponse)
async def job_status(request: Request, job_id: str) -> HTMLResponse:
    _guard_job_id(job_id)
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
    _guard_job_id(job_id)
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
    # Per-step deep links to the source video at the step's start time.
    step_links = [_video_deep_link(m.get("url", ""), s.get("start", 0)) for s in steps]
    # vts-v1.AC8.5
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


@app.get("/job/{job_id}/frame/{name}.jpg")
async def job_frame(job_id: str, name: str) -> FileResponse:
    _guard_job_id(job_id)
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
