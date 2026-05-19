"""Tests for server.py — routes, validation, fragments."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from PIL import Image

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
    job_id = "ab12ab12ab12"
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
async def test_status_fragment_error_stops_polling(jobs_root):
    job_id = "cd34cd34cd34"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(job_id=job_id, url="https://youtu.be/x", status="error",
                 error="simulated failure", cost=CostBreakdown(total_usd=0.001))
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/status")
    assert r.status_code == 200
    # Error message must be visible.
    assert "Error" in r.text or "error" in r.text
    assert "simulated failure" in r.text
    # Polling must STOP for error status too.
    assert 'hx-trigger="every 2s"' not in r.text


@pytest.mark.asyncio
async def test_result_page_renders_steps_and_meta(jobs_root):
    # vts-v1.AC8.5
    job_id = "ef56ef56ef56"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(
        job_id=job_id, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
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
            {"index": 2, "start": 20.5, "end": 30.0,
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
    # Per-step deep links to the source video — open in new tab.
    # Jinja2 escapes & → &amp; in attribute values; browsers decode both.
    assert 'href="https://www.youtube.com/watch?v=dQw4w9WgXcQ&amp;t=0s"' in r.text
    assert 'href="https://www.youtube.com/watch?v=dQw4w9WgXcQ&amp;t=10s"' in r.text
    assert 'href="https://www.youtube.com/watch?v=dQw4w9WgXcQ&amp;t=20s"' in r.text  # 20.5 truncates to 20
    assert 'target="_blank"' in r.text
    assert 'rel="noopener noreferrer"' in r.text


@pytest.mark.asyncio
async def test_result_redirects_when_not_done(jobs_root):
    job_id = "ab78ab78ab78"
    ensure_job_dir(jobs_root, job_id)
    m = Manifest(job_id=job_id, url="https://youtu.be/x", status="running")
    write_json_atomic(jobs_root / job_id / "meta.json", m)
    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/result", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/job/{job_id}"


@pytest.mark.asyncio
async def test_frame_route_serves_jpeg(jobs_root):
    job_id = "cd56cd56cd56"
    ensure_job_dir(jobs_root, job_id)
    # Create a tiny JPEG image.
    frames_dir = jobs_root / job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (100, 100), color="red")
    img.save(frames_dir / "0001.jpg", "JPEG")

    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/frame/0001.jpg")
    assert r.status_code == 200
    assert "image/jpeg" in r.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_frame_route_rejects_bad_name(jobs_root):
    job_id = "ef78ef78ef78"
    ensure_job_dir(jobs_root, job_id)
    # Reject frame names that aren't exactly 4 digits.
    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/frame/abcd.jpg")
    assert r.status_code == 400

    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/frame/12345.jpg")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_frame_route_404_when_missing(jobs_root):
    job_id = "ab90ab90ab90"
    ensure_job_dir(jobs_root, job_id)
    # Create frames dir but no file.
    (jobs_root / job_id / "frames").mkdir(parents=True, exist_ok=True)

    async with await _client() as c:
        r = await c.get(f"/job/{job_id}/frame/9999.jpg")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_job_routes_reject_invalid_job_id(jobs_root):
    # All /job/{job_id}/* routes should reject bad job_id format.
    bad_job_id = "invalid-job-id"

    async with await _client() as c:
        r = await c.get(f"/job/{bad_job_id}/status")
    assert r.status_code == 400

    async with await _client() as c:
        r = await c.get(f"/job/{bad_job_id}/result")
    assert r.status_code == 400

    async with await _client() as c:
        r = await c.get(f"/job/{bad_job_id}/frame/0001.jpg")
    assert r.status_code == 400

    # Also test the job_page route.
    async with await _client() as c:
        r = await c.get(f"/job/{bad_job_id}")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_unknown_job_id_404(jobs_root):
    # Use a valid job_id format (hex) for a job that doesn't exist.
    async with await _client() as c:
        r = await c.get("/job/fedcfedcfedc/status")
    assert r.status_code == 404


def test_video_deep_link_for_watch_url():
    from server import _video_deep_link
    assert (_video_deep_link("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 42.7)
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s")


def test_video_deep_link_for_shortlink():
    from server import _video_deep_link
    assert (_video_deep_link("https://youtu.be/dQw4w9WgXcQ", 15)
            == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=15s")


def test_video_deep_link_clamps_negative_and_truncates_decimal():
    from server import _video_deep_link
    assert _video_deep_link("https://youtu.be/dQw4w9WgXcQ", -3.0).endswith("&t=0s")
    assert _video_deep_link("https://youtu.be/dQw4w9WgXcQ", 9.999).endswith("&t=9s")


def test_video_deep_link_returns_none_for_non_youtube():
    from server import _video_deep_link
    assert _video_deep_link("https://example.com/some-video", 10) is None
    assert _video_deep_link("", 10) is None
