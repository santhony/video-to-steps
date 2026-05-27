"""Integration tests for the publish UI + routes."""

from __future__ import annotations

import json
import re

import pytest
from httpx import ASGITransport, AsyncClient

import server
from pipeline.types import PublishError, StaticBundle


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


async def _client():
    transport = ASGITransport(app=server.app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_result_page_hides_publish_button_when_disabled(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "false")
    async with await _client() as c:
        r = await c.get(f"/job/{done_job}/result")
    assert r.status_code == 200
    assert "Publish to GitHub Pages" not in r.text


@pytest.mark.asyncio
async def test_result_page_shows_publish_button_when_enabled(jobs_root, done_job, monkeypatch):
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    async with await _client() as c:
        r = await c.get(f"/job/{done_job}/result")
    assert r.status_code == 200
    assert "Publish to GitHub Pages" in r.text
    assert f'hx-post="/job/{done_job}/publish"' in r.text


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

    async with await _client() as c:
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
    async with await _client() as c:
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

    async with await _client() as c:
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

    async with await _client() as c:
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
    monkeypatch.setenv("PUBLISH_ENABLED", "true")

    class FailingRepo(FakePublishRepo):
        async def publish_job(self, job_id, bundle):
            raise PublishError("git push failed: rejected")

    monkeypatch.setattr(server, "build_publish_repo", lambda settings: FailingRepo())

    async with await _client() as c:
        r = await c.post(f"/job/{done_job}/publish")

    assert r.status_code == 500
    # Manifest unchanged — both publish fields stay null
    meta = json.loads((jobs_root / done_job / "meta.json").read_text())
    assert meta["published_url"] is None
    assert meta["published_at"] is None


@pytest.mark.asyncio
async def test_unpublish_route_clears_manifest_for_older_job_without_publish_keys(jobs_root, monkeypatch):
    """Unpublishing a job with an older manifest (no published_* keys) should return 200 with cleared fields."""
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    job_id = "dQw4w9WgXcQ_a1b2c3"
    (jobs_root / job_id).mkdir()
    (jobs_root / job_id / "frames").mkdir()
    # Older manifest WITHOUT published_* keys
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
    }
    (jobs_root / job_id / "meta.json").write_text(json.dumps(meta))

    fake = FakePublishRepo()
    monkeypatch.setattr(server, "build_publish_repo", lambda settings: fake)

    async with await _client() as c:
        r = await c.post(f"/job/{job_id}/unpublish")

    assert r.status_code == 200
    assert fake.unpublished == [job_id]
    assert "Publish to GitHub Pages" in r.text
    meta_after = json.loads((jobs_root / job_id / "meta.json").read_text())
    assert meta_after.get("published_url") is None
    assert meta_after.get("published_at") is None
