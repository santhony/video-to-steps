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
