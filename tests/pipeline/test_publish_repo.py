"""Tests for pipeline.publish_repo.

These tests verify the shell logic by stubbing subprocess execution.
The end-to-end smoke that actually hits gh + GitHub lives in
`scripts/smoke_publish.py` and is guarded by RUN_PUBLISH_SMOKE=1.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.publish_repo import PublishRepo, build_publish_repo
from pipeline.types import PublishError, StaticBundle


@pytest.fixture
def settings(tmp_path):
    return SimpleNamespace(
        publish_repo="testuser/testrepo",
        publish_branch="main",
        publish_base_url="https://testuser.github.io/testrepo",
        publish_clone_dir=tmp_path / "publish_repo",
        publish_enabled=True,
    )


@pytest.fixture
def bundle():
    return StaticBundle(
        html="<html><body>hi</body></html>",
        file_map={
            "main.css": Path("/nonexistent/main.css"),
            "frames/0001.jpg": Path("/nonexistent/0001.jpg"),
        },
    )


def _ok_proc(stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    """Return a fake `Process` whose `.communicate()` resolves to (stdout, stderr) and returncode 0."""
    proc = MagicMock()
    proc.returncode = 0
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _fail_proc(stderr: bytes = b"boom") -> MagicMock:
    proc = MagicMock()
    proc.returncode = 1
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    return proc


@pytest.mark.asyncio
async def test_build_publish_repo_returns_instance(settings):
    pr = build_publish_repo(settings)
    assert isinstance(pr, PublishRepo)


@pytest.mark.asyncio
async def test_ensure_ready_creates_repo_when_missing(settings, monkeypatch):
    """No local clone yet → ensure_ready probes gh, creates repo, enables Pages, clones."""
    pr = build_publish_repo(settings)

    calls: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        # gh repo view → fail (repo missing); everything else → succeed.
        # Also: when the fake `gh repo clone` runs, materialize a `.git` dir so
        # any subsequent ensure_ready calls in the same test would take the
        # already-cloned branch.
        if args[:3] == ("gh", "repo", "view"):
            return _fail_proc(b"not found")
        if args[:2] == ("git", "clone"):
            clone_target = Path(args[3])
            (clone_target / ".git").mkdir(parents=True, exist_ok=True)
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    # Note: do NOT pre-create the clone dir. ensure_ready's first branch
    # ("no .git present yet") must run for this test to be meaningful.
    await pr.ensure_ready()

    cmds = [" ".join(c) for c in calls]
    assert any("gh repo view" in c for c in cmds)
    assert any("gh repo create" in c for c in cmds)
    assert any("gh api" in c and "pages" in c for c in cmds)
    assert any(c.startswith("git clone https://github.com/") for c in cmds)


@pytest.mark.asyncio
async def test_ensure_ready_fetches_when_clone_exists(settings, monkeypatch):
    """A local clone already exists → ensure_ready fetches + hard-resets, no gh calls."""
    pr = build_publish_repo(settings)
    settings.publish_clone_dir.mkdir(parents=True, exist_ok=True)
    (settings.publish_clone_dir / ".git").mkdir()

    calls: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    await pr.ensure_ready()

    cmds = [" ".join(c) for c in calls]
    assert not any(c.startswith("gh ") for c in cmds)
    assert any("git" in c and "fetch" in c for c in cmds)
    assert any("git" in c and "reset" in c and "--hard" in c for c in cmds)


@pytest.mark.asyncio
async def test_publish_job_writes_bundle_and_pushes(settings, bundle, monkeypatch, tmp_path):
    """publish_job copies bundle files into the clone and runs git add/commit/push."""
    pr = build_publish_repo(settings)

    # Pre-create clone dir as if ensure_ready ran
    settings.publish_clone_dir.mkdir(parents=True, exist_ok=True)
    (settings.publish_clone_dir / ".git").mkdir()

    # Make the bundle file_map point at real files we create on disk so
    # the shutil.copy doesn't blow up.
    src_css = tmp_path / "main.css"
    src_css.write_text("/* css */")
    src_frame = tmp_path / "0001.jpg"
    src_frame.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    bundle.file_map["main.css"] = src_css
    bundle.file_map["frames/0001.jpg"] = src_frame

    calls: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    # Skip ensure_ready's gh probing in this test by stubbing it
    pr.ensure_ready = AsyncMock()

    url = await pr.publish_job("dQw4w9WgXcQ_a1b2c3", bundle)

    assert url == "https://testuser.github.io/testrepo/dQw4w9WgXcQ_a1b2c3/"
    # Bundle was written to the clone
    job_dir = settings.publish_clone_dir / "dQw4w9WgXcQ_a1b2c3"
    assert (job_dir / "index.html").read_text() == "<html><body>hi</body></html>"
    assert (job_dir / "main.css").read_text() == "/* css */"
    assert (job_dir / "frames" / "0001.jpg").exists()
    # git add/commit/push were invoked
    cmds = [" ".join(c) for c in calls if c and c[0] == "git"]
    assert any("add" in c for c in cmds)
    assert any("commit" in c for c in cmds)
    assert any("push" in c for c in cmds)


@pytest.mark.asyncio
async def test_publish_job_raises_publish_error_on_push_failure(settings, bundle, monkeypatch, tmp_path):
    pr = build_publish_repo(settings)
    settings.publish_clone_dir.mkdir(parents=True, exist_ok=True)
    (settings.publish_clone_dir / ".git").mkdir()
    src_css = tmp_path / "main.css"; src_css.write_text("c")
    src_frame = tmp_path / "0001.jpg"; src_frame.write_bytes(b"j")
    bundle.file_map["main.css"] = src_css
    bundle.file_map["frames/0001.jpg"] = src_frame

    async def fake_exec(*args, **kwargs):
        # args = ("git", "-C", clone_dir, subcommand, ...)
        if args[0] == "git" and "push" in args:
            return _fail_proc(b"rejected: non-fast-forward")
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    pr.ensure_ready = AsyncMock()

    with pytest.raises(PublishError) as exc:
        await pr.publish_job("dQw4w9WgXcQ_a1b2c3", bundle)
    assert "rejected" in str(exc.value)


@pytest.mark.asyncio
async def test_unpublish_job_removes_dir_and_pushes(settings, monkeypatch):
    pr = build_publish_repo(settings)
    settings.publish_clone_dir.mkdir(parents=True, exist_ok=True)
    (settings.publish_clone_dir / ".git").mkdir()
    (settings.publish_clone_dir / "dQw4w9WgXcQ_a1b2c3").mkdir()

    calls: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    pr.ensure_ready = AsyncMock()

    await pr.unpublish_job("dQw4w9WgXcQ_a1b2c3")

    cmds = [" ".join(c) for c in calls if c and c[0] == "git"]
    assert any("rm" in c and "-r" in c for c in cmds)
    assert any("commit" in c for c in cmds)
    assert any("push" in c for c in cmds)


@pytest.mark.asyncio
async def test_unpublish_job_noop_when_dir_absent(settings, monkeypatch):
    """Unpublishing an already-absent job should not call git rm."""
    pr = build_publish_repo(settings)
    settings.publish_clone_dir.mkdir(parents=True, exist_ok=True)
    (settings.publish_clone_dir / ".git").mkdir()
    # Note: no job_id subdir created

    calls: list[list[str]] = []

    async def fake_exec(*args, **kwargs):
        calls.append(list(args))
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    pr.ensure_ready = AsyncMock()

    await pr.unpublish_job("dQw4w9WgXcQ_a1b2c3")

    cmds = [" ".join(c) for c in calls if c]
    assert not any(c.startswith("git") and "rm" in c for c in cmds)


@pytest.mark.asyncio
async def test_publish_job_raises_publish_error_on_missing_bundle_file(settings, bundle, monkeypatch, tmp_path):
    """If a bundle file_map points to a nonexistent src path, PublishError is raised."""
    pr = build_publish_repo(settings)
    settings.publish_clone_dir.mkdir(parents=True, exist_ok=True)
    (settings.publish_clone_dir / ".git").mkdir()

    # Create one file but leave another missing
    src_css = tmp_path / "main.css"
    src_css.write_text("/* css */")
    bundle.file_map["main.css"] = src_css
    bundle.file_map["frames/0001.jpg"] = Path("/nonexistent/file/path.jpg")

    async def fake_exec(*args, **kwargs):
        return _ok_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    pr.ensure_ready = AsyncMock()

    with pytest.raises(PublishError) as exc:
        await pr.publish_job("dQw4w9WgXcQ_a1b2c3", bundle)
    assert "copying bundle file" in str(exc.value)
    assert "frames/0001.jpg" in str(exc.value)
