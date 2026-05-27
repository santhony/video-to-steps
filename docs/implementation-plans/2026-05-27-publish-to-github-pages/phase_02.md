# Publish to GitHub Pages — Phase 2: Imperative Shell + Manifest Fields

**Goal:** Add `published_url`/`published_at` to `Manifest`, the `pipeline/publish_repo.py` shell that drives `gh`/`git` subprocesses, settings fields, and a smoke script.

**Architecture:** Imperative Shell. All `git`/`gh` subprocess calls, local-clone state, asyncio serialization. Pure bundling is delegated to `pipeline.publish` (Phase 1). Factory `build_publish_repo(settings)` mirrors the existing `build_llm` / `build_embedder` / `build_whisper` shape.

**Tech Stack:** Python 3.11, `asyncio.create_subprocess_exec`, `asyncio.Lock`, `gh` CLI (host-installed), `git` (host-installed), `pydantic-settings`.

**Scope:** Phase 2 of 4 from `docs/design-plans/2026-05-27-publish-to-github-pages.md`.

**Codebase verified:** 2026-05-27

---

## Reference: verified codebase state

- `Manifest` is at `pipeline/types.py:67-82`. Two new optional fields (`published_url: str | None = None`, `published_at: datetime | None = None`) append cleanly after `cost`.
- **Important gap surfaced by investigator:** `pipeline/storage._to_jsonable` (lines 34-46) does **not** handle `datetime`. `json.dump` will raise on a `datetime` value. Task 1 below adds a `datetime` branch.
- Settings live in `config.py:15-78`; new fields go after `whisper_model` (line 73) and before `get_settings()` (line 76).
- Factory pattern: `providers/llm.py:152` `build_llm(settings: Any) -> LLMClient`; mirror this signature exactly.
- Smoke-script style: `scripts/smoke_phase3.py`. Guarded by an env-var presence check.
- `gh` CLI: `gh repo view <owner/repo>` returns non-zero (exit 1) when the repo doesn't exist; we use that to detect "needs to be created". `gh repo create <repo> --public --add-readme` creates it. `gh api -X POST repos/<repo>/pages -f source[branch]=main -f source[path]=/` enables Pages (returns 422 with "already exists" body if already enabled — we treat any "already" error as success).

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Teach `_to_jsonable` to serialize `datetime`

**Files:**
- Modify: `pipeline/storage.py` (insert before line 46, the final `return value`)
- Modify/create: `tests/pipeline/test_storage.py` (add `datetime` round-trip test; create file if absent)

**Step 1: Write the failing test**

Create or append to `tests/pipeline/test_storage.py`:

```python
"""Round-trip tests for pipeline.storage helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.storage import read_json, write_json_atomic


def test_write_json_atomic_serializes_datetime(tmp_path: Path):
    """A dict with a datetime value must serialize as ISO-8601."""
    ts = datetime(2026, 5, 27, 12, 34, 56, tzinfo=timezone.utc)
    target = tmp_path / "meta.json"

    write_json_atomic(target, {"published_at": ts, "published_url": None})

    loaded = read_json(target)
    assert loaded["published_url"] is None
    assert loaded["published_at"] == "2026-05-27T12:34:56+00:00"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/santhony/Documents/dev_claude/video-to-steps/.worktrees/publish-to-github-pages && python -m pytest tests/pipeline/test_storage.py::test_write_json_atomic_serializes_datetime -v`
Expected: FAIL with `TypeError: Object of type datetime is not JSON serializable`.

**Step 3: Patch `pipeline/storage.py`**

At the top of the file, add a `datetime` import (modify the existing imports near line 16):

```python
from datetime import datetime
```

In `_to_jsonable`, insert this branch *before* the final `return value` (current line 46):

```python
    if isinstance(value, datetime):
        return value.isoformat()
```

The function ends up looking like:

```python
def _to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, Paths, sets, and datetimes to JSON-safe shapes."""
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_storage.py -v`
Expected: PASS.

Run the full suite to confirm no regressions: `python -m pytest`
Expected: All pre-existing tests still pass.

**Step 5: Commit**

```bash
git add pipeline/storage.py tests/pipeline/test_storage.py
git commit -m "Serialize datetime as ISO-8601 in _to_jsonable"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add `published_url`/`published_at` to `Manifest`

**Files:**
- Modify: `pipeline/types.py` (extend imports + append fields to `Manifest`)
- Modify: `tests/pipeline/test_publish_types.py` (add round-trip test through `write_json_atomic`)

**Step 1: Write the failing test**

Append to `tests/pipeline/test_publish_types.py`:

```python


def test_manifest_publish_fields_default_to_none():
    """Existing manifests without these fields must continue to construct."""
    from pipeline.types import Manifest

    m = Manifest(job_id="abc", url="https://example.com")
    assert m.published_url is None
    assert m.published_at is None


def test_manifest_publish_fields_round_trip(tmp_path):
    """A manifest with a datetime survives write+read via the storage helpers."""
    from datetime import datetime, timezone
    from pipeline.storage import read_json, write_json_atomic
    from pipeline.types import Manifest

    m = Manifest(
        job_id="abc",
        url="https://example.com",
        published_url="https://santhony.github.io/vts-publish/abc/",
        published_at=datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    p = tmp_path / "meta.json"
    write_json_atomic(p, m)
    loaded = read_json(p)
    assert loaded["published_url"] == "https://santhony.github.io/vts-publish/abc/"
    assert loaded["published_at"] == "2026-05-27T12:00:00+00:00"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_publish_types.py::test_manifest_publish_fields_default_to_none -v`
Expected: FAIL — `AttributeError: 'Manifest' object has no attribute 'published_url'`.

**Step 3: Modify `pipeline/types.py`**

Update the top imports block (lines 8-12) to add `datetime`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
```

Append two fields to `Manifest` (after the `cost` line, currently line 81):

```python
    published_url: str | None = None   # set when the result page is live on GitHub Pages
    published_at: datetime | None = None
```

The full `Manifest` should now look like:

```python
@dataclass(slots=True)
class Manifest:
    """Per-job record persisted to meta.json.

    Mutated by the orchestrator (`pipeline.pipeline._update`) and by the
    server in two narrow carve-outs: the initial `queued` write in
    `/process`, and the publish-state updates in `/job/{id}/publish`
    and `/job/{id}/unpublish`. All writes go through `write_json_atomic`.
    """
    job_id: str
    url: str
    title: str = ""
    status: str = "queued"
    progress: str = ""
    error: str = ""
    mode: str = ""
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    published_url: str | None = None
    published_at: datetime | None = None
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_publish_types.py -v`
Expected: 5 tests pass.

Run: `python -m pytest`
Expected: All offline tests pass.

**Step 5: Commit**

```bash
git add pipeline/types.py tests/pipeline/test_publish_types.py
git commit -m "Add published_url and published_at to Manifest"
```
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->

<!-- START_TASK_3 -->
### Task 3: Add publish settings to `config.py` and `.env.example`

**Files:**
- Modify: `config.py` (insert after line 73, the `whisper_model` field)
- Modify: `.env.example` (append publish section)

**Step 1: Write the failing test**

Create or append to `tests/test_config.py`:

```python
"""Settings field smoke tests for publish-related env vars."""

from __future__ import annotations

from pathlib import Path


def test_settings_have_publish_defaults(monkeypatch):
    # Make sure no .env in CWD leaks values into the test
    monkeypatch.delenv("PUBLISH_REPO", raising=False)
    monkeypatch.delenv("PUBLISH_BRANCH", raising=False)
    monkeypatch.delenv("PUBLISH_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLISH_CLONE_DIR", raising=False)
    monkeypatch.delenv("PUBLISH_ENABLED", raising=False)

    from config import Settings

    s = Settings(_env_file=None)
    assert s.publish_repo == "santhony/vts-publish"
    assert s.publish_branch == "main"
    assert s.publish_base_url == "https://santhony.github.io/vts-publish"
    assert s.publish_clone_dir == Path("data/publish_repo")
    assert s.publish_enabled is False


def test_settings_publish_repo_override(monkeypatch):
    monkeypatch.setenv("PUBLISH_REPO", "someone/elsewhere")
    monkeypatch.setenv("PUBLISH_ENABLED", "true")
    from config import Settings

    s = Settings(_env_file=None)
    assert s.publish_repo == "someone/elsewhere"
    assert s.publish_enabled is True
```

If `tests/test_config.py` doesn't exist yet, create it with just the imports + these tests.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — fields don't exist.

**Step 3: Edit `config.py`**

Insert below the `whisper_model` field (current line 73, before line 76 `def get_settings`):

```python

    # Publish to GitHub Pages
    publish_repo: str = Field(default="santhony/vts-publish", alias="PUBLISH_REPO")
    publish_branch: str = Field(default="main", alias="PUBLISH_BRANCH")
    publish_base_url: str = Field(
        default="https://santhony.github.io/vts-publish",
        alias="PUBLISH_BASE_URL",
    )
    publish_clone_dir: Path = Field(
        default=Path("data/publish_repo"), alias="PUBLISH_CLONE_DIR"
    )
    publish_enabled: bool = Field(default=False, alias="PUBLISH_ENABLED")
```

**Step 4: Edit `.env.example`**

Append to the end of the file:

```dotenv

# Publish to GitHub Pages
# Requires `gh auth login` with `repo` scope on the host.
# Enabling renders a "Publish" button on the result page. The first publish
# auto-creates `PUBLISH_REPO` (public) and enables Pages on `PUBLISH_BRANCH`.
PUBLISH_ENABLED=false
PUBLISH_REPO=santhony/vts-publish
PUBLISH_BRANCH=main
PUBLISH_BASE_URL=https://santhony.github.io/vts-publish
PUBLISH_CLONE_DIR=data/publish_repo
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add config.py .env.example tests/test_config.py
git commit -m "Add publish settings to config + .env.example"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Implement `pipeline/publish_repo.py` with `PublishRepo` + `build_publish_repo`

**Files:**
- Create: `pipeline/publish_repo.py`
- Create: `tests/pipeline/test_publish_repo.py`

**Step 1: Write the failing test**

Create `tests/pipeline/test_publish_repo.py`:

```python
"""Tests for pipeline.publish_repo.

These tests verify the shell logic by stubbing subprocess execution.
The end-to-end smoke that actually hits gh + GitHub lives in
`scripts/smoke_publish.py` and is guarded by RUN_PUBLISH_SMOKE=1.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
        if args[:3] == ("gh", "repo", "clone"):
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
    assert any("gh repo clone" in c for c in cmds)


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
    cmds = [" ".join(c[:3]) for c in calls if c and c[0] == "git"]
    assert any("git add" in c for c in cmds)
    assert any("git commit" in c for c in cmds)
    assert any("git push" in c for c in cmds)


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
        if args[0] == "git" and args[1] == "push":
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
    assert any("git rm -r" in c for c in cmds)
    assert any("git commit" in c for c in cmds)
    assert any("git push" in c for c in cmds)


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
    assert not any("git rm" in c for c in cmds)
```

Tests are async — confirm `pytest-asyncio` is already installed (it is — `tests/test_server.py` uses `@pytest.mark.asyncio`). If your test file doesn't auto-discover the asyncio marker, add to the top: `pytestmark = pytest.mark.asyncio`.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_publish_repo.py -v`
Expected: ImportError on `pipeline.publish_repo`.

**Step 3: Create `pipeline/publish_repo.py`**

```python
"""Push the static bundle for a job to a shared GitHub Pages repo.

pattern: Imperative Shell
Owns the local clone of the publish repo (`settings.publish_clone_dir`),
the asyncio lock that serializes concurrent publishes on a single
process, and all `gh`/`git` subprocess calls. The pure rendering and
file-map composition lives in `pipeline.publish`.

Operator prerequisites:
- `gh` CLI installed and authenticated (`gh auth login`) with `repo` scope.
- `git` configured (user.name + user.email) so commits don't fail.
- Network reachable to github.com.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from pipeline.types import PublishError, StaticBundle


class PublishRepo:
    """Drives publish/unpublish against a single shared GitHub Pages repo."""

    def __init__(
        self,
        *,
        publish_repo: str,
        publish_branch: str,
        publish_base_url: str,
        publish_clone_dir: Path,
    ) -> None:
        self._repo = publish_repo                # "owner/name"
        self._branch = publish_branch
        self._base_url = publish_base_url.rstrip("/")
        self._clone_dir = Path(publish_clone_dir)
        self._lock = asyncio.Lock()

    # ---- public API -------------------------------------------------

    async def ensure_ready(self) -> None:
        """Make the local clone usable, creating the remote repo if needed.

        Idempotent — safe to call before every publish.
        """
        if not (self._clone_dir / ".git").exists():
            # New host or clone dir wiped. Confirm the remote exists (or create it),
            # enable Pages, then clone.
            await self._ensure_remote_exists()
            await self._ensure_pages_enabled()
            await self._clone()
        else:
            # Existing clone — sync with origin so the next push doesn't reject.
            await self._git("fetch", "origin", self._branch)
            await self._git("reset", "--hard", f"origin/{self._branch}")

    async def publish_job(self, job_id: str, bundle: StaticBundle) -> str:
        """Write `bundle` into `<clone>/<job_id>/`, commit, push. Returns the public URL."""
        async with self._lock:
            await self.ensure_ready()

            job_dir = self._clone_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
            job_dir.mkdir(parents=True)

            (job_dir / "index.html").write_text(bundle.html, encoding="utf-8")
            for rel, src in bundle.file_map.items():
                dest = job_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)

            await self._git("add", f"{job_id}/")
            await self._git("commit", "-m", f"publish {job_id}")
            await self._git("push", "origin", self._branch)

            return f"{self._base_url}/{job_id}/"

    async def unpublish_job(self, job_id: str) -> None:
        """Remove `<job_id>/` from the publish repo, commit, push."""
        async with self._lock:
            await self.ensure_ready()

            job_dir = self._clone_dir / job_id
            if not job_dir.exists():
                return  # already absent; manifest will still be cleared by caller

            await self._git("rm", "-r", f"{job_id}/")
            await self._git("commit", "-m", f"unpublish {job_id}")
            await self._git("push", "origin", self._branch)

    # ---- internals --------------------------------------------------

    async def _ensure_remote_exists(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "gh", "repo", "view", self._repo,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode == 0:
            return  # exists

        # Create it. --add-readme so the first push has a parent commit on the branch.
        create = await asyncio.create_subprocess_exec(
            "gh", "repo", "create", self._repo, "--public", "--add-readme",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await create.communicate()
        if create.returncode != 0:
            raise PublishError(f"gh repo create failed: {err.decode(errors='replace')}")

    async def _ensure_pages_enabled(self) -> None:
        """Enable Pages on `_branch`/'/'. 422 'already exists' is treated as success."""
        proc = await asyncio.create_subprocess_exec(
            "gh", "api", "-X", "POST", f"repos/{self._repo}/pages",
            "-f", f"source[branch]={self._branch}",
            "-f", "source[path]=/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode == 0:
            return
        text = err.decode(errors="replace")
        # GitHub returns 422 with a body containing "already exists" when Pages
        # is already on. Anything else is a real failure.
        if "already" in text.lower():
            return
        raise PublishError(f"enabling Pages failed: {text}")

    async def _clone(self) -> None:
        self._clone_dir.parent.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_exec(
            "gh", "repo", "clone", self._repo, str(self._clone_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise PublishError(f"gh repo clone failed: {err.decode(errors='replace')}")

    async def _git(self, *args: str) -> None:
        """Run `git <args>` inside the clone dir. Raises PublishError on non-zero exit."""
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(self._clone_dir), *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise PublishError(
                f"git {' '.join(args)} failed: {err.decode(errors='replace').strip()}"
            )


def build_publish_repo(settings: Any) -> PublishRepo:
    """Factory mirroring build_llm / build_embedder / build_whisper."""
    return PublishRepo(
        publish_repo=settings.publish_repo,
        publish_branch=settings.publish_branch,
        publish_base_url=settings.publish_base_url,
        publish_clone_dir=Path(settings.publish_clone_dir),
    )
```

**Note about the test stubs:** The tests in Step 1 monkeypatch `asyncio.create_subprocess_exec` to a `fake_exec` and rely on it being called with positional `args`. The implementation above always uses positional args for the command and its flags, so the test passes through cleanly. The `git -C <clone_dir>` form means tests can inspect `args` starting at index `0='git'`, index `1='-C'`, index `2=<clone_dir>`, and the actual subcommand at index `3`. Adjust the test assertions if you change the form — current tests check for substrings via `" ".join(c)` which handles either form.

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_publish_repo.py -v`
Expected: 7 passed (the two `test_ensure_ready_*` tests, `test_build_publish_repo_returns_instance`, two `test_publish_job_*` tests, two `test_unpublish_job_*` tests).

If any "git X" assertions fail because tests checked the wrong index, update them to use `any(...substring... in " ".join(call) for call in calls)` — the substring approach is forgiving of positional differences.

Run full suite: `python -m pytest`
Expected: All offline tests pass.

**Step 5: Commit**

```bash
git add pipeline/publish_repo.py tests/pipeline/test_publish_repo.py
git commit -m "Add pipeline.publish_repo with PublishRepo + build_publish_repo"
```
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_TASK_5 -->
### Task 5: Smoke script `scripts/smoke_publish.py`

**Files:**
- Create: `scripts/smoke_publish.py`

This is an operator diagnostic that actually pushes to the configured `PUBLISH_REPO`. It is guarded by `RUN_PUBLISH_SMOKE=1` so accidentally running `python -m scripts.smoke_publish` does nothing.

**Step 1: Create the script**

```python
"""End-to-end smoke for the publish path.

Bundles a synthetic single-step "job" and pushes it to the configured
PUBLISH_REPO under a `smoke-<timestamp>/` slug, then waits for the user
to hit Enter and unpublishes it.

Run:
    RUN_PUBLISH_SMOKE=1 python -m scripts.smoke_publish

Requires:
    - PUBLISH_ENABLED=1
    - gh auth login (with `repo` scope)
    - A writable PUBLISH_CLONE_DIR
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config import get_settings
from pipeline.publish import build_static_bundle
from pipeline.publish_repo import build_publish_repo
from pipeline.types import CostBreakdown, Frame, Manifest, Step


REPO_ROOT = Path(__file__).resolve().parents[1]


async def _main() -> int:
    if os.getenv("RUN_PUBLISH_SMOKE") != "1":
        print("Refusing to run without RUN_PUBLISH_SMOKE=1.", file=sys.stderr)
        return 2

    settings = get_settings()
    if not settings.publish_enabled:
        print("PUBLISH_ENABLED is false. Set it to true in .env and retry.", file=sys.stderr)
        return 2

    env = Environment(
        loader=FileSystemLoader(str(REPO_ROOT / "templates")),
        autoescape=True,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"smoke-{ts}"

    # Synthetic manifest + a single step pointing at a single dummy frame.
    with tempfile.TemporaryDirectory() as td:
        frames_dir = Path(td) / "frames"
        frames_dir.mkdir()
        # Tiny JPEG header bytes — Pages will render the missing image
        # as a broken icon, which is fine for smoke purposes.
        (frames_dir / "0001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

        manifest = Manifest(
            job_id=job_id,
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            title=f"Smoke publish {ts}",
            status="done",
            mode="cloud",
            config_snapshot={
                "embed_backend": "smoke",
                "llm_model": "smoke",
                "vision_model": "smoke",
            },
            cost=CostBreakdown(total_usd=0.0),
        )
        steps = [
            Step(
                index=0, start=0.0, end=5.0,
                instruction="If you can read this on github.io, the publish path works.",
                frames=[Frame(index=0, timestamp=1.0, path=frames_dir / "0001.jpg")],
            ),
        ]

        bundle = build_static_bundle(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None],
            frames_dir=frames_dir,
            css_path=REPO_ROOT / "static" / "css" / "main.css",
            templates_env=env,
        )

        pr = build_publish_repo(settings)
        url = await pr.publish_job(job_id, bundle)
        print(f"Published: {url}")
        print("Open the URL in a browser. When done, press Enter to unpublish.")
        try:
            input()
        except EOFError:
            pass
        await pr.unpublish_job(job_id)
        print(f"Unpublished {job_id}.")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
```

**Step 2: Smoke verification (operator step — optional in CI)**

This step requires `gh auth login` and a real network. It is not part of the test suite — it's a manual check the operator performs once.

```bash
# Confirm gh is authed
gh auth status

# Set the env in .env (or export inline)
export PUBLISH_ENABLED=true
RUN_PUBLISH_SMOKE=1 python -m scripts.smoke_publish
```

Expected: A `Published: https://<base>/smoke-<timestamp>/` line; opening that URL in a browser shows the rendered page within ~30 seconds; pressing Enter unpublishes and the URL goes 404 within ~30 seconds.

If `gh repo create` fails on a non-existent owner (e.g., the default `santhony/vts-publish` for a different operator), update `PUBLISH_REPO` to a writable destination first.

**Step 3: Commit**

```bash
git add scripts/smoke_publish.py
git commit -m "Add scripts.smoke_publish for end-to-end publish verification"
```
<!-- END_TASK_5 -->
