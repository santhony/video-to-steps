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
            try:
                for rel, src in bundle.file_map.items():
                    dest = job_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(src, dest)
            except (OSError, FileNotFoundError) as e:
                raise PublishError(f"copying bundle file {rel}: {e}") from e

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
        # gh derives the default branch from `git config init.defaultBranch`.
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
        # If the clone dir exists but lacks .git (interrupted prior clone),
        # remove it so gh repo clone succeeds.
        if self._clone_dir.exists() and not (self._clone_dir / ".git").exists():
            shutil.rmtree(self._clone_dir, ignore_errors=True)
        # Force HTTPS clone so gh's HTTPS token credential helper handles
        # auth — `gh repo clone <repo>` honors the user's git_protocol pref
        # (often ssh), which can fail if no SSH key is registered on the
        # newly-created repo.
        clone_url = f"https://github.com/{self._repo}.git"
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", clone_url, str(self._clone_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise PublishError(f"git clone failed: {err.decode(errors='replace')}")

    async def _git(self, *args: str) -> None:
        """Run `git <args>` inside the clone dir. Raises PublishError on non-zero exit."""
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(self._clone_dir), *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
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
