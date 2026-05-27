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
