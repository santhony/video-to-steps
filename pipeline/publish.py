"""Bundle a finished job's result page into a self-contained static snapshot.

pattern: Functional Core
Pure module: no disk writes, no network, no subprocesses. The caller
resolves all source paths (frames dir, CSS path) and the Jinja2
Environment. We return a `StaticBundle` describing what bytes to write
where; the publisher (`pipeline.publish_repo`, the Imperative Shell)
performs the actual copy + push.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment

from pipeline.types import Manifest, StaticBundle, Step


def render_static_html(
    manifest: Manifest,
    steps: list[Step],
    frame_captions: dict[str, str | None],
    step_links: list[str | None],
    templates_env: Environment,
) -> str:
    """Render `result.html` with `static_mode=True`.

    The base + result templates handle the conditional drop of the
    htmx tag, the rewrite of `/static/css/main.css` → `main.css`, the
    non-clickable brand, and the `/job/<id>/frame/...` → `frames/...`
    image-path rewrite. We just pass the flag.
    """
    template = templates_env.get_template("result.html")
    return template.render(
        manifest=manifest,
        steps=steps,
        frame_captions=frame_captions,
        step_links=step_links,
        static_mode=True,
        publish_enabled=False,  # defense-in-depth: snapshot must never render publish controls
    )


def _winner_frame_indices(steps: list[Step]) -> list[int]:
    """Return the sorted, de-duplicated list of frame indices referenced by any step."""
    seen: set[int] = set()
    for step in steps:
        for frame in step.frames:
            seen.add(frame.index)
    return sorted(seen)


def build_static_bundle(
    manifest: Manifest,
    steps: list[Step],
    frame_captions: dict[str, str | None],
    step_links: list[str | None],
    frames_dir: Path,
    css_path: Path,
    templates_env: Environment,
) -> StaticBundle:
    """Compose the static bundle for a job.

    `frames_dir` is the on-disk dir containing the original 720p frames
    (`data/jobs/<id>/frames/`). `css_path` is the on-disk source of
    `main.css`. Both are read by the publisher when it copies files into
    the bundle directory — this function only records the source paths.

    Frame filenames mirror the live convention: index N → `(N+1):04d.jpg`.
    """
    html = render_static_html(
        manifest=manifest,
        steps=steps,
        frame_captions=frame_captions,
        step_links=step_links,
        templates_env=templates_env,
    )

    file_map: dict[str, Path] = {"main.css": css_path}
    for idx in _winner_frame_indices(steps):
        filename = f"{idx + 1:04d}.jpg"
        file_map[f"frames/{filename}"] = frames_dir / filename

    return StaticBundle(html=html, file_map=file_map)
