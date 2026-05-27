# Publish to GitHub Pages — Phase 1: Functional Core + Tests

**Goal:** Add the pure `StaticBundle` type and `pipeline/publish.py` module that renders the result page as a self-contained HTML string + file map, with full unit-test coverage.

**Architecture:** Functional Core. Pure module — no disk writes, no network, no subprocesses. Reads source files (CSS) only via paths the caller resolved. Renders via a Jinja2 `Environment` the caller passes in (so `pipeline/publish.py` does not import from `server.py`).

**Tech Stack:** Python 3.11, Jinja2 (existing), `@dataclass(slots=True)` (existing convention).

**Scope:** Phase 1 of 4 from `docs/design-plans/2026-05-27-publish-to-github-pages.md`.

**Codebase verified:** 2026-05-27

---

## Reference: verified codebase state

- `pipeline/types.py` (lines 15-82) holds all current `@dataclass(slots=True)` types. `Manifest` ends at line 82 with no `published_url`/`published_at` (these are Phase 2's job — not added here).
- `templates/base.html` (15 lines) owns the htmx script (line 8), the `main.css` link (line 7), and the brand anchor (line 11). `result.html` extends it.
- `templates/result.html` (43 lines) renders image refs as `/job/{{ manifest.job_id }}/frame/{{ '%04d' % (frame.index + 1) }}.jpg` (lines 32-33). Frame `.index` is 0-based; filenames are 1-padded.
- Jinja `templates` object is constructed in `server.py:39-40` and is NOT imported by any pipeline module. The pure render function must accept a Jinja2 `Environment` argument.
- `static/css/main.css` exists at the repo root.
- `tests/pipeline/` is the test location; `test_match.py` is a good style reference (plain pytest, no fixtures for pure modules, classes for grouping).

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Add `StaticBundle` and `PublishError` to `pipeline/types.py`

**Files:**
- Modify: `pipeline/types.py` (append after current line 82)

**Step 1: Write the failing test**

Create `tests/pipeline/test_publish_types.py`:

```python
"""Smoke tests for the new publish types."""

from pathlib import Path

from pipeline.types import PublishError, StaticBundle


def test_static_bundle_is_a_dataclass():
    b = StaticBundle(html="<html></html>", file_map={"main.css": Path("static/css/main.css")})
    assert b.html == "<html></html>"
    assert b.file_map == {"main.css": Path("static/css/main.css")}


def test_static_bundle_has_slots():
    b = StaticBundle(html="", file_map={})
    # slots=True means assigning an unknown attribute raises
    try:
        b.unknown_attr = 1  # type: ignore[attr-defined]
    except AttributeError:
        return
    raise AssertionError("StaticBundle should have slots=True")


def test_publish_error_is_runtime_error():
    e = PublishError("git push failed")
    assert isinstance(e, RuntimeError)
    assert str(e) == "git push failed"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/santhony/Documents/dev_claude/video-to-steps/.worktrees/publish-to-github-pages && python -m pytest tests/pipeline/test_publish_types.py -v`
Expected: ImportError — `cannot import name 'StaticBundle' from 'pipeline.types'`

**Step 3: Append to `pipeline/types.py`**

Append below line 82 (after the `Manifest` dataclass):

```python


@dataclass(slots=True)
class StaticBundle:
    """A self-contained snapshot of a job's result page.

    `html` is the full text of `index.html`; `file_map` maps each
    bundle-relative path (e.g. `"frames/0001.jpg"`, `"main.css"`) to the
    source file on disk that the publisher should copy into the bundle.
    """
    html: str
    file_map: dict[str, Path] = field(default_factory=dict)


class PublishError(RuntimeError):
    """Raised when a publish or unpublish operation fails.

    Carries the underlying stderr / message so server routes can render
    a useful error fragment.
    """
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/pipeline/test_publish_types.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add pipeline/types.py tests/pipeline/test_publish_types.py
git commit -m "Add StaticBundle and PublishError to pipeline.types"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add `static_mode` branches to `templates/base.html` and `templates/result.html`

**Files:**
- Modify: `templates/base.html` (lines 7, 8, 11)
- Modify: `templates/result.html` (lines 6, 32-33)

The flag `static_mode` defaults to falsy in live renders (we never set it in the server's `TemplateResponse` contexts), so the live page is unchanged. When `static_mode=True`, the snapshot:

- Uses `main.css` (relative) instead of `/static/css/main.css`
- Drops the `<script src="/static/js/htmx.min.js" defer>` tag (no HTMX on the snapshot)
- Renders a non-clickable brand label (preserves layout)
- Drops the `manifest.url` "Source video ↗" link to `manifest.url` (the page is a static copy; the link is fine to keep since it points at a public YouTube URL — verify with test)
- Rewrites image src/href from `/job/<id>/frame/NNNN.jpg` to `frames/NNNN.jpg`

**Step 1: Write the failing test**

Create `tests/pipeline/test_publish.py`:

```python
"""Tests for pipeline.publish (pure, no I/O beyond reading CSS path)."""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from pipeline.types import (
    CostBreakdown,
    Frame,
    Manifest,
    StaticBundle,
    Step,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "templates"
CSS_PATH = REPO_ROOT / "static" / "css" / "main.css"


@pytest.fixture
def jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )


@pytest.fixture
def manifest() -> Manifest:
    return Manifest(
        job_id="dQw4w9WgXcQ_a1b2c3",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Test Video",
        status="done",
        mode="cloud",
        config_snapshot={
            "embed_backend": "jina_v4",
            "llm_model": "deepseek-v4-flash",
            "vision_model": "gpt-4o-mini",
        },
        cost=CostBreakdown(total_usd=0.12),
    )


@pytest.fixture
def steps() -> list[Step]:
    return [
        Step(
            index=0,
            start=0.0,
            end=10.0,
            instruction="Mix the flour and water.",
            frames=[
                Frame(index=0, timestamp=1.0, path=Path("/tmp/0001.jpg")),
                Frame(index=4, timestamp=5.0, path=Path("/tmp/0005.jpg")),
            ],
        ),
        Step(
            index=1,
            start=10.0,
            end=20.0,
            instruction="Knead until smooth.",
            frames=[
                Frame(index=4, timestamp=11.0, path=Path("/tmp/0005.jpg")),  # duplicate index 4
                Frame(index=9, timestamp=15.0, path=Path("/tmp/0010.jpg")),
            ],
        ),
    ]


class TestStaticModeTemplate:
    """Static-mode renders must drop live-only chrome and use relative paths."""

    def test_static_mode_uses_relative_main_css(self, jinja_env, manifest, steps):
        tpl = jinja_env.get_template("result.html")
        html = tpl.render(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            static_mode=True,
        )
        assert 'href="main.css"' in html
        assert '/static/css/main.css' not in html

    def test_static_mode_drops_htmx_script(self, jinja_env, manifest, steps):
        tpl = jinja_env.get_template("result.html")
        html = tpl.render(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            static_mode=True,
        )
        assert 'htmx.min.js' not in html

    def test_static_mode_brand_is_not_clickable(self, jinja_env, manifest, steps):
        tpl = jinja_env.get_template("result.html")
        html = tpl.render(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            static_mode=True,
        )
        # No <a href="/"> brand link in the snapshot
        assert '<a href="/"' not in html

    def test_static_mode_rewrites_frame_image_paths(self, jinja_env, manifest, steps):
        tpl = jinja_env.get_template("result.html")
        html = tpl.render(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            static_mode=True,
        )
        # Image refs become relative: frames/NNNN.jpg
        assert 'src="frames/0001.jpg"' in html
        assert 'src="frames/0010.jpg"' in html
        # No absolute /job/<id>/frame/ URLs in static mode
        assert '/job/' not in html

    def test_static_mode_keeps_source_video_link(self, jinja_env, manifest, steps):
        """The snapshot is the result page verbatim — keep the source-video anchor."""
        tpl = jinja_env.get_template("result.html")
        html = tpl.render(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            static_mode=True,
        )
        assert "Source video" in html

    def test_live_mode_keeps_absolute_paths(self, jinja_env, manifest, steps):
        """Sanity: omitting static_mode (= live mode) preserves current behavior."""
        tpl = jinja_env.get_template("result.html")
        html = tpl.render(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
        )
        assert '/job/dQw4w9WgXcQ_a1b2c3/frame/0001.jpg' in html
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_publish.py::TestStaticModeTemplate -v`
Expected: All 6 `TestStaticModeTemplate` tests FAIL (templates do not honor `static_mode` yet). The `test_static_mode_keeps_source_video_link` test will pass once the live-mode anchor is preserved through the static-mode branch (which it is — we don't touch lines 4-6 of `result.html`).

**Step 3: Edit `templates/base.html`** to honor `static_mode`

Replace the entire file with:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}video-to-steps{% endblock %}</title>
  {% if static_mode %}
  <link rel="stylesheet" href="main.css">
  {% else %}
  <link rel="stylesheet" href="/static/css/main.css">
  <script src="/static/js/htmx.min.js" defer></script>
  {% endif %}
</head>
<body>
  <header>
    {% if static_mode %}
    <span class="brand">video-to-steps</span>
    {% else %}
    <a href="/" class="brand">video-to-steps</a>
    {% endif %}
  </header>
  <main>{% block main %}{% endblock %}</main>
</body>
</html>
```

**Step 4: Edit `templates/result.html` image anchor + img**

Replace lines 32-33 (the `<a href=...>` and `<img src=...>` lines inside the `frame-grid` loop) with:

```jinja
        {% if static_mode %}
        <a href="frames/{{ '%04d' % (frame.index + 1) }}.jpg">
          <img src="frames/{{ '%04d' % (frame.index + 1) }}.jpg"
               alt="{{ frame_captions.get(frame.index|string, '') or step.instruction }}"
               loading="lazy">
        </a>
        {% else %}
        <a href="/job/{{ manifest.job_id }}/frame/{{ '%04d' % (frame.index + 1) }}.jpg">
          <img src="/job/{{ manifest.job_id }}/frame/{{ '%04d' % (frame.index + 1) }}.jpg"
               alt="{{ frame_captions.get(frame.index|string, '') or step.instruction }}"
               loading="lazy">
        </a>
        {% endif %}
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_publish.py::TestStaticModeTemplate -v`
Expected: 5 passed

Also run the full offline suite to confirm no regressions:

Run: `python -m pytest`
Expected: existing tests still pass; new tests pass.

**Step 6: Commit**

```bash
git add templates/base.html templates/result.html tests/pipeline/test_publish.py
git commit -m "Add static_mode branches to base.html and result.html"
```
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->

<!-- START_TASK_3 -->
### Task 3: Implement `pipeline/publish.py` — pure `render_static_html` + `build_static_bundle`

**Files:**
- Create: `pipeline/publish.py`

**Step 1: Write the failing tests**

Append the following classes to `tests/pipeline/test_publish.py`:

```python


class TestBuildStaticBundle:
    """build_static_bundle composes render + file_map deterministically."""

    def test_bundle_html_uses_static_mode(self, jinja_env, manifest, steps):
        from pipeline.publish import build_static_bundle

        bundle = build_static_bundle(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            frames_dir=Path("/tmp/job/frames"),
            css_path=CSS_PATH,
            templates_env=jinja_env,
        )
        assert 'href="main.css"' in bundle.html
        assert '/job/' not in bundle.html
        assert 'htmx.min.js' not in bundle.html

    def test_file_map_contains_main_css(self, jinja_env, manifest, steps):
        from pipeline.publish import build_static_bundle

        bundle = build_static_bundle(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            frames_dir=Path("/tmp/job/frames"),
            css_path=CSS_PATH,
            templates_env=jinja_env,
        )
        assert bundle.file_map["main.css"] == CSS_PATH

    def test_file_map_contains_union_of_frames_deduped(self, jinja_env, manifest, steps):
        from pipeline.publish import build_static_bundle

        bundle = build_static_bundle(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            frames_dir=Path("/tmp/job/frames"),
            css_path=CSS_PATH,
            templates_env=jinja_env,
        )
        # steps reference frame indices {0, 4, 9} → filenames 0001, 0005, 0010
        frame_keys = sorted(k for k in bundle.file_map if k.startswith("frames/"))
        assert frame_keys == ["frames/0001.jpg", "frames/0005.jpg", "frames/0010.jpg"]
        assert bundle.file_map["frames/0001.jpg"] == Path("/tmp/job/frames/0001.jpg")
        assert bundle.file_map["frames/0005.jpg"] == Path("/tmp/job/frames/0005.jpg")
        assert bundle.file_map["frames/0010.jpg"] == Path("/tmp/job/frames/0010.jpg")

    def test_file_map_excludes_unreferenced_frames(self, jinja_env, manifest, steps):
        """A frame on disk that no step references must not be in the bundle."""
        from pipeline.publish import build_static_bundle

        bundle = build_static_bundle(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            frames_dir=Path("/tmp/job/frames"),
            css_path=CSS_PATH,
            templates_env=jinja_env,
        )
        # Only 3 frame entries (plus main.css = 4 total)
        assert sum(1 for k in bundle.file_map if k.startswith("frames/")) == 3

    def test_no_absolute_urls_in_html(self, jinja_env, manifest, steps):
        """The published page must be self-contained — no live-server URLs."""
        from pipeline.publish import build_static_bundle

        bundle = build_static_bundle(
            manifest=manifest,
            steps=steps,
            frame_captions={},
            step_links=[None, None],
            frames_dir=Path("/tmp/job/frames"),
            css_path=CSS_PATH,
            templates_env=jinja_env,
        )
        # No internal absolute paths leak
        assert '/static/' not in bundle.html
        assert '/job/' not in bundle.html
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_publish.py::TestBuildStaticBundle -v`
Expected: ImportError or ModuleNotFoundError on `pipeline.publish`.

**Step 3: Create `pipeline/publish.py`**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_publish.py -v`
Expected: All `TestStaticModeTemplate` + `TestBuildStaticBundle` tests pass (10 tests total).

Also run the full offline suite:

Run: `python -m pytest`
Expected: All existing tests still pass; new tests pass.

**Step 5: Commit**

```bash
git add pipeline/publish.py tests/pipeline/test_publish.py
git commit -m "Add pipeline.publish with pure render_static_html + build_static_bundle"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Verify Phase 1 end-to-end

**Files:** none (verification only).

**Step 1: Run the full offline suite**

Run: `cd /Users/santhony/Documents/dev_claude/video-to-steps/.worktrees/publish-to-github-pages && python -m pytest -v`
Expected: All tests pass, including:
- `tests/pipeline/test_publish_types.py` (3 tests)
- `tests/pipeline/test_publish.py::TestStaticModeTemplate` (5 tests)
- `tests/pipeline/test_publish.py::TestBuildStaticBundle` (5 tests)
- All pre-existing tests still pass.

**Step 2: Sanity-check live mode still works**

Start the server and confirm an existing job's result page still renders normally:

Run: `./start.sh`
Then in a browser open `http://127.0.0.1:8090/` and confirm the form loads.

If you have an existing `data/jobs/<id>/` directory from a prior run, open `/job/<id>/result` and verify:
- The page renders with the htmx script tag present
- CSS is loaded from `/static/css/main.css`
- Frame `<img>` src attributes start with `/job/<id>/frame/`
- The brand "video-to-steps" is a clickable link to `/`

Stop the server:

Run: `./stop.sh`

Expected: Live mode unchanged; static_mode confined to publish path (not yet exercised by routes — that's Phase 3).
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
