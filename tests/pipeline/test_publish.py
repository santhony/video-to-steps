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
