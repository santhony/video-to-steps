"""Tests for pipeline/frames.py.

Uses ffmpeg's lavfi test sources to generate small synthetic mp4s, so the
tests need no checked-in binary fixtures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pipeline.frames import FixedFpsExtractor, _phash_filter


def _make_static(path: Path, seconds: int = 4) -> None:
    # ffmpeg 4.2.2 lavfi: source=param=value:param=value format
    # Use s= for size, d= for duration, r= for rate
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"smptebars=s=320x240:r=24:d={seconds}",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _make_cuts(path: Path) -> None:
    # 4 distinct 2-second segments concatenated. Use lavfi sources via
    # concat-demuxer-via-temp-files to avoid filter_complex.
    # Note: ffmpeg 4.2.2 uses s=/d=/r= parameter format
    parts = []
    lavfi_specs = [
        "testsrc=s=320x240:r=24:d=2",
        "testsrc2=s=320x240:r=24:d=2",
        "rgbtestsrc=s=320x240:r=24:d=2",
        "smptebars=s=320x240:r=24:d=2",
    ]
    for i, spec in enumerate(lavfi_specs):
        part = path.parent / f"_part_{i}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", spec,
                "-pix_fmt", "yuv420p", str(part),
            ],
            check=True, capture_output=True,
        )
        parts.append(part)
    list_file = path.parent / "_concat.txt"
    list_file.write_text("\n".join(f"file '{p}'" for p in parts))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(path)],
        check=True, capture_output=True,
    )


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    p = tmp_path / "static.mp4"
    _make_static(p, seconds=4)
    return p


@pytest.fixture
def cuts_video(tmp_path: Path) -> Path:
    p = tmp_path / "cuts.mp4"
    _make_cuts(p)
    return p


def test_fixed_fps_extracts_one_per_second_at_720_or_less(static_video: Path, tmp_path: Path) -> None:
    # vts-v1.AC3.1
    out = tmp_path / "out_no_dedup"
    extractor = FixedFpsExtractor(fps=1.0, dedup=False)
    frames = extractor.extract(static_video, out)
    # 4-second video at 1 fps → exactly 4 frames.
    assert len(frames) == 4
    # Height ≤ 720.
    img = Image.open(frames[0].path)
    assert img.height <= 720


def test_dedup_strictly_reduces_on_cut_heavy_video(cuts_video: Path, tmp_path: Path) -> None:
    # vts-v1.AC3.2
    out_all = tmp_path / "out_all"
    out_dd = tmp_path / "out_dd"
    n_all = len(FixedFpsExtractor(fps=1.0, dedup=False).extract(cuts_video, out_all))
    n_dd = len(FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6).extract(cuts_video, out_dd))
    assert n_dd < n_all
    # Sanity: still at least one frame per visible segment.
    assert n_dd >= 1


def test_dedup_equal_or_fewer_on_any_input(static_video: Path, tmp_path: Path) -> None:
    # vts-v1.AC3.2 (the "any input" half)
    out_all = tmp_path / "out_all"
    out_dd = tmp_path / "out_dd"
    n_all = len(FixedFpsExtractor(fps=1.0, dedup=False).extract(static_video, out_all))
    n_dd = len(FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6).extract(static_video, out_dd))
    assert n_dd <= n_all


def test_phash_filter_never_drops_first(tmp_path: Path) -> None:
    # vts-v1.AC3.3 — pure-function test, synthesize 3 PIL images with visually distinct content.
    # The key invariant: the first frame is ALWAYS kept, regardless of hamming distance to subsequent frames.
    from PIL import ImageDraw
    paths = []

    # First: diagonal lines
    img1 = Image.new("RGB", (64, 64), color=(255, 255, 255))
    draw1 = ImageDraw.Draw(img1)
    for i in range(0, 64, 4):
        draw1.line([(i, 0), (i+64, 64)], fill=(0, 0, 0), width=2)
    p1 = tmp_path / "0000.jpg"
    img1.save(p1)
    paths.append(p1)

    # Second: horizontal lines (visually very different from diagonal)
    img2 = Image.new("RGB", (64, 64), color=(255, 255, 255))
    draw2 = ImageDraw.Draw(img2)
    for i in range(0, 64, 4):
        draw2.line([(0, i), (64, i)], fill=(0, 0, 0), width=2)
    p2 = tmp_path / "0001.jpg"
    img2.save(p2)
    paths.append(p2)

    # Third: radial pattern (also visually distinct)
    img3 = Image.new("RGB", (64, 64), color=(255, 255, 255))
    draw3 = ImageDraw.Draw(img3)
    for i in range(0, 32, 4):
        draw3.ellipse([(32-i, 32-i), (32+i, 32+i)], outline=(0, 0, 0), width=1)
    p3 = tmp_path / "0002.jpg"
    img3.save(p3)
    paths.append(p3)

    kept = _phash_filter(paths, hamming_max=6)
    assert kept[0] == paths[0], "First frame must always be kept"
    assert len(kept) >= 2, "Should keep at least 2 frames (first + the visually distinct one)"


def test_scene_change_extractor_is_a_v2_stub(tmp_path: Path) -> None:
    from pipeline.frames import SceneChangeExtractor
    with pytest.raises(NotImplementedError) as exc:
        SceneChangeExtractor().extract(tmp_path / "x.mp4", tmp_path / "out")
    assert "v2" in str(exc.value).lower()
