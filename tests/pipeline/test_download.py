"""Unit tests for pipeline.download module."""

from __future__ import annotations

from pathlib import Path

from pipeline.download import _discover_video


def test_discover_video_excludes_subtitles(tmp_path: Path) -> None:
    """_discover_video filters out .vtt, .srt, .ass files and language-specific files."""
    # Create mixed subtitle and video files
    (tmp_path / "video.en.vtt").touch()
    (tmp_path / "video.srt").touch()
    (tmp_path / "video.ass").touch()
    (tmp_path / "video.webm").touch()

    # Alphabetically, video.ass comes before video.webm, but should be skipped
    result = _discover_video(tmp_path)
    assert result is not None
    assert result.name == "video.webm"


def test_discover_video_prefers_mp4(tmp_path: Path) -> None:
    """_discover_video prefers .mp4 over other video formats."""
    (tmp_path / "video.webm").touch()
    (tmp_path / "video.mp4").touch()

    result = _discover_video(tmp_path)
    assert result is not None
    assert result.name == "video.mp4"


def test_discover_video_returns_none_when_no_video(tmp_path: Path) -> None:
    """_discover_video returns None when only subtitles exist."""
    (tmp_path / "video.en.vtt").touch()
    (tmp_path / "video.srt").touch()

    result = _discover_video(tmp_path)
    assert result is None


def test_discover_video_accepts_any_video_format(tmp_path: Path) -> None:
    """_discover_video accepts .avi, .mov, .flv, etc. (non-subtitle formats)."""
    (tmp_path / "video.avi").touch()

    result = _discover_video(tmp_path)
    assert result is not None
    assert result.name == "video.avi"
