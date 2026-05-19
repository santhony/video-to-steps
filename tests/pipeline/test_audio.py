"""Tests for pipeline/audio.py — ffmpeg WAV extraction."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

from pipeline.audio import extract_audio


def _make_video_with_audio(path: Path, seconds: int = 2) -> None:
    """Generate a synthetic mp4 with a 440 Hz sine tone via ffmpeg lavfi."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-f", "lavfi", "-i", f"testsrc=s=320x240:d={seconds}:r=24",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            str(path),
        ],
        check=True, capture_output=True,
    )


@pytest.fixture
def video_with_audio(tmp_path: Path) -> Path:
    p = tmp_path / "with_audio.mp4"
    _make_video_with_audio(p)
    return p


def test_extract_audio_produces_16khz_mono_wav(video_with_audio: Path, tmp_path: Path) -> None:
    out = extract_audio(video_with_audio, tmp_path / "out.wav")
    assert out.exists()
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_extract_audio_raises_when_input_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ffmpeg .* failed"):
        extract_audio(tmp_path / "nope.mp4", tmp_path / "out.wav")
