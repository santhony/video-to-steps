"""Tests for providers/whisper.py — FasterWhisperTranscriber adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.types import Cue
from providers.whisper import (
    FasterWhisperTranscriber,
    _segments_to_cues,
    build_whisper,
)


@dataclass
class _Seg:
    start: float
    end: float
    text: str


def test_segments_to_cues_drops_empty_and_strips_whitespace():
    segs = [
        _Seg(0.0, 1.0, "  hello  "),
        _Seg(1.0, 2.0, "   "),       # whitespace only — drop
        _Seg(2.0, 3.0, "world"),
        _Seg(3.0, 4.0, ""),          # empty — drop
    ]
    cues = _segments_to_cues(segs)
    assert cues == [
        Cue(start=0.0, end=1.0, text="hello"),
        Cue(start=2.0, end=3.0, text="world"),
    ]


def test_segments_to_cues_handles_none_text():
    seg = _Seg(0.0, 1.0, None)  # type: ignore[arg-type]
    assert _segments_to_cues([seg]) == []


@pytest.mark.asyncio
async def test_faster_whisper_transcriber_lazy_loads_and_calls_model(tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")  # path just needs to exist for the str(path) cast

    fake_segments = [
        _Seg(0.0, 1.5, "Hello there."),
        _Seg(1.5, 3.0, "Today we tie a Windsor."),
    ]

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.init_args = (args, kwargs)

        def transcribe(self, audio_path, **kwargs):
            return iter(fake_segments), {}

    with patch("faster_whisper.WhisperModel", FakeModel):
        t = FasterWhisperTranscriber(model="base.en")
        assert t._model is None  # not loaded yet
        cues = await t.transcribe(audio)
        assert t._model is not None  # loaded on first transcribe
    assert cues == [
        Cue(start=0.0, end=1.5, text="Hello there."),
        Cue(start=1.5, end=3.0, text="Today we tie a Windsor."),
    ]


def test_faster_whisper_transcriber_exposes_name_as_attribute():
    t = FasterWhisperTranscriber(model="base.en")
    assert t.name == "faster-whisper:base.en"


@dataclass
class _StubSettings:
    whisper_model: str = "tiny.en"


def test_build_whisper_returns_faster_whisper_transcriber():
    transcriber = build_whisper(_StubSettings(whisper_model="tiny.en"))
    assert isinstance(transcriber, FasterWhisperTranscriber)
    assert transcriber.name == "faster-whisper:tiny.en"
