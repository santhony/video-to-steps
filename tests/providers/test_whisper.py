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


def test_check_speech_or_raise_empty_segments():
    from providers.whisper import NoSpeechDetectedError, _check_speech_or_raise
    with pytest.raises(NoSpeechDetectedError, match="no transcribable audio"):
        _check_speech_or_raise([])


def test_check_speech_or_raise_high_no_speech_prob():
    from providers.whisper import NoSpeechDetectedError, _check_speech_or_raise
    # Simulate a music-only video: faster-whisper hallucinates some text
    # but flags each segment as probably non-speech.
    fake_segments = [
        _Seg(0.0, 2.0, "♪♪♪"),
        _Seg(2.0, 4.0, "Thanks for watching!"),
        _Seg(4.0, 6.0, "♪♪♪"),
    ]
    # Inject no_speech_prob onto the dataclass dynamically (Whisper segments
    # carry it natively; our test _Seg doesn't, so simulate it).
    for s in fake_segments:
        s.no_speech_prob = 0.9
    with pytest.raises(NoSpeechDetectedError, match="doesn't appear to have a spoken narration"):
        _check_speech_or_raise(fake_segments)


def test_check_speech_or_raise_passes_for_real_speech():
    from providers.whisper import _check_speech_or_raise
    fake_segments = [
        _Seg(0.0, 2.0, "First heat the pan."),
        _Seg(2.0, 5.0, "Then add the onion."),
    ]
    for s in fake_segments:
        s.no_speech_prob = 0.05
    # Should not raise
    _check_speech_or_raise(fake_segments)


@pytest.mark.asyncio
async def test_faster_whisper_transcriber_raises_on_silent_audio(tmp_path: Path):
    """Integration: the transcriber path raises NoSpeechDetectedError when
    the underlying model returns high no_speech_prob segments."""
    from providers.whisper import NoSpeechDetectedError

    audio = tmp_path / "silent.wav"
    audio.write_bytes(b"")

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio_path, **kwargs):
            # Simulate music-only: a few segments, all flagged non-speech.
            class _S:
                def __init__(self, start, end, text, prob):
                    self.start = start
                    self.end = end
                    self.text = text
                    self.no_speech_prob = prob
            return iter([
                _S(0.0, 2.0, "[Music]", 0.95),
                _S(2.0, 4.0, "♪", 0.92),
            ]), {}

    with patch("faster_whisper.WhisperModel", FakeModel):
        t = FasterWhisperTranscriber(model="base.en")
        with pytest.raises(NoSpeechDetectedError):
            await t.transcribe(audio)
