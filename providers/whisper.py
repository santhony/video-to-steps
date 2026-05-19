"""WhisperTranscriber — local faster-whisper audio → list[Cue] adapter.

pattern: Imperative Shell
Thin wrapper around `faster_whisper.WhisperModel`. The model loads
lazily on first `transcribe` call and is cached for the instance
lifetime. Inference is synchronous (CTranslate2 inside), so we wrap it
with `asyncio.to_thread` to keep the orchestrator's event loop free.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

from pipeline.types import Cue

log = logging.getLogger(__name__)


# Threshold for the "is this audio actually speech?" sanity check. Whisper
# emits `no_speech_prob` per segment (0..1); 0.6 is the well-known default
# above which the segment is "probably not speech". We average across the
# whole transcript so a single noisy segment can't trip the check.
_NO_SPEECH_THRESHOLD = 0.6


class NoSpeechDetectedError(RuntimeError):
    """Raised when Whisper transcription concludes the audio isn't really
    speech (silent video, music-only, ambient noise, etc.).

    The orchestrator catches this and surfaces a clear "this doesn't look
    like an instructional video" message to the operator instead of letting
    hallucinated cues drive the rest of the pipeline.
    """


class WhisperTranscriber(Protocol):
    name: str

    async def transcribe(self, audio: Path) -> list[Cue]: ...


def _segments_to_cues(segments: Any) -> list[Cue]:
    """Convert faster-whisper Segment iterator to list[Cue], skipping empties."""
    cues: list[Cue] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        cues.append(Cue(start=float(seg.start), end=float(seg.end), text=text))
    return cues


def _check_speech_or_raise(segments_list: list[Any]) -> None:
    """Raise NoSpeechDetectedError when the transcript looks like non-speech.

    Two triggers:
    1. The materialized list is empty — Whisper found nothing transcribable.
    2. The average `no_speech_prob` across segments exceeds the threshold —
       Whisper's own confidence signal that what it heard isn't really
       speech (music, ambient noise, silence with hallucinated cues).
    """
    if not segments_list:
        raise NoSpeechDetectedError(
            "Whisper found no transcribable audio in this video."
        )
    probs = [getattr(s, "no_speech_prob", 0.0) for s in segments_list]
    avg = sum(probs) / len(probs)
    if avg > _NO_SPEECH_THRESHOLD:
        raise NoSpeechDetectedError(
            f"This video doesn't appear to have a spoken narration "
            f"(Whisper no_speech_prob avg {avg:.2f} > {_NO_SPEECH_THRESHOLD}). "
            f"video-to-steps relies on either YouTube captions or spoken audio; "
            f"music-only or silent videos aren't supported."
        )


class FasterWhisperTranscriber:
    name: str

    def __init__(
        self,
        *,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.name = f"faster-whisper:{model}"
        self._model_id = model
        self._device = device
        self._compute_type = compute_type
        self._model: Any | None = None  # lazy

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("Loading faster-whisper model %r (device=%s, compute=%s)",
                     self._model_id, self._device, self._compute_type)
            self._model = WhisperModel(
                self._model_id,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model

    def _transcribe_sync(self, audio: Path) -> list[Cue]:
        model = self._load()
        segments, _info = model.transcribe(str(audio), beam_size=1)
        # Materialize so we can inspect no_speech_prob before returning cues.
        segments_list = list(segments)
        _check_speech_or_raise(segments_list)
        return _segments_to_cues(segments_list)

    async def transcribe(self, audio: Path) -> list[Cue]:
        return await asyncio.to_thread(self._transcribe_sync, audio)


def build_whisper(settings: Any) -> WhisperTranscriber:
    """Returns a configured WhisperTranscriber.

    Only one backend is wired in v1.1: local faster-whisper. Cloud
    backends (OpenAI / Groq / Together) can be added behind the same
    Protocol when the operator needs them.
    """
    return FasterWhisperTranscriber(model=settings.whisper_model)
