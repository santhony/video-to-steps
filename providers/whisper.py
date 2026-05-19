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
        return _segments_to_cues(segments)

    async def transcribe(self, audio: Path) -> list[Cue]:
        return await asyncio.to_thread(self._transcribe_sync, audio)


def build_whisper(settings: Any) -> WhisperTranscriber:
    """Returns a configured WhisperTranscriber.

    Only one backend is wired in v1.1: local faster-whisper. Cloud
    backends (OpenAI / Groq / Together) can be added behind the same
    Protocol when the operator needs them.
    """
    return FasterWhisperTranscriber(model=settings.whisper_model)
