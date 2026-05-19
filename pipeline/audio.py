"""Audio extraction for Whisper transcription fallback.

pattern: Imperative Shell
Wraps ffmpeg as a subprocess to extract a 16 kHz mono WAV from the
downloaded video file. 16 kHz mono is Whisper's native input format,
so resampling at extract time avoids a second pass inside the model.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def extract_audio(video: Path, out_path: Path) -> Path:
    """Extract a 16 kHz mono WAV from `video` to `out_path`.

    Returns:
        `out_path` on success.

    Raises:
        RuntimeError: when ffmpeg is missing or the subprocess exits non-zero.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not on PATH; install ffmpeg before using the Whisper fallback."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}"
        )
    return out_path
