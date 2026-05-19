"""Frame extraction strategies.

`FixedFpsExtractor` is the v1 default: ffmpeg at fixed fps, optional pHash
dedup pass.

`SceneChangeExtractor` is a v2 stub — exposed so the FrameExtractor protocol
shape is exercised, but it raises NotImplementedError on invocation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from PIL import Image
import imagehash

from .types import Frame

log = logging.getLogger(__name__)


def _run_ffmpeg(video: Path, out_dir: Path, fps: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"fps={fps},scale=-2:720",
        "-qscale:v", "4",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        # Surface stderr for diagnostic; orchestrator will catch and record.
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.strip()[-500:]}"
        )


def _list_frames(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("*.jpg"))


def _phash_filter(paths: list[Path], hamming_max: int) -> list[Path]:
    """Keeps frames whose pHash differs from the last kept by > hamming_max.

    The first frame is ALWAYS kept (vts-v1.AC3.3).
    """
    if not paths:
        return []

    kept: list[Path] = [paths[0]]
    last_hash = imagehash.phash(Image.open(paths[0]))
    for p in paths[1:]:
        h = imagehash.phash(Image.open(p))
        if (h - last_hash) > hamming_max:
            kept.append(p)
            last_hash = h
    return kept


class FixedFpsExtractor:
    def __init__(self, *, fps: float = 1.0, dedup: bool = True, hamming_max: int = 6) -> None:
        self._fps = fps
        self._dedup = dedup
        self._hamming_max = hamming_max
        self.name = f"FixedFpsExtractor(fps={fps},dedup={dedup},hamming_max={hamming_max})"

    def extract(self, video: Path, out_dir: Path) -> list[Frame]:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not on PATH; install ffmpeg before running the pipeline.")
        _run_ffmpeg(video, out_dir, self._fps)
        all_paths = _list_frames(out_dir)
        kept = _phash_filter(all_paths, self._hamming_max) if self._dedup else all_paths

        # Delete dropped frames so disk reflects what the pipeline sees.
        kept_set = set(kept)
        for p in all_paths:
            if p not in kept_set:
                p.unlink(missing_ok=True)

        # Capture each kept frame's ORIGINAL ordinal (= its source-video
        # second under fps=1.0) for timestamp computation, then RENAME the
        # kept frames to a dense `0001.jpg, 0002.jpg, ...` sequence so the
        # server can derive a frame's URL from `Frame.index` alone.
        # The rename is safe in forward order: dropped frames are already
        # unlinked, so target names never collide with surviving source
        # names.
        frames: list[Frame] = []
        for new_idx, src in enumerate(kept):
            original_ordinal = int(src.stem)            # 1-indexed from ffmpeg
            timestamp = (original_ordinal - 1) / self._fps
            dst = out_dir / f"{new_idx + 1:04d}.jpg"
            if src != dst:
                src.rename(dst)
            frames.append(Frame(index=new_idx, timestamp=timestamp, path=dst))
        return frames


class SceneChangeExtractor:
    """v2 stub. Surfaces the FrameExtractor shape without behavior."""

    def __init__(self) -> None:
        self.name = "SceneChangeExtractor(v2-stub)"

    def extract(self, video: Path, out_dir: Path) -> list[Frame]:
        raise NotImplementedError(
            "SceneChangeExtractor is a v2 roadmap item. "
            "Use FixedFpsExtractor for v1."
        )
