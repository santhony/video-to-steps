"""yt-dlp wrapper for downloading the source video + English VTT.

Returns (video_path, vtt_path_or_None). The orchestrator (Phase 5) uses the
None case to set status='error' with a clear message when WHISPER_FALLBACK
is disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

log = logging.getLogger(__name__)


def download_video_and_captions(url: str, job_dir: Path) -> tuple[Path, Path | None]:
    """Downloads the video MP4 (≤720p) and the English auto-caption VTT into job_dir.

    Returns:
        (video_path, vtt_path) — vtt_path is None when no captions were available.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(job_dir / "video.%(ext)s")

    opts: dict[str, Any] = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "vtt",
        "writesubtitles": False,           # we want auto-captions specifically
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with YoutubeDL(opts) as ydl:
        ydl.download([url])

    # Find the artifacts. yt-dlp writes `video.mp4` and `video.en.vtt`
    # (subtitleslangs=['en'], subtitlesformat='vtt') alongside.
    video_candidates = sorted(job_dir.glob("video.*"))
    video: Path | None = None
    for c in video_candidates:
        if c.suffix.lower() == ".mp4":
            video = c
            break
    if video is None and video_candidates:
        # yt-dlp picked a non-mp4 container despite the format hint; accept it.
        video = video_candidates[0]
    if video is None:
        raise RuntimeError(f"yt-dlp produced no video file in {job_dir}")

    vtt = job_dir / "video.en.vtt"
    if not vtt.exists():
        # Try alternate naming yt-dlp may use.
        alt = sorted(job_dir.glob("video*.vtt"))
        vtt = alt[0] if alt else None  # type: ignore[assignment]

    return video, vtt if (vtt is not None and Path(vtt).exists()) else None
