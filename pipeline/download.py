"""yt-dlp wrapper for downloading the source video + English VTT.

Returns (video_path, vtt_path_or_None). The orchestrator (Phase 5) uses the
None case to set status='error' with a clear message when WHISPER_FALLBACK
is disabled.

pattern: Imperative Shell
This module orchestrates subprocess I/O (yt-dlp, filesystem discovery) with
minimal logic. Video discovery filters out subtitle files to ensure correctness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

log = logging.getLogger(__name__)

# Subtitle extensions that may be picked up by glob("video.*") but should not be treated as video
SUB_EXTS = {".vtt", ".srt", ".ass"}


def _discover_video(job_dir: Path) -> Path | None:
    """Discovers the video file in job_dir, excluding subtitle files.

    Returns the first non-subtitle video.* file in sorted order, or None if
    no video file is found.
    """
    video_candidates = [
        c for c in sorted(job_dir.glob("video.*"))
        if c.suffix.lower() not in SUB_EXTS and ".en." not in c.name
    ]

    # Prefer .mp4 if available
    for c in video_candidates:
        if c.suffix.lower() == ".mp4":
            return c

    # Fall back to the first non-subtitle file
    return video_candidates[0] if video_candidates else None


def download_video_and_captions(url: str, job_dir: Path) -> tuple[Path, Path | None, str]:
    """Downloads the video MP4 (≤720p) and the English auto-caption VTT into job_dir.

    Returns:
        (video_path, vtt_path, title) — vtt_path is None when no captions
        were available; title is the video title from yt-dlp's metadata
        (empty string if extraction succeeded but the metadata lacked one).
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
        # YouTube extraction needs a JS runtime since the 2025 player rework.
        # yt-dlp only enables `deno` by default; on Macs deno is rarely
        # installed but Homebrew's `node` usually is, so accept either.
        # If neither is present yt-dlp emits a warning and falls back to
        # signature-free formats (typically itag 18: 360p mp4) which is
        # fine for instructional video matching.
        "js_runtimes": {"node": {}, "deno": {}},
    }

    with YoutubeDL(opts) as ydl:
        # extract_info(..., download=True) does the same work as ydl.download
        # but also returns the parsed info dict so we can read the title.
        info = ydl.extract_info(url, download=True) or {}

    title = (info.get("title") or "").strip()

    # Find the artifacts. yt-dlp writes `video.mp4` and `video.en.vtt`
    # (subtitleslangs=['en'], subtitlesformat='vtt') alongside.
    video = _discover_video(job_dir)
    if video is None:
        raise RuntimeError(f"yt-dlp produced no video file in {job_dir}")

    vtt_path: Path | None = job_dir / "video.en.vtt"
    if not vtt_path.exists():
        # Try alternate naming yt-dlp may use.
        alt = sorted(job_dir.glob("video*.vtt"))
        vtt_path = alt[0] if alt else None

    return (
        video,
        vtt_path if (vtt_path is not None and vtt_path.exists()) else None,
        title,
    )
