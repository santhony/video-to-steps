"""End-to-end Phase 3 smoke: download → parse VTT → extract+dedupe frames.

Usage: python scripts/smoke_phase3.py <youtube-url>

Run only with a known short instructional video (~2-5 minutes). Writes
artifacts to ./data/jobs/<rand>/ for inspection.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from pipeline.captions import dedupe_rolling, parse_vtt
from pipeline.download import download_video_and_captions
from pipeline.frames import FixedFpsExtractor


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <youtube-url>", file=sys.stderr)
        return 2
    url = argv[1]
    job_id = uuid.uuid4().hex[:12]
    job_dir = Path("./data/jobs") / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"job: {job_id} → {job_dir}")
    video, vtt = download_video_and_captions(url, job_dir)
    print(f"video: {video} ({video.stat().st_size:,} bytes)")
    if vtt is None:
        print("vtt:   MISSING (Whisper fallback would be needed)")
    else:
        cues = parse_vtt(vtt)
        deduped = dedupe_rolling(cues)
        print(f"vtt:   {len(cues)} raw cues → {len(deduped)} after dedupe_rolling")

    extractor = FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6)
    frames = extractor.extract(video, job_dir / "frames")
    print(f"frames: {len(frames)} kept after pHash dedup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
