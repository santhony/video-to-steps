# video-to-steps Implementation Plan — Phase 3: Source-to-frames pipeline

**Goal:** Given a YouTube URL, produce a deduplicated frame set and a parsed
caption track. All artifacts land in the per-job directory.

**Architecture:** Three modules — `download` wraps `yt-dlp`, `captions`
parses the VTT and collapses YouTube's rolling-repeat auto-caption pattern,
`frames` runs an `ffmpeg` extraction followed by a pure-Python pHash
dedup pass. The `FrameExtractor` protocol allows future
`SceneChangeExtractor` swap-in without touching downstream code.

**Tech Stack:** `yt-dlp`, `webvtt-python`, `Pillow`, `imagehash`,
`ffmpeg` (external binary, host-installed).

**Scope:** 3 of 7 phases.

**Codebase verified:** 2026-05-18. Phase 1 (`pipeline/types.py`,
`pipeline/storage.py`) and Phase 2 protocols expected. ffmpeg 4.2.2 present
on host.

**External dependency findings:**
- ✓ yt-dlp Python API: `YoutubeDL(opts).download([url])`; opts include
  `outtmpl`, `format`, `writeautomaticsub`, `subtitleslangs`,
  `subtitlesformat`. Captionless videos do NOT raise — yt-dlp silently
  skips the VTT, and absence of a `*.en.vtt` file is the signal.
- ✓ webvtt-py: `webvtt.read(path)` yields cues with `start`, `end`, `text`
  string attributes (times like `"00:01:23.456"`); no built-in
  rolling-repeat dedup — we write one.
- ✓ ffmpeg `-vf "fps=1.0,scale=-2:720" -qscale:v 4` produces 1 frame per
  source second, height-capped at 720, with even-width auto-calc, at good
  JPEG quality (scale range 1–31, lower=better). Best-practice invocation:
  `subprocess.run([...], check=False, capture_output=True, text=True)` then
  inspect return code.
- ✓ `imagehash.phash(PIL.Image)` returns an `ImageHash`; `h1 - h2` returns
  integer Hamming distance. Works on RGB and RGBA equally.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### vts-v1.AC2: YouTube download and caption parsing
- **vts-v1.AC2.1 Success:** yt-dlp downloads MP4 at ≤720p and the English VTT auto-caption track for a valid YouTube URL.
- **vts-v1.AC2.2 Success:** VTT parsing returns a `list[Cue]` in temporal order.
- **vts-v1.AC2.3 Success:** `dedupe_rolling` collapses YouTube auto-caption rolling-repeat patterns into single Cues.

### vts-v1.AC3: Frame extraction with deduplication
- **vts-v1.AC3.1 Success:** `FixedFpsExtractor(fps=1.0, dedup=False)` extracts one frame per source second at ≤720p height.
- **vts-v1.AC3.2 Success:** `FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6)` produces strictly fewer frames than `dedup=False` on a cut-heavy test video; equal-or-fewer on any input.
- **vts-v1.AC3.3 Edge:** pHash dedup never drops the first frame.

vts-v1.AC2.4 ("orchestrator writes status=error when no captions") is wired
in Phase 5 (orchestrator); the detection mechanism — absence of a `.vtt`
file — is provided here.

---

<!-- START_TASK_1 -->
### Task 1: Bump requirements for video/caption/hash deps

**Files:**
- Modify: `requirements.txt` (append)

**Implementation:**

```
yt-dlp>=2024.5.27
webvtt-python>=0.5.0
imagehash>=4.3.1
```

`Pillow` was already added in Phase 2 (vision); no change.

**Verification:**

```bash
source venv/bin/activate
uv pip install -r requirements-dev.txt
python -c "import yt_dlp, webvtt, imagehash; print(yt_dlp.version.__version__, imagehash.__version__)"
```
Expected: two version strings.

**Commit:**

```bash
git add requirements.txt
git commit -m "chore(vts-v1): add yt-dlp + webvtt-python + imagehash"
```
<!-- END_TASK_1 -->

<!-- START_SUBCOMPONENT_A (tasks 2-3) -->

<!-- START_TASK_2 -->
### Task 2: `pipeline/captions.py::parse_vtt`

**Verifies:** vts-v1.AC2.2

**Files:**
- Create: `pipeline/captions.py`
- Create: `tests/pipeline/__init__.py` (empty)
- Create: `tests/pipeline/test_captions.py` (unit)
- Create: `tests/pipeline/fixtures/sample.vtt` (checked-in 8-cue YouTube auto-caption fragment — see below)

**Implementation:**

```python
"""VTT parsing and rolling-repeat dedup for YouTube auto-captions.

`parse_vtt` is a thin wrapper over `webvtt.read` that converts string
timestamps to float seconds and yields `Cue` instances in file order.

`dedupe_rolling` collapses YouTube's two-line karaoke-style auto-caption
pattern, where the same text appears across consecutive overlapping cues
because the player needs to redraw a "current line" each tick.
"""

from __future__ import annotations

from pathlib import Path

import webvtt

from .types import Cue


def _ts_to_seconds(ts: str) -> float:
    # webvtt-python timestamps look like "00:01:23.456" or "01:23:45.678".
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, rest = parts
    elif len(parts) == 2:
        h, m, rest = "0", parts[0], parts[1]
    else:
        raise ValueError(f"unrecognized VTT timestamp: {ts!r}")
    s, _, ms = rest.partition(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms or "0") / 1000.0)


def parse_vtt(path: Path) -> list[Cue]:
    """Parses a .vtt file into a list of Cue, in file/temporal order.

    Newlines inside cue text are collapsed to spaces so downstream consumers
    see a single line per cue.
    """
    cues: list[Cue] = []
    for v in webvtt.read(str(path)):
        text = " ".join(v.text.splitlines()).strip()
        if not text:
            continue
        cues.append(Cue(start=_ts_to_seconds(v.start), end=_ts_to_seconds(v.end), text=text))
    return cues
```

`tests/pipeline/fixtures/sample.vtt` — checked-in fixture with at least
**3 distinct cues in temporal order** plus YouTube-style overlapping
rolling repeats (used in Task 3). Example (≈8 cues; create exactly this
content):

```
WEBVTT
Kind: captions
Language: en

00:00:00.480 --> 00:00:02.640
first heat the pan

00:00:01.000 --> 00:00:03.200
first heat the pan over medium

00:00:02.640 --> 00:00:04.880
heat the pan over medium heat

00:00:04.880 --> 00:00:07.000
then add a splash of olive oil

00:00:06.000 --> 00:00:08.200
then add a splash of olive oil to the pan

00:00:08.200 --> 00:00:10.400
slice the onion thinly

00:00:09.000 --> 00:00:11.500
slice the onion thinly into half-moons

00:00:11.500 --> 00:00:13.500
add the onion to the pan
```

**Testing:**

Tests must verify:
- **vts-v1.AC2.2:** `parse_vtt(fixture)` returns ≥3 Cue objects with strictly
  non-decreasing `start` times; first cue.start ≈ 0.48; last cue.start ≈
  11.5; every cue has non-empty `text`.

Test file: `tests/pipeline/test_captions.py` (unit).

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_captions.py::test_parse_vtt -v
```
Expected: pass.

**Commit:**

```bash
git add pipeline/captions.py tests/pipeline/
git commit -m "feat(vts-v1): parse_vtt — VTT → list[Cue] with float-seconds"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: `pipeline/captions.py::dedupe_rolling`

**Verifies:** vts-v1.AC2.3

**Files:**
- Modify: `pipeline/captions.py` (append)
- Modify: `tests/pipeline/test_captions.py` (append)

**Implementation:**

The YouTube rolling-repeat pattern: each new cue starts before the previous
ends, and its text is the previous cue's text with extra words appended (or
sometimes vice-versa). Dedup rule:

1. Walk cues in order.
2. For each candidate cue, compare its text to the last kept cue's text. If
   either is a prefix of the other (after lowercasing + whitespace
   normalize), they describe the same content — keep the LONGER text and
   extend the kept cue's `end` to the candidate's `end`. Drop the shorter.
3. Otherwise, keep the candidate as a new entry.

This catches both directions of the rolling overlap (text grows then
sometimes shrinks) and works without time-based heuristics.

```python
def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe_rolling(cues: list[Cue]) -> list[Cue]:
    """Collapses YouTube rolling-repeat auto-captions into stable cues.

    Adjacent cues where one's normalized text is a prefix of the other's
    are merged: the longer text wins, and the merged cue's `end` is the
    later of the two `end`s.
    """
    if not cues:
        return []

    out: list[Cue] = [cues[0]]
    for cue in cues[1:]:
        last = out[-1]
        a, b = _norm(last.text), _norm(cue.text)
        if a == b or a.startswith(b) or b.startswith(a):
            # Same content; keep the longer text and extend the time range.
            keep_text = last.text if len(last.text) >= len(cue.text) else cue.text
            out[-1] = Cue(start=last.start, end=max(last.end, cue.end), text=keep_text)
        else:
            out.append(cue)
    return out
```

**Testing:**

Tests must verify:
- **vts-v1.AC2.3:** `dedupe_rolling(parse_vtt(fixture))` returns
  STRICTLY FEWER cues than `parse_vtt(fixture)` alone (the 8-cue fixture
  should collapse to 4). Each output cue's text is at least as long as any
  input cue it absorbed.
- Output cues are still in temporal order with non-decreasing `start`.
- Empty input returns empty list (edge case).

Test file: `tests/pipeline/test_captions.py` (append).

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_captions.py -v
```
Expected: all pass.

**Commit:**

```bash
git add pipeline/captions.py tests/pipeline/test_captions.py
git commit -m "feat(vts-v1): dedupe_rolling — collapse YouTube rolling repeats"
```
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-6) -->

<!-- START_TASK_4 -->
### Task 4: `pipeline/frames.py::FixedFpsExtractor` (ffmpeg pass)

**Verifies:** vts-v1.AC3.1 (partial — full verification with the pHash pass in Task 5).

**Files:**
- Create: `pipeline/frames.py`

**Implementation:**

Splits into:
- `_run_ffmpeg(video, out_dir, fps)` — subprocess call producing
  `out_dir/####.jpg` with `-vf "fps=N,scale=-2:720" -qscale:v 4`.
- `_phash_filter(paths, hamming_max)` — pure function: walks paths in
  order, keeps the first frame unconditionally, then keeps each subsequent
  frame iff its pHash differs from the LAST KEPT frame by more than
  `hamming_max`. Returns the filtered list.
- `FixedFpsExtractor.extract(...)` — orchestrates the two and returns
  `list[Frame]`.

The Hamming-vs-last-kept rule (NOT vs previous frame) ensures runs of
near-duplicates collapse to one, regardless of slow drift.

```python
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

    def name(self) -> str:
        return f"FixedFpsExtractor(fps={self._fps},dedup={self._dedup},hamming_max={self._hamming_max})"

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

    def name(self) -> str:
        return "SceneChangeExtractor(v2-stub)"

    def extract(self, video: Path, out_dir: Path) -> list[Frame]:
        raise NotImplementedError(
            "SceneChangeExtractor is a v2 roadmap item. "
            "Use FixedFpsExtractor for v1."
        )
```

**Verification:** (no test in this task — Task 5 has the test fixture and
runs the real extractor).

**Commit:**

```bash
git add pipeline/frames.py
git commit -m "feat(vts-v1): FixedFpsExtractor (ffmpeg + pHash dedup) + SceneChangeExtractor v2 stub"
```
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Frame extractor tests (synthetic videos + pHash pure-function tests)

**Verifies:** vts-v1.AC3.1, vts-v1.AC3.2, vts-v1.AC3.3

**Files:**
- Create: `tests/pipeline/test_frames.py` (unit + small-integration)
- Create: `tests/pipeline/fixtures/.gitkeep` (and any generated mp4s are
  produced by test fixtures, not committed)

**Implementation:**

Tests synthesize tiny mp4s on the fly using ffmpeg's `lavfi` test source.
Two clips:

- `static.mp4` — 4 seconds of a single SMPTE color bar pattern (highly
  duplicate frames; dedup should collapse to 1).
- `cuts.mp4` — concatenation of 2 seconds each of `testsrc`, `mandelbrot`,
  `rgbtestsrc`, `smptebars` (8 seconds, 4 visually distinct segments;
  dedup should keep at least 4 frames, and strictly fewer than the
  no-dedup count of 8).

A pytest fixture generates these in a `tmp_path` so nothing is checked in.

```python
"""Tests for pipeline/frames.py.

Uses ffmpeg's lavfi test sources to generate small synthetic mp4s, so the
tests need no checked-in binary fixtures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image

from pipeline.frames import FixedFpsExtractor, _phash_filter


def _make_static(path: Path, seconds: int = 4) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"smptebars=duration={seconds}:size=320x240:rate=24",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True, capture_output=True,
    )


def _make_cuts(path: Path) -> None:
    # 4 distinct 2-second segments concatenated. Use lavfi sources via
    # concat-demuxer-via-temp-files to avoid filter_complex.
    parts = []
    for i, src in enumerate(["testsrc", "mandelbrot", "rgbtestsrc=size=320x240", "smptebars=size=320x240"]):
        part = path.parent / f"_part_{i}.mp4"
        # lavfi accepts chained params with ':' regardless of whether src
        # already has an '=' assignment (e.g., "testsrc:duration=2" and
        # "rgbtestsrc=size=320x240:duration=2" both parse correctly).
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"{src}:duration=2:rate=24",
                "-pix_fmt", "yuv420p", str(part),
            ],
            check=True, capture_output=True,
        )
        parts.append(part)
    list_file = path.parent / "_concat.txt"
    list_file.write_text("\n".join(f"file '{p}'" for p in parts))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(list_file), "-c", "copy", str(path)],
        check=True, capture_output=True,
    )


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    p = tmp_path / "static.mp4"
    _make_static(p, seconds=4)
    return p


@pytest.fixture
def cuts_video(tmp_path: Path) -> Path:
    p = tmp_path / "cuts.mp4"
    _make_cuts(p)
    return p


def test_fixed_fps_extracts_one_per_second_at_720_or_less(static_video: Path, tmp_path: Path) -> None:
    # vts-v1.AC3.1
    out = tmp_path / "out_no_dedup"
    extractor = FixedFpsExtractor(fps=1.0, dedup=False)
    frames = extractor.extract(static_video, out)
    # 4-second video at 1 fps → exactly 4 frames.
    assert len(frames) == 4
    # Height ≤ 720.
    img = Image.open(frames[0].path)
    assert img.height <= 720


def test_dedup_strictly_reduces_on_cut_heavy_video(cuts_video: Path, tmp_path: Path) -> None:
    # vts-v1.AC3.2
    out_all = tmp_path / "out_all"
    out_dd = tmp_path / "out_dd"
    n_all = len(FixedFpsExtractor(fps=1.0, dedup=False).extract(cuts_video, out_all))
    n_dd = len(FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6).extract(cuts_video, out_dd))
    assert n_dd < n_all
    # Sanity: still at least one frame per visible segment.
    assert n_dd >= 1


def test_dedup_equal_or_fewer_on_any_input(static_video: Path, tmp_path: Path) -> None:
    # vts-v1.AC3.2 (the "any input" half)
    out_all = tmp_path / "out_all"
    out_dd = tmp_path / "out_dd"
    n_all = len(FixedFpsExtractor(fps=1.0, dedup=False).extract(static_video, out_all))
    n_dd = len(FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6).extract(static_video, out_dd))
    assert n_dd <= n_all


def test_phash_filter_never_drops_first(tmp_path: Path) -> None:
    # vts-v1.AC3.3 — pure-function test, synthesize 3 PIL images.
    paths = []
    for i, color in enumerate([(255, 0, 0), (255, 0, 0), (0, 255, 0)]):
        p = tmp_path / f"{i:04d}.jpg"
        Image.new("RGB", (64, 64), color=color).save(p)
        paths.append(p)
    kept = _phash_filter(paths, hamming_max=6)
    assert kept[0] == paths[0]
    assert len(kept) >= 2  # at least red and green
```

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_frames.py -v
```
Expected: 4 tests pass.

**Commit:**

```bash
git add tests/pipeline/test_frames.py tests/pipeline/fixtures/.gitkeep
git commit -m "test(vts-v1): FixedFpsExtractor frame-count + pHash invariants"
```
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: `SceneChangeExtractor` is a stub — confirm via test

**Verifies:** None directly; documents protocol surface.

**Files:**
- Modify: `tests/pipeline/test_frames.py` (append one short test)

**Implementation:**

```python
def test_scene_change_extractor_is_a_v2_stub(tmp_path: Path) -> None:
    from pipeline.frames import SceneChangeExtractor
    with pytest.raises(NotImplementedError) as exc:
        SceneChangeExtractor().extract(tmp_path / "x.mp4", tmp_path / "out")
    assert "v2" in str(exc.value).lower()
```

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_frames.py::test_scene_change_extractor_is_a_v2_stub -v
```
Expected: pass.

**Commit:**

```bash
git add tests/pipeline/test_frames.py
git commit -m "test(vts-v1): SceneChangeExtractor v2 stub asserted"
```
<!-- END_TASK_6 -->

<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 7-8) -->

<!-- START_TASK_7 -->
### Task 7: `pipeline/download.py::download_video_and_captions`

**Verifies:** vts-v1.AC2.1 (via Task 8 ad-hoc smoke; library behavior is
trusted, not unit-tested against a live URL).

**Files:**
- Create: `pipeline/download.py`

**Implementation:**

```python
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
```

**Verification:** Exercised in Task 8.

**Commit:**

```bash
git add pipeline/download.py
git commit -m "feat(vts-v1): download_video_and_captions — yt-dlp wrapper"
```
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: End-to-end ad-hoc smoke script + Phase 3 gate

**Verifies:** vts-v1.AC2.1 (operational), and overall Phase 3 "Done when".

**Files:**
- Create: `scripts/smoke_phase3.py`

**Implementation:**

A script that takes a YouTube URL on argv, invokes the three modules
end-to-end into a temp `data/jobs/<rand>/` directory, and prints a one-line
summary. This is the operator-level proof for Phase 3 — NOT an automated
test (live URLs are flaky and we don't want CI to depend on YouTube).

```python
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
```

**Verification:**

Manual operator test (NOT part of automated CI), with any ~3-minute
instructional YouTube URL:

```bash
source venv/bin/activate
python scripts/smoke_phase3.py "https://www.youtube.com/watch?v=<some-id>"
```
Expected: prints video path + size, raw and deduped cue counts (deduped <
raw), and a kept-frames count (< 1 × video-duration-in-seconds).

Then run the full Phase 3 test suite:

```bash
pytest tests/pipeline/ -v
```
Expected: all tests pass.

**Commit:**

```bash
git add scripts/smoke_phase3.py
git commit -m "feat(vts-v1): smoke_phase3.py — end-to-end download+parse+extract"
```

**Done when:** All `tests/pipeline/` tests pass AND a manual run of
`smoke_phase3.py` against any short instructional YouTube video produces
the expected output shape.
<!-- END_TASK_8 -->

<!-- END_SUBCOMPONENT_C -->
