# video-to-steps Implementation Plan — Phase 5: Match, caption-winners, orchestrator

**Goal:** Wire all prior phases into a single async `run_job(...)` function.
End-to-end run produces `steps.json` and a final `meta.json` with running
and final cost.

**Architecture:**
- `match` is a pure function: per step, restrict frames to the time window
  `[start - pad, end + pad]`, score by `frame_emb @ step_emb`, take top-k.
  Empty-window fallback: pick the single frame nearest to step midpoint.
- `caption_winners` is a bounded-concurrency vision-LLM fanout across the
  unique union of winning frame indexes (so a frame shared by two steps is
  captioned once). Per-frame failures degrade to `caption=None`.
- `run_job` chains everything, updates `manifest.status` and
  `manifest.progress` at every stage boundary, accumulates token counts
  into `manifest.cost`, and writes atomically via Phase 1's
  `write_json_atomic`.

**Tech Stack:** numpy (already a dep), `asyncio.Semaphore`, all earlier
phases.

**Scope:** 5 of 7 phases.

**Codebase verified:** 2026-05-18. Phases 1-4 expected; this phase composes
them and adds no new external deps.

**External dependency findings:**
- ✓ `numpy` matrix-vector multiply: `frame_emb @ step_emb` where
  `frame_emb.shape == (n, d)` and `step_emb.shape == (d,)` yields a 1-D
  array of length n. Since both inputs are L2-normalized (per Phase 2 /
  AC4.3), the dot product IS the cosine similarity — no extra division.
- ✓ `np.save(path, arr)` and `np.load(path)` for the persisted
  `frame_embeddings.npy`. Suitable for v1 single-job scale (tens of MB
  max).
- ✓ `asyncio.gather(*tasks, return_exceptions=True)` lets per-frame
  vision failures surface as exceptions without aborting the gather —
  caller filters and records `None` per failure.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### vts-v1.AC1: End-to-end Mode C pipeline produces a usable result
- **vts-v1.AC1.1 Success:** A YouTube URL of a short instructional video produces a `steps.json` with ≥3 ordered steps, each with ≥1 frame.
- **vts-v1.AC1.2 Success:** Each step's `instruction` field is 1–3 second-person imperative sentences.
- **vts-v1.AC1.3 Success:** Each step's frames are visibly relevant to the instruction text (manual judgment on a checked-in test video).

### vts-v1.AC2: YouTube download and caption parsing (orchestrator side)
- **vts-v1.AC2.4 Failure:** When yt-dlp returns no captions and `WHISPER_FALLBACK` is disabled, the orchestrator writes `status="error"` with a clear human-readable message in `manifest.error`.

### vts-v1.AC6: Frame-to-step matching
- **vts-v1.AC6.1 Success:** `match` restricts candidate frames per step to `[step.start - pad, step.end + pad]` before scoring.
- **vts-v1.AC6.2 Success:** `match` picks the top-k frames by cosine via `frame_emb @ step_emb` within the candidate window.
- **vts-v1.AC6.3 Edge:** When the candidate window is empty, `match` falls back to the single nearest frame to step midpoint.

### vts-v1.AC7: Cost tracking and manifest
- **vts-v1.AC7.1 Success:** At the end of a successful job, `meta.json.cost.total_usd` is non-zero and equals the sum of per-call costs from `pricing.py`.
- **vts-v1.AC7.2 Success:** `meta.json` is atomically written; the status-poll endpoint always returns parseable JSON, never a torn write.
- **vts-v1.AC7.3 Edge:** A configured model not present in `pricing.py` records zero for that line item and logs a startup warning; the job still completes.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: `pipeline/match.py` — pure cosine match with window + fallback

**Verifies:** vts-v1.AC6.1, vts-v1.AC6.2, vts-v1.AC6.3

**Files:**
- Create: `pipeline/match.py`
- Create: `tests/pipeline/test_match.py` (unit)

**Implementation:**

```python
"""Frame-to-step matching.

Pure function: given per-step time bounds and L2-normalized embeddings for
both frames and steps, return per-step top-k Frame winners. The candidate
window restriction respects each step's [start - pad, end + pad] span
before scoring. When that window is empty, fall back to the single frame
nearest the step midpoint.
"""

from __future__ import annotations

import numpy as np

from .types import Frame, StepOutline


def match(
    steps: list[StepOutline],
    frames: list[Frame],
    frame_emb: np.ndarray,         # shape (n_frames, d), L2-normalized float32
    step_emb: np.ndarray,          # shape (n_steps, d),  L2-normalized float32
    *,
    pad_sec: float = 2.0,
    top_k: int = 3,
) -> list[list[Frame]]:
    """Returns winners-per-step in the same order as `steps`.

    Cosine similarity between an L2-normalized frame vector and an
    L2-normalized step vector IS their dot product — so the scoring is a
    plain `frame_emb @ step_emb_row`.
    """
    if not steps:
        return []
    if not frames:
        return [[] for _ in steps]
    if frame_emb.shape[0] != len(frames):
        raise ValueError(f"frame_emb rows ({frame_emb.shape[0]}) != len(frames) ({len(frames)})")
    if step_emb.shape[0] != len(steps):
        raise ValueError(f"step_emb rows ({step_emb.shape[0]}) != len(steps) ({len(steps)})")

    frame_times = np.array([f.timestamp for f in frames], dtype=np.float32)
    out: list[list[Frame]] = []

    for i, step in enumerate(steps):
        lo, hi = step.start - pad_sec, step.end + pad_sec
        in_window = np.where((frame_times >= lo) & (frame_times <= hi))[0]
        if in_window.size == 0:
            # Empty window: nearest-to-midpoint fallback.
            mid = (step.start + step.end) / 2.0
            nearest = int(np.argmin(np.abs(frame_times - mid)))
            out.append([frames[nearest]])
            continue

        scores = frame_emb[in_window] @ step_emb[i]
        order = np.argsort(-scores)              # descending
        winners_local = in_window[order][:top_k]
        out.append([frames[int(j)] for j in winners_local])

    return out
```

**Testing:**

Tests must verify:
- **vts-v1.AC6.1:** Construct 10 frames at timestamps 0..9 and a step at
  `[3.0, 5.0]` with `pad_sec=1.0`. Use uniform embeddings (all rows
  identical) so scoring is a tie; assert the returned set is a subset of
  frame indexes `{2,3,4,5,6}`. (Tie-breaking is implementation-defined; the
  window restriction is what we're checking.)
- **vts-v1.AC6.2:** Construct 5 frames and 1 step, with hand-crafted
  embeddings such that frame 3 has cosine 1.0 with the step, frames 1
  and 2 have cosine ~0.5, others 0.0. Assert `match(..., top_k=3)`
  returns frames `[3, 1, 2]` (or `[3, 2, 1]`) — frame 3 first.
- **vts-v1.AC6.3:** Construct 5 frames at timestamps 100, 200, 300, 400,
  500 and a step at `[10, 15]` (entirely before all frames). With
  `pad_sec=1.0` the window is empty. Assert the returned list is `[frames[0]]`
  (the frame closest to midpoint 12.5 → 100, which is frame index 0).

Test file: `tests/pipeline/test_match.py` (unit).

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_match.py -v
```
Expected: all tests pass.

**Commit:**

```bash
git add pipeline/match.py tests/pipeline/test_match.py
git commit -m "feat(vts-v1): match — cosine pick within time window + midpoint fallback"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: `pipeline/caption_winners.py`

**Files:**
- Create: `pipeline/caption_winners.py`
- Create: `tests/pipeline/test_caption_winners.py` (unit)

**Implementation:**

```python
"""Vision-caption fanout over the unique union of per-step winner frames.

A frame indexed by `Frame.index` is captioned exactly once even if it wins
for multiple steps. Captions are persisted to `job_dir/frame_captions.json`
keyed by frame index. Per-frame failures degrade to `captions[index] = None`
and a warning log.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pipeline.storage import write_json_atomic
from pipeline.types import Frame
from providers.vision import VisionCaptioner

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CaptionUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


async def _caption_one(
    frame: Frame,
    captioner: VisionCaptioner,
    sem: asyncio.Semaphore,
) -> tuple[int, str | None, int, int]:
    async with sem:
        try:
            result = await captioner.caption(frame.path)
            return frame.index, result.text or None, result.prompt_tokens, result.completion_tokens
        except Exception as exc:  # noqa: BLE001
            log.warning("caption_winners frame %d failed: %s", frame.index, exc)
            return frame.index, None, 0, 0


async def caption_winners(
    winners_by_step: dict[int, list[Frame]],
    job_dir: Path,
    captioner: VisionCaptioner,
    *,
    max_in_flight: int = 16,
) -> tuple[dict[int, str | None], CaptionUsage]:
    """Captions each unique winning frame; persists results to frame_captions.json."""
    unique: dict[int, Frame] = {}
    for ws in winners_by_step.values():
        for f in ws:
            unique.setdefault(f.index, f)
    if not unique:
        write_json_atomic(job_dir / "frame_captions.json", {})
        return {}, CaptionUsage()

    sem = asyncio.Semaphore(max_in_flight)
    tasks = [asyncio.create_task(_caption_one(f, captioner, sem)) for f in unique.values()]
    rows = await asyncio.gather(*tasks)

    captions: dict[int, str | None] = {idx: text for idx, text, _, _ in rows}
    usage = CaptionUsage(
        prompt_tokens=sum(p for _, _, p, _ in rows),
        completion_tokens=sum(c for _, _, _, c in rows),
    )

    # Persist (None values are written as JSON null).
    write_json_atomic(job_dir / "frame_captions.json", captions)
    return captions, usage
```

**Testing:**

Tests must verify:
- A single fake `VisionCaptioner` is called once per unique frame even when
  the frame appears in multiple steps' winner lists.
- When the fake raises on every call, the returned `captions` dict has
  `None` values for every frame and the job continues (no propagation).
- `frame_captions.json` is written to disk and contains the expected keys.

Test file: `tests/pipeline/test_caption_winners.py` (unit). Use a stub
captioner class with a per-call counter; use `tmp_path` for the job dir.

**Verification:**

```bash
source venv/bin/activate
pytest tests/pipeline/test_caption_winners.py -v
```
Expected: all tests pass.

**Commit:**

```bash
git add pipeline/caption_winners.py tests/pipeline/test_caption_winners.py
git commit -m "feat(vts-v1): caption_winners — fanout, dedupe, persist, tolerate failures"
```
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_3 -->
### Task 3: `pipeline/pipeline.py::run_job` orchestrator

**Verifies:** vts-v1.AC2.4, vts-v1.AC7.1, vts-v1.AC7.2, vts-v1.AC7.3

**Files:**
- Create: `pipeline/pipeline.py`

**Implementation:**

The orchestrator is the only place where:
- Manifest is mutated and persisted.
- Costs are accumulated.
- The `<no captions + WHISPER_FALLBACK=false>` branch sets `status=error`
  with a clear message and returns (without raising).
- Any unhandled exception in a pipeline stage is caught at the outer level
  and recorded into `manifest.error` (then re-raised so the background
  task surfaces).

```python
"""End-to-end orchestrator: URL in, steps.json out, meta.json updated.

The orchestrator's job is composition. It does not implement any pipeline
logic itself — every stage delegates to a module from earlier phases.

Manifest lifecycle:
- 'queued' → set on creation by the server (not here)
- 'running' → set immediately at run_job start
- 'done' → set on successful completion
- 'error' → set in the outer try/except OR by the captionless-video early-return
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import Settings
from pipeline.caption_winners import caption_winners
from pipeline.captions import dedupe_rolling, parse_vtt
from pipeline.download import download_video_and_captions
from pipeline.frames import FixedFpsExtractor
from pipeline.llm_outline import llm_outline
from pipeline.llm_refine import llm_refine
from pipeline.match import match
from pipeline.storage import ensure_job_dir, read_json, write_json_atomic
from pipeline.types import CostBreakdown, Manifest
from pricing import compute_chat_cost, compute_embed_cost, compute_vision_cost
from providers.embed import build_embedder
from providers.llm import build_llm
from providers.vision import build_vision

log = logging.getLogger(__name__)


def _update(manifest: Manifest, jobs_root: Path, **fields) -> None:
    for k, v in fields.items():
        setattr(manifest, k, v)
    write_json_atomic(jobs_root / manifest.job_id / "meta.json", manifest)


def _config_snapshot(settings: Settings) -> dict:
    return {
        "embed_backend": settings.embed_backend,
        "llm_model": settings.llm_model,
        "vision_model": settings.vision_model,
        "jina_model": settings.jina_model,
        "refine_max_in_flight": settings.refine_max_in_flight,
        "caption_max_in_flight": settings.caption_max_in_flight,
        "whisper_fallback": settings.whisper_fallback,
    }


def _mode_label(settings: Settings) -> str:
    if settings.embed_backend == "mlx_clip":
        return "hybrid" if not settings.llm_base_url.startswith("http://127.") else "local"
    return "cloud"


async def run_job(job_id: str, url: str, settings: Settings, jobs_root: Path) -> None:
    """Runs the full pipeline for one job. Writes meta.json + steps.json."""
    job_dir = ensure_job_dir(jobs_root, job_id)
    manifest = Manifest(
        job_id=job_id,
        url=url,
        status="running",
        progress="starting",
        mode=_mode_label(settings),
        config_snapshot=_config_snapshot(settings),
    )
    _update(manifest, jobs_root)

    try:
        # ── Stage 1: download ──────────────────────────────────────────────
        _update(manifest, jobs_root, progress="downloading video")
        video, vtt = download_video_and_captions(url, job_dir)

        if vtt is None:
            if not settings.whisper_fallback:
                _update(
                    manifest, jobs_root,
                    status="error",
                    progress="",
                    error="This video has no captions. Whisper fallback is on the v2 roadmap.",
                )
                return
            raise RuntimeError("Whisper fallback enabled but not implemented in v1.")

        # ── Stage 2: parse + dedupe captions ───────────────────────────────
        _update(manifest, jobs_root, progress="parsing captions")
        cues = dedupe_rolling(parse_vtt(vtt))

        # ── Stage 3: extract + dedupe frames ───────────────────────────────
        _update(manifest, jobs_root, progress="extracting frames")
        extractor = FixedFpsExtractor(fps=1.0, dedup=True, hamming_max=6)
        frames = extractor.extract(video, job_dir / "frames")

        # ── Stage 4: embed every frame ─────────────────────────────────────
        _update(manifest, jobs_root, progress="embedding frames")
        embedder = build_embedder(settings)
        frame_paths = [f.path for f in frames]
        frame_res = await embedder.embed_images(frame_paths)
        np.save(job_dir / "frame_embeddings.npy", frame_res.vectors)

        # ── Stage 5: outline ──────────────────────────────────────────────
        _update(manifest, jobs_root, progress="outlining steps")
        llm = build_llm(settings)
        outlines, outline_usage = await llm_outline(cues, llm)
        write_json_atomic(job_dir / "outline.json", [asdict(o) for o in outlines])

        # ── Stage 6: embed step briefs ────────────────────────────────────
        _update(manifest, jobs_root, progress="embedding step briefs")
        step_res = await embedder.embed_texts([o.brief for o in outlines])

        # ── Stage 7: match frames to steps ─────────────────────────────────
        _update(manifest, jobs_root, progress="matching frames to steps")
        winners = match(outlines, frames, frame_res.vectors, step_res.vectors)
        winners_by_step = {o.index: ws for o, ws in zip(outlines, winners)}

        # ── Stage 8: caption winning frames ────────────────────────────────
        _update(manifest, jobs_root, progress="captioning representative frames")
        captioner = build_vision(settings)
        captions, caption_usage = await caption_winners(
            winners_by_step, job_dir, captioner,
            max_in_flight=settings.caption_max_in_flight,
        )

        # ── Stage 9: refine each step ──────────────────────────────────────
        _update(manifest, jobs_root, progress="refining step text")
        steps, refine_usage = await llm_refine(
            outlines=outlines,
            cues=cues,
            winners_by_step=winners_by_step,
            captions=captions,
            llm=llm,
            max_in_flight=settings.refine_max_in_flight,
        )

        # ── Stage 10: persist + accumulate cost ────────────────────────────
        write_json_atomic(job_dir / "steps.json", [asdict(s) for s in steps])

        chat_cost = (
            compute_chat_cost(settings.llm_model, outline_usage.prompt_tokens, outline_usage.completion_tokens)
            + compute_chat_cost(settings.llm_model, refine_usage.prompt_tokens, refine_usage.completion_tokens)
        )
        vision_cost = compute_vision_cost(
            settings.vision_model, caption_usage.prompt_tokens, caption_usage.completion_tokens
        )
        embed_cost = (
            compute_embed_cost(embedder.name(), frame_res.billable_tokens)
            + compute_embed_cost(embedder.name(), step_res.billable_tokens)
        )
        cost = CostBreakdown(
            chat_usd=chat_cost,
            vision_usd=vision_cost,
            embed_usd=embed_cost,
            total_usd=chat_cost + vision_cost + embed_cost,
        )

        _update(manifest, jobs_root, status="done", progress="done", cost=cost, error="")

    except Exception as exc:  # noqa: BLE001 — orchestrator policy: surface, don't paper over.
        log.exception("run_job %s failed", job_id)
        _update(
            manifest, jobs_root,
            status="error",
            progress="",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        # Close any provider clients that opened HTTP sessions. Mode A's
        # MlxClipEmbedder has no client to close.
        for closer in (
            getattr(locals().get("embedder"), "aclose", None),
            getattr(locals().get("llm"), "aclose", None),
            getattr(locals().get("captioner"), "aclose", None),
        ):
            if closer is not None:
                try:
                    await closer()
                except Exception:  # noqa: BLE001
                    pass
```

**Verification:** Exercised in Task 4.

**Commit:**

```bash
git add pipeline/pipeline.py
git commit -m "feat(vts-v1): run_job orchestrator (stages, manifest, cost accumulation)"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Orchestrator unit + integration tests

**Verifies:** vts-v1.AC1.1, vts-v1.AC1.2, vts-v1.AC1.3, vts-v1.AC2.4, vts-v1.AC7.1, vts-v1.AC7.2, vts-v1.AC7.3

**Files:**
- Create: `tests/pipeline/test_pipeline.py` (unit + integration; integration marked `@pytest.mark.cloud`)

**Implementation:**

Three classes of test in this file:

**A. Unit: captionless-video error path (AC2.4)** — monkeypatch
`download_video_and_captions` to return `(some_path, None)`; run
`run_job` to completion; assert `meta.json.status == "error"` and the error
message contains "no captions" and "Whisper".

**B. Unit: atomic write contract (AC7.2)** — call `write_json_atomic` in a
loop while a reader thread `read_json`s in another thread. Assert the
reader never sees a JSONDecodeError. (This is a property test of
`pipeline/storage.write_json_atomic`, not of `run_job`, but lives here
because the orchestrator depends on the property.)

**C. Unit: unknown-model pricing zeroing (AC7.3)** — Configure
`settings.llm_model = "totally-bogus-model"`. Build a mocked pipeline (use
monkeypatch to replace the stages with stubs that return zero-token usage
EXCEPT a single mocked `compute_chat_cost('totally-bogus-model', 100, 50)`
which must return 0.0 by `pricing.py` contract. Assert
`meta.json.cost.chat_usd == 0.0` and the job finishes with
`status="done"`. (The startup-warning log is verified by checking
`caplog.records` at WARNING level for the model name.)

**D. Integration (cloud, slow) (AC1.1/AC1.2/AC1.3/AC7.1)** — gated by
`@pytest.mark.cloud` (skipped unless `RUN_CLOUD_TESTS=1`). Uses a
hard-coded short instructional YouTube URL (≤ 3 minutes; choose a knot-tying
or simple recipe video). Asserts:

- `meta.json.status == "done"`
- `len(steps) >= 3`
- For each step: `len(frames) >= 1`, `1 <= sentence_count(instruction) <= 3`
- `meta.json.cost.total_usd > 0`
- `steps.json` parses cleanly.

**vts-v1.AC1.3** (frames visibly relevant) is a manual judgment per the
design plan — the integration test prints step text + frame paths to the
test stdout so a human reviewer can spot-check. The test does NOT assert
visual relevance automatically; it asserts only the structural properties
above.

Test file: `tests/pipeline/test_pipeline.py`.

The exact YouTube URL to use is a planning-time decision; pick one in
collaboration with the user during execution. Pin it as a module-level
constant with a comment naming the video so it can be re-checked.

**Verification:**

```bash
source venv/bin/activate
# Fast path: unit tests only.
pytest tests/pipeline/test_pipeline.py -v -m "not cloud"

# Slow path: integration test (requires .env with real API keys).
RUN_CLOUD_TESTS=1 pytest tests/pipeline/test_pipeline.py -v -m cloud
```
Expected: unit tests pass unconditionally; cloud test passes when keys
configured and skips otherwise.

**Add a default-skip rule** for the `cloud` marker in `pyproject.toml`
(append to `[tool.pytest.ini_options]`):

```toml
addopts = "-m 'not cloud' --strict-markers"
```

This change ensures that `pytest tests/` without args runs the fast
subset; the `RUN_CLOUD_TESTS=1` env override is interpreted by a
`conftest.py` hook.

Create `conftest.py` at repo root:

```python
"""Pytest config — interprets RUN_CLOUD_TESTS to enable @pytest.mark.cloud."""

import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_CLOUD_TESTS") == "1":
        # Cloud tests opt-in: remove the default `-m 'not cloud'` filter
        # by overriding the marker expression.
        config.option.markexpr = ""
```

**Commit:**

```bash
git add tests/pipeline/test_pipeline.py conftest.py pyproject.toml
git commit -m "test(vts-v1): orchestrator covers AC1, AC2.4, AC7.* (cloud opt-in)"
```

**Done when:** Unit tests pass on every developer run; the cloud-gated
integration test passes on a developer's box with `.env` configured. The
"manual frame-relevance check" of AC1.3 is satisfied by the developer
eyeballing the printed step-frame mapping the first time the cloud test is
run, and committing a one-paragraph note to `docs/implementation-plans/2026-05-18-vts-v1/run-notes.md` recording the URL, the kept-frame count, and a thumbs-up/down on relevance.
<!-- END_TASK_4 -->
