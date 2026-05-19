"""Vision-caption fanout over the unique union of per-step winner frames.

A frame indexed by `Frame.index` is captioned exactly once even if it wins
for multiple steps. Captions are persisted to `job_dir/frame_captions.json`
keyed by frame index. Per-frame failures degrade to `captions[index] = None`
and a warning log.

pattern: Imperative Shell
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from pipeline.storage import write_json_atomic
from pipeline.types import Frame, TokenUsage
from providers.vision import VisionCaptioner

log = logging.getLogger(__name__)


async def _caption_one(
    frame: Frame,
    captioner: VisionCaptioner,
    sem: asyncio.Semaphore,
) -> tuple[int, str | None, int, int]:
    """Caption a single frame; return (index, text or None, prompt_tokens, completion_tokens)."""
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
) -> tuple[dict[int, str | None], TokenUsage]:
    """Captions each unique winning frame; persists results to frame_captions.json.

    Returns a tuple of (captions_dict, usage) where captions_dict maps
    frame index to caption text (or None on failure), and usage aggregates
    token counts across all captioned frames.
    """
    # Deduplicate by Frame.index
    unique: dict[int, Frame] = {}
    for ws in winners_by_step.values():
        for f in ws:
            unique.setdefault(f.index, f)

    if not unique:
        write_json_atomic(job_dir / "frame_captions.json", {})
        return {}, TokenUsage()

    # Fanout with concurrency limit
    sem = asyncio.Semaphore(max_in_flight)
    tasks = [asyncio.create_task(_caption_one(f, captioner, sem)) for f in unique.values()]
    rows = await asyncio.gather(*tasks, return_exceptions=False)

    # Extract captions and aggregate token usage
    captions: dict[int, str | None] = {idx: text for idx, text, _, _ in rows}
    usage = TokenUsage(
        prompt_tokens=sum(p for _, _, p, _ in rows),
        completion_tokens=sum(c for _, _, _, c in rows),
    )

    # Persist (None values are written as JSON null)
    write_json_atomic(job_dir / "frame_captions.json", captions)
    return captions, usage
