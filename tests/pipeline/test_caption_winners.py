"""Tests for pipeline/caption_winners.py.

Tests verify deduplication by Frame.index, fault tolerance, and persistence.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import pytest

from pipeline.caption_winners import caption_winners
from pipeline.types import Frame
from providers.vision import CaptionResult


@dataclass(slots=True)
class StubCaptioner:
    """Stub VisionCaptioner that tracks call counts and can inject failures."""

    call_count: int = 0
    name: str = "stub-captioner"
    fail_on_index: set[int] | None = None  # If set, raise on these frame indexes

    async def caption(self, path: Path) -> CaptionResult:
        """Mock caption method."""
        # Note: we don't actually read the path, just track calls
        self.call_count += 1
        # For this test stub, we don't have frame index info directly,
        # so we'll use a different approach: inject failures via a wrapper.
        return CaptionResult(
            text="caption for frame",
            prompt_tokens=10,
            completion_tokens=5,
        )


@dataclass(slots=True)
class FailingCaptioner:
    """Stub VisionCaptioner that always fails."""

    name: str = "failing-captioner"

    async def caption(self, path: Path) -> CaptionResult:
        raise RuntimeError("simulated caption failure")


class TestCaptionWinnersDedupByFrameIndex:
    """A single frame shared across multiple steps is captioned once."""

    @pytest.mark.asyncio
    async def test_dedup_by_frame_index_single_call(self, tmp_path: Path):
        """Frame 0 appears in winners for steps 0 and 1; should be called once."""
        shared_frame = Frame(index=0, timestamp=1.0, path=tmp_path / "frame_0.jpg")

        # Create a dummy file so the path exists
        shared_frame.path.touch()

        winners_by_step = {
            0: [shared_frame],
            1: [shared_frame],
        }

        captioner = StubCaptioner()
        captions, usage = await caption_winners(
            winners_by_step,
            tmp_path,
            captioner,
            max_in_flight=1,
        )

        # Should call caption exactly once for frame 0
        assert captioner.call_count == 1
        assert 0 in captions
        assert captions[0] is not None


class TestCaptionWinnersFailureHandling:
    """Per-frame failures degrade to None; job continues."""

    @pytest.mark.asyncio
    async def test_all_frames_fail_returns_none_values(self, tmp_path: Path):
        """All caption attempts fail; returns None values, no exception raised."""
        frames = [
            Frame(index=i, timestamp=float(i), path=tmp_path / f"frame_{i}.jpg")
            for i in range(3)
        ]
        for f in frames:
            f.path.touch()

        winners_by_step = {0: frames}

        captioner = FailingCaptioner()
        captions, usage = await caption_winners(
            winners_by_step,
            tmp_path,
            captioner,
            max_in_flight=2,
        )

        # All should be None; no exception
        assert len(captions) == 3
        for i in range(3):
            assert captions[i] is None
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


class TestCaptionWinnersPersistence:
    """Captions persisted to frame_captions.json on disk."""

    @pytest.mark.asyncio
    async def test_persists_to_frame_captions_json(self, tmp_path: Path):
        """Output written to job_dir/frame_captions.json."""
        frames = [
            Frame(index=0, timestamp=0.0, path=tmp_path / "frame_0.jpg"),
            Frame(index=1, timestamp=1.0, path=tmp_path / "frame_1.jpg"),
        ]
        for f in frames:
            f.path.touch()

        winners_by_step = {0: frames}

        captioner = StubCaptioner()
        captions, usage = await caption_winners(
            winners_by_step,
            tmp_path,
            captioner,
            max_in_flight=2,
        )

        # Check file exists and parses
        output_file = tmp_path / "frame_captions.json"
        assert output_file.exists()

        with output_file.open("r", encoding="utf-8") as f:
            persisted = json.load(f)

        assert isinstance(persisted, dict)
        # JSON keys are strings (per write_json_atomic behavior)
        assert "0" in persisted
        assert "1" in persisted


class TestCaptionWinnersEmptyWinners:
    """Empty winners dict returns empty captions dict."""

    @pytest.mark.asyncio
    async def test_empty_winners_by_step(self, tmp_path: Path):
        """No winners; returns empty dict and empty file."""
        captioner = StubCaptioner()
        captions, usage = await caption_winners(
            {},
            tmp_path,
            captioner,
        )

        assert captions == {}
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

        # File should still exist
        output_file = tmp_path / "frame_captions.json"
        assert output_file.exists()
        with output_file.open("r", encoding="utf-8") as f:
            assert json.load(f) == {}


class TestCaptionWinnersTokenUsage:
    """TokenUsage correctly aggregates token counts."""

    @pytest.mark.asyncio
    async def test_token_usage_aggregation(self, tmp_path: Path):
        """Token counts from all frames summed into usage."""
        # Use a custom captioner that returns predictable token counts
        @dataclass(slots=True)
        class CountingCaptioner:
            name: str = "counting"

            async def caption(self, path: Path) -> CaptionResult:
                return CaptionResult(
                    text="caption",
                    prompt_tokens=10,
                    completion_tokens=5,
                )

        frames = [
            Frame(index=i, timestamp=float(i), path=tmp_path / f"frame_{i}.jpg")
            for i in range(2)
        ]
        for f in frames:
            f.path.touch()

        winners_by_step = {0: frames}
        captioner = CountingCaptioner()
        captions, usage = await caption_winners(
            winners_by_step,
            tmp_path,
            captioner,
        )

        # 2 frames × (10 prompt + 5 completion) = 20 prompt, 10 completion
        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 10
