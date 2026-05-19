"""Tests for pipeline/match.py.

Covers vts-v1.AC6.1, vts-v1.AC6.2, vts-v1.AC6.3 — frame-to-step matching
with time window restriction and fallback to nearest midpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.match import match
from pipeline.types import Frame, StepOutline


class TestMatchWindowRestriction:
    """vts-v1.AC6.1: Restrict candidate frames per step to time window."""

    def test_window_restriction_with_padding(self):
        """Frames outside [step.start - pad, step.end + pad] excluded from scoring."""
        # 10 frames at timestamps 0, 1, 2, ..., 9 seconds
        frames = [Frame(index=i, timestamp=float(i), path=None) for i in range(10)]
        # One step at [3.0, 5.0] with pad_sec=1.0 → window [2.0, 6.0]
        # So frames 2, 3, 4, 5, 6 are in window; frames 0, 1, 7, 8, 9 excluded
        steps = [StepOutline(index=0, start=3.0, end=5.0, brief="test")]

        # Uniform embeddings (all 1.0 normalized) — all in-window frames tie
        frame_emb = np.ones((10, 1), dtype=np.float32)
        step_emb = np.ones((1, 1), dtype=np.float32)

        winners = match(steps, frames, frame_emb, step_emb, pad_sec=1.0, top_k=3)

        # Should return frames from window [2, 6]: indexes 2, 3, 4, 5, 6
        assert len(winners) == 1
        assert len(winners[0]) == 3  # top_k=3
        winner_indexes = {f.index for f in winners[0]}
        assert winner_indexes.issubset({2, 3, 4, 5, 6}), \
            f"Winners {winner_indexes} should be subset of {{2,3,4,5,6}}"


class TestMatchTopK:
    """vts-v1.AC6.2: Pick top-k frames by cosine similarity."""

    def test_top_k_ranking_by_cosine(self):
        """Frame 3 has cosine 1.0, frames 1,2 have ~0.5, others ~0.0."""
        # 5 frames, 1 step
        frames = [
            Frame(index=0, timestamp=0.0, path=None),
            Frame(index=1, timestamp=1.0, path=None),
            Frame(index=2, timestamp=2.0, path=None),
            Frame(index=3, timestamp=3.0, path=None),
            Frame(index=4, timestamp=4.0, path=None),
        ]
        steps = [StepOutline(index=0, start=2.0, end=3.0, brief="test")]

        # Hand-crafted L2-normalized embeddings
        # Frame embeddings: [0.0, 0.707, 0.707, 1.0, 0.0] (norms all 1.0)
        # Step embedding:   [1.0]
        # Cosine scores: [0.0, 0.707, 0.707, 1.0, 0.0]
        frame_emb = np.array([
            [0.0],
            [0.707],
            [0.707],
            [1.0],
            [0.0],
        ], dtype=np.float32)
        step_emb = np.array([[1.0]], dtype=np.float32)

        winners = match(steps, frames, frame_emb, step_emb, pad_sec=10.0, top_k=3)

        assert len(winners) == 1
        assert len(winners[0]) == 3
        # First winner should be frame 3 (cosine 1.0)
        assert winners[0][0].index == 3
        # Next two should be frames 1, 2 (both cosine 0.707), order undefined
        assert {winners[0][1].index, winners[0][2].index} == {1, 2}


class TestMatchEmptyWindowFallback:
    """vts-v1.AC6.3: Empty window fallback to nearest frame by midpoint."""

    def test_fallback_to_nearest_midpoint(self):
        """When window is empty, return single frame nearest step midpoint."""
        # 5 frames at times 100, 200, 300, 400, 500
        frames = [
            Frame(index=0, timestamp=100.0, path=None),
            Frame(index=1, timestamp=200.0, path=None),
            Frame(index=2, timestamp=300.0, path=None),
            Frame(index=3, timestamp=400.0, path=None),
            Frame(index=4, timestamp=500.0, path=None),
        ]
        # Step at [10, 15] (entirely before all frames)
        # Midpoint = 12.5, nearest frame is frame 0 (distance 100 - 12.5 = 87.5)
        steps = [StepOutline(index=0, start=10.0, end=15.0, brief="test")]

        frame_emb = np.ones((5, 1), dtype=np.float32)
        step_emb = np.ones((1, 1), dtype=np.float32)

        winners = match(steps, frames, frame_emb, step_emb, pad_sec=1.0, top_k=3)

        assert len(winners) == 1
        assert len(winners[0]) == 1  # Fallback returns single frame
        assert winners[0][0].index == 0


class TestMatchEdgeCases:
    """Additional edge cases."""

    def test_empty_steps_returns_empty_list(self):
        """Empty steps list returns empty list."""
        frames = [Frame(index=0, timestamp=0.0, path=None)]
        frame_emb = np.ones((1, 1), dtype=np.float32)
        step_emb = np.ones((0, 1), dtype=np.float32)

        winners = match([], frames, frame_emb, step_emb)
        assert winners == []

    def test_empty_frames_returns_empty_lists_per_step(self):
        """Empty frames list returns list of empty lists (one per step)."""
        steps = [
            StepOutline(index=0, start=0.0, end=1.0, brief="a"),
            StepOutline(index=1, start=1.0, end=2.0, brief="b"),
        ]
        frame_emb = np.ones((0, 1), dtype=np.float32)
        step_emb = np.ones((2, 1), dtype=np.float32)

        winners = match(steps, [], frame_emb, step_emb)
        assert len(winners) == 2
        assert winners[0] == []
        assert winners[1] == []

    def test_shape_mismatch_raises_value_error(self):
        """Mismatched frame_emb rows and len(frames) raises ValueError."""
        frames = [Frame(index=0, timestamp=0.0, path=None)]
        steps = [StepOutline(index=0, start=0.0, end=1.0, brief="test")]

        # frame_emb has 2 rows but only 1 frame
        frame_emb = np.ones((2, 1), dtype=np.float32)
        step_emb = np.ones((1, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="frame_emb rows.*!=.*len\\(frames\\)"):
            match(steps, frames, frame_emb, step_emb)

    def test_step_emb_shape_mismatch_raises_value_error(self):
        """Mismatched step_emb rows and len(steps) raises ValueError."""
        frames = [Frame(index=0, timestamp=0.0, path=None)]
        steps = [StepOutline(index=0, start=0.0, end=1.0, brief="test")]

        frame_emb = np.ones((1, 1), dtype=np.float32)
        # step_emb has 2 rows but only 1 step
        step_emb = np.ones((2, 1), dtype=np.float32)

        with pytest.raises(ValueError, match="step_emb rows.*!=.*len\\(steps\\)"):
            match(steps, frames, frame_emb, step_emb)
