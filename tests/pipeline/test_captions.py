"""Tests for pipeline/captions.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.captions import parse_vtt


@pytest.fixture
def sample_vtt_path() -> Path:
    """Path to the sample.vtt fixture."""
    return Path(__file__).parent / "fixtures" / "sample.vtt"


def test_parse_vtt(sample_vtt_path: Path) -> None:
    """Verifies AC2.2: parse_vtt returns Cues in temporal order with correct bounds.

    - Returns ≥3 Cue objects
    - Strictly non-decreasing start times
    - First cue.start ≈ 0.48 seconds
    - Last cue.start ≈ 11.5 seconds
    - Every cue has non-empty text
    """
    cues = parse_vtt(sample_vtt_path)

    # At least 3 cues
    assert len(cues) >= 3

    # Non-decreasing start times
    starts = [cue.start for cue in cues]
    assert starts == sorted(starts)

    # First cue starts at approximately 0.48 seconds
    assert abs(cues[0].start - 0.48) < 0.01

    # Last cue starts at approximately 11.5 seconds
    assert abs(cues[-1].start - 11.5) < 0.1

    # Every cue has non-empty text
    for cue in cues:
        assert cue.text
        assert len(cue.text) > 0
