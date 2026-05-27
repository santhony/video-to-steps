"""Round-trip tests for pipeline.storage helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pipeline.storage import read_json, write_json_atomic


def test_write_json_atomic_serializes_datetime(tmp_path: Path):
    """A dict with a datetime value must serialize as ISO-8601."""
    ts = datetime(2026, 5, 27, 12, 34, 56, tzinfo=timezone.utc)
    target = tmp_path / "meta.json"

    write_json_atomic(target, {"published_at": ts, "published_url": None})

    loaded = read_json(target)
    assert loaded["published_url"] is None
    assert loaded["published_at"] == "2026-05-27T12:34:56+00:00"
