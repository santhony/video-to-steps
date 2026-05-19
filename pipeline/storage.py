"""Atomic JSON write helpers and job-directory resolver.

`write_json_atomic` writes to `path.tmp` then `os.replace(tmp, path)` — on
POSIX this is a single rename inode operation, so concurrent readers (the
HTMX status poll) see either the old or new file but never a torn write.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def job_dir(jobs_root: Path, job_id: str) -> Path:
    """Returns the per-job artifact directory; does NOT create it."""
    return Path(jobs_root) / job_id


def ensure_job_dir(jobs_root: Path, job_id: str) -> Path:
    """Returns the per-job artifact directory, creating it if absent."""
    d = job_dir(jobs_root, job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames").mkdir(exist_ok=True)
    return d


def _to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, Paths, and sets to JSON-safe shapes."""
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        return [_to_jsonable(v) for v in sorted(value, key=str)]
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    """Write `value` as JSON to `path` atomically.

    Writes to `path.with_suffix(path.suffix + ".tmp")` in the same directory,
    then atomically renames. The renaming guarantees readers never see a
    half-written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = _to_jsonable(value)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    """Read JSON from `path`. Raises FileNotFoundError if absent."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
