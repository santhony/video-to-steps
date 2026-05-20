"""Core pipeline data types.

These are plain dataclasses (not Pydantic) — they're internal contracts, not
external boundaries. Pydantic is reserved for env-driven Settings and any
future incoming HTTP payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Cue:
    """A single timed caption segment parsed from a VTT track."""
    start: float          # seconds
    end: float            # seconds
    text: str


@dataclass(slots=True)
class Frame:
    """One extracted still frame, located in time."""
    index: int            # 0-based ordinal within the frame set
    timestamp: float      # seconds since start of video
    path: Path            # absolute path to the .jpg


@dataclass(slots=True)
class StepOutline:
    """LLM Pass 1 output — coarse step boundary + brief description."""
    index: int            # 0-based ordinal
    start: float          # seconds
    end: float            # seconds
    brief: str            # ≤ 1 sentence describing the step


@dataclass(slots=True)
class Step:
    """LLM Pass 2 output — polished step text + selected illustrating frames."""
    index: int
    start: float
    end: float
    instruction: str      # 1–3 second-person imperative sentences
    frames: list[Frame] = field(default_factory=list)


@dataclass(slots=True)
class TokenUsage:
    """Billable-token counts for a single provider call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0


@dataclass(slots=True)
class CostBreakdown:
    """Running cost totals for a job, in USD."""
    chat_usd: float = 0.0
    vision_usd: float = 0.0
    embed_usd: float = 0.0
    total_usd: float = 0.0


@dataclass(slots=True)
class Manifest:
    """Per-job record persisted to meta.json.

    Only the orchestrator mutates this; the server reads from disk.
    """
    job_id: str
    url: str
    title: str = ""                    # video title from yt-dlp; populated after stage 1
    status: str = "queued"             # queued | running | done | error
    progress: str = ""                 # free-form short description of current stage
    error: str = ""                    # populated when status == "error"
    mode: str = ""                     # "cloud" | "local" | "hybrid" — informational
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    cost: CostBreakdown = field(default_factory=CostBreakdown)
