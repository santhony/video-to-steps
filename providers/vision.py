"""VisionCaptioner protocol + factory stub.

Concrete implementation lands in Phase 2. Caption-of-winners only; this is
NOT used to caption every frame. See design § Architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class CaptionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class VisionCaptioner(Protocol):
    name: str

    async def caption(self, image: Path) -> CaptionResult: ...


def build_vision(settings: Any) -> VisionCaptioner:
    """Returns a configured VisionCaptioner. Implemented in Phase 2."""
    raise NotImplementedError("VisionCaptioner factory implemented in Phase 2")
