"""Embedder + FrameExtractor protocols, embedder factory stub.

Embedder vectors MUST be float32, shape (n, d), L2-normalized. The protocol
makes this explicit so cosine similarity reduces to a plain `frame_emb @
step_emb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class EmbedResult:
    vectors: np.ndarray   # shape (n, d), dtype float32, L2-normalized
    billable_tokens: int


class Embedder(Protocol):
    name: str
    async def embed_images(self, paths: list[Path]) -> EmbedResult: ...
    async def embed_texts(self, texts: list[str]) -> EmbedResult: ...


class FrameExtractor(Protocol):
    name: str

    def extract(self, video: Path, out_dir: Path) -> list:
        """Returns a list[pipeline.types.Frame]. Implementations in Phase 3."""
        ...


def build_embedder(settings: Any) -> Embedder:
    """Returns a configured Embedder. Implemented in Phase 2."""
    raise NotImplementedError("Embedder factory implemented in Phase 2")
