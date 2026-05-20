"""Embedder + FrameExtractor protocols, embedder factory stub.

Embedder vectors MUST be float32, shape (n, d), L2-normalized. The protocol
makes this explicit so cosine similarity reduces to a plain `frame_emb @
step_emb`.

pattern: Imperative Shell
This module exposes Protocols and factory functions. The factory dynamically
imports concrete embedder implementations; the protocols define the I/O contracts.
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
    """Returns a configured Embedder based on settings.embed_backend.

    Routes to JinaEmbedder for jina backends (jina_v4, jina, jina-v4) or
    MlxClipEmbedder for mlx_clip backends. Raises ValueError for unknown
    backends and RuntimeError if mlx_clip is not installed.
    """
    backend = (settings.embed_backend or "").strip().lower()
    if backend in ("jina_v4", "jina", "jina-v4"):
        from providers.embed_jina import JinaEmbedder
        return JinaEmbedder(
            api_key=settings.jina_api_key,
            model=settings.jina_model,
            batch=settings.jina_embed_batch,
        )
    if backend in ("mlx_clip", "mlx-clip", "mlx"):
        from providers.embed_mlx_clip import MlxClipEmbedder  # may raise at instantiation
        cache_dir = (settings.mlx_clip_cache_dir or "").strip() or None
        return MlxClipEmbedder(model=settings.mlx_clip_model, cache_dir=cache_dir)
    raise ValueError(
        f"Unknown EMBED_BACKEND={settings.embed_backend!r}. "
        "Valid values: jina_v4, mlx_clip."
    )
