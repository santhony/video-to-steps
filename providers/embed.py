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

    Recognized values:
      - jina_v4 / jina / jina-v4       — JinaEmbedder (multimodal)
      - mlx_clip / mlx-clip / mlx      — MlxClipEmbedder (Apple Silicon)
      - vision_caption / vc / cap      — VisionCaptionEmbedder
          (caption every frame with the vision LLM, then text-embed the
          captions; single-key OpenAI deploys use this so they don't need
          a separate multimodal-embedding provider)
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
    if backend in ("vision_caption", "vc", "cap"):
        from providers.embed_vision_caption import VisionCaptionEmbedder
        from providers.vision import build_vision
        # The text-embed endpoint defaults to the vision provider's API
        # base when text_embed_base_url is empty — the single-key case
        # where vision and embeddings live on the same OpenAI account.
        text_url = (settings.text_embed_base_url or "").strip() or settings.vision_base_url
        text_path = (settings.text_embed_path or "").strip() or "/v1/embeddings"
        text_key = (settings.text_embed_api_key or "").strip() or settings.vision_api_key
        text_model = (settings.text_embed_model or "").strip() or "text-embedding-3-small"
        return VisionCaptionEmbedder(
            captioner=build_vision(settings),
            base_url=text_url,
            path=text_path,
            api_key=text_key,
            model=text_model,
        )
    raise ValueError(
        f"Unknown EMBED_BACKEND={settings.embed_backend!r}. "
        "Valid values: jina_v4, mlx_clip, vision_caption."
    )
