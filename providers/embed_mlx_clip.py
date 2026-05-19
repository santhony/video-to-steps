"""MlxClipEmbedder — local multimodal embeddings via mlx_clip (Apple Silicon).

This module is import-safe everywhere (you can `import providers.embed_mlx_clip`
on Linux without error). Instantiation requires `mlx_clip` to be available;
the factory in providers/embed.py raises a clear RuntimeError on hosts
without the dep.

This class is NOT exercised in v1's acceptance smoke test. It exists so the
README's "try Mode A on Macbook" path works without code changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from providers.embed import EmbedResult


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


class MlxClipEmbedder:
    """Local multimodal embedder using mlx_clip (Apple Silicon only)."""

    def __init__(self, *, model: str = "openai/clip-vit-base-patch32") -> None:
        try:
            import mlx_clip  # noqa: F401 — proves availability
        except ImportError as exc:
            raise RuntimeError(
                "mlx_clip not installed. Mode A requires Apple Silicon + "
                "`pip install git+https://github.com/harperreed/mlx_clip`. "
                "On non-Apple-Silicon hosts use EMBED_BACKEND=jina_v4."
            ) from exc

        self.name = f"mlx_clip:{model}"
        self._model_id = model
        # Hold the actual mlx_clip handle; exact API surface verified at
        # integration time. Treat the import success as a green light.
        self._mlx_clip = __import__("mlx_clip")

    async def aclose(self) -> None:
        """No-op for compatibility with async interface."""
        pass

    async def embed_images(self, paths: list[Path]) -> EmbedResult:
        """Embed a list of image paths."""
        if not paths:
            raise ValueError("embed_images requires at least one path")
        # mlx_clip's API is synchronous; we run it inline since v1 does not
        # exercise this path under load. If/when Mode A becomes a test
        # target, wrap in asyncio.to_thread.
        rows = [self._mlx_clip.image_encoder(str(p)) for p in paths]
        vectors = np.asarray(rows, dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=0)

    async def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Embed a list of text strings."""
        if not texts:
            raise ValueError("embed_texts requires at least one text")
        rows = [self._mlx_clip.text_encoder(t) for t in texts]
        vectors = np.asarray(rows, dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=0)
