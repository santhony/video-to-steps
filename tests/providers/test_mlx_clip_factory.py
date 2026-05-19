"""Unit tests for MlxClipEmbedder — import-guarded local embedding model."""

from __future__ import annotations

import pytest

from providers.embed_mlx_clip import MlxClipEmbedder


def test_mlx_clip_embedder_raises_on_missing_mlx_clip():
    """AC4.4: MlxClipEmbedder raises RuntimeError on hosts without mlx_clip."""
    with pytest.raises(RuntimeError, match=r"mlx_clip not installed"):
        MlxClipEmbedder()
