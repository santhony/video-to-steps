"""Unit tests for MlxClipEmbedder — import-guarded local embedding model.

Two categories:

* `test_mlx_clip_embedder_raises_on_missing_mlx_clip` — fast, always
  runs; uses monkeypatch to simulate the ImportError path even on
  hosts where mlx_clip is installed locally.
* `test_mlx_clip_embedder_encodes_*` — gated on `RUN_MLX_TESTS=1` AND
  the dep actually being importable. These download (~600 MB on first
  run) and run the real CLIP model, so they're opt-in like the cloud
  tests are.
"""

from __future__ import annotations

import os
import sys
from importlib import util as importlib_util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from providers.embed_mlx_clip import MlxClipEmbedder


def _have_mlx_clip() -> bool:
    """Return True if `import mlx_clip` would currently succeed."""
    try:
        return importlib_util.find_spec("mlx_clip") is not None
    except Exception:
        return False


# ── Always-on: ImportError path ───────────────────────────────────────────────

def test_mlx_clip_embedder_raises_on_missing_mlx_clip(monkeypatch):
    """The constructor must surface a clear RuntimeError on hosts where
    mlx_clip isn't importable, regardless of whether it's installed in
    the current venv."""
    # Force the import inside the constructor to fail by stubbing the
    # entry in sys.modules to None — Python's import machinery treats
    # this as a cached ImportError.
    monkeypatch.setitem(sys.modules, "mlx_clip", None)
    with pytest.raises(RuntimeError, match=r"mlx_clip not installed"):
        MlxClipEmbedder()


# ── Opt-in: real encode round-trip ────────────────────────────────────────────

_RUN_MLX = os.getenv("RUN_MLX_TESTS") == "1" and _have_mlx_clip()
_SKIP_REASON = (
    "RUN_MLX_TESTS != 1 or mlx_clip not importable; "
    "set RUN_MLX_TESTS=1 on Apple Silicon to enable."
)


@pytest.mark.skipif(not _RUN_MLX, reason=_SKIP_REASON)
async def test_mlx_clip_embedder_encodes_images_and_text(tmp_path: Path):
    """End-to-end: a red image + 'red square' should cosine-match higher
    than the same image + 'blue square'. Confirms shape, dtype, and
    L2-normalization invariants the matcher relies on."""
    cache = tmp_path / "mlx_clip_cache"
    embedder = MlxClipEmbedder(
        model="openai/clip-vit-base-patch32", cache_dir=cache
    )
    assert embedder.name.startswith("mlx_clip:")

    red = tmp_path / "red.jpg"
    Image.new("RGB", (224, 224), color=(255, 0, 0)).save(red)

    img_res = await embedder.embed_images([red])
    txt_res = await embedder.embed_texts(["a red square", "a blue square"])

    assert img_res.vectors.dtype == np.float32
    assert txt_res.vectors.dtype == np.float32
    assert img_res.vectors.shape == (1, txt_res.vectors.shape[1])

    # Vectors are L2-normalized; cosine = dot product.
    np.testing.assert_allclose(
        np.linalg.norm(img_res.vectors, axis=1), 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        np.linalg.norm(txt_res.vectors, axis=1), 1.0, atol=1e-5
    )

    sim_red = float(img_res.vectors[0] @ txt_res.vectors[0])
    sim_blue = float(img_res.vectors[0] @ txt_res.vectors[1])
    assert sim_red > sim_blue, (
        f"CLIP sanity failed: red img matched 'blue square' ({sim_blue:.4f}) "
        f"as well or better than 'red square' ({sim_red:.4f})"
    )

    # Billable tokens are always zero for local embedding.
    assert img_res.billable_tokens == 0
    assert txt_res.billable_tokens == 0


@pytest.mark.skipif(not _RUN_MLX, reason=_SKIP_REASON)
async def test_mlx_clip_embedder_rejects_empty_input(tmp_path: Path):
    """Both encoder methods refuse empty input lists with ValueError."""
    embedder = MlxClipEmbedder(cache_dir=tmp_path / "mlx_clip_cache")
    with pytest.raises(ValueError):
        await embedder.embed_images([])
    with pytest.raises(ValueError):
        await embedder.embed_texts([])
