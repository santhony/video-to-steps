"""MlxClipEmbedder — local multimodal embeddings via mlx_clip (Apple Silicon).

This module is import-safe everywhere — you can `import
providers.embed_mlx_clip` on Linux without error. Instantiation requires
`mlx_clip` to be importable AND the host to run Apple Silicon (the
package's transitive `mlx` dep is macOS-arm64 only). The factory in
providers/embed.py raises a clear RuntimeError on hosts without the dep.

pattern: Imperative Shell
The class wraps the synchronous `mlx_clip.mlx_clip` instance. Public
encoder methods are async by protocol but run the sync MLX call inline
on the event-loop thread because MLX's GPU stream is bound per-thread —
hopping into `asyncio.to_thread` raises "no Stream(gpu, 0)". For an
instructional video's worth of frames the block is short.

Notes on the upstream API (harperreed/mlx_clip):
  - `mlx_clip.mlx_clip(model_dir, hf_repo=...)` is the only constructor.
    If `model_dir` doesn't exist it downloads from `hf_repo` and writes
    the MLX-converted weights to `model_dir`. So the dir doubles as a
    persistent cache between runs.
  - `image_encoder(path)` returns a Python list of floats (512-d for
    `clip-vit-base-patch32`), already L2-normalized.
  - `text_encoder(text)` likewise returns a list of floats in the same
    space.
  - Both methods are blocking; they reuse the loaded MLX weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from providers.embed import EmbedResult


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise. Idempotent — safe to call on
    already-normalized vectors (mlx_clip returns them so), but guards
    against any future model that doesn't."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


# CLIP's BPE vocabulary doesn't cover many Unicode glyphs common in
# real-world instructional text (Vulgar Fractions block, typographic
# quotes, en/em dashes, the degree sign). Hitting one of those raises
# `KeyError: '⅜</w>'` from the tokenizer, which kills the embedding
# call entirely. Map the most common offenders to ASCII before
# tokenizing. Anything still outside vocab gets stripped at the end as
# a defense in depth — better to lose the glyph than the whole step.
_CLIP_TEXT_REWRITES = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5",
    "⅙": "1/6", "⅚": "5/6", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8",
    "°": " degrees", "×": "x", "÷": "/", "±": "+/-",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-", "…": "...",
    "′": "'", "″": '"',
}


def _sanitize_clip_text(text: str) -> str:
    """ASCII-fold the Unicode glyphs CLIP's BPE vocab can't tokenize."""
    for src, dst in _CLIP_TEXT_REWRITES.items():
        if src in text:
            text = text.replace(src, dst)
    # Catch-all: drop anything outside Latin-1 — CLIP's English tokenizer
    # can't represent CJK / emoji either, but those would silently degrade.
    # `errors="ignore"` is intentional: if the glyph survives the explicit
    # map above, lose it rather than fail the entire embed call.
    return text.encode("ascii", errors="ignore").decode("ascii") or " "


def _default_cache_dir(hf_repo: str) -> Path:
    """Pick a stable per-model cache dir under the user's cache.

    The dir name is derived from the HF repo so different models
    coexist. `mlx_clip` writes MLX-converted weights here on first use
    and reloads them next time.
    """
    slug = hf_repo.replace("/", "__")
    return Path.home() / ".cache" / "video-to-steps" / "mlx_clip" / slug


class MlxClipEmbedder:
    """Local multimodal embedder using mlx_clip (Apple Silicon only)."""

    def __init__(
        self,
        *,
        model: str = "openai/clip-vit-base-patch32",
        cache_dir: Path | str | None = None,
    ) -> None:
        try:
            import mlx_clip
        except ImportError as exc:
            raise RuntimeError(
                "mlx_clip not installed. Mode A requires Apple Silicon + "
                "`pip install git+https://github.com/harperreed/mlx_clip`. "
                "On non-Apple-Silicon hosts use EMBED_BACKEND=jina_v4."
            ) from exc

        self.name = f"mlx_clip:{model}"
        self._model_id = model
        cache_path = Path(cache_dir) if cache_dir else _default_cache_dir(model)
        # Upstream quirk: `mlx_clip.download_and_convert_weights` only
        # downloads when the model dir does NOT exist. If we pre-create
        # it, the download is skipped and load fails on the empty dir.
        # So we make sure the PARENT exists and leave the model dir
        # itself for mlx_clip to populate. On subsequent runs the dir
        # is already there (with weights) and gets loaded directly.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # `mlx_clip.mlx_clip` is the package's top-level class. The
        # constructor downloads + converts weights on first call if the
        # directory doesn't exist, then loads them. Subsequent runs
        # reuse the converted weights from `cache_path` and skip the
        # network.
        self._clip = mlx_clip.mlx_clip(str(cache_path), hf_repo=model)

    async def aclose(self) -> None:
        """No-op for compatibility with the async interface."""
        return None

    async def embed_images(self, paths: list[Path]) -> EmbedResult:
        if not paths:
            raise ValueError("embed_images requires at least one path")
        # NOTE: MLX uses per-thread GPU streams. The model is loaded on
        # the thread where this embedder was constructed (the asyncio
        # main thread). Running the encoder via `asyncio.to_thread`
        # raises "There is no Stream(gpu, 0) in current thread." So we
        # do the call inline. For instructional videos (~tens of frames
        # after dedup) the block is short; the cost of yielding control
        # isn't worth the stream-marshalling complexity.
        rows = [self._clip.image_encoder(str(p)) for p in paths]
        vectors = np.asarray(rows, dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=0)

    async def embed_texts(self, texts: list[str]) -> EmbedResult:
        if not texts:
            raise ValueError("embed_texts requires at least one text")
        # Same MLX-thread constraint as embed_images. Sanitize each text
        # first — CLIP's BPE vocab raises KeyError on common DIY glyphs
        # like ⅜ that the model has never seen.
        rows = [self._clip.text_encoder(_sanitize_clip_text(t)) for t in texts]
        vectors = np.asarray(rows, dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=0)
