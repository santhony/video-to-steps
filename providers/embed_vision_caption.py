"""VisionCaptionEmbedder — single-provider multimodal embeddings.

Embed images by first captioning each frame with a vision LLM, then text-
embedding the caption via an OpenAI-shape `/v1/embeddings` endpoint. Step
texts embed directly through the same endpoint. Both vectors live in the
same text-embedding space, so cosine similarity in `pipeline/match.py` is
unchanged.

This embedder exists primarily for the single-key deployment story: with
`OPENAI_API_KEY` alone, this class plus `LLMClient` + `VisionCaptioner` can
cover all three model roles (chat, vision, embeddings) — no Jina account
needed. Trade-off versus JinaEmbedder: every kept frame is captioned
(typically 3–10× more vision-LLM calls than the existing `caption_winners`
stage), which raises cost on long videos. Caption results are buffered
in-process per-call; we do not currently dedupe captions across calls.

pattern: Imperative Shell
This module orchestrates HTTP I/O (httpx) and delegates per-frame
captioning to a `VisionCaptioner` (also imperative shell). The pure
L2-normalization helper is internal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from providers.embed import EmbedResult
from providers.vision import VisionCaptioner


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


class VisionCaptionEmbedder:
    """Multimodal embedder that fans out per-frame captions then text-embeds."""

    def __init__(
        self,
        *,
        captioner: VisionCaptioner,
        base_url: str,
        path: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch: int = 96,
        caption_concurrency: int = 16,
        timeout: float = 120.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = f"vision_caption:{captioner.name}+{model}"
        self._captioner = captioner
        self._text_url = base_url.rstrip("/") + path
        self._text_model = model
        self._batch = batch
        # Cap concurrent vision calls so the provider's per-minute rate limit
        # doesn't get hammered when a job has 100+ frames. The existing
        # caption_winners stage uses the same default ceiling.
        self._caption_sem = asyncio.Semaphore(caption_concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=_transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._captioner.aclose()

    async def _caption_one(self, path: Path) -> str:
        async with self._caption_sem:
            try:
                result = await self._captioner.caption(path)
                return (result.text or "").strip() or "(empty caption)"
            except Exception as exc:  # noqa: BLE001
                # Per-frame caption failures (content policy, transient 5xx)
                # degrade gracefully — embed a placeholder so the frame still
                # has SOME vector and downstream matching doesn't crash. The
                # placeholder mentions the failure so debugging is possible.
                return f"(caption unavailable: {type(exc).__name__})"

    async def embed_images(self, paths: list[Path]) -> EmbedResult:
        if not paths:
            raise ValueError("embed_images requires at least one path")
        captions = await asyncio.gather(*[self._caption_one(p) for p in paths])
        # Text-embed the captions. Token counts from the embed step are the
        # "billable_tokens" for the embed; the vision-caption calls are
        # accounted separately (they go through pricing.compute_vision via
        # the existing CostBreakdown flow if/when the orchestrator records
        # their usage).
        return await self.embed_texts(captions)

    async def embed_texts(self, texts: list[str]) -> EmbedResult:
        if not texts:
            raise ValueError("embed_texts requires at least one text")
        # Batch inputs to /v1/embeddings — OpenAI accepts an array of strings
        # in one call. Batch size matches Jina's default for parity.
        all_vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(texts), self._batch):
            chunk = texts[start : start + self._batch]
            resp = await self._client.post(
                self._text_url,
                json={"model": self._text_model, "input": chunk},
            )
            resp.raise_for_status()
            data = resp.json()
            # OpenAI shape: {"data":[{"embedding":[...]},...], "usage":{...}}
            for entry in data.get("data", []):
                all_vectors.append(entry["embedding"])
            usage = data.get("usage") or {}
            total_tokens += int(usage.get("total_tokens", 0) or usage.get("prompt_tokens", 0))
        vectors = np.asarray(all_vectors, dtype=np.float32)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=total_tokens)
