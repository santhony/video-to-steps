"""JinaEmbedder — multimodal embeddings via Jina /v1/embeddings.

Both image and text inputs go through the same endpoint; the request body's
`input` array carries `{"text": "..."}` or `{"image": "<data-url-or-url>"}`
objects. Vectors come back float32 (cast on receipt) and L2-normalized
(re-normalized client-side defensively).

Batches are sized by `settings.jina_embed_batch` to stay within the API's
per-request limits; results are concatenated in input order.

pattern: Imperative Shell
This module handles HTTP I/O (httpx) and embedding normalization. Pure
vector normalization logic is internal; the class exposes only the async
I/O interface.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from providers.embed import EmbedResult


def _data_url(image: Path) -> str:
    """Convert an image file to a base64 data URL."""
    raw = Path(image).read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = image.suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{b64}"


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize vectors row-wise."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero for any zero vectors (shouldn't happen, but defensive).
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32, copy=False)


class JinaEmbedder:
    """Multimodal embedder using Jina API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "jina-embeddings-v4",
        batch: int = 64,
        base_url: str = "https://api.jina.ai",
        timeout: float = 120.0,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = model
        self._model = model
        self._batch = batch
        self._url = base_url.rstrip("/") + "/v1/embeddings"
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=_transport,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _post_batch(self, inputs: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
        """Post a batch of inputs to Jina; retry on 429 honoring Retry-After.

        Up to 5 retries with exponential backoff (max 60s sleep). Honors the
        provider's `Retry-After` header when present; otherwise uses 2^n
        seconds. Any non-429 4xx/5xx raises immediately.
        """
        body = {
            "model": self._model,
            "input": inputs,
            "normalized": True,
        }
        for attempt in range(5):
            resp = await self._client.post(self._url, json=body)
            if resp.status_code != 429:
                resp.raise_for_status()
                data = resp.json()
                rows = [item["embedding"] for item in data.get("data", [])]
                vectors = np.asarray(rows, dtype=np.float32)
                usage = data.get("usage") or {}
                tokens = int(usage.get("total_tokens", 0))
                return vectors, tokens
            # 429: honor Retry-After (seconds) or fall back to exponential.
            retry_after = resp.headers.get("Retry-After")
            try:
                sleep_s = float(retry_after) if retry_after else min(2 ** (attempt + 1), 60)
            except ValueError:
                sleep_s = min(2 ** (attempt + 1), 60)
            await asyncio.sleep(sleep_s)
        resp.raise_for_status()  # final raise after retries exhausted
        raise RuntimeError("unreachable")  # for type-checkers

    async def embed_images(self, paths: list[Path]) -> EmbedResult:
        """Embed a list of image paths."""
        if not paths:
            raise ValueError("embed_images requires at least one path")

        all_vectors: list[np.ndarray] = []
        total_tokens = 0
        for i in range(0, len(paths), self._batch):
            chunk = paths[i : i + self._batch]
            inputs = [{"image": _data_url(p)} for p in chunk]
            vecs, tokens = await self._post_batch(inputs)
            all_vectors.append(vecs)
            total_tokens += tokens

        vectors = np.concatenate(all_vectors, axis=0)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=total_tokens)

    async def embed_texts(self, texts: list[str]) -> EmbedResult:
        """Embed a list of text strings."""
        if not texts:
            raise ValueError("embed_texts requires at least one text")

        all_vectors: list[np.ndarray] = []
        total_tokens = 0
        for i in range(0, len(texts), self._batch):
            chunk = texts[i : i + self._batch]
            inputs = [{"text": t} for t in chunk]
            vecs, tokens = await self._post_batch(inputs)
            all_vectors.append(vecs)
            total_tokens += tokens

        vectors = np.concatenate(all_vectors, axis=0)
        return EmbedResult(vectors=_l2_normalize(vectors), billable_tokens=total_tokens)
