"""Unit tests for JinaEmbedder — multimodal embeddings via Jina API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from providers.embed_jina import JinaEmbedder


def build_jina_response(num_vectors: int, dim: int = 2048) -> dict:
    """Build a Jina API response with synthetic embeddings."""
    import json

    data = []
    for _ in range(num_vectors):
        # Create a random vector with non-trivial magnitude
        vec = np.random.randn(dim).astype(np.float32).tolist()
        data.append({"embedding": vec, "index": len(data)})

    return {
        "model": "jina-embeddings-v4",
        "object": "list",
        "data": data,
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }


@pytest.mark.asyncio
async def test_jina_embedder_embed_images():
    """AC4.3: embed_images returns float32, L2-normalized vectors with consistent dim."""
    response_data = build_jina_response(3, dim=2048)

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=response_data,
        )

    embedder = JinaEmbedder(
        api_key="test-key",
        model="jina-embeddings-v4",
        batch=64,
        _transport=httpx.MockTransport(transport),
    )

    paths = [Path(f"/tmp/image{i}.jpg") for i in range(3)]

    # Mock the file reading to avoid needing actual files
    with patch("pathlib.Path.read_bytes", return_value=b"\xff\xd8\xff"):
        result = await embedder.embed_images(paths)

    assert result.vectors.dtype == np.float32
    assert result.vectors.shape == (3, 2048)
    # Check L2 norm (should be close to 1.0 for L2-normalized vectors)
    norms = np.linalg.norm(result.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert result.billable_tokens == 5


@pytest.mark.asyncio
async def test_jina_embedder_embed_texts():
    """AC4.3: embed_texts returns float32, L2-normalized vectors with consistent dim."""
    response_data = build_jina_response(3, dim=2048)

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=response_data,
        )

    embedder = JinaEmbedder(
        api_key="test-key",
        model="jina-embeddings-v4",
        batch=64,
        _transport=httpx.MockTransport(transport),
    )

    texts = ["text1", "text2", "text3"]
    result = await embedder.embed_texts(texts)

    assert result.vectors.dtype == np.float32
    assert result.vectors.shape == (3, 2048)
    norms = np.linalg.norm(result.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert result.billable_tokens == 5


@pytest.mark.asyncio
async def test_jina_embedder_consistent_dim():
    """AC4.3: embed_images and embed_texts return matching dimensions."""
    response_images = build_jina_response(2, dim=2048)
    response_texts = build_jina_response(2, dim=2048)

    call_count = [0]

    def transport(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] <= 1:
            return httpx.Response(status_code=200, json=response_images)
        else:
            return httpx.Response(status_code=200, json=response_texts)

    embedder = JinaEmbedder(
        api_key="test-key",
        model="jina-embeddings-v4",
        batch=64,
        _transport=httpx.MockTransport(transport),
    )

    with patch("pathlib.Path.read_bytes", return_value=b"\xff\xd8\xff"):
        image_result = await embedder.embed_images([Path("/tmp/img.jpg"), Path("/tmp/img2.jpg")])
    text_result = await embedder.embed_texts(["text1", "text2"])

    assert image_result.vectors.shape[1] == text_result.vectors.shape[1]
    assert image_result.vectors.shape[1] == 2048


@pytest.mark.asyncio
async def test_jina_embedder_batching():
    """AC4.3: Batching works correctly with batch=2 and 5 inputs (3 batches)."""
    # Track how many times _post_batch is called
    responses = [
        build_jina_response(2, dim=2048),  # batch 1: 2 vectors
        build_jina_response(2, dim=2048),  # batch 2: 2 vectors
        build_jina_response(1, dim=2048),  # batch 3: 1 vector
    ]
    call_count = [0]

    def transport(request: httpx.Request) -> httpx.Response:
        idx = call_count[0]
        call_count[0] += 1
        return httpx.Response(status_code=200, json=responses[idx])

    embedder = JinaEmbedder(
        api_key="test-key",
        model="jina-embeddings-v4",
        batch=2,
        _transport=httpx.MockTransport(transport),
    )

    texts = ["text1", "text2", "text3", "text4", "text5"]
    result = await embedder.embed_texts(texts)

    # Should have made 3 batches
    assert call_count[0] == 3
    # Total vectors should be 5
    assert result.vectors.shape[0] == 5
    assert result.vectors.shape[1] == 2048


@pytest.mark.asyncio
async def test_jina_embedder_empty_input_raises():
    """I-1: empty input raises ValueError instead of returning (0, 0) shape."""
    embedder = JinaEmbedder(api_key="test-key")

    with pytest.raises(ValueError, match="embed_images requires at least one path"):
        await embedder.embed_images([])

    with pytest.raises(ValueError, match="embed_texts requires at least one text"):
        await embedder.embed_texts([])

    await embedder.aclose()
