"""Unit tests for embedder factory routing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from providers.embed import build_embedder
from providers.embed_jina import JinaEmbedder


def test_embed_factory_jina_v4_backend():
    """AC4.4: build_embedder returns JinaEmbedder for jina_v4 backend."""
    settings = Mock()
    settings.embed_backend = "jina_v4"
    settings.jina_api_key = "test-key"
    settings.jina_model = "jina-embeddings-v4"
    settings.jina_embed_batch = 64

    embedder = build_embedder(settings)

    assert isinstance(embedder, JinaEmbedder)
    assert embedder.name == "jina-embeddings-v4"


def test_embed_factory_jina_alias():
    """AC4.4: build_embedder recognizes 'jina' as alias for jina_v4."""
    settings = Mock()
    settings.embed_backend = "jina"
    settings.jina_api_key = "test-key"
    settings.jina_model = "jina-embeddings-v4"
    settings.jina_embed_batch = 64

    embedder = build_embedder(settings)

    assert isinstance(embedder, JinaEmbedder)


def test_embed_factory_jina_dash_v4_alias():
    """AC4.4: build_embedder recognizes 'jina-v4' as alias for jina_v4."""
    settings = Mock()
    settings.embed_backend = "jina-v4"
    settings.jina_api_key = "test-key"
    settings.jina_model = "jina-embeddings-v4"
    settings.jina_embed_batch = 64

    embedder = build_embedder(settings)

    assert isinstance(embedder, JinaEmbedder)


def test_embed_factory_mlx_clip_raises_on_linux():
    """AC4.4: build_embedder raises RuntimeError for mlx_clip on Linux."""
    settings = Mock()
    settings.embed_backend = "mlx_clip"
    settings.mlx_clip_model = "openai/clip-vit-base-patch32"

    with pytest.raises(RuntimeError, match=r"mlx_clip not installed"):
        build_embedder(settings)


def test_embed_factory_unknown_backend():
    """AC4.4: build_embedder raises ValueError for unknown backend."""
    settings = Mock()
    settings.embed_backend = "totally-bogus"

    with pytest.raises(ValueError, match=r"Unknown EMBED_BACKEND.*Valid values.*jina_v4.*mlx_clip"):
        build_embedder(settings)
