"""Unit tests for embedder factory routing."""

from __future__ import annotations

import sys
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


def test_embed_factory_mlx_clip_raises_when_mlx_clip_unavailable(monkeypatch):
    """AC4.4: build_embedder raises RuntimeError for mlx_clip on hosts
    where the dep isn't importable. Uses monkeypatch to simulate the
    ImportError even on Apple Silicon hosts where the package may now
    be installed in the project venv (which is the case on the dev
    machine since v1.0)."""
    # Stub mlx_clip to a sentinel that fails import (Python's import
    # machinery treats `None` in sys.modules as cached ImportError).
    monkeypatch.setitem(sys.modules, "mlx_clip", None)

    settings = Mock()
    settings.embed_backend = "mlx_clip"
    settings.mlx_clip_model = "openai/clip-vit-base-patch32"
    settings.mlx_clip_cache_dir = ""

    with pytest.raises(RuntimeError, match=r"mlx_clip not installed"):
        build_embedder(settings)


def test_embed_factory_unknown_backend():
    """AC4.4: build_embedder raises ValueError for unknown backend."""
    settings = Mock()
    settings.embed_backend = "totally-bogus"

    with pytest.raises(ValueError, match=r"Unknown EMBED_BACKEND.*Valid values.*jina_v4.*mlx_clip"):
        build_embedder(settings)
