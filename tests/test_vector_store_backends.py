"""Tests for src/retrieval/vector_store_backends.py -- backend selection and
Qdrant Cloud API-key wiring (needed for the render.yaml/fly.toml default, which
points QDRANT_URL at a Qdrant Cloud cluster). No live network calls."""

from __future__ import annotations

import pytest

from src.config import Config
from src.retrieval.vector_store_backends import get_vector_store


def test_qdrant_memory_mode_ignores_api_key(mocker):
    fake_client_cls = mocker.patch("qdrant_client.QdrantClient")
    mocker.patch("llama_index.vector_stores.qdrant.QdrantVectorStore")

    config = Config(VECTOR_DB="qdrant", QDRANT_MODE=":memory:", QDRANT_API_KEY="unused-key")
    get_vector_store(config)

    fake_client_cls.assert_called_once_with(location=":memory:")


def test_qdrant_url_mode_passes_api_key(mocker):
    fake_client_cls = mocker.patch("qdrant_client.QdrantClient")
    mocker.patch("llama_index.vector_stores.qdrant.QdrantVectorStore")

    config = Config(
        VECTOR_DB="qdrant", QDRANT_MODE="https://my-cluster.qdrant.io", QDRANT_API_KEY="secret-key"
    )
    get_vector_store(config)

    fake_client_cls.assert_called_once_with(url="https://my-cluster.qdrant.io", api_key="secret-key")


def test_qdrant_url_mode_without_api_key_passes_none(mocker):
    fake_client_cls = mocker.patch("qdrant_client.QdrantClient")
    mocker.patch("llama_index.vector_stores.qdrant.QdrantVectorStore")

    config = Config(VECTOR_DB="qdrant", QDRANT_MODE="http://qdrant:6333", QDRANT_API_KEY="")
    get_vector_store(config)

    fake_client_cls.assert_called_once_with(url="http://qdrant:6333", api_key=None)


def test_unknown_vector_db_raises():
    config = Config(VECTOR_DB="not_a_real_backend")
    with pytest.raises(ValueError, match="Unknown VECTOR_DB backend"):
        get_vector_store(config)
