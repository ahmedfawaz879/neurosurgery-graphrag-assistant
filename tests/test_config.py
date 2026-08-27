"""Tests for src/config.py env-var loading, including the QDRANT_URL/QDRANT_MODE alias
used by the Docker/deployment configs added in Prompt #9."""

from __future__ import annotations

from src.config import Config


def test_from_env_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_MODE", raising=False)

    config = Config.from_env()

    assert config.QDRANT_MODE == ":memory:"
    assert config.USE_OPENAI is False  # no API key present -> auto-disabled
    assert config.GRAPH_BACKEND == "networkx"
    assert config.VECTOR_DB == "qdrant"


def test_from_env_prefers_qdrant_url_over_qdrant_mode(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_MODE", ":memory:")

    config = Config.from_env()

    assert config.QDRANT_MODE == "http://qdrant:6333"


def test_from_env_falls_back_to_qdrant_mode_alias(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("QDRANT_MODE", "http://legacy-qdrant:6333")

    config = Config.from_env()

    assert config.QDRANT_MODE == "http://legacy-qdrant:6333"


def test_use_openai_auto_disables_without_api_key(monkeypatch):
    monkeypatch.setenv("USE_OPENAI", "true")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = Config.from_env()

    assert config.USE_OPENAI is False
