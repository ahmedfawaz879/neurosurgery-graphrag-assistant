"""Central runtime configuration, loaded from environment variables.

Mirrors the notebook's Section 0 config block exactly, including its defaults --
see `notebooks/neurosurgery_graphrag_assistant.ipynb`, cell 3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    """Loaded once from `os.environ` at import time via `Config.from_env()`."""

    USE_OPENAI: bool = True
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = ""
    LOCAL_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"

    CORPUS_MODE: str = "shipped"  # "shipped" | "real_pdf_dir"
    REAL_PDF_DIR: str = "/kaggle/input/your-neurosurgery-ai-papers"

    QDRANT_MODE: str = ":memory:"  # or a URL like "http://qdrant:6333" in production
    VECTOR_DB: str = "qdrant"  # "qdrant" | "pinecone"

    GRAPH_BACKEND: str = "networkx"  # "networkx" | "neo4j"
    NEO4J_URI: str = ""
    NEO4J_USER: str = ""
    NEO4J_PASSWORD: str = ""

    ALLOWED_ORIGIN: str = "*"  # restrict to the deployed frontend origin in production

    @classmethod
    def from_env(cls) -> Config:
        use_openai = _env_bool("USE_OPENAI", True)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if use_openai and not api_key:
            use_openai = False

        return cls(
            USE_OPENAI=use_openai,
            OPENAI_MODEL=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            OPENAI_API_KEY=api_key,
            LOCAL_MODEL_ID=os.environ.get("LOCAL_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"),
            CORPUS_MODE=os.environ.get("CORPUS_MODE", "shipped"),
            REAL_PDF_DIR=os.environ.get("REAL_PDF_DIR", "/kaggle/input/your-neurosurgery-ai-papers"),
            QDRANT_MODE=os.environ.get("QDRANT_MODE", ":memory:"),
            VECTOR_DB=os.environ.get("VECTOR_DB", "qdrant"),
            GRAPH_BACKEND=os.environ.get("GRAPH_BACKEND", "networkx"),
            NEO4J_URI=os.environ.get("NEO4J_URI", ""),
            NEO4J_USER=os.environ.get("NEO4J_USER", ""),
            NEO4J_PASSWORD=os.environ.get("NEO4J_PASSWORD", ""),
            ALLOWED_ORIGIN=os.environ.get("ALLOWED_ORIGIN", "*"),
        )
