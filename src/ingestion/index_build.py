"""Ingestion: papers -> LlamaIndex Documents -> chunked, embedded, and indexed.

Ported from the notebook's Section 4 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 15). `qdrant_mode` is read from Config rather than hardcoded ":memory:" --
production config should point at the `qdrant` service defined in
docker-compose.yml, not in-process mode.

Embeddings: OpenAI by default, with an automatic local fallback. The chat-
completion path (src/llm/backend.py) already has an OpenAI-quota fallback to
a local Qwen model; embeddings are a SEPARATE OpenAI dependency
(LlamaIndex's OpenAIEmbedding calls the /embeddings endpoint directly) and
need their own fallback -- this probes the embedding endpoint with one tiny
call BEFORE committing to build the full index, and falls back to a local
sentence-transformers model if that probe hits the same quota wall.
"""

from __future__ import annotations

from dataclasses import replace

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter

from src.config import Config
from src.data.schemas import Paper
from src.llm.backend import is_quota_error
from src.retrieval.vector_store_backends import get_vector_store

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
OPENAI_EMBED_MODEL = "text-embedding-3-small"
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def papers_to_documents(corpus: list[Paper]) -> list[Document]:
    return [Document(text=p.abstract, metadata={"paper_id": p.id, "title": p.title}) for p in corpus]


def configure_embed_model(config: Config) -> str:
    """Selects and installs `Settings.embed_model`, returning "openai" or "local".

    Probes OpenAI embeddings with one tiny call before committing to it; falls
    back to a local sentence-transformers model on a quota/billing failure. A
    non-quota error is re-raised, never masked as a fallback trigger.
    """
    if config.USE_OPENAI:
        from llama_index.embeddings.openai import OpenAIEmbedding

        openai_embed_model = OpenAIEmbedding(model=OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY or None)
        try:
            openai_embed_model.get_text_embedding("connectivity probe")
            Settings.embed_model = openai_embed_model
            return "openai"
        except Exception as e:
            if not is_quota_error(e):
                raise

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    Settings.embed_model = HuggingFaceEmbedding(model_name=LOCAL_EMBED_MODEL)
    return "local"


def build_index(
    corpus: list[Paper],
    qdrant_mode: str | None = None,
    config: Config | None = None,
    embed_model: BaseEmbedding | None = None,
) -> VectorStoreIndex:
    """Builds (or connects to) the vector index over `corpus`.

    Args:
        corpus: papers to index.
        qdrant_mode: overrides `config.QDRANT_MODE` if given.
        config: defaults to `Config.from_env()`.
        embed_model: if given, skips the OpenAI-probe/fallback dance and uses this
            embedding model directly -- a testability hook so tests never need a
            real OpenAI key or a real local-model download.
    """
    config = config or Config.from_env()
    if qdrant_mode is not None:
        config = replace(config, QDRANT_MODE=qdrant_mode)

    if embed_model is not None:
        Settings.embed_model = embed_model
    else:
        configure_embed_model(config)

    Settings.node_parser = SentenceSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

    vector_store = get_vector_store(config)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    docs = papers_to_documents(corpus)
    return VectorStoreIndex.from_documents(docs, storage_context=storage_context)
