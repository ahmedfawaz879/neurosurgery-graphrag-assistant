"""Vector-store backend selection.

Qdrant is the default and only backend exercised by tests/CI (its `:memory:`
mode needs no external service). A Pinecone-based alternative is implemented
for real behind the same interface, selected via `Config.VECTOR_DB`
("qdrant" | "pinecone"), so the repo demonstrates both without committing to
only one -- swapping backends is a one-line config change, not a code change.
"""

from __future__ import annotations

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from src.config import Config

QDRANT_COLLECTION_NAME = "neurosurgery_ai_papers"

# Dimension of OpenAI's text-embedding-3-small, the default embedding model
# (see src/ingestion/index_build.py). Override if pairing Pinecone with a
# different embedding model.
PINECONE_INDEX_DIMENSION = 1536


def _build_qdrant_vector_store(config: Config) -> BasePydanticVectorStore:
    import qdrant_client
    from llama_index.vector_stores.qdrant import QdrantVectorStore

    if config.QDRANT_MODE in (":memory:", ""):
        client = qdrant_client.QdrantClient(location=":memory:")
    else:
        # api_key is required for Qdrant Cloud, ignored by a self-hosted instance.
        client = qdrant_client.QdrantClient(url=config.QDRANT_MODE, api_key=config.QDRANT_API_KEY or None)
    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME)


def _build_pinecone_vector_store(config: Config) -> BasePydanticVectorStore:
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec

    if not config.PINECONE_API_KEY:
        raise ValueError("Config.VECTOR_DB='pinecone' requires PINECONE_API_KEY to be set")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = {i["name"] for i in pc.list_indexes()}
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=PINECONE_INDEX_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    return PineconeVectorStore(pinecone_index=pinecone_index)


def get_vector_store(config: Config) -> BasePydanticVectorStore:
    """Builds the configured vector store. Real, complete implementations for
    both backends -- only Qdrant is exercised without external network access."""
    if config.VECTOR_DB == "qdrant":
        return _build_qdrant_vector_store(config)
    if config.VECTOR_DB == "pinecone":
        return _build_pinecone_vector_store(config)
    raise ValueError(f"Unknown VECTOR_DB backend: {config.VECTOR_DB!r} (expected 'qdrant' or 'pinecone')")
