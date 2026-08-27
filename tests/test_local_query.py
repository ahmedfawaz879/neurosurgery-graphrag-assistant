"""Tests for src/ingestion/index_build.py and src/retrieval/local_query.py.

No live embedding or LLM calls: embeddings use LlamaIndex's deterministic
MockEmbedding, generation uses a fake backend.
"""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter

from src.config import Config
from src.data.schemas import Paper
from src.ingestion.index_build import build_index, papers_to_documents
from src.retrieval.local_query import LocalQueryEngine
from src.retrieval.result import QueryResult

FIXTURE_CORPUS = [
    Paper(
        id="paper_a",
        title="Alpha Study",
        citation="Author A, Journal, 2020",
        url="https://example.com/a",
        abstract="This paper studies calibration of prognostic models after external validation.",
        gap_tags=["G1"],
    ),
    Paper(
        id="paper_b",
        title="Beta Study",
        citation="Author B, Journal, 2021",
        url="https://example.com/b",
        abstract="This paper studies scarce annotated video datasets for skull-base surgery.",
        gap_tags=["G7"],
    ),
]


def _fixture_index():
    config = Config(VECTOR_DB="qdrant", QDRANT_MODE=":memory:", USE_OPENAI=False)
    Settings.node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)
    return build_index(FIXTURE_CORPUS, config=config, embed_model=MockEmbedding(embed_dim=8))


def test_papers_to_documents_preserves_metadata():
    docs = papers_to_documents(FIXTURE_CORPUS)
    assert len(docs) == 2
    assert docs[0].metadata["paper_id"] == "paper_a"
    assert docs[0].metadata["title"] == "Alpha Study"
    assert docs[0].text == FIXTURE_CORPUS[0].abstract


def test_local_query_returns_nonempty_retrieved_paper_ids(mocker):
    index = _fixture_index()

    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "Mocked answer attributing the claim to paper_a."

    engine = LocalQueryEngine(index, llm_backend=fake_backend, similarity_top_k=2)
    result = engine.query("What did the calibration study find?")

    assert isinstance(result, QueryResult)
    assert result.retrieved_paper_ids, "expected non-empty retrieved_paper_ids"
    assert set(result.retrieved_paper_ids) <= {"paper_a", "paper_b"}
    assert result.answer == "Mocked answer attributing the claim to paper_a."
    assert result.latency_s >= 0
    fake_backend.generate.assert_called_once()
