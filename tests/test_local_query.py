"""Tests for src/ingestion/index_build.py and src/retrieval/local_query.py.

No live embedding or LLM calls: embeddings use LlamaIndex's deterministic
MockEmbedding, generation uses a fake backend.
"""

from __future__ import annotations

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
from llama_index.core.node_parser import SentenceSplitter

from src.config import Config
from src.data.schemas import Paper
from src.ingestion.index_build import build_index, configure_embed_model, papers_to_documents
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


# ---- embeddings quota fallback (Rule: a SEPARATE OpenAI dependency from chat
# completions -- see src/ingestion/index_build.py's configure_embed_model()) -----


def test_configure_embed_model_falls_back_to_local_on_quota_error(mocker):
    mock_openai_embed = mocker.Mock(spec=BaseEmbedding)
    mock_openai_embed.get_text_embedding.side_effect = RuntimeError(
        "Error code: 429 - You exceeded your current quota, insufficient_quota"
    )
    mocker.patch("llama_index.embeddings.openai.OpenAIEmbedding", return_value=mock_openai_embed)

    fake_local_embed = mocker.Mock(spec=BaseEmbedding)
    mock_hf_cls = mocker.patch(
        "llama_index.embeddings.huggingface.HuggingFaceEmbedding", return_value=fake_local_embed
    )

    config = Config(USE_OPENAI=True, OPENAI_API_KEY="sk-test")
    backend_name = configure_embed_model(config)

    assert backend_name == "local"
    mock_hf_cls.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert Settings.embed_model is fake_local_embed


def test_configure_embed_model_uses_openai_when_probe_succeeds(mocker):
    mock_openai_embed = mocker.Mock(spec=BaseEmbedding)
    mock_openai_embed.get_text_embedding.return_value = [0.1] * 1536
    mocker.patch("llama_index.embeddings.openai.OpenAIEmbedding", return_value=mock_openai_embed)

    config = Config(USE_OPENAI=True, OPENAI_API_KEY="sk-test")
    backend_name = configure_embed_model(config)

    assert backend_name == "openai"
    assert Settings.embed_model is mock_openai_embed


def test_configure_embed_model_reraises_non_quota_error(mocker):
    mock_openai_embed = mocker.Mock(spec=BaseEmbedding)
    mock_openai_embed.get_text_embedding.side_effect = ValueError("an unrelated bug")
    mocker.patch("llama_index.embeddings.openai.OpenAIEmbedding", return_value=mock_openai_embed)

    config = Config(USE_OPENAI=True, OPENAI_API_KEY="sk-test")

    with pytest.raises(ValueError, match="an unrelated bug"):
        configure_embed_model(config)


def test_configure_embed_model_goes_straight_to_local_when_use_openai_false(mocker):
    fake_local_embed = mocker.Mock(spec=BaseEmbedding)
    mock_hf_cls = mocker.patch(
        "llama_index.embeddings.huggingface.HuggingFaceEmbedding", return_value=fake_local_embed
    )
    mock_openai_cls = mocker.patch("llama_index.embeddings.openai.OpenAIEmbedding")

    config = Config(USE_OPENAI=False)
    backend_name = configure_embed_model(config)

    assert backend_name == "local"
    mock_hf_cls.assert_called_once()
    mock_openai_cls.assert_not_called()
