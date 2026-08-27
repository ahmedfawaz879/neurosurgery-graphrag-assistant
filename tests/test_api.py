"""Tests for src/api/main.py -- FastAPI TestClient, no real index/graph build,
no live LLM calls. The orchestration graph is mocked at both entry points:
`build_startup_deps` (so startup doesn't build a real Qdrant index/graph) and
`ask_assistant` (so /ask doesn't invoke a real LangGraph run)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.orchestration.nodes import Deps


@pytest.fixture
def client(mocker):
    fake_deps = Deps(
        llm_backend=mocker.Mock(), local_query_engine=mocker.Mock(), community_summaries=[], corpus=[]
    )
    mocker.patch("src.api.main.build_startup_deps", return_value=fake_deps)
    mocker.patch("src.api.main.init_assistant_graph")

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_expected_shape(client, mocker):
    mocker.patch(
        "src.api.main.ask_assistant",
        return_value={"answer": "the answer", "retrieved_paper_ids": ["paper_01"], "intent": "local"},
    )

    response = client.post("/ask", json={"question": "What did paper_01 find?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "the answer",
        "retrieved_paper_ids": ["paper_01"],
        "intent": "local",
    }


def test_ask_empty_question_returns_422_not_500(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_missing_question_returns_422_not_500(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_missing_question_error_body_is_clear_not_a_bare_500(client):
    response = client.post("/ask", json={})
    body = response.json()
    assert "detail" in body


def test_ask_backend_failure_returns_clear_502_not_bare_500(client, mocker):
    mocker.patch("src.api.main.ask_assistant", side_effect=RuntimeError("invalid API key"))

    response = client.post("/ask", json={"question": "What did paper_01 find?"})

    assert response.status_code == 502
    assert "invalid API key" in response.json()["detail"]


def test_index_page_serves_static_html(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Neurosurgery" in response.text
