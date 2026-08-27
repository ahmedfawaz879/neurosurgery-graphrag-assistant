"""Tests for src/orchestration/ -- intent routing, generation dispatch, and the
attribution verify/revise loop. No live LLM calls."""

from __future__ import annotations

from src.orchestration.graph import build_assistant_graph
from src.orchestration.nodes import Deps, generate_node
from src.retrieval.result import QueryResult


def _make_deps(mocker, llm_side_effect=None):
    llm_backend = mocker.Mock()
    if llm_side_effect is not None:
        llm_backend.generate.side_effect = llm_side_effect

    local_query_engine = mocker.Mock()
    local_query_engine.query.return_value = QueryResult(
        answer="local answer", retrieved_paper_ids=["paper_a"]
    )

    deps = Deps(
        llm_backend=llm_backend,
        local_query_engine=local_query_engine,
        community_summaries=["summary 1"],
        corpus=[],
    )
    return deps, local_query_engine


# ---- routing ------------------------------------------------------------------


def test_generate_node_calls_global_path_when_intent_is_global(mocker):
    fake_result = QueryResult(answer="ans", retrieved_paper_ids=["p1"])
    mock_global_search = mocker.patch(
        "src.orchestration.nodes.graphrag_global_search", return_value=fake_result
    )
    deps, local_query_engine = _make_deps(mocker)

    new_state = generate_node({"question": "q", "intent": "global"}, deps)

    mock_global_search.assert_called_once_with("q", deps.community_summaries, deps.corpus, deps.llm_backend)
    local_query_engine.query.assert_not_called()
    assert new_state["answer"] == "ans"
    assert new_state["retrieved_paper_ids"] == ["p1"]


def test_generate_node_calls_local_query_engine_when_intent_is_local(mocker):
    mock_global_search = mocker.patch("src.orchestration.nodes.graphrag_global_search")
    deps, local_query_engine = _make_deps(mocker)

    new_state = generate_node({"question": "q", "intent": "local"}, deps)

    local_query_engine.query.assert_called_once_with("q")
    mock_global_search.assert_not_called()
    assert new_state["answer"] == "local answer"


def test_synthesis_question_routes_to_global_end_to_end(mocker):
    """A question containing synthesis language ('across ...') should route to the
    global path once the classifier LLM call returns GLOBAL."""
    fake_global_result = QueryResult(
        answer="global synthesized answer", retrieved_paper_ids=["paper_a", "paper_b"]
    )
    mock_global_search = mocker.patch(
        "src.orchestration.nodes.graphrag_global_search", return_value=fake_global_result
    )
    deps, local_query_engine = _make_deps(mocker, llm_side_effect=["GLOBAL", "ATTRIBUTED"])

    graph = build_assistant_graph(deps)
    final_state = graph.invoke({"question": "What recurs across this corpus?", "revised": False})

    assert final_state["intent"] == "global"
    mock_global_search.assert_called_once()
    local_query_engine.query.assert_not_called()
    assert final_state["answer"] == "global synthesized answer"


def test_local_question_routes_to_local_end_to_end(mocker):
    mock_global_search = mocker.patch("src.orchestration.nodes.graphrag_global_search")
    deps, local_query_engine = _make_deps(mocker, llm_side_effect=["LOCAL", "ATTRIBUTED"])

    graph = build_assistant_graph(deps)
    final_state = graph.invoke({"question": "What did paper X find?", "revised": False})

    assert final_state["intent"] == "local"
    local_query_engine.query.assert_called_once()
    mock_global_search.assert_not_called()
    assert final_state["answer"] == "local answer"


# ---- attribution verify/revise loop --------------------------------------------


def test_unattributed_triggers_exactly_one_revision_and_terminates(mocker):
    deps, _ = _make_deps(
        mocker,
        llm_side_effect=[
            "LOCAL",  # classify_intent
            "UNATTRIBUTED",  # verify_attribution, 1st pass
            "revised answer text",  # revise_node's llm call
            "ATTRIBUTED",  # verify_attribution, 2nd pass (post-revision)
        ],
    )

    graph = build_assistant_graph(deps)
    final_state = graph.invoke({"question": "What did paper X find?", "revised": False})

    assert final_state["revised"] is True
    assert final_state["attribution_flag"] == "ATTRIBUTED"
    assert final_state["answer"] == "revised answer text"
    # exactly 4 LLM calls -- no repeated revise pass (no infinite loop)
    assert deps.llm_backend.generate.call_count == 4


def test_final_state_never_silently_unattributed_without_a_revision_attempt(mocker):
    """Rule: the graph never silently returns an unverified UNATTRIBUTED answer
    without at least attempting one revision. Even if attribution is STILL
    UNATTRIBUTED after the one allowed revision pass, `revised` must be True and
    the graph must terminate (not loop again)."""
    deps, _ = _make_deps(
        mocker,
        llm_side_effect=[
            "LOCAL",
            "UNATTRIBUTED",  # 1st verify
            "revised answer text",
            "UNATTRIBUTED",  # 2nd verify -- still bad
        ],
    )

    graph = build_assistant_graph(deps)
    final_state = graph.invoke({"question": "What did paper X find?", "revised": False})

    assert final_state["attribution_flag"] == "ATTRIBUTED" or final_state["revised"] is True
    assert deps.llm_backend.generate.call_count == 4  # terminates after one revise, does not loop


def test_attributed_on_first_pass_never_revises(mocker):
    deps, _ = _make_deps(mocker, llm_side_effect=["LOCAL", "ATTRIBUTED"])

    graph = build_assistant_graph(deps)
    final_state = graph.invoke({"question": "What did paper X find?", "revised": False})

    assert final_state["attribution_flag"] == "ATTRIBUTED"
    assert final_state.get("revised") is False
    assert deps.llm_backend.generate.call_count == 2
