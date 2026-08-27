"""Tests for src/graphrag/ -- triple extraction, graph build, community detection,
local/global search, Neo4j export. No live LLM or Neo4j calls."""

from __future__ import annotations

import json

import networkx as nx

from src.data.schemas import Paper
from src.graphrag.community import detect_communities, summarize_community
from src.graphrag.global_search import graphrag_global_search
from src.graphrag.graph_build import build_graph, load_graph, load_or_build_graph, save_graph
from src.graphrag.local_search import graphrag_local_search
from src.graphrag.neo4j_export import to_neo4j_cypher
from src.graphrag.triple_extraction import Triple, extract_triples_and_gap, parse_llm_json

GAP_TAXONOMY = {"G1": "External validation rarity", "G3": "Poor calibration reporting"}

FIXTURE_CORPUS = [
    Paper(id="paper_a", title="Alpha", citation="c", url="u", abstract="Alpha abstract.", gap_tags=["G1"]),
    Paper(id="paper_b", title="Beta", citation="c", url="u", abstract="Beta abstract.", gap_tags=["G3"]),
]


def _fixed_json(subject: str, predicate: str, obj: str, gap_code) -> str:
    return json.dumps(
        {"triples": [{"subject": subject, "predicate": predicate, "object": obj}], "gap_code": gap_code}
    )


# ---- triple extraction -------------------------------------------------------------


def test_parse_llm_json_handles_plain_json():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_llm_json_handles_markdown_fences():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_llm_json_handles_surrounding_text():
    assert parse_llm_json('Sure, here you go:\n{"a": 1}\nHope that helps!') == {"a": 1}


def test_triple_validator_coerces_invalid_gap_code_to_none():
    t = Triple.model_validate(
        {"subject": "A", "predicate": "rel", "object": "B", "gap_code": "BOGUS"},
        context={"valid_gap_codes": set(GAP_TAXONOMY.keys())},
    )
    assert t.gap_code is None


def test_triple_validator_accepts_valid_gap_code():
    t = Triple.model_validate(
        {"subject": "A", "predicate": "rel", "object": "B", "gap_code": "G1"},
        context={"valid_gap_codes": set(GAP_TAXONOMY.keys())},
    )
    assert t.gap_code == "G1"


def test_extract_triples_and_gap_happy_path(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = _fixed_json("Model X", "validated on", "Cohort Y", "G1")

    triples, gap_code = extract_triples_and_gap(FIXTURE_CORPUS[0], GAP_TAXONOMY, fake_backend)

    assert gap_code == "G1"
    assert len(triples) == 1
    assert triples[0].subject == "Model X"
    assert triples[0].gap_code == "G1"


def test_extract_triples_and_gap_coerces_invalid_gap_code(mocker):
    """An invalid gap_code from a malformed LLM response is coerced to None,
    not silently accepted -- at both the function-return level and the
    per-Triple pydantic-validator level."""
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = _fixed_json("A", "rel", "B", "NOT_A_REAL_GAP")

    triples, gap_code = extract_triples_and_gap(FIXTURE_CORPUS[0], GAP_TAXONOMY, fake_backend)

    assert gap_code is None
    assert triples[0].gap_code is None


def test_extract_triples_and_gap_fails_closed_on_malformed_json(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "not json at all { broken"

    triples, gap_code = extract_triples_and_gap(FIXTURE_CORPUS[0], GAP_TAXONOMY, fake_backend)

    assert triples == []
    assert gap_code is None


# ---- graph build ---------------------------------------------------------------


def test_build_graph_produces_expected_nodes_edges_and_gap_assignment(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.side_effect = [
        _fixed_json("Model X", "validated on", "Cohort Y", "G1"),
        _fixed_json("Tool Z", "degrades under", "Deployment W", "G3"),
    ]

    g = build_graph(FIXTURE_CORPUS, GAP_TAXONOMY, fake_backend)

    assert isinstance(g, nx.MultiDiGraph)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 2

    gaps_by_paper = {d["source_paper"]: d["extracted_gap"] for _, _, d in g.edges(data=True)}
    assert gaps_by_paper == {"paper_a": "G1", "paper_b": "G3"}

    labels = {d["label"] for _, d in g.nodes(data=True)}
    assert labels == {"Model X", "Cohort Y", "Tool Z", "Deployment W"}


def test_build_graph_invalid_gap_code_becomes_none_edge_attribute(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = _fixed_json("A", "rel", "B", "BOGUS_CODE")

    g = build_graph([FIXTURE_CORPUS[0]], GAP_TAXONOMY, fake_backend)

    edge_data = next(iter(g.edges(data=True)))[2]
    assert edge_data["extracted_gap"] is None


def test_save_and_load_graph_roundtrip(tmp_path):
    g = nx.MultiDiGraph()
    g.add_edge(
        "a", "b", predicate="rel", source_paper="paper_a", extracted_gap="G1", tagged_gaps=["G1", "G3"]
    )
    g.add_edge("b", "c", predicate="rel2", source_paper="paper_b", extracted_gap=None, tagged_gaps=[])
    g.nodes["a"]["label"] = "A"
    g.nodes["b"]["label"] = "B"
    g.nodes["c"]["label"] = "C"

    path = tmp_path / "graph.graphml"
    save_graph(g, path)
    loaded = load_graph(path)

    assert loaded.number_of_nodes() == 3
    assert loaded.number_of_edges() == 2
    by_paper = {d["source_paper"]: d for _, _, d in loaded.edges(data=True)}
    assert by_paper["paper_a"]["extracted_gap"] == "G1"
    assert by_paper["paper_a"]["tagged_gaps"] == ["G1", "G3"]
    assert by_paper["paper_b"]["extracted_gap"] is None
    assert by_paper["paper_b"]["tagged_gaps"] == []


def test_load_or_build_graph_uses_persisted_graph_without_llm_calls(tmp_path, mocker):
    g = nx.MultiDiGraph()
    g.add_edge("a", "b", predicate="rel", source_paper="paper_a", extracted_gap="G1", tagged_gaps=[])
    g.nodes["a"]["label"] = "A"
    g.nodes["b"]["label"] = "B"
    path = tmp_path / "graph.graphml"
    save_graph(g, path)

    fake_backend = mocker.Mock()
    result = load_or_build_graph(FIXTURE_CORPUS, GAP_TAXONOMY, fake_backend, path=path, rebuild=False)

    assert result.number_of_nodes() == 2
    fake_backend.generate.assert_not_called()


def test_load_or_build_graph_rebuilds_when_flagged(tmp_path, mocker):
    g = nx.MultiDiGraph()
    g.add_edge("a", "b", predicate="rel", source_paper="paper_a", extracted_gap="G1", tagged_gaps=[])
    g.nodes["a"]["label"] = "A"
    g.nodes["b"]["label"] = "B"
    path = tmp_path / "graph.graphml"
    save_graph(g, path)

    fake_backend = mocker.Mock()
    fake_backend.generate.side_effect = [
        _fixed_json("New Subject", "rel", "New Object", "G1"),
        _fixed_json("Another", "rel", "Thing", "G3"),
    ]

    result = load_or_build_graph(FIXTURE_CORPUS, GAP_TAXONOMY, fake_backend, path=path, rebuild=True)

    assert fake_backend.generate.call_count == 2
    labels = {d["label"] for _, d in result.nodes(data=True)}
    assert "New Subject" in labels


# ---- community detection / summarization -----------------------------------


def test_detect_communities_louvain_default():
    g = nx.MultiDiGraph()
    g.add_edge("a", "b", predicate="p", source_paper="paper_a", extracted_gap="G1", tagged_gaps=["G1"])
    for n in g.nodes():
        g.nodes[n]["label"] = n

    communities = detect_communities(g)
    assert len(communities) >= 1
    assert {"a", "b"} <= set().union(*communities)


def test_summarize_community_calls_llm_and_returns_text(mocker):
    g = nx.MultiDiGraph()
    g.add_edge(
        "model x", "cohort y", predicate="validated on", source_paper="paper_a",
        extracted_gap="G1", tagged_gaps=["G1"],
    )
    g.nodes["model x"]["label"] = "Model X"
    g.nodes["cohort y"]["label"] = "Cohort Y"

    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "This community illustrates external validation issues (G1)."

    summary = summarize_community(g, {"model x", "cohort y"}, GAP_TAXONOMY, fake_backend)

    assert "external validation" in summary
    fake_backend.generate.assert_called_once()


def test_summarize_community_empty_when_no_internal_edges(mocker):
    g = nx.MultiDiGraph()
    g.add_node("isolated")
    g.nodes["isolated"]["label"] = "Isolated"
    fake_backend = mocker.Mock()

    result = summarize_community(g, {"isolated"}, GAP_TAXONOMY, fake_backend)

    assert result == ""
    fake_backend.generate.assert_not_called()


# ---- local / global search --------------------------------------------------


def test_graphrag_local_search_returns_matched_result(mocker):
    g = nx.MultiDiGraph()
    g.add_edge(
        "calibration", "prognostic model", predicate="degrades", source_paper="paper_a",
        extracted_gap="G1", tagged_gaps=["G1"],
    )
    g.nodes["calibration"]["label"] = "Calibration"
    g.nodes["prognostic model"]["label"] = "Prognostic Model"

    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "Calibration degrades according to paper_a."

    result = graphrag_local_search("What happens to calibration?", g, fake_backend)

    assert result.retrieved_paper_ids == ["paper_a"]
    assert result.answer == "Calibration degrades according to paper_a."


def test_graphrag_local_search_no_match_returns_empty_result_without_llm_call(mocker):
    g = nx.MultiDiGraph()
    g.add_edge("x", "y", predicate="p", source_paper="paper_a", extracted_gap=None, tagged_gaps=[])
    g.nodes["x"]["label"] = "Unrelated Thing"
    g.nodes["y"]["label"] = "Another Thing"

    fake_backend = mocker.Mock()
    result = graphrag_local_search("zzz nonmatching qqq", g, fake_backend)

    assert result.retrieved_paper_ids == []
    fake_backend.generate.assert_not_called()


def test_graphrag_global_search_synthesizes_from_relevant_summaries(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.side_effect = [
        "Relevant: calibration issues (G1)",
        "NOT RELEVANT",
        "Synthesized final answer citing G1.",
    ]
    result = graphrag_global_search(
        "What recurs across the corpus?",
        community_summaries=["summary 1", "summary 2"],
        corpus=FIXTURE_CORPUS,
        llm_backend=fake_backend,
    )
    assert result.answer == "Synthesized final answer citing G1."
    assert result.retrieved_paper_ids == ["paper_a", "paper_b"]


def test_graphrag_global_search_no_relevant_summaries(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "NOT RELEVANT"

    result = graphrag_global_search("q", ["s1"], FIXTURE_CORPUS, fake_backend)

    assert result.retrieved_paper_ids == []
    assert "No community summaries" in result.answer


# ---- Neo4j export (generation only -- no live connection) -------------------


def test_to_neo4j_cypher_generates_statements_without_connecting():
    g = nx.MultiDiGraph()
    g.add_edge("a", "b", predicate="rel", source_paper="paper_a", extracted_gap="G1", tagged_gaps=["G1"])
    g.nodes["a"]["label"] = "A"
    g.nodes["b"]["label"] = "B"

    statements = to_neo4j_cypher(g)

    assert len(statements) == 3  # 2 node MERGEs + 1 relationship MERGE
    assert all("MERGE" in s for s in statements)
