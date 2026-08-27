"""Builds a NetworkX knowledge graph over LLM-extracted triples across the corpus.

Ported from the notebook's Section 5 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 19), `build_graph()`. Persists to `data/graph/neurosurgery_graph.graphml` so LLM
calls aren't repeated on every run -- pass `--rebuild` (or `rebuild=True`) to force
re-extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from src.config import Config
from src.data.loaders import load_corpus, load_gap_taxonomy
from src.data.schemas import Paper
from src.graphrag.triple_extraction import extract_triples_and_gap
from src.llm.backend import OpenAIBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = REPO_ROOT / "data" / "graph" / "neurosurgery_graph.graphml"


def build_graph(
    corpus: list[Paper], gap_taxonomy: dict[str, str], llm_backend: OpenAIBackend
) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for paper in corpus:
        triples, gap_code = extract_triples_and_gap(paper, gap_taxonomy, llm_backend)

        for triple in triples:
            subject_id = triple.subject.strip().lower()
            object_id = triple.object.strip().lower()

            g.add_node(subject_id, label=triple.subject)
            g.add_node(object_id, label=triple.object)
            g.add_edge(
                subject_id,
                object_id,
                predicate=triple.predicate,
                source_paper=paper.id,
                extracted_gap=gap_code,
                tagged_gaps=list(paper.gap_tags),
            )

    return g


def _to_graphml_safe(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """GraphML attributes must be primitive (str/int/float/bool) -- serialize the
    list-valued `tagged_gaps` attribute and map `None` -> "" for `extracted_gap`."""
    safe = g.copy()
    for _, _, data in safe.edges(data=True):
        data["tagged_gaps"] = json.dumps(data.get("tagged_gaps", []))
        if data.get("extracted_gap") is None:
            data["extracted_gap"] = ""
    return safe


def _from_graphml_safe(g: nx.MultiDiGraph) -> nx.MultiDiGraph:
    for _, _, data in g.edges(data=True):
        raw = data.get("tagged_gaps", "[]")
        data["tagged_gaps"] = json.loads(raw) if raw else []
        if data.get("extracted_gap") == "":
            data["extracted_gap"] = None
    return g


def save_graph(g: nx.MultiDiGraph, path: Path = DEFAULT_GRAPH_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(_to_graphml_safe(g), path)


def load_graph(path: Path = DEFAULT_GRAPH_PATH) -> nx.MultiDiGraph:
    return _from_graphml_safe(nx.read_graphml(path, node_type=str))


def load_or_build_graph(
    corpus: list[Paper],
    gap_taxonomy: dict[str, str],
    llm_backend: OpenAIBackend,
    path: Path = DEFAULT_GRAPH_PATH,
    rebuild: bool = False,
) -> nx.MultiDiGraph:
    """Loads the persisted graph unless `rebuild` is set or no persisted graph exists."""
    path = Path(path)
    if not rebuild and path.exists():
        return load_graph(path)
    g = build_graph(corpus, gap_taxonomy, llm_backend)
    save_graph(g, path)
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="Build (or rebuild) the neurosurgery-AI knowledge graph.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-extraction via LLM calls even if a persisted graph exists.",
    )
    parser.add_argument("--out", default=str(DEFAULT_GRAPH_PATH), help="Output .graphml path.")
    args = parser.parse_args()

    config = Config.from_env()
    corpus = load_corpus(mode=config.CORPUS_MODE)
    gap_taxonomy = load_gap_taxonomy().root
    llm_backend = OpenAIBackend(config)

    g = load_or_build_graph(corpus, gap_taxonomy, llm_backend, path=Path(args.out), rebuild=args.rebuild)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {args.out}")


if __name__ == "__main__":
    main()
