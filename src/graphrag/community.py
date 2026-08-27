"""Community detection and summarization over the knowledge graph.

Ported from the notebook's Section 6 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 21). Louvain is the default community-detection algorithm (no extra external
dependencies); an optional Leiden path is available if the optional `python-igraph`
and `leidenalg` packages are installed.
"""

from __future__ import annotations

import json

import networkx as nx
from networkx.algorithms.community import louvain_communities

from src.llm.backend import OpenAIBackend

COMMUNITY_SUMMARY_PROMPT = """Summarize the shared research pattern in these triples in 2-3 sentences, and
name which gap category (or categories) from this taxonomy they collectively illustrate.

Gap taxonomy: {taxonomy}
Triples:
{triples}

Summary:"""


def detect_communities(g: nx.MultiDiGraph, algorithm: str = "louvain", seed: int = 7) -> list[set[str]]:
    undirected = g.to_undirected()

    if algorithm == "louvain":
        return list(louvain_communities(undirected, seed=seed))

    if algorithm == "leiden":
        try:
            import igraph as ig
            import leidenalg
        except ImportError as e:
            raise ImportError(
                "algorithm='leiden' requires the optional 'python-igraph' and 'leidenalg' packages, "
                "which are not part of this repo's default dependencies -- Louvain is the default "
                "detector for exactly this reason: zero extra external deps out of the box."
            ) from e
        node_list = list(undirected.nodes())
        index_of = {n: i for i, n in enumerate(node_list)}
        edges = [(index_of[u], index_of[v]) for u, v in undirected.edges()]
        ig_graph = ig.Graph(n=len(node_list), edges=edges)
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition, seed=seed)
        return [{node_list[i] for i in community} for community in partition]

    raise ValueError(f"Unknown community-detection algorithm: {algorithm!r} (expected 'louvain' or 'leiden')")


def summarize_community(
    g: nx.MultiDiGraph, nodes: set[str], gap_taxonomy: dict[str, str], llm_backend: OpenAIBackend
) -> str:
    triples_text = "\n".join(
        f"({g.nodes[u]['label']}) --{d['predicate']}--> "
        f"({g.nodes[v]['label']}) [gap: {d.get('extracted_gap')}]"
        for u, v, d in g.edges(data=True)
        if u in nodes and v in nodes
    )
    if not triples_text:
        return ""
    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        taxonomy=json.dumps(gap_taxonomy, ensure_ascii=False), triples=triples_text
    )
    return llm_backend.generate(prompt, max_tokens=180)
