"""Graph-based local search: traverses the graph from entities mentioned in the
question -- an alternative path to `src/retrieval/local_query.py`'s embedding-based
`LocalQueryEngine`, useful when a question is about a *relationship* rather than a
retrievable passage.

Ported from the notebook's Section 7 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 23), `graphrag_local_search()`.

Deliberately a simple keyword-overlap entity match against node labels -- NOT tuned to
compete with the embedding-based `LocalQueryEngine` path. It demonstrates the
graph-traversal alternative explicitly, not a production-grade competitor (see the
README's Honesty / Limitations section).
"""

from __future__ import annotations

import re
import time

import networkx as nx

from src.llm.backend import OpenAIBackend
from src.retrieval.result import QueryResult


def graphrag_local_search(question: str, g: nx.MultiDiGraph, llm_backend: OpenAIBackend) -> QueryResult:
    t0 = time.perf_counter()

    q_words = set(re.findall(r"[a-z]+", question.lower()))
    matched_nodes = [
        n for n, d in g.nodes(data=True) if set(re.findall(r"[a-z]+", d["label"].lower())) & q_words
    ]
    relevant_edges = [(u, v, d) for u, v, d in g.edges(data=True) if u in matched_nodes or v in matched_nodes]

    if not relevant_edges:
        return QueryResult(
            answer="No relevant graph neighborhood found for this question.",
            retrieved_paper_ids=[],
            latency_s=time.perf_counter() - t0,
        )

    triples_text = "\n".join(
        f"({g.nodes[u]['label']}) --{d['predicate']}--> ({g.nodes[v]['label']}) [source: {d['source_paper']}]"
        for u, v, d in relevant_edges
    )
    prompt = (
        "Answer using ONLY these graph facts. If they don't support the question's premise, say so.\n\n"
        f"Facts:\n{triples_text}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = llm_backend.generate(prompt, max_tokens=300)
    retrieved_paper_ids = sorted({d["source_paper"] for _, _, d in relevant_edges})

    return QueryResult(
        answer=answer, retrieved_paper_ids=retrieved_paper_ids, latency_s=time.perf_counter() - t0
    )
