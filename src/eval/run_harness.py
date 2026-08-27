"""Runs every applicable system variant over the QA set and tabulates metrics.

Ported from the notebook's Section 10 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 32). Unlike the companion clinical-rag-eval-harness repo's original gap, this
module DOES call `df.to_csv(args.out, index=False)` explicitly -- the notebook already
demonstrates saving it correctly, so this does not repeat that earlier project's
missing-CSV mistake.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.data.loaders import load_corpus, load_gap_taxonomy, load_qa_set
from src.data.schemas import Paper, QAItem
from src.eval.metrics import (
    citation_attribution_accuracy,
    gap_resolution_accuracy,
    global_coverage_recall,
    is_falsely_confident,
)
from src.graphrag.community import detect_communities, summarize_community
from src.graphrag.global_search import graphrag_global_search
from src.graphrag.graph_build import load_or_build_graph
from src.graphrag.local_search import graphrag_local_search
from src.ingestion.index_build import build_index
from src.llm.backend import OpenAIBackend
from src.orchestration.graph import ask, init_assistant_graph
from src.orchestration.nodes import Deps
from src.retrieval.local_query import LocalQueryEngine

DEFAULT_APPLICABLE: dict[str, list[str]] = {
    "local": ["local_query_only", "graphrag_local_only", "orchestrated"],
    "trap": ["local_query_only", "graphrag_local_only", "orchestrated"],
    # local_query_only kept deliberately for "global" questions, to show it
    # structurally fails a synthesis question no single paper can answer.
    "global": ["local_query_only", "graphrag_global_only", "orchestrated"],
}


@dataclass
class HarnessDeps:
    """Dependencies the harness needs beyond the system callables themselves --
    the LLM backend and gap taxonomy used by the LLM-judged metrics."""

    llm_backend: OpenAIBackend
    gap_taxonomy: dict[str, str]


def _extract_answer_and_retrieved(result: Any) -> tuple[str, list[str]]:
    """Normalizes a system callable's return value -- either a QueryResult
    dataclass or the AssistantState dict `ask()` returns -- into (answer, ids)."""
    if isinstance(result, dict):
        return result.get("answer", ""), result.get("retrieved_paper_ids", [])
    return result.answer, result.retrieved_paper_ids


def run_harness(
    qa_set: list[QAItem],
    systems: dict[str, Callable[[str], Any]],
    applicable_map: dict[str, list[str]],
    deps: HarnessDeps,
) -> pd.DataFrame:
    rows = []

    for item in qa_set:
        qtype = item.type
        for system_name in applicable_map[qtype]:
            t0 = time.perf_counter()
            try:
                result = systems[system_name](item.question)
                answer_text, retrieved = _extract_answer_and_retrieved(result)
            except Exception as e:
                answer_text, retrieved = f"[ERROR: {e}]", []
            latency = time.perf_counter() - t0

            row: dict[str, Any] = {
                "question_id": item.id,
                "question_type": qtype,
                "system": system_name,
                "latency_s": latency,
                "answer": answer_text,
            }

            if qtype == "trap":
                row["falsely_confident"] = float(is_falsely_confident(item, answer_text, deps.llm_backend))
                row["citation_attribution_acc"] = np.nan
                row["gap_resolution_acc"] = np.nan
                row["global_coverage_recall"] = np.nan
            else:
                row["falsely_confident"] = np.nan
                row["citation_attribution_acc"] = citation_attribution_accuracy(retrieved, item.paper_ids)
                row["gap_resolution_acc"] = gap_resolution_accuracy(
                    answer_text, item.gold_gaps, deps.gap_taxonomy, deps.llm_backend
                )
                row["global_coverage_recall"] = (
                    global_coverage_recall(answer_text, item.gold_gaps, deps.gap_taxonomy, deps.llm_backend)
                    if qtype == "global"
                    else np.nan
                )

            rows.append(row)

    return pd.DataFrame(rows)


def build_default_systems(config: Config) -> tuple[dict[str, Callable[[str], Any]], HarnessDeps, list[Paper]]:
    """Wires up the four real system variants (local_query_only, graphrag_local_only,
    graphrag_global_only, orchestrated) against the shipped corpus and graph."""
    llm_backend = OpenAIBackend(config)
    corpus = load_corpus(mode=config.CORPUS_MODE)
    gap_taxonomy = load_gap_taxonomy().root

    index = build_index(corpus, config=config)
    local_query_engine = LocalQueryEngine(index, llm_backend)

    graph = load_or_build_graph(corpus, gap_taxonomy, llm_backend)
    communities = detect_communities(graph)
    community_summaries = [summarize_community(graph, c, gap_taxonomy, llm_backend) for c in communities]

    orchestration_deps = Deps(
        llm_backend=llm_backend,
        local_query_engine=local_query_engine,
        community_summaries=community_summaries,
        corpus=corpus,
    )
    init_assistant_graph(orchestration_deps)

    systems: dict[str, Callable[[str], Any]] = {
        "local_query_only": lambda q: local_query_engine.query(q),
        "graphrag_local_only": lambda q: graphrag_local_search(q, graph, llm_backend),
        "graphrag_global_only": lambda q: graphrag_global_search(q, community_summaries, corpus, llm_backend),
        "orchestrated": lambda q: ask(q),
    }
    harness_deps = HarnessDeps(llm_backend=llm_backend, gap_taxonomy=gap_taxonomy)
    return systems, harness_deps, corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation harness over the QA set.")
    parser.add_argument("--out", default="results/run.csv", help="Output CSV path.")
    args = parser.parse_args()

    config = Config.from_env()
    qa_set = load_qa_set()
    systems, deps, _corpus = build_default_systems(config)

    df = run_harness(qa_set, systems, DEFAULT_APPLICABLE, deps)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Harness run complete: {len(df)} (question x system) results -> {args.out}")


if __name__ == "__main__":
    main()
