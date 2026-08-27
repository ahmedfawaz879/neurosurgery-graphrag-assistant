"""Shared result-object shape for every retrieval path (vector, graph-local, graph-global).

Same result-object shape convention as the companion clinical-rag-eval-harness repo, so the
orchestration graph (Prompt #5) and the API layer (Prompt #10) can treat every retrieval path
interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    answer: str
    retrieved_paper_ids: list[str] = field(default_factory=list)
    latency_s: float = 0.0
