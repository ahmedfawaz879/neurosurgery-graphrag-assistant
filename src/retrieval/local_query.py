"""Vector-index-based local query engine.

Ported from the notebook's Section 4 retrieval-only path
(`notebooks/neurosurgery_graphrag_assistant.ipynb`, cell 16), which
deliberately uses the notebook's own `llm_call()`-equivalent backend for
generation rather than handing LlamaIndex a default OpenAI LLM object --
generation always goes through `OpenAIBackend`, so the quota fallback stays
in one place.
"""

from __future__ import annotations

import time

from llama_index.core import VectorStoreIndex

from src.llm.backend import OpenAIBackend
from src.retrieval.result import QueryResult

LOCAL_QUERY_PROMPT_TEMPLATE = (
    "Answer using ONLY the retrieved paper excerpts below. "
    "Attribute every claim to its paper_id. "
    "If the excerpts do not support the question's premise, say so explicitly "
    "rather than guessing.\n\n"
    "{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


class LocalQueryEngine:
    """Wraps vector-index retrieval + generation behind a single `.query()` call."""

    def __init__(self, index: VectorStoreIndex, llm_backend: OpenAIBackend, similarity_top_k: int = 3):
        self.retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        self.llm_backend = llm_backend

    def query(self, question: str) -> QueryResult:
        t0 = time.perf_counter()

        source_nodes = self.retriever.retrieve(question)
        retrieved_ids = sorted(
            {n.metadata.get("paper_id") for n in source_nodes if n.metadata.get("paper_id") is not None}
        )

        context_parts = []
        for node in source_nodes:
            paper_id = node.metadata.get("paper_id", "UNKNOWN")
            title = node.metadata.get("title", "UNKNOWN")
            text = node.get_content()
            context_parts.append(f"[paper_id={paper_id}]\nTitle: {title}\nExcerpt:\n{text}")
        context = "\n\n".join(context_parts)

        prompt = LOCAL_QUERY_PROMPT_TEMPLATE.format(context_str=context, query_str=question)
        answer = self.llm_backend.generate(prompt)

        latency_s = time.perf_counter() - t0
        return QueryResult(answer=str(answer), retrieved_paper_ids=retrieved_ids, latency_s=latency_s)
