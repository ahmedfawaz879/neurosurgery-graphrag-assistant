"""Global search: map-reduce synthesis over every community summary.

Ported from the notebook's Section 7 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 23), `graphrag_global_search()`. The automated counterpart to the author's by-hand
gap-dossier synthesis -- no single paper answers a corpus-wide synthesis question, so
this is the path that answers `global`-type questions.
"""

from __future__ import annotations

import time

from src.data.schemas import Paper
from src.llm.backend import OpenAIBackend
from src.retrieval.result import QueryResult


def graphrag_global_search(
    question: str,
    community_summaries: list[str],
    corpus: list[Paper],
    llm_backend: OpenAIBackend,
) -> QueryResult:
    t0 = time.perf_counter()

    partial_answers = []
    for summary in community_summaries:
        if not summary:
            continue
        p = (
            f"Community summary: {summary}\n\nQuestion: {question}\n\n"
            "Does this summary help answer the question? If yes, extract the relevant point in 1 "
            "sentence and name the gap code(s) it supports; if no, reply exactly 'NOT RELEVANT'."
        )
        r = llm_backend.generate(p, max_tokens=120)
        if "NOT RELEVANT" not in r.upper():
            partial_answers.append(r)

    if not partial_answers:
        return QueryResult(
            answer="No community summaries were relevant to this global question.",
            retrieved_paper_ids=[],
            latency_s=time.perf_counter() - t0,
        )

    reduce_prompt = (
        "Synthesize these partial findings into one coherent answer, explicitly naming which "
        f"gap categories the answer draws on.\n\nQuestion: {question}\n\nPartial findings:\n"
        + "\n".join(f"- {p}" for p in partial_answers)
        + "\n\nSynthesized answer:"
    )
    final = llm_backend.generate(reduce_prompt, max_tokens=350)

    return QueryResult(
        answer=final,
        retrieved_paper_ids=[p.id for p in corpus],
        latency_s=time.perf_counter() - t0,
    )
