"""Orchestration graph nodes: intent classification, generation, attribution
verification, and revision.

Ported from the notebook's Section 8 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 26). Each node is a free function taking `(state, deps)` -- dependency injection,
matching the companion clinical-rag-eval-harness repo's pattern -- rather than closing
over module-level globals, so the graph is testable with fakes.

This targets the false-attribution failure mode directly: does every claim in the
draft answer trace to a `retrieved_paper_id` that was actually returned? A research
assistant that cites the wrong paper for a real claim is arguably worse than one that
says "I don't know," because a researcher may not double-check a plausible-looking
citation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.schemas import Paper
from src.graphrag.global_search import graphrag_global_search
from src.llm.backend import OpenAIBackend
from src.orchestration.state import AssistantState
from src.retrieval.local_query import LocalQueryEngine

INTENT_PROMPT = """Classify this research question as exactly one word: LOCAL (about a specific paper's
finding) or GLOBAL (asking for a pattern/synthesis across multiple papers).

Question: {question}
Classification:"""

ATTRIBUTION_CHECK_PROMPT = """Check whether every specific claim in this draft answer is attributable to
one of the listed retrieved papers. Reply with exactly one word: ATTRIBUTED or UNATTRIBUTED, then a
one-sentence reason.

Retrieved paper IDs: {paper_ids}
Draft answer: {answer}
"""


@dataclass
class Deps:
    """Bundles everything the orchestration nodes need, injected rather than
    imported as module globals so the graph is testable with fakes."""

    llm_backend: OpenAIBackend
    local_query_engine: LocalQueryEngine
    community_summaries: list[str]
    corpus: list[Paper]


def classify_intent_node(state: AssistantState, deps: Deps) -> AssistantState:
    verdict = deps.llm_backend.generate(INTENT_PROMPT.format(question=state["question"]), max_tokens=5)
    intent = "global" if "GLOBAL" in verdict.upper() else "local"
    return {**state, "intent": intent}


def generate_node(state: AssistantState, deps: Deps) -> AssistantState:
    if state["intent"] == "global":
        result = graphrag_global_search(
            state["question"], deps.community_summaries, deps.corpus, deps.llm_backend
        )
    else:
        result = deps.local_query_engine.query(state["question"])
    return {**state, "answer": result.answer, "retrieved_paper_ids": result.retrieved_paper_ids}


def verify_attribution_node(state: AssistantState, deps: Deps) -> AssistantState:
    v = deps.llm_backend.generate(
        ATTRIBUTION_CHECK_PROMPT.format(paper_ids=state["retrieved_paper_ids"], answer=state["answer"]),
        max_tokens=60,
    )
    flag = "UNATTRIBUTED" if "UNATTRIBUTED" in v.upper() else "ATTRIBUTED"
    return {**state, "attribution_flag": flag}


def revise_node(state: AssistantState, deps: Deps) -> AssistantState:
    revise_prompt = (
        "Your previous answer was flagged as containing claims not attributable to the "
        "retrieved papers. Re-answer using ONLY what is directly attributable, explicitly "
        "noting anything you're removing due to insufficient support.\n\n"
        f"Question: {state['question']}\nPrevious answer: {state['answer']}\n\nRevised answer:"
    )
    revised = deps.llm_backend.generate(revise_prompt, max_tokens=350)
    return {**state, "answer": revised, "revised": True}


def should_revise(state: AssistantState) -> str:
    return "revise" if state.get("attribution_flag") == "UNATTRIBUTED" and not state.get("revised") else "end"
