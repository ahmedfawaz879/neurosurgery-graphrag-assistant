"""Compiles the orchestration StateGraph and exposes the assistant's public entrypoint.

Ported from the notebook's Section 8 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 26). This is the module a live deployment's API layer (src/api/main.py) calls
directly -- `ask()`'s signature stays exactly `ask(question: str) -> dict` so the API
layer's request handler is a thin wrapper.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from src.orchestration.nodes import (
    Deps,
    classify_intent_node,
    generate_node,
    revise_node,
    should_revise,
    verify_attribution_node,
)
from src.orchestration.state import AssistantState

_ASSISTANT_GRAPH = None


def build_assistant_graph(deps: Deps):
    """Builds and compiles the intent-routing / generation / attribution-verification
    StateGraph, with `deps` bound into every node via `functools.partial`."""
    builder = StateGraph(AssistantState)
    builder.add_node("classify_intent", partial(classify_intent_node, deps=deps))
    builder.add_node("generate", partial(generate_node, deps=deps))
    builder.add_node("verify_attribution", partial(verify_attribution_node, deps=deps))
    builder.add_node("revise", partial(revise_node, deps=deps))

    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "generate")
    builder.add_edge("generate", "verify_attribution")
    builder.add_conditional_edges("verify_attribution", should_revise, {"revise": "revise", "end": END})
    builder.add_edge("revise", "verify_attribution")

    return builder.compile()


def init_assistant_graph(deps: Deps):
    """Compiles the assistant graph once and stores it as a module-level singleton.
    Call this once at process startup (see src/api/main.py's startup event) so `ask()`
    can stay a pure `(question: str) -> dict` function."""
    global _ASSISTANT_GRAPH
    _ASSISTANT_GRAPH = build_assistant_graph(deps)
    return _ASSISTANT_GRAPH


def ask(question: str) -> dict:
    """The single public entrypoint. Requires `init_assistant_graph(deps)` to have
    been called first (e.g. at API startup)."""
    if _ASSISTANT_GRAPH is None:
        raise RuntimeError(
            "Assistant graph not initialized -- call init_assistant_graph(deps) once "
            "at process startup before calling ask()."
        )
    return _ASSISTANT_GRAPH.invoke({"question": question, "revised": False})
