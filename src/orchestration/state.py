"""Shared state schema for the orchestration graph.

Ported from the notebook's Section 8 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 26), `AssistantState`.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class AssistantState(TypedDict, total=False):
    question: str
    intent: str
    answer: str
    retrieved_paper_ids: list[str]
    attribution_flag: str
    revised: bool
