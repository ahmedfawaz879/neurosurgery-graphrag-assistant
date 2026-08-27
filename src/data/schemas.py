"""Pydantic schemas for the corpus, QA set, and gap taxonomy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, RootModel


class Paper(BaseModel):
    id: str
    title: str
    citation: str
    url: str
    abstract: str
    gap_tags: list[str] = Field(default_factory=list)


class QAItem(BaseModel):
    id: str
    type: Literal["local", "global", "trap"]
    paper_ids: list[str]
    question: str
    gold_answer: str
    gold_gaps: list[str] = Field(default_factory=list)
    false_premise: str | None = None


class GapTaxonomy(RootModel[dict[str, str]]):
    """Maps a gap code (e.g. "G1") to its human-readable name."""

    def __contains__(self, code: object) -> bool:
        return code in self.root

    def __getitem__(self, code: str) -> str:
        return self.root[code]

    def __iter__(self):
        return iter(self.root)

    def keys(self):
        return self.root.keys()
