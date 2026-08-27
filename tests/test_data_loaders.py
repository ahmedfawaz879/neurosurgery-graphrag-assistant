"""Tests for the data layer: schema parsing + cross-file referential integrity."""

from __future__ import annotations

from src.data.loaders import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_QA_PATH,
    DEFAULT_TAXONOMY_PATH,
    load_corpus,
    load_gap_taxonomy,
    load_qa_set,
)
from src.data.schemas import Paper, QAItem


def test_literature_corpus_parses_into_paper_schema():
    papers = load_corpus(mode="shipped")
    assert len(papers) == 14
    assert all(isinstance(p, Paper) for p in papers)


def test_qa_set_parses_into_qaitem_schema():
    items = load_qa_set()
    assert len(items) == 8
    assert all(isinstance(q, QAItem) for q in items)
    assert {q.type for q in items} == {"local", "global", "trap"}


def test_every_paper_has_nonempty_citation_and_url():
    """Catches an entry that lost its provenance during editing (Rule 1.3)."""
    papers = load_corpus(mode="shipped")
    for p in papers:
        assert p.citation.strip(), f"{p.id} has an empty citation"
        assert p.url.strip(), f"{p.id} has an empty url"


def test_qa_paper_ids_exist_in_corpus():
    papers = load_corpus(mode="shipped")
    corpus_ids = {p.id for p in papers}
    qa_items = load_qa_set()
    for item in qa_items:
        for pid in item.paper_ids:
            assert pid in corpus_ids, f"{item.id} references unknown paper_id {pid!r}"


def test_qa_gold_gaps_exist_in_taxonomy():
    taxonomy = load_gap_taxonomy()
    qa_items = load_qa_set()
    for item in qa_items:
        for code in item.gold_gaps:
            assert code in taxonomy, f"{item.id} references unknown gap code {code!r}"


def test_corpus_gap_tags_exist_in_taxonomy():
    taxonomy = load_gap_taxonomy()
    papers = load_corpus(mode="shipped")
    for p in papers:
        for code in p.gap_tags:
            assert code in taxonomy, f"{p.id} references unknown gap code {code!r}"


def test_gap_taxonomy_has_13_categories():
    taxonomy = load_gap_taxonomy()
    assert len(list(taxonomy.keys())) == 13


def test_default_paths_exist():
    assert DEFAULT_CORPUS_PATH.exists()
    assert DEFAULT_QA_PATH.exists()
    assert DEFAULT_TAXONOMY_PATH.exists()


def test_load_corpus_rejects_unknown_mode():
    import pytest

    with pytest.raises(ValueError, match="Unknown corpus mode"):
        load_corpus(mode="not_a_real_mode")
