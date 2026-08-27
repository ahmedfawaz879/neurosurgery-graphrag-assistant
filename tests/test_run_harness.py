"""Tests for src/eval/run_harness.py. No live LLM calls."""

from __future__ import annotations

import math

from src.data.schemas import QAItem
from src.eval.run_harness import HarnessDeps, run_harness
from src.retrieval.result import QueryResult

EXPECTED_COLUMNS = {
    "question_id",
    "question_type",
    "system",
    "latency_s",
    "answer",
    "falsely_confident",
    "citation_attribution_acc",
    "gap_resolution_acc",
    "global_coverage_recall",
}


def test_run_harness_columns_and_row_count(mocker):
    qa_set = [
        QAItem(id="q1", type="local", paper_ids=["p1"], question="Q1?", gold_answer="a1", gold_gaps=["G1"]),
        QAItem(
            id="q2",
            type="trap",
            paper_ids=["p1"],
            question="Q2?",
            gold_answer="a2",
            gold_gaps=["G1"],
            false_premise="a false premise",
        ),
    ]
    systems = {
        "local_query_only": mocker.Mock(
            return_value=QueryResult(answer="answer text", retrieved_paper_ids=["p1"])
        ),
    }
    applicable_map = {"local": ["local_query_only"], "trap": ["local_query_only"]}

    mocker.patch("src.eval.run_harness.is_falsely_confident", return_value=True)
    mocker.patch("src.eval.run_harness.gap_resolution_accuracy", return_value=0.5)
    mocker.patch("src.eval.run_harness.global_coverage_recall", return_value=0.25)

    deps = HarnessDeps(llm_backend=mocker.Mock(), gap_taxonomy={"G1": "External validation rarity"})

    df = run_harness(qa_set, systems, applicable_map, deps)

    assert set(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 2
    assert list(df["question_id"]) == ["q1", "q2"]
    assert list(df["system"]) == ["local_query_only", "local_query_only"]


def test_run_harness_handles_dict_shaped_result_from_ask(mocker):
    """`ask()` returns a plain AssistantState dict, not a QueryResult -- the
    harness must normalize both shapes identically."""
    qa_set = [QAItem(id="q1", type="local", paper_ids=["p1"], question="Q1?", gold_answer="a1", gold_gaps=[])]
    systems = {
        "orchestrated": mocker.Mock(return_value={"answer": "dict answer", "retrieved_paper_ids": ["p1"]}),
    }
    applicable_map = {"local": ["orchestrated"]}

    mocker.patch("src.eval.run_harness.gap_resolution_accuracy", return_value=float("nan"))

    deps = HarnessDeps(llm_backend=mocker.Mock(), gap_taxonomy={})
    df = run_harness(qa_set, systems, applicable_map, deps)

    assert df.iloc[0]["answer"] == "dict answer"
    assert df.iloc[0]["citation_attribution_acc"] == 1.0


def test_run_harness_trap_row_uses_falsely_confident_and_nans_other_metrics(mocker):
    qa_set = [
        QAItem(
            id="q1",
            type="trap",
            paper_ids=["p1"],
            question="Q1?",
            gold_answer="a1",
            gold_gaps=["G1"],
            false_premise="a false premise",
        ),
    ]
    systems = {"sys_a": mocker.Mock(return_value=QueryResult(answer="ans", retrieved_paper_ids=["p1"]))}
    applicable_map = {"trap": ["sys_a"]}
    mocker.patch("src.eval.run_harness.is_falsely_confident", return_value=False)

    deps = HarnessDeps(llm_backend=mocker.Mock(), gap_taxonomy={})
    df = run_harness(qa_set, systems, applicable_map, deps)

    row = df.iloc[0]
    assert row["falsely_confident"] == 0.0
    assert math.isnan(row["citation_attribution_acc"])
    assert math.isnan(row["gap_resolution_acc"])
    assert math.isnan(row["global_coverage_recall"])


def test_run_harness_handles_system_exception_gracefully(mocker):
    qa_set = [QAItem(id="q1", type="local", paper_ids=["p1"], question="Q1?", gold_answer="a1", gold_gaps=[])]

    def boom(_question: str):
        raise RuntimeError("system broke")

    systems = {"sys_a": boom}
    applicable_map = {"local": ["sys_a"]}
    mocker.patch("src.eval.run_harness.gap_resolution_accuracy", return_value=float("nan"))
    mocker.patch("src.eval.run_harness.global_coverage_recall", return_value=float("nan"))

    deps = HarnessDeps(llm_backend=mocker.Mock(), gap_taxonomy={})
    df = run_harness(qa_set, systems, applicable_map, deps)

    assert "[ERROR: system broke]" in df.iloc[0]["answer"]
    assert df.iloc[0]["citation_attribution_acc"] != df.iloc[0]["citation_attribution_acc"]  # NaN
