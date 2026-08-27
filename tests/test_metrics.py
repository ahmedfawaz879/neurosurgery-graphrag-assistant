"""Tests for src/eval/metrics.py and src/eval/bootstrap.py. No live LLM calls."""

from __future__ import annotations

import math

from src.data.schemas import QAItem
from src.eval.bootstrap import bootstrap_ci
from src.eval.metrics import (
    citation_attribution_accuracy,
    classify_answer_gaps,
    gap_resolution_accuracy,
    global_coverage_recall,
    is_falsely_confident,
)

GAP_TAXONOMY = {
    "G1": "External validation rarity",
    "G3": "Poor calibration reporting",
    "G4": "Deployed-tool degradation",
}


def _trap_item(false_premise="some false premise") -> QAItem:
    return QAItem(
        id="q1",
        type="trap",
        paper_ids=["p1"],
        question="q",
        gold_answer="a",
        gold_gaps=["G1"],
        false_premise=false_premise,
    )


# ---- citation_attribution_accuracy --------------------------------------------


def test_citation_attribution_accuracy_hand_computed():
    # retrieved={p1,p2,p3}, gold={p1,p2} -> correct=2, precision = 2/3
    result = citation_attribution_accuracy(["p1", "p2", "p3"], ["p1", "p2"])
    assert math.isclose(result, 2 / 3)


def test_citation_attribution_accuracy_perfect_precision():
    assert citation_attribution_accuracy(["p1"], ["p1"]) == 1.0


def test_citation_attribution_accuracy_zero_overlap():
    assert citation_attribution_accuracy(["p9"], ["p1"]) == 0.0


def test_citation_attribution_accuracy_no_retrieved_is_nan_not_zero():
    """Ungradeable is not the same as wrong -- must not be silently scored as 0."""
    assert math.isnan(citation_attribution_accuracy([], ["p1"]))


# ---- classify_answer_gaps ------------------------------------------------------


def test_classify_answer_gaps_filters_invalid_codes(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = '["G1", "BOGUS", "G3"]'

    codes = classify_answer_gaps("some answer", GAP_TAXONOMY, fake_backend)

    assert codes == ["G1", "G3"]


def test_classify_answer_gaps_returns_empty_on_malformed_json(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "not json"

    assert classify_answer_gaps("some answer", GAP_TAXONOMY, fake_backend) == []


# ---- gap_resolution_accuracy ----------------------------------------------------


def test_gap_resolution_accuracy_hand_computed(mocker):
    mocker.patch("src.eval.metrics.classify_answer_gaps", return_value=["G1", "G4"])
    # predicted={G1,G4}, gold={G1,G3} -> correct=1, /len(predicted)=2 -> 0.5
    result = gap_resolution_accuracy("answer", ["G1", "G3"], GAP_TAXONOMY, mocker.Mock())
    assert math.isclose(result, 0.5)


def test_gap_resolution_accuracy_no_predicted_is_nan_not_zero(mocker):
    mocker.patch("src.eval.metrics.classify_answer_gaps", return_value=[])
    result = gap_resolution_accuracy("answer", ["G1"], GAP_TAXONOMY, mocker.Mock())
    assert math.isnan(result)


# ---- global_coverage_recall ------------------------------------------------------


def test_global_coverage_recall_hand_computed(mocker):
    mocker.patch("src.eval.metrics.classify_answer_gaps", return_value=["G1", "G4"])
    # gold={G1,G3}, predicted∩gold={G1} -> 1/len(gold)=2 -> 0.5
    result = global_coverage_recall("answer", ["G1", "G3"], GAP_TAXONOMY, mocker.Mock())
    assert math.isclose(result, 0.5)


def test_global_coverage_recall_no_gold_is_nan_not_zero(mocker):
    mocker.patch("src.eval.metrics.classify_answer_gaps", return_value=["G1"])
    result = global_coverage_recall("answer", [], GAP_TAXONOMY, mocker.Mock())
    assert math.isnan(result)


# ---- is_falsely_confident --------------------------------------------------------


def test_is_falsely_confident_true_when_judge_says_a(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "A"
    assert is_falsely_confident(_trap_item(), "an answer accepting the premise", fake_backend) is True


def test_is_falsely_confident_false_when_judge_says_b(mocker):
    fake_backend = mocker.Mock()
    fake_backend.generate.return_value = "B"
    assert is_falsely_confident(_trap_item(), "an answer refuting the premise", fake_backend) is False


# ---- bootstrap_ci -----------------------------------------------------------------


def test_bootstrap_ci_hand_computed_mean():
    mean, lo, hi = bootstrap_ci([0.7, 0.8, 0.6, 0.9])
    assert math.isclose(mean, 0.75)
    assert lo <= mean <= hi


def test_bootstrap_ci_empty_returns_all_nan():
    mean, lo, hi = bootstrap_ci([])
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_ci_filters_nan_values():
    mean, _, _ = bootstrap_ci([0.5, float("nan"), 0.7])
    assert math.isclose(mean, 0.6)
