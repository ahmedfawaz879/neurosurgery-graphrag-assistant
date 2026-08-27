"""Tests for src/reporting/aggregate.py and src/reporting/plots.py."""

from __future__ import annotations

import math

import pandas as pd

from src.reporting.aggregate import agg_with_ci, build_summary_table
from src.reporting.plots import plot_bar_with_ci


def test_agg_with_ci_mean_matches_hand_computed_value():
    df = pd.DataFrame(
        {
            "system": ["sys_a", "sys_a", "sys_b", "sys_b"],
            "citation_attribution_acc": [0.5, 1.0, 0.0, 0.4],
        }
    )

    agg = agg_with_ci(df, "citation_attribution_acc")
    by_system = agg.set_index("system")

    assert math.isclose(by_system.loc["sys_a", "mean"], 0.75)
    assert math.isclose(by_system.loc["sys_b", "mean"], 0.2)
    assert by_system.loc["sys_a", "n"] == 2
    lo, mean, hi = by_system.loc["sys_a", ["ci_lo", "mean", "ci_hi"]]
    assert lo <= mean <= hi


def test_agg_with_ci_excludes_nan_rows_from_n_count():
    df = pd.DataFrame({"system": ["sys_a", "sys_a", "sys_a"], "col": [0.5, float("nan"), 0.7]})

    agg = agg_with_ci(df, "col")

    assert agg.iloc[0]["n"] == 2
    assert math.isclose(agg.iloc[0]["mean"], 0.6)


def test_build_summary_table_merges_all_default_metrics():
    df = pd.DataFrame(
        {
            "system": ["sys_a", "sys_a"],
            "citation_attribution_acc": [1.0, 0.5],
            "gap_resolution_acc": [0.8, 0.6],
            "global_coverage_recall": [0.2, 0.4],
            "falsely_confident": [0.0, 1.0],
        }
    )

    summary = build_summary_table(df)

    assert set(summary.columns) == {
        "system",
        "citation_attribution_acc",
        "gap_resolution_acc",
        "global_coverage_recall",
        "false_attribution_rate",
    }
    assert len(summary) == 1
    assert math.isclose(summary.iloc[0]["citation_attribution_acc"], 0.75)


def test_plot_bar_with_ci_writes_a_nonempty_file(tmp_path):
    df = pd.DataFrame(
        {
            "system": ["sys_a", "sys_b", "sys_a", "sys_b"],
            "citation_attribution_acc": [0.5, 0.6, 0.7, 0.8],
        }
    )
    out_path = tmp_path / "fig.png"

    result = plot_bar_with_ci(df, panels=[("citation_attribution_acc", "Title", "ylabel")], out_path=out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_bar_with_ci_supports_arbitrary_panel_count(tmp_path):
    df = pd.DataFrame(
        {
            "system": ["sys_a", "sys_b"],
            "metric_a": [0.5, 0.6],
            "metric_b": [0.1, 0.2],
        }
    )
    out_path = tmp_path / "fig2.png"

    panels = [("metric_a", "A", "a"), ("metric_b", "B", "b")]
    result = plot_bar_with_ci(df, panels=panels, out_path=out_path)

    assert result.exists()
