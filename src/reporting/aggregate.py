"""Aggregation of harness results with bootstrap confidence intervals.

Ported from the notebook's Section 11 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 35).
"""

from __future__ import annotations

import pandas as pd

from src.eval.bootstrap import bootstrap_ci

DEFAULT_SUMMARY_METRICS: dict[str, str] = {
    "citation_attribution_acc": "citation_attribution_acc",
    "gap_resolution_acc": "gap_resolution_acc",
    "global_coverage_recall": "global_coverage_recall",
    "falsely_confident": "false_attribution_rate",
}


def agg_with_ci(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for system, g in df.groupby("system"):
        mean, lo, hi = bootstrap_ci(g[col].tolist())
        out.append({"system": system, "mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(g[col].notna().sum())})
    return pd.DataFrame(out)


def build_summary_table(df: pd.DataFrame, metric_columns: dict[str, str] | None = None) -> pd.DataFrame:
    """Builds the per-system summary table: one row per system, one mean-value
    column per metric in `metric_columns` ({source_column: output_column_name})."""
    metric_columns = metric_columns or DEFAULT_SUMMARY_METRICS

    summary: pd.DataFrame | None = None
    for source_col, out_col in metric_columns.items():
        agg = agg_with_ci(df, source_col).rename(columns={"mean": out_col})[["system", out_col]]
        summary = agg if summary is None else summary.merge(agg, on="system")

    return summary if summary is not None else pd.DataFrame(columns=["system"])
