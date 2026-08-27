"""CLI: results/run.csv -> results/summary.csv + results/figures/gaprag_metrics_by_system.png.

Used by the README and CI. Ported from the notebook's Section 11
(`notebooks/neurosurgery_graphrag_assistant.ipynb`, cells 35-36).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reporting.aggregate import build_summary_table  # noqa: E402
from src.reporting.plots import plot_bar_with_ci  # noqa: E402

DEFAULT_PANELS = [
    ("citation_attribution_acc", "Citation-attribution accuracy", "accuracy"),
    ("global_coverage_recall", "Global-coverage recall (global Qs only)", "recall"),
    ("falsely_confident", "False-attribution rate on trap Qs (LOWER = safer)", "rate"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a harness run CSV into a summary table + figure.")
    parser.add_argument(
        "--run-csv",
        default="results/run.csv",
        help="Input harness run CSV (from `python -m src.eval.run_harness`).",
    )
    parser.add_argument("--summary-out", default="results/summary.csv", help="Output summary CSV path.")
    parser.add_argument(
        "--figure-out",
        default="results/figures/gaprag_metrics_by_system.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.run_csv)

    summary = build_summary_table(df)
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    print(f"Summary table -> {summary_out}")

    figure_out = plot_bar_with_ci(df, DEFAULT_PANELS, args.figure_out)
    print(f"Figure -> {figure_out}")


if __name__ == "__main__":
    main()
