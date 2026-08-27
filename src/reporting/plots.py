"""Bar-with-CI plots for harness results.

Ported from the notebook's Section 11 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 36). Unlike the notebook's hardcoded three-panel figure, the panel list is a
function argument here -- add or remove metrics without touching this module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.reporting.aggregate import agg_with_ci  # noqa: E402


def _bar_with_ci(ax, agg_df: pd.DataFrame, title: str, ylabel: str) -> None:
    order = agg_df.sort_values("mean", ascending=False)
    ax.bar(
        order["system"],
        order["mean"],
        yerr=[order["mean"] - order["ci_lo"], order["ci_hi"] - order["mean"]],
        capsize=4,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=40)


def plot_bar_with_ci(
    df: pd.DataFrame,
    panels: list[tuple[str, str, str]],
    out_path: str | Path,
    figsize_per_panel: tuple[float, float] = (5.3, 4.5),
) -> Path:
    """Renders one bar-with-CI subplot per `(column, title, ylabel)` tuple in `panels`,
    laid out in a single row, and saves the figure to `out_path`."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))
    if n == 1:
        axes = [axes]

    for ax, (column, title, ylabel) in zip(axes, panels, strict=True):
        agg_df = agg_with_ci(df, column)
        _bar_with_ci(ax, agg_df, title, ylabel)

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
