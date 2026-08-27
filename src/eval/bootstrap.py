"""Nonparametric bootstrap confidence intervals.

Ported verbatim from the notebook's Section 9 (`notebooks/neurosurgery_graphrag_assistant.ipynb`,
cell 29); duplicated here rather than cross-imported from the companion clinical-rag-eval-harness
repo, to avoid a cross-repo dependency.
"""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 7
) -> tuple[float, float, float]:
    vals = np.array([v for v in values if not (isinstance(v, float) and math.isnan(v))])
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return (float(vals.mean()), float(lo), float(hi))
