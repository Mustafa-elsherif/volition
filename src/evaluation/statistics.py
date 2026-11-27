"""Resampling statistics: confidence intervals and paired significance tests.

These support honest reporting. Confidence intervals come from case resampling of
trials; the paired test compares two decoders on the same held-out trials so the
comparison is not confounded by which trials happened to be easy.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(
    y_true,
    y_prob,
    metric: Callable,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point, low, high) for a metric by resampling trials with replacement.

    ``metric`` maps (y_true, y_prob) to a float. Resamples that end up with a single
    class are skipped so metrics such as ROC AUC stay defined.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = y_true.size
    point = float(metric(y_true, y_prob))
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        draws.append(float(metric(y_true[idx], y_prob[idx])))
    if not draws:
        return point, float("nan"), float("nan")
    lo = (1.0 - ci) / 2.0
    low, high = np.quantile(draws, [lo, 1.0 - lo])
    return point, float(low), float(high)


def paired_difference(scores_a, scores_b) -> dict:
    """Compare two methods scored on the same units (e.g. per-session accuracy).

    Returns the mean paired difference (a minus b), a Wilcoxon signed-rank p-value,
    and the number of pairs. The Wilcoxon test is used because the per-session
    sample is small and not assumed normal.
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("Paired score arrays must have the same length.")
    diff = a - b
    mean_diff = float(np.mean(diff))
    if np.allclose(diff, 0.0) or a.size < 2:
        p_value = float("nan")
    else:
        try:
            p_value = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            p_value = float("nan")
    return {"mean_difference": mean_diff, "p_value": p_value, "n_pairs": int(a.size)}


def summarize_seeds(values) -> dict:
    """Mean and sample standard deviation across seeds, for 'mean +/- std' reporting."""
    v = np.asarray(values, dtype=float).ravel()
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "n_seeds": int(v.size),
    }
