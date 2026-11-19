"""S1.5: exploratory summaries (owner: Omar).

The three functions named in the team guide:
- trials_overview(trials): one row per column of a raw IBL trials table.
- find_blocks(prob_left): the blocks of constant block bias, in trial order.
- sessions_overview(sessions): one row per ready session.

region_counts(regions) is an extra helper used by notebook 01.
Tables and numbers only; plotting belongs to plots.py (Stage 2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRIAL_FIELDS = (
    "y", "trial_number", "stim_side", "contrast",
    "is_error", "is_zero_contrast", "prob_left", "reaction_time",
)
NEURON_FIELDS = ("neuron_id", "neuron_region")
REQUIRED_FIELDS = ("X",) + TRIAL_FIELDS + NEURON_FIELDS + ("session_id",)


def trials_overview(trials):
    """Return one row per column of a trials table: its type, missing values and an example."""
    if not isinstance(trials, pd.DataFrame):
        raise TypeError("trials must be a pandas DataFrame.")
    rows = []
    for col in trials.columns:
        values = trials[col]
        present = values.dropna()
        rows.append({
            "column": str(col),
            "type": str(values.dtype),
            "n_missing": int(values.isna().sum()),
            "example": present.iloc[0] if len(present) else None,
        })
    return pd.DataFrame(rows, columns=["column", "type", "n_missing", "example"])


def find_blocks(prob_left):
    """Return the blocks of constant prob_left: start, end, prob_left and length per block.

    start and end are zero-based trial positions and both are inside the block.
    A missing value starts its own block, so a gap in the data is never hidden.
    """
    p = pd.Series(np.asarray(prob_left, dtype=float))
    columns = ["start", "end", "prob_left", "length"]
    if p.empty:
        return pd.DataFrame(columns=columns)
    key = p.fillna(np.inf)
    block = key.ne(key.shift()).cumsum()
    positions = pd.Series(np.arange(len(p)))
    out = pd.DataFrame({
        "start": positions.groupby(block).min(),
        "end": positions.groupby(block).max(),
        "prob_left": p.groupby(block).first(),
        "length": positions.groupby(block).size(),
    })
    return out.reset_index(drop=True)[columns]


def _check_fields(data):
    """Raise ValueError if a ready session misses a field or its shapes do not line up."""
    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")
    X = np.asarray(data["X"])
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix (trials x neurons).")
    n_trials, n_neurons = X.shape
    for k in TRIAL_FIELDS:
        if len(data[k]) != n_trials:
            raise ValueError(f"{k} has {len(data[k])} entries, expected {n_trials}.")
    for k in NEURON_FIELDS:
        if len(data[k]) != n_neurons:
            raise ValueError(f"{k} has {len(data[k])} entries, expected {n_neurons}.")


def _session_row(data):
    """One ready session as a flat dict of numbers."""
    _check_fields(data)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    rt = np.asarray(data["reaction_time"], dtype=float)
    counts = np.bincount(y.astype(int), minlength=2)
    has_rt = np.isfinite(rt).any()
    return {
        "session_id": str(data["session_id"]),
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "n_regions": len(set(map(str, data["neuron_region"]))),
        "share_y1": float(y.mean()),
        "majority_class_rate": float(counts.max() / y.size),
        "n_error_trials": int(np.sum(data["is_error"])),
        "n_zero_contrast_trials": int(np.sum(data["is_zero_contrast"])),
        "rt_median": float(np.nanmedian(rt)) if has_rt else float("nan"),
        "mean_rate_hz": float(X.mean()),
        "silent_neuron_fraction": float(np.mean(X.mean(axis=0) == 0)),
    }


def sessions_overview(sessions):
    """Return one row per ready session: trials, neurons, regions and the share of y = 1, plus extras."""
    return pd.DataFrame([_session_row(d) for d in sessions])


def region_counts(regions):
    """Return the number of neurons per brain region, largest first."""
    return pd.Series(np.asarray(regions).astype(str)).value_counts()
