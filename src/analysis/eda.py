"""S1.5 (draft): exploratory summaries of the processed session data.

Works on the dict returned by build_dataset.assemble_session (12 fields).
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


def check_fields(data):
    """Raise ValueError if a field is missing or the shapes do not line up."""
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


def session_summary(data):
    """One-session summary as a flat dict of numbers."""
    check_fields(data)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    rt = np.asarray(data["reaction_time"], dtype=float)

    _, class_counts = np.unique(y, return_counts=True)
    neuron_mean = X.mean(axis=0)
    has_rt = np.isfinite(rt).any()

    return {
        "session_id": str(data["session_id"]),
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "n_regions": len(set(map(str, data["neuron_region"]))),
        "majority_class_rate": float(class_counts.max() / y.size),
        "n_error_trials": int(np.sum(data["is_error"])),
        "n_zero_contrast_trials": int(np.sum(data["is_zero_contrast"])),
        "rt_median": float(np.nanmedian(rt)) if has_rt else float("nan"),
        "mean_rate_hz": float(X.mean()),
        "silent_neuron_fraction": float(np.mean(neuron_mean == 0)),
    }


def summarize_sessions(sessions):
    """DataFrame with one row per session."""
    return pd.DataFrame([session_summary(d) for d in sessions])


def region_counts(data):
    """Neurons per brain region, largest first."""
    return pd.Series(np.asarray(data["neuron_region"]).astype(str)).value_counts()


def block_counts(data):
    """Trials per block prior (probability of a left stimulus)."""
    return pd.Series(np.asarray(data["prob_left"], dtype=float)).value_counts().sort_index()