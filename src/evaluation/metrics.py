"""Decoding metrics for binary choice prediction.

Every metric takes true labels in {0, 1} and predicted probabilities for class 1.
The functions are small and dependency-light so they can be reused by the model
training code, the evaluation tables, and the serving API without pulling in the
whole modelling stack.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def _as_arrays(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length.")
    if y_true.size == 0:
        raise ValueError("Cannot score an empty set of trials.")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1.")
    return y_true, np.clip(y_prob, 0.0, 1.0)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins.

    Returns the average gap between confidence and accuracy, weighted by how many
    trials land in each bin. Lower is better; 0 means perfectly calibrated.
    """
    y_true, y_prob = _as_arrays(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    total = y_true.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_id == b
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (count / total) * abs(confidence - accuracy)
    return ece


def decoding_scores(y_true, y_prob, threshold: float = 0.5, n_bins: int = 10) -> dict:
    """Return the standard metric set for one group of trials.

    balanced_accuracy and f1 use a hard threshold on the probability; roc_auc,
    brier and ece use the probability directly. roc_auc is undefined when only one
    class is present, in which case it is returned as NaN rather than raising.
    """
    y_true, y_prob = _as_arrays(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(y_true.size),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins)),
    }
