"""Format checks (owner: Mustafa, S1.1). Never imports settings.py."""
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]

SESSION_KEYS = ["X", "y", "trial_number", "stim_side", "contrast", "is_error",
                "is_zero_contrast", "prob_left", "reaction_time", "neuron_id",
                "neuron_region", "session_id"]

PRED_COLUMNS = ["session_id", "model", "calibration", "experiment", "setting", "value",
                "repeat", "fold", "trial_number", "y_true", "y_prob", "y_pred", "n_train",
                "n_neurons", "is_error", "is_zero_contrast", "prob_left"]

TABLE_COLUMNS = {
    "scores": ["session_id", "model", "calibration", "experiment", "setting", "value",
               "repeat", "n_test", "balanced_accuracy", "roc_auc", "f1", "brier", "ece"],
    "summary": ["model", "calibration", "experiment", "setting", "value", "metric",
                "n_sessions", "mean", "ci_low", "ci_high"],
    "regions": ["session_id", "model", "region", "n_removed", "score_full",
                "score_removed", "score_random_mean", "p_value", "p_value_fdr"],
    "reliability": ["model", "calibration", "experiment", "value", "bin",
                    "mean_predicted", "fraction_positive", "count"],
    "model_tests": ["experiment", "value", "metric", "model_a", "model_b", "n_sessions",
                    "mean_difference", "p_value", "p_value_holm"],
    "controls": ["session_id", "model", "group", "n_trials", "balanced_accuracy"],
    "shuffles": ["session_id", "model", "repeat", "balanced_accuracy"],
    "shuffle_tests": ["session_id", "model", "observed", "shuffled_mean", "p_value",
                      "p_value_holm"],
}

PER_TRIAL = ["y", "trial_number", "stim_side", "contrast", "is_error",
             "is_zero_contrast", "prob_left", "reaction_time"]
PER_NEURON = ["neuron_id", "neuron_region"]


def _config():
    with open(_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_session(data):
    """Raise ValueError if a ready session does not follow the format of the guide (2.6)."""
    keys = list(data.keys())
    if sorted(keys) != sorted(SESSION_KEYS):
        missing = sorted(set(SESSION_KEYS) - set(keys))
        extra = sorted(set(keys) - set(SESSION_KEYS))
        raise ValueError(f"session keys are wrong. missing: {missing}, extra: {extra}")

    X = np.asarray(data["X"])
    if X.ndim != 2:
        raise ValueError(f"X must be trials x neurons (2D), got shape {X.shape}")
    if np.isnan(X).any():
        raise ValueError(f"X has {int(np.isnan(X).sum())} NaN values")
    n_trials, n_neurons = X.shape

    for k in PER_TRIAL:
        if len(data[k]) != n_trials:
            raise ValueError(f"{k} has {len(data[k])} values but X has {n_trials} trials")
    for k in PER_NEURON:
        if len(data[k]) != n_neurons:
            raise ValueError(f"{k} has {len(data[k])} values but X has {n_neurons} neurons")

    bad_y = set(np.unique(data["y"])) - {0, 1}
    if bad_y:
        raise ValueError(f"y must contain only 0 and 1, found {sorted(bad_y)}")

    if not np.all(np.diff(np.asarray(data["trial_number"])) > 0):
        raise ValueError("trial_number must be strictly increasing")

    bad_p = set(np.round(np.unique(data["prob_left"]), 6)) - {0.2, 0.5, 0.8}
    if bad_p:
        raise ValueError(f"prob_left must be 0.2, 0.5 or 0.8, found {sorted(bad_p)}")


def check_predictions(df):
    """Raise ValueError if a predictions table does not follow the format of the guide (2.6)."""
    if list(df.columns) != PRED_COLUMNS:
        raise ValueError(f"prediction columns must be exactly {PRED_COLUMNS} in this order, "
                         f"got {list(df.columns)}")
    cfg = _config()
    allowed = {"model": cfg["models"]["names"],
               "experiment": cfg["experiments"]["names"],
               "calibration": cfg["models"]["calibration"]}
    for col, names in allowed.items():
        bad = set(df[col].unique()) - set(names)
        if bad:
            raise ValueError(f"unknown {col} values: {sorted(bad)}. allowed: {names}")
    if ((df["y_prob"] < 0) | (df["y_prob"] > 1)).any():
        raise ValueError("y_prob must be between 0 and 1")


def check_table(df, name):
    """Raise ValueError if a result table does not have the columns of the guide (2.6)."""
    if name not in TABLE_COLUMNS:
        raise ValueError(f"unknown table '{name}'. allowed: {list(TABLE_COLUMNS)}")
    if list(df.columns) != TABLE_COLUMNS[name]:
        raise ValueError(f"table '{name}' must have columns {TABLE_COLUMNS[name]}, "
                         f"got {list(df.columns)}")
