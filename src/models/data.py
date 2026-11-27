"""Feature views over the processed sessions.

Three views are built from the same source arrays so every model sees the same
trials:

* ``session_matrix``    firing rates as (trials, neurons), for within-session
  baselines that are retrained per session.
* ``region_pooled``     a fixed-length vector per trial, one entry per Beryl
  region in a shared vocabulary, for a baseline that transfers across sessions.
* ``neuron_tokens``     a per-trial set of (rate, region-id) tokens, for the
  permutation-invariant decoder that transfers across sessions.

All firing rates are z-scored per neuron using only that session's own trials.
Standardization uses no labels, so it is applied to held-out sessions as well
without leaking the thing we are trying to predict.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.settings import list_processed_sessions, load_session


@dataclass
class Session:
    session_id: str
    X: np.ndarray            # (trials, neurons) z-scored firing rates
    y: np.ndarray            # (trials,) 0/1 choice
    regions: np.ndarray      # (neurons,) Beryl acronym per neuron
    region_ids: np.ndarray   # (neurons,) index into the shared vocabulary
    is_error: np.ndarray
    is_zero_contrast: np.ndarray
    contrast: np.ndarray
    prob_left: np.ndarray

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.X.shape[1]


def _zscore_per_neuron(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std


def build_region_vocabulary(session_ids: list[str] | None = None) -> list[str]:
    """Return the sorted list of Beryl regions that appear in the cohort.

    Index 0 is reserved as a padding slot for the token models, so the returned
    list is what region id ``k + 1`` refers to.
    """
    ids = session_ids or list_processed_sessions()
    regions: set[str] = set()
    for sid in ids:
        regions.update(load_session(sid)["neuron_region"].tolist())
    return sorted(regions)


def load_sessions(vocab: list[str] | None = None) -> tuple[list[Session], list[str]]:
    """Load every processed session as a :class:`Session`, with a shared vocabulary."""
    ids = list_processed_sessions()
    if not ids:
        raise RuntimeError("No processed sessions found. Build data/processed first.")
    if vocab is None:
        vocab = build_region_vocabulary(ids)
    region_to_id = {r: i + 1 for i, r in enumerate(vocab)}  # 0 reserved for padding

    sessions: list[Session] = []
    for sid in ids:
        d = load_session(sid)
        regions = d["neuron_region"].astype(str)
        sessions.append(
            Session(
                session_id=str(d["session_id"]),
                X=_zscore_per_neuron(d["X"].astype(np.float32)),
                y=d["y"].astype(np.int64),
                regions=regions,
                region_ids=np.array([region_to_id[r] for r in regions], dtype=np.int64),
                is_error=d["is_error"].astype(bool),
                is_zero_contrast=d["is_zero_contrast"].astype(bool),
                contrast=d["contrast"].astype(np.float32),
                prob_left=d["prob_left"].astype(np.float32),
            )
        )
    return sessions, vocab


def region_pooled_matrix(session: Session, vocab: list[str]) -> np.ndarray:
    """Average the z-scored rate of the neurons in each region, per trial.

    Returns (trials, len(vocab)). A region with no neuron in this session stays
    zero, which is the shared value that lets one linear model span sessions.
    """
    region_to_col = {r: i for i, r in enumerate(vocab)}
    out = np.zeros((session.n_trials, len(vocab)), dtype=np.float32)
    counts = np.zeros(len(vocab), dtype=np.float32)
    acc_cols = np.array([region_to_col[r] for r in session.regions])
    for col in np.unique(acc_cols):
        members = acc_cols == col
        out[:, col] = session.X[:, members].mean(axis=1)
        counts[col] = members.sum()
    return out
