"""Cross-validation splits used across every model, so comparisons are paired.

* ``within_session_folds``  stratified k-fold inside one session, for decoders
  that are retrained per session.
* ``leave_one_session_out``  one session held out entirely, for measuring whether
  a decoder transfers to an unseen mouse.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold


def within_session_folds(
    y: np.ndarray, n_folds: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for stratified k-fold within one session."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros_like(y), y)


def leave_one_session_out(n_sessions: int) -> Iterator[tuple[list[int], int]]:
    """Yield (train_session_indices, test_session_index) for each held-out session."""
    for test in range(n_sessions):
        train = [i for i in range(n_sessions) if i != test]
        yield train, test
