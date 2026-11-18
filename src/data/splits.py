"""Blocked forward-chaining splits with a gap (dead zone) between train and test.

Trials MUST be in chronological order. Every fold trains only on trials that
happened before the test block, and the last `gap` trials before the test
block are dropped so slow drift / trial carry-over cannot cross the boundary.
"""
from __future__ import annotations

import numpy as np


class BlockedGapSplit:
    """Forward-chaining CV with contiguous test blocks and a train/test gap.

    Layout for n_splits=3, gap=g (T = train, - = excluded gap, E = test):

        fold 0: TTTTTTTT--EEEE
        fold 1: TTTTTTTTTTTT--EEEE
        fold 2: TTTTTTTTTTTTTTTT--EEEE

    Parameters
    ----------
    n_splits : number of folds (test blocks).
    gap : number of trials removed between the end of train and start of test.
    test_size : trials per test block. Default: n_trials // (n_splits + 1).
    """

    def __init__(self, n_splits: int = 5, gap: int = 5, test_size: int | None = None):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        test_size = self.test_size or n // (self.n_splits + 1)
        if test_size < 1:
            raise ValueError("Too few trials for this many splits.")

        first_test_start = n - self.n_splits * test_size
        if first_test_start - self.gap < 1:
            raise ValueError(
                f"No training data left in fold 0 (n={n}, n_splits={self.n_splits}, "
                f"test_size={test_size}, gap={self.gap}). Reduce n_splits or gap."
            )

        for k in range(self.n_splits):
            test_start = first_test_start + k * test_size
            test_end = test_start + test_size
            train_end = test_start - self.gap  # exclusive
            yield np.arange(0, train_end), np.arange(test_start, test_end)