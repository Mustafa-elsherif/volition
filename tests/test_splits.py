import numpy as np
import pytest

from src.data.splits import BlockedGapSplit


def _splits(n=200, **kw):
    X = np.zeros((n, 3))
    return list(BlockedGapSplit(**kw).split(X))


def test_train_strictly_before_test_with_gap():
    for tr, te in _splits(n_splits=4, gap=7):
        assert tr.max() < te.min()
        assert te.min() - tr.max() - 1 == 7  # exactly `gap` trials excluded


def test_no_overlap_and_contiguous_test_blocks():
    for tr, te in _splits(n_splits=4, gap=5):
        assert len(np.intersect1d(tr, te)) == 0
        assert np.all(np.diff(te) == 1)


def test_train_grows_forward_and_last_test_ends_at_last_trial():
    s = _splits(n=200, n_splits=4, gap=5)
    sizes = [len(tr) for tr, _ in s]
    assert sizes == sorted(sizes)
    assert s[-1][1].max() == 199


def test_gap_zero_is_adjacent():
    for tr, te in _splits(n_splits=3, gap=0):
        assert te.min() == tr.max() + 1


def test_raises_when_no_training_data():
    with pytest.raises(ValueError):
        _splits(n=20, n_splits=5, gap=10)