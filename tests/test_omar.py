import numpy as np
import pytest

from src.data.features import firing_rates


def _toy():
    t = np.array([0.1, 0.2, 0.3, 1.1, 1.2])
    n = np.array([0, 1, 0, 1, 1])
    return t, n, 2, np.array([0.0, 1.0]), np.array([0.5, 1.5])


def test_known_rates_in_hz():
    t, n, k, s, e = _toy()
    x = firing_rates(t, n, k, s, e)
    assert x.shape == (2, 2)
    np.testing.assert_allclose(x, [[4.0, 2.0], [0.0, 4.0]])


def test_spike_at_window_end_is_excluded_start_is_included():
    # window [1.0, 2.0): spike at 1.0 and 1.5 count, spike at 2.0 must not
    x = firing_rates([1.0, 1.5, 2.0], [0, 0, 0], 1, [1.0], [2.0])
    assert x[0, 0] == pytest.approx(2.0)


def test_no_post_onset_leak():
    # movement onset at 5.0, window (-0.2, 0.0): spikes at/after 5.0 are ignored
    t = np.array([4.9, 5.0, 5.1, 6.0])
    x = firing_rates(t, [0, 0, 0, 0], 1, [4.8], [5.0])
    assert x[0, 0] == pytest.approx(1 / 0.2)


def test_unsorted_spikes_give_same_result():
    t, n, k, s, e = _toy()
    perm = np.random.default_rng(0).permutation(len(t))
    np.testing.assert_allclose(
        firing_rates(t[perm], n[perm], k, s, e), firing_rates(t, n, k, s, e)
    )


def test_silent_neuron_and_empty_spikes_give_zeros():
    x = firing_rates([], [], 3, [0.0, 1.0], [0.5, 1.5])
    assert x.shape == (2, 3)
    assert np.all(x == 0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        firing_rates([0.1], [0], 1, [1.0], [1.0])        # ends == starts
    with pytest.raises(ValueError):
        firing_rates([0.1], [5], 2, [0.0], [1.0])        # neuron out of range
    with pytest.raises(ValueError):
        firing_rates([0.1, 0.2], [0], 1, [0.0], [1.0])   # length mismatch
    with pytest.raises(ValueError):
        firing_rates([0.1], [0], 1, [0.0, 1.0], [1.0])   # starts/ends mismatch