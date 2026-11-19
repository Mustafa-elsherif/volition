import numpy as np
import pytest

import pandas as pd

from src.analysis.eda import (
    find_blocks,
    region_counts,
    sessions_overview,
    trials_overview,
)
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

def _fake_session(session_id="s1"):
    X = np.zeros((6, 3))
    X[:, 0] = 5
    X[1, 1] = 2
    X[3, 1] = 4
    return {
        "X": X,
        "y": np.array([1, 0, 1, 1, 0, 1]),
        "trial_number": np.arange(6),
        "stim_side": np.array([1, -1, 1, 1, -1, 1]),
        "contrast": np.array([1.0, 0.5, 0.25, 0.0, 0.125, 0.0]),
        "is_error": np.array([0, 1, 0, 0, 1, 0], dtype=bool),
        "is_zero_contrast": np.array([1, 0, 0, 0, 0, 1], dtype=bool),
        "prob_left": np.array([0.5, 0.5, 0.2, 0.2, 0.8, 0.8]),
        "reaction_time": np.arange(1, 7) / 10,
        "neuron_id": np.array(["a", "b", "c"]),
        "neuron_region": np.array(["CA1", "CA1", "VISp"]),
        "session_id": np.asarray(session_id),
    }


def test_trials_overview_one_row_per_column():
    trials = pd.DataFrame({
        "choice": [1.0, -1.0, np.nan],
        "probabilityLeft": [0.5, 0.5, 0.2],
        "name": [None, "b", "c"],
    })
    t = trials_overview(trials)
    assert list(t.columns) == ["column", "type", "n_missing", "example"]
    assert list(t["column"]) == ["choice", "probabilityLeft", "name"]
    assert list(t["n_missing"]) == [1, 0, 1]
    assert t.loc[0, "example"] == 1.0
    assert t.loc[2, "example"] == "b"


def test_trials_overview_rejects_non_dataframe():
    with pytest.raises(TypeError):
        trials_overview([1, 2, 3])


def test_find_blocks_runs_in_order():
    b = find_blocks([0.5, 0.5, 0.2, 0.2, 0.2, 0.8])
    assert list(b.columns) == ["start", "end", "prob_left", "length"]
    assert list(b["start"]) == [0, 2, 5]
    assert list(b["end"]) == [1, 4, 5]
    assert list(b["prob_left"]) == [0.5, 0.2, 0.8]
    assert list(b["length"]) == [2, 3, 1]
    assert b["length"].sum() == 6


def test_find_blocks_missing_value_is_its_own_block():
    b = find_blocks([0.8, np.nan, 0.8])
    assert list(b["length"]) == [1, 1, 1]
    assert np.isnan(b.loc[1, "prob_left"])


def test_find_blocks_empty():
    b = find_blocks([])
    assert b.empty
    assert list(b.columns) == ["start", "end", "prob_left", "length"]


def test_sessions_overview_values():
    s = sessions_overview([_fake_session()]).iloc[0]
    assert s["n_trials"] == 6
    assert s["n_neurons"] == 3
    assert s["n_regions"] == 2
    assert s["share_y1"] == pytest.approx(4 / 6)
    assert s["n_error_trials"] == 2
    assert s["n_zero_contrast_trials"] == 2
    assert s["majority_class_rate"] == pytest.approx(4 / 6)
    assert s["rt_median"] == pytest.approx(0.35)
    assert s["mean_rate_hz"] == pytest.approx(2.0)
    assert s["silent_neuron_fraction"] == pytest.approx(1 / 3)


def test_sessions_overview_one_row_each():
    df = sessions_overview([_fake_session("s1"), _fake_session("s2")])
    assert list(df["session_id"]) == ["s1", "s2"]


def test_sessions_overview_catches_bad_data():
    d = _fake_session()
    del d["prob_left"]
    with pytest.raises(ValueError):
        sessions_overview([d])
    d = _fake_session()
    d["y"] = d["y"][:-1]
    with pytest.raises(ValueError):
        sessions_overview([d])


def test_region_counts_largest_first():
    rc = region_counts(np.array(["CA1", "VISp", "CA1"]))
    assert list(rc.index) == ["CA1", "VISp"]
    assert rc["CA1"] == 2
