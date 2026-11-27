"""Tests for Mustafa's files: settings.py and checks.py. No internet needed."""
import numpy as np
import pandas as pd
import pytest

from src.common import settings
from src.common.checks import (PRED_COLUMNS, TABLE_COLUMNS, check_predictions,
                               check_session, check_table)


def make_session(n_trials=20, n_neurons=5):
    rng = np.random.default_rng(0)
    return {
        "X": rng.random((n_trials, n_neurons)),
        "y": np.array([0, 1] * (n_trials // 2)),
        "trial_number": np.arange(n_trials),
        "stim_side": np.ones(n_trials, int),
        "contrast": np.full(n_trials, 0.25),
        "is_error": np.zeros(n_trials, bool),
        "is_zero_contrast": np.zeros(n_trials, bool),
        "prob_left": np.full(n_trials, 0.5),
        "reaction_time": np.full(n_trials, 0.3),
        "neuron_id": np.array([f"n{i}" for i in range(n_neurons)]),
        "neuron_region": np.array(["CA1"] * n_neurons),
        "session_id": np.array("test-session"),
    }


def make_predictions(n=6):
    return pd.DataFrame({
        "session_id": "s1", "model": "logreg", "calibration": "none",
        "experiment": "full", "setting": "none", "value": 1.0, "repeat": 0,
        "fold": 0, "trial_number": np.arange(n), "y_true": [0, 1] * (n // 2),
        "y_prob": np.linspace(0.1, 0.9, n), "y_pred": [0, 1] * (n // 2),
        "n_train": 100, "n_neurons": 5, "is_error": False,
        "is_zero_contrast": False, "prob_left": 0.5,
    })[PRED_COLUMNS]


# ---------- check_session ----------

def test_valid_session_passes():
    check_session(make_session())


def test_missing_key_fails():
    s = make_session()
    del s["contrast"]
    with pytest.raises(ValueError, match="missing"):
        check_session(s)


def test_nan_in_x_fails():
    s = make_session()
    s["X"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        check_session(s)


def test_bad_y_fails():
    s = make_session()
    s["y"][0] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        check_session(s)


def test_trial_number_not_increasing_fails():
    s = make_session()
    s["trial_number"][5] = 0
    with pytest.raises(ValueError, match="increasing"):
        check_session(s)


def test_bad_prob_left_fails():
    s = make_session()
    s["prob_left"][0] = 0.3
    with pytest.raises(ValueError, match="prob_left"):
        check_session(s)


def test_length_mismatch_fails():
    s = make_session()
    s["neuron_region"] = s["neuron_region"][:-1]
    with pytest.raises(ValueError, match="neurons"):
        check_session(s)


# ---------- check_predictions ----------

def test_valid_predictions_pass():
    check_predictions(make_predictions())


def test_wrong_column_order_fails():
    df = make_predictions()
    df = df[list(reversed(df.columns))]
    with pytest.raises(ValueError, match="columns"):
        check_predictions(df)


def test_unknown_model_fails():
    df = make_predictions()
    df["model"] = "LogReg"
    with pytest.raises(ValueError, match="model"):
        check_predictions(df)


def test_y_prob_out_of_range_fails():
    df = make_predictions()
    df.loc[0, "y_prob"] = 1.5
    with pytest.raises(ValueError, match="y_prob"):
        check_predictions(df)


# ---------- check_table ----------

def test_valid_table_passes():
    check_table(pd.DataFrame(columns=TABLE_COLUMNS["shuffles"]), "shuffles")


def test_unknown_table_fails():
    with pytest.raises(ValueError, match="unknown table"):
        check_table(pd.DataFrame(), "my_table")


def test_wrong_table_columns_fails():
    with pytest.raises(ValueError, match="must have columns"):
        check_table(pd.DataFrame(columns=["a", "b"]), "shuffles")


# ---------- settings ----------

def test_load_config():
    assert settings.load_config()["project_name"] == "volition"


def test_make_rng_same_names_same_numbers():
    a = settings.make_rng("s1", "full", 0).random(5)
    b = settings.make_rng("s1", "full", 0).random(5)
    assert np.array_equal(a, b)


def test_make_rng_different_names_different_numbers():
    a = settings.make_rng("s1", "full", 0).random(5)
    b = settings.make_rng("s1", "full", 1).random(5)
    assert not np.array_equal(a, b)


def test_save_and_load_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_processed_dir", lambda: tmp_path)
    s = make_session()
    settings.save_session(s)
    loaded = settings.load_session("test-session")
    assert sorted(loaded) == sorted(s)
    assert np.array_equal(loaded["X"], s["X"])
    assert list(loaded["neuron_id"]) == list(s["neuron_id"])


def test_save_session_rejects_bad_data(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_processed_dir", lambda: tmp_path)
    s = make_session()
    s["y"][0] = 5
    with pytest.raises(ValueError):
        settings.save_session(s)
    assert not list(tmp_path.glob("*.npz"))


# ---------- neurons.py ----------
from src.data import neurons


def fake_probe(n_neurons, region_ids, spike_rows):
    clusters = pd.DataFrame({"uuids": [f"u{i}" for i in range(n_neurons)],
                             "atlas_id": region_ids})
    spikes = {"times": np.linspace(0, 10, len(spike_rows)),
              "clusters": np.array(spike_rows)}
    return spikes, clusters


def test_probe_neurons_drops_excluded_regions():
    spikes, clusters = fake_probe(3, [1, 2, 3], [0, 1, 2, 0, 2])
    regions = np.array(["CA1", "root", "VISp"])
    table, times, index = neurons.probe_neurons(spikes, clusters, regions, ["root", "void"])
    assert list(table["region"]) == ["CA1", "VISp"]
    assert len(times) == 4
    assert set(index) == {0, 1}


def test_probe_neurons_bad_cluster_rows_fail():
    spikes, clusters = fake_probe(2, [1, 2], [0, 5])
    with pytest.raises(ValueError, match="row numbers"):
        neurons.probe_neurons(spikes, clusters, np.array(["CA1", "CA1"]), ["root"])


def test_load_neurons_joins_probes(monkeypatch):
    probes = {"p1": fake_probe(2, [1, 1], [0, 1, 1]),
              "p2": fake_probe(3, [2, 2, 2], [0, 1, 2, 2])}
    monkeypatch.setattr(neurons, "load_good_units", lambda one, pid, qc: probes[pid])

    class FakeRegions:
        def id2acronym(self, ids, mapping):
            return np.array(["CA1" if i == 1 else "VISp" for i in ids])

    monkeypatch.setattr(neurons, "BrainRegions", FakeRegions)
    table, spikes = neurons.load_neurons(None, ["p1", "p2"], settings.load_config())
    assert len(table) == 5
    assert spikes["neuron_index"].min() == 0
    assert spikes["neuron_index"].max() == len(table) - 1
    assert set(spikes["neuron_index"]) == set(range(len(table)))
