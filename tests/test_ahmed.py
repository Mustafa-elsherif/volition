
"""Tests for Ahmed's Stage 1 dataset builder."""

import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import assemble_session


def test_assemble_session():
    """Check the required keys and dimensions."""
    trials = pd.DataFrame({
        "y": [1, 0],
        "trial_number": [1, 2],
        "stim_side": [1, -1],
        "contrast": [0.5, -0.5],
        "is_error": [False, True],
        "is_zero_contrast": [False, False],
        "prob_left": [0.5, 0.5],
        "reaction_time": [0.2, 0.3],
    })

    neurons = pd.DataFrame({
        "neuron_id": ["n1", "n2", "n3"],
        "region": ["VISp", "MOs", "CA1"],
    })

    x = np.array([
        [10.0, 20.0, 0.0],
        [0.0, 10.0, 30.0],
    ])

    result = assemble_session(
        "test-session", trials, neurons, x
    )

    assert len(result) == 12
    assert result["X"].shape == (2, 3)
    assert result["y"].tolist() == [1, 0]
    assert result["neuron_id"].tolist() == [
        "n1", "n2", "n3"
    ]


def test_invalid_dimensions():
    """Reject a feature matrix with incorrect dimensions."""
    trials = pd.DataFrame({
        "y": [1, 0],
        "trial_number": [1, 2],
        "stim_side": [1, -1],
        "contrast": [0.5, -0.5],
        "is_error": [False, True],
        "is_zero_contrast": [False, False],
        "prob_left": [0.5, 0.5],
        "reaction_time": [0.2, 0.3],
    })

    neurons = pd.DataFrame({
        "neuron_id": ["n1", "n2", "n3"],
        "region": ["VISp", "MOs", "CA1"],
    })

    invalid_x = np.zeros((2, 2))

    with pytest.raises(ValueError):
        assemble_session(
            "test-session",
            trials,
            neurons,
            invalid_x,
        )


# ---- session selection (src/data/sessions.py), offline with small fake tables ----

from src.common.settings import load_config
from src.data import sessions as sel


def _selection_config():
    config = load_config()
    config["data"].update({
        "n_sessions": 2, "min_neurons": 60, "min_regions": 3,
        "neurons_per_region": 10, "max_sessions_per_lab": 2,
        "region_mapping": "Beryl",
    })
    return config


def _fake_release():
    """10 sessions; each one is removed by a different rule, or kept."""
    meta = [  # eid, mouse, lab, neurons per region
        ("e1", "m1", "L1", {"A": 30, "B": 30, "C": 20}),   # kept, most neurons
        ("e2", "m1", "L1", {"A": 30, "B": 20, "C": 20}),   # same mouse as e1
        ("e3", "m3", "L1", {"A": 25, "B": 25, "C": 25}),   # kept
        ("e4", "m4", "L1", {"A": 25, "B": 20, "C": 20}),   # lab L1 already has 2
        ("e5", "m5", "L2", {"A": 22, "B": 20, "C": 20}),   # trials fail to load
        ("e6", "m6", "L2", {"A": 21, "B": 20, "C": 20}),   # kept, as a reserve
        ("e7", "m7", "L3", {"A": 20, "B": 20, "C": 10}),   # only 50 neurons
        ("e8", "m8", "L3", {"A": 55, "B": 10, "C": 5}),    # only 2 qualifying regions
        ("e9", "m9", "L3", {"A": 40, "B": 40, "C": 40}),   # fails the IBL trial rules
        ("e10", "m10", "L3", {}),                          # no good neurons
    ]
    bwm = pd.DataFrame([{"eid": e, "pid": e + "-p0", "subject": m, "lab": lab}
                        for e, m, lab, _ in meta])
    units = pd.DataFrame([{"eid": e, "Beryl": r}
                          for e, _, _, regions in meta
                          for r, n in regions.items() for _ in range(n)])
    valid = [e for e, *_ in meta if e != "e9"]
    return bwm, units, valid


def test_make_candidates_applies_neuron_and_region_rules():
    bwm, units, valid = _fake_release()
    units = units[units["eid"].isin(valid)]
    cands = sel.make_candidates(bwm, units, _selection_config())
    ids = [c["session_id"] for c in cands]
    assert ids == ["e1", "e3", "e2", "e4", "e5", "e6"]   # most neurons first
    assert "e7" not in ids and "e8" not in ids
    assert cands[0]["n_neurons"] == 80 and cands[0]["pids"] == ["e1-p0"]


def test_choose_sessions_mouse_lab_and_load_rules(monkeypatch):
    bwm, units, valid = _fake_release()
    config = _selection_config()
    cands = sel.make_candidates(bwm, units[units["eid"].isin(valid)], config)
    monkeypatch.setattr(sel, "verify_trials", lambda one, eid, cfg: eid != "e5")
    selected, reserves = sel.choose_sessions(cands, None, config)
    assert [s["session_id"] for s in selected] == ["e1", "e3"]
    assert reserves == ["e6"]


def test_selection_audit_counts_every_rule():
    bwm, units, valid = _fake_release()
    audit = sel.selection_audit(bwm, units, valid, _selection_config(), ["e1", "e3"], ["e6"])
    assert list(audit.columns) == ["step", "n_sessions", "n_removed"]
    assert list(audit["n_sessions"]) == [10, 9, 8, 7, 6, 5, 4, 3]
    assert list(audit["n_removed"]) == [0, 1, 1, 1, 1, 1, 1, 1]


def test_selection_audit_rejects_a_selection_the_rules_do_not_give():
    bwm, units, valid = _fake_release()
    with pytest.raises(ValueError):
        sel.selection_audit(bwm, units, valid, _selection_config(), ["e1", "e2"], [])
