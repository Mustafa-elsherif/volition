
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
