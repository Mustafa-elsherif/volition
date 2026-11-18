"""S1.6: firing-rate feature matrix (trials x neurons).

Counts the spikes of every neuron inside each trial's window and divides by
the window length, so the result is in Hz. The caller decides the window
(build_dataset.py uses movement time + (-0.2, 0.0)), so nothing after movement
onset can enter the features as long as `ends` is at or before movement onset.
"""
from __future__ import annotations

import numpy as np


def firing_rates(spike_times, neuron_index, n_neurons, starts, ends):
    """Return firing rates in Hz, shape (n_trials, n_neurons).

    Each window is half-open: [start, end). A spike exactly at `end` is NOT
    counted, so a spike at the movement-onset time never leaks in.

    Parameters
    ----------
    spike_times : 1-D array of spike times in seconds (any order).
    neuron_index : 1-D array, same length, column index of each spike's neuron.
    n_neurons : total number of neurons (number of output columns).
    starts, ends : 1-D arrays, one window per trial, in seconds.
    """
    spike_times = np.asarray(spike_times, dtype=float)
    neuron_index = np.asarray(neuron_index)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)

    if spike_times.ndim != 1 or neuron_index.shape != spike_times.shape:
        raise ValueError("spike_times and neuron_index must be 1-D and the same length.")
    if starts.ndim != 1 or starts.shape != ends.shape:
        raise ValueError("starts and ends must be 1-D and the same length.")
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("starts and ends must be finite.")
    if np.any(ends <= starts):
        raise ValueError("Every window must have ends > starts.")
    if n_neurons < 1:
        raise ValueError("n_neurons must be >= 1.")

    idx = neuron_index.astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n_neurons):
        raise ValueError("neuron_index values must be in [0, n_neurons).")

    order = np.argsort(spike_times, kind="stable")
    t = spike_times[order]
    idx = idx[order]

    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, ends, side="left")

    x = np.zeros((starts.size, n_neurons), dtype=float)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            x[i] = np.bincount(idx[a:b], minlength=n_neurons)
    x /= (ends - starts)[:, None]
    return x