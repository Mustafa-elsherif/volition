"""Turn sessions into padded token batches for the set transformer.

A trial becomes a variable-length set of neuron tokens. Because sessions have
different neuron counts, a batch is padded to its largest set and carries a
padding mask so attention ignores the pad slots.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.data import Session


class TrialTokenDataset(Dataset):
    """Flat list of trials drawn from one or more sessions.

    Each item keeps a reference to its session's neuron region ids, so trials from
    sessions with different neurons can live in the same dataset and batch.
    """

    def __init__(self, sessions: list[Session], indices: list[tuple[int, int]] | None = None):
        self.rates: list[np.ndarray] = []
        self.region_ids: list[np.ndarray] = []
        self.y: list[int] = []
        src = indices if indices is not None else [
            (si, ti) for si, s in enumerate(sessions) for ti in range(s.n_trials)
        ]
        for si, ti in src:
            s = sessions[si]
            self.rates.append(s.X[ti])
            self.region_ids.append(s.region_ids)
            self.y.append(int(s.y[ti]))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.rates[i], self.region_ids[i], self.y[i]


def collate(batch):
    """Pad a batch to its largest neuron set. Returns rates, region_ids, pad_mask, y."""
    rates, region_ids, ys = zip(*batch)
    n_max = max(len(r) for r in rates)
    b = len(rates)
    rate_t = torch.zeros(b, n_max, dtype=torch.float32)
    region_t = torch.zeros(b, n_max, dtype=torch.long)
    pad_mask = torch.ones(b, n_max, dtype=torch.bool)  # True = padding
    for i, (r, rid) in enumerate(zip(rates, region_ids)):
        n = len(r)
        rate_t[i, :n] = torch.from_numpy(np.ascontiguousarray(r))
        region_t[i, :n] = torch.from_numpy(np.ascontiguousarray(rid))
        pad_mask[i, :n] = False
    y_t = torch.tensor(ys, dtype=torch.float32)
    return rate_t, region_t, pad_mask, y_t
