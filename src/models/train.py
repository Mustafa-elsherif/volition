"""Train and run the neuron-set transformer.

Two phases:

* ``pretrain_ssl`` learns the backbone by masked-neuron reconstruction across all
  given sessions, using no choice labels.
* ``train_choice`` fine-tunes (optionally from a pretrained backbone) to predict
  the binary choice, with early stopping on a held-out validation split.

The data is small, so every trial is packed once into padded device tensors and
training iterates by slicing them. That removes per-step host-to-device copies and
Python collation, which otherwise dominate the run time on this workload.

The public helpers return plain numpy probabilities so the evaluation code and the
serving API never need to know about torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from src.models.data import Session
from src.models.set_transformer import NeuronSetTransformer


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    ssl_epochs: int = 30
    ssl_mask_frac: float = 0.15
    choice_epochs: int = 120
    patience: int = 12
    seed: int = 42
    n_regions: int = field(default=65)


def _max_neurons(sessions: list[Session]) -> int:
    return max(s.n_neurons for s in sessions)


def _pack(sessions, indices, device, n_max):
    """Pad the given (session, trial) indices into device tensors, built once."""
    m = len(indices)
    rates = torch.zeros(m, n_max, dtype=torch.float32)
    region_ids = torch.zeros(m, n_max, dtype=torch.long)
    pad_mask = torch.ones(m, n_max, dtype=torch.bool)
    y = torch.zeros(m, dtype=torch.float32)
    for row, (si, ti) in enumerate(indices):
        s = sessions[si]
        n = s.n_neurons
        rates[row, :n] = torch.from_numpy(np.ascontiguousarray(s.X[ti]))
        region_ids[row, :n] = torch.from_numpy(np.ascontiguousarray(s.region_ids))
        pad_mask[row, :n] = False
        y[row] = s.y[ti]
    return (rates.to(device), region_ids.to(device), pad_mask.to(device), y.to(device))


def pretrain_ssl(sessions: list[Session], cfg: TrainConfig) -> NeuronSetTransformer:
    """Masked-neuron reconstruction across sessions. Returns the pretrained model."""
    set_seed(cfg.seed)
    device = get_device()
    n_max = _max_neurons(sessions)
    model = NeuronSetTransformer(cfg.n_regions, cfg.d_model, cfg.n_heads,
                                 cfg.n_layers, cfg.dropout).to(device)
    idx = [(si, ti) for si, s in enumerate(sessions) for ti in range(s.n_trials)]
    rates, region_ids, pad_mask, _ = _pack(sessions, idx, device, n_max)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    m = rates.size(0)
    model.train()
    for _ in range(cfg.ssl_epochs):
        perm = torch.randperm(m, generator=gen, device=device)
        for b in range(0, m, cfg.batch_size):
            sel = perm[b:b + cfg.batch_size]
            r, rid, pm = rates[sel], region_ids[sel], pad_mask[sel]
            real = ~pm
            token_mask = real & (torch.rand(r.shape, generator=gen, device=device) < cfg.ssl_mask_frac)
            if not token_mask.any():
                continue
            pred = model.reconstruct(r, rid, pm, token_mask)
            loss = ((pred - r) ** 2)[token_mask].mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def _predict_packed(model, rates, region_ids, pad_mask, batch_size=512):
    model.eval()
    out = []
    for b in range(0, rates.size(0), batch_size):
        out.append(model(rates[b:b + batch_size], region_ids[b:b + batch_size],
                         pad_mask[b:b + batch_size]))
    return torch.cat(out)


def train_choice(
    sessions: list[Session],
    train_idx: list[tuple[int, int]],
    val_idx: list[tuple[int, int]],
    cfg: TrainConfig,
    pretrained: NeuronSetTransformer | None = None,
) -> NeuronSetTransformer:
    """Fine-tune the choice head with early stopping on validation loss."""
    set_seed(cfg.seed)
    device = get_device()
    n_max = _max_neurons(sessions)
    model = (pretrained or NeuronSetTransformer(
        cfg.n_regions, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout)).to(device)

    tr = _pack(sessions, train_idx, device, n_max)
    va = _pack(sessions, val_idx, device, n_max)
    rates, region_ids, pad_mask, y = tr
    y_val = va[3]

    pos = max(1, int((y == 1).sum())); neg = max(1, int((y == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    m = rates.size(0)

    best_val, best_state, waited = np.inf, None, 0
    for _ in range(cfg.choice_epochs):
        model.train()
        perm = torch.randperm(m, generator=gen, device=device)
        for b in range(0, m, cfg.batch_size):
            sel = perm[b:b + cfg.batch_size]
            logits = model(rates[sel], region_ids[sel], pad_mask[sel])
            loss = loss_fn(logits, y[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        val_logits = _predict_packed(model, va[0], va[1], va[2])
        val_loss = float(nn.functional.binary_cross_entropy_with_logits(val_logits, y_val))
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba(model: NeuronSetTransformer, sessions: list[Session],
                  indices: list[tuple[int, int]]) -> np.ndarray:
    """Return class-1 probabilities for the given (session, trial) indices."""
    device = get_device()
    model = model.to(device)
    rates, region_ids, pad_mask, _ = _pack(sessions, indices, device, _max_neurons(sessions))
    logits = _predict_packed(model, rates, region_ids, pad_mask)
    return torch.sigmoid(logits).cpu().numpy()
