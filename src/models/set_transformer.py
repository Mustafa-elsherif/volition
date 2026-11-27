"""Permutation-invariant decoder over a set of neuron tokens.

Each trial is a set of neurons. A neuron token is its z-scored firing rate in the
pre-movement window plus an embedding of its Beryl region. Because region ids are
shared across sessions, one set of weights can read sessions with different and
non-overlapping neurons. There is no positional encoding, so the model is
invariant to neuron order and copes with a variable neuron count by masking.

The same backbone supports two heads:

* a choice head (one logit) for decoding, and
* a reconstruction head used only for self-supervised masked-neuron pretraining.
"""
from __future__ import annotations

import torch
from torch import nn


class NeuronSetTransformer(nn.Module):
    def __init__(
        self,
        n_regions: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # region id 0 is the padding slot; the embedding for it is masked out anyway.
        self.region_embed = nn.Embedding(n_regions + 1, d_model, padding_idx=0)
        self.rate_proj = nn.Linear(1, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.input_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # attention pooling: a learned query reads the encoded set into one vector.
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.choice_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )
        self.recon_head = nn.Linear(d_model, 1)

    def encode(self, rates, region_ids, pad_mask, token_mask=None):
        """Run the backbone. Returns encoded tokens (batch, neurons, d_model)."""
        tok = self.rate_proj(rates.unsqueeze(-1)) + self.region_embed(region_ids)
        if token_mask is not None:
            tok = torch.where(token_mask.unsqueeze(-1), self.mask_token, tok)
        tok = self.input_norm(tok)
        return self.encoder(tok, src_key_padding_mask=pad_mask)

    def pool(self, encoded, pad_mask, return_weights: bool = False):
        q = self.pool_query.expand(encoded.size(0), 1, self.d_model)
        pooled, weights = self.pool_attn(
            q, encoded, encoded, key_padding_mask=pad_mask,
            need_weights=return_weights, average_attn_weights=True,
        )
        pooled = pooled.squeeze(1)
        if return_weights:
            return pooled, weights.squeeze(1)  # (batch, neurons), attention mass per neuron
        return pooled

    def forward(self, rates, region_ids, pad_mask):
        """Return choice logits (batch,). pad_mask is True where a slot is padding."""
        encoded = self.encode(rates, region_ids, pad_mask)
        return self.choice_head(self.pool(encoded, pad_mask)).squeeze(-1)

    def forward_with_attention(self, rates, region_ids, pad_mask):
        """Return (logits, per-neuron attention mass) for interpretable attribution."""
        encoded = self.encode(rates, region_ids, pad_mask)
        pooled, weights = self.pool(encoded, pad_mask, return_weights=True)
        return self.choice_head(pooled).squeeze(-1), weights

    def reconstruct(self, rates, region_ids, pad_mask, token_mask):
        """Predict the rate at masked tokens, for self-supervised pretraining."""
        encoded = self.encode(rates, region_ids, pad_mask, token_mask)
        return self.recon_head(encoded).squeeze(-1)
