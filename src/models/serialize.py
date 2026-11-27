"""Save, load, and export the shipped decoder.

The serving layer and the browser demo both load a frozen artifact, so this
module is the single definition of what an artifact contains:

* ``model.pt``       torch weights plus the config needed to rebuild the module.
* ``decoder.onnx``   the same model for ONNX Runtime (FastAPI) and the browser
  (ONNX Runtime Web), exported with a dynamic neuron axis so any neuron count
  works at inference.
* ``metadata.json``  region vocabulary and honest held-out metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.set_transformer import NeuronSetTransformer
from src.models.train import TrainConfig

# Fixed padded neuron length for the exported graph. ONNX Runtime (and the browser
# runtime) handle fixed shapes far more reliably than a dynamic attention axis, so
# clients pad every neuron set to this length and pass a padding mask. The largest
# cohort session has 429 neurons, so 512 leaves headroom.
MAX_NEURONS = 512


class _ExportWrapper(nn.Module):
    """Fixed-length inference with an explicit padding mask, for a stable ONNX graph."""

    def __init__(self, model: NeuronSetTransformer):
        super().__init__()
        self.model = model

    def forward(self, rates, region_ids, pad_mask):  # each (B, MAX_NEURONS)
        logit, attention = self.model.forward_with_attention(rates, region_ids, pad_mask.bool())
        return torch.sigmoid(logit), attention


def save_artifact(model: NeuronSetTransformer, cfg: TrainConfig, vocab: list[str],
                  metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    torch.save(
        {"state_dict": model.state_dict(),
         "config": vars(cfg),
         "vocab": vocab},
        out_dir / "model.pt",
    )
    (out_dir / "metadata.json").write_text(
        json.dumps({"vocab": vocab, "metrics": metrics, "n_regions": len(vocab),
                    "d_model": cfg.d_model, "max_neurons": MAX_NEURONS,
                    "region_to_id": {r: i + 1 for i, r in enumerate(vocab)}}, indent=2),
        encoding="utf-8",
    )
    _export_onnx(model, out_dir / "decoder.onnx")


def _export_onnx(model: NeuronSetTransformer, path: Path) -> None:
    wrapper = _ExportWrapper(model).eval()
    n = MAX_NEURONS
    rates = torch.randn(1, n, dtype=torch.float32)
    region_ids = torch.randint(0, model.region_embed.num_embeddings, (1, n))
    pad_mask = torch.zeros(1, n, dtype=torch.bool)
    pad_mask[:, n // 2:] = True  # exercise the mask path during tracing
    region_ids[:, n // 2:] = 0
    torch.onnx.export(
        wrapper, (rates, region_ids, pad_mask), str(path),
        input_names=["rates", "region_ids", "pad_mask"],
        output_names=["probability", "attention"],
        dynamic_axes={"rates": {0: "batch"}, "region_ids": {0: "batch"},
                      "pad_mask": {0: "batch"}, "probability": {0: "batch"},
                      "attention": {0: "batch"}},
        opset_version=17,
    )


def load_model(out_dir: Path) -> tuple[NeuronSetTransformer, list[str], dict]:
    blob = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**blob["config"])
    model = NeuronSetTransformer(cfg.n_regions, cfg.d_model, cfg.n_heads,
                                 cfg.n_layers, cfg.dropout)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, blob["vocab"], meta
