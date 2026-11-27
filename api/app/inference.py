"""ONNX inference for the neuron-set decoder.

Loads the exported artifact once and runs single trials. The same ONNX graph powers
the browser demo, so server and browser return identical probabilities. The graph
returns both the choice probability and the per-neuron pooling attention, which is
turned into interpretable per-neuron and per-region contributions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class Decoder:
    def __init__(self, artifact_dir: Path):
        artifact_dir = Path(artifact_dir)
        meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        self.vocab: list[str] = meta["vocab"]
        self.region_to_id: dict[str, int] = meta["region_to_id"]
        self.max_neurons: int = meta["max_neurons"]
        self.metrics: dict = meta.get("metrics", {})
        self.session = ort.InferenceSession(
            str(artifact_dir / "decoder.onnx"), providers=["CPUExecutionProvider"]
        )

    def _pack(self, rates: list[float], regions: list[str]):
        n = len(rates)
        if n == 0 or n > self.max_neurons:
            raise ValueError(f"A trial needs 1 to {self.max_neurons} neurons, got {n}.")
        R = np.zeros((1, self.max_neurons), np.float32)
        I = np.zeros((1, self.max_neurons), np.int64)
        M = np.ones((1, self.max_neurons), bool)
        R[0, :n] = np.asarray(rates, np.float32)
        I[0, :n] = [self.region_to_id.get(r, 0) for r in regions]
        M[0, :n] = False
        return R, I, M, n

    def decode(self, rates: list[float], regions: list[str], top_k: int = 12) -> dict:
        R, I, M, n = self._pack(rates, regions)
        prob, attn = self.session.run(None, {"rates": R, "region_ids": I, "pad_mask": M})
        p = float(prob.ravel()[0])
        a = attn.ravel()[:n]

        order = np.argsort(a)[::-1][:top_k]
        top = [{"region": regions[i], "rate": round(float(rates[i]), 3),
                "attention": round(float(a[i]), 4)} for i in order]

        region_attn: dict[str, list[float]] = {}
        for reg, w in zip(regions, a):
            region_attn.setdefault(reg, []).append(float(w))
        regions_out = sorted(
            ({"region": reg, "attention": round(float(np.sum(ws)), 4), "n_neurons": len(ws)}
             for reg, ws in region_attn.items()),
            key=lambda r: r["attention"], reverse=True,
        )

        return {
            "probability": round(p, 4),
            "choice": "Right (choice = +1)" if p >= 0.5 else "Left (choice = -1)",
            "confidence": round(abs(p - 0.5) * 2.0, 4),
            "n_neurons": n,
            "top_neurons": top,
            "region_contributions": regions_out,
        }
