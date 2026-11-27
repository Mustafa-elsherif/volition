"""Unit tests for the Volition decoding pipeline.

These use small synthetic inputs and do not require the IBL dataset, so they run
anywhere. They cover the metrics, the resampling statistics, the cross-validation
splits, the set-transformer forward pass and attention, and the ONNX serialization
round trip that the serving layer and the browser demo depend on.
"""
import numpy as np
import pytest

from src.evaluation.metrics import decoding_scores, expected_calibration_error
from src.evaluation.statistics import bootstrap_ci, paired_difference, summarize_seeds
from src.models.splits import leave_one_session_out, within_session_folds


# ---------- metrics ----------

def test_decoding_scores_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    perfect = decoding_scores(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["roc_auc"] == 1.0
    assert perfect["balanced_accuracy"] == 1.0
    chance = decoding_scores(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert chance["balanced_accuracy"] == pytest.approx(0.5)


def test_roc_auc_is_nan_with_one_class():
    scores = decoding_scores(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))
    assert np.isnan(scores["roc_auc"])


def test_ece_zero_when_calibrated():
    y = np.array([0, 1] * 50)
    # probabilities equal to the empirical rate in every bin are calibrated
    ece = expected_calibration_error(y, np.full(100, 0.5), n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        decoding_scores(np.array([0, 2]), np.array([0.5, 0.5]))


# ---------- statistics ----------

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0, 0.2, 200), 0, 1)
    from sklearn.metrics import roc_auc_score
    point, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_boot=500, seed=0)
    assert lo <= point <= hi


def test_paired_difference_direction():
    out = paired_difference([0.9, 0.8, 0.85], [0.7, 0.6, 0.65])
    assert out["mean_difference"] > 0
    assert out["n_pairs"] == 3


def test_summarize_seeds():
    s = summarize_seeds([0.7, 0.72, 0.74])
    assert s["mean"] == pytest.approx(0.72)
    assert s["n_seeds"] == 3


# ---------- splits ----------

def test_leave_one_session_out_covers_each_session():
    folds = list(leave_one_session_out(4))
    assert [test for _, test in folds] == [0, 1, 2, 3]
    for train, test in folds:
        assert test not in train and len(train) == 3


def test_within_session_folds_partition_test_indices():
    y = np.array([0, 1] * 25)
    seen = np.concatenate([test for _, test in within_session_folds(y, n_folds=5, seed=0)])
    assert sorted(seen) == list(range(50))


# ---------- model and serialization ----------

def test_set_transformer_forward_and_attention():
    import torch
    from src.models.set_transformer import NeuronSetTransformer
    model = NeuronSetTransformer(n_regions=10, d_model=32, n_heads=4, n_layers=2).eval()
    rates = torch.randn(2, 20)
    region_ids = torch.randint(1, 11, (2, 20))
    pad_mask = torch.zeros(2, 20, dtype=torch.bool)
    pad_mask[:, 15:] = True
    logits = model(rates, region_ids, pad_mask)
    assert logits.shape == (2,)
    _, attn = model.forward_with_attention(rates, region_ids, pad_mask)
    # attention sums to one over the real tokens and is zero on padding
    assert attn.shape == (2, 20)
    assert torch.allclose(attn[:, :15].sum(dim=1), torch.ones(2), atol=1e-4)
    assert attn[:, 15:].abs().sum().item() == pytest.approx(0.0, abs=1e-5)


def test_onnx_round_trip_matches_torch(tmp_path):
    import torch
    import onnxruntime as ort
    from src.models.serialize import save_artifact, load_model, MAX_NEURONS
    from src.models.set_transformer import NeuronSetTransformer
    from src.models.train import TrainConfig

    vocab = [f"R{i}" for i in range(10)]
    cfg = TrainConfig(n_regions=len(vocab))
    model = NeuronSetTransformer(len(vocab), cfg.d_model, cfg.n_heads, cfg.n_layers).eval()
    save_artifact(model, cfg, vocab, {"auc": 0.75}, tmp_path)

    reloaded, vocab2, meta = load_model(tmp_path)
    assert vocab2 == vocab and meta["max_neurons"] == MAX_NEURONS

    n = 12
    R = np.zeros((1, MAX_NEURONS), np.float32); R[0, :n] = np.random.randn(n)
    I = np.zeros((1, MAX_NEURONS), np.int64); I[0, :n] = np.random.randint(1, 11, n)
    M = np.ones((1, MAX_NEURONS), bool); M[0, :n] = False
    with torch.no_grad():
        ref = torch.sigmoid(model(torch.from_numpy(R), torch.from_numpy(I), torch.from_numpy(M))).item()
    sess = ort.InferenceSession(str(tmp_path / "decoder.onnx"), providers=["CPUExecutionProvider"])
    prob = sess.run(None, {"rates": R, "region_ids": I, "pad_mask": M})[0].ravel()[0]
    assert abs(ref - prob) < 1e-5
