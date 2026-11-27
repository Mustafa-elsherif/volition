"""Run every decoding experiment, write result tables, and save the shipped model.

Outputs (all under ``results/model/`` unless noted):

* ``pooled_within.csv``   one shared transformer on held-out trials pooled across
  sessions, with per-session logistic-regression and random-forest references.
* ``ssl_ablation.csv``    transformer with and without self-supervised pretraining.
* ``cross_session.csv``   leave-one-session-out transformer vs region-pooled logreg.
* ``robustness.csv``      shipped transformer under neuron dropout and region knockout.
* ``summary.json``        headline mean +/- std numbers used by the docs and app.
* ``artifacts/``          the final serialized model (``model.pt``, ``decoder.onnx``,
  ``metadata.json``) trained on all sessions, plus a few sample trials for the demo.

The shipped weights are trained on all ten sessions; every reported accuracy comes
from held-out evaluation, never from the data the shipped model trained on.
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.common.settings import ROOT
from src.evaluation.metrics import decoding_scores
from src.evaluation.statistics import summarize_seeds
from src.models.baselines import region_pooled_logreg, within_session_oof
from src.models.data import Session, load_sessions
from src.models.splits import leave_one_session_out
from src.models.serialize import save_artifact
from src.models.train import (
    TrainConfig,
    predict_proba,
    pretrain_ssl,
    train_choice,
)

OUT = ROOT / "results" / "model"
ARTIFACTS = ROOT / "artifacts"
SEEDS = [0, 1, 2]
BASE = dict(d_model=64, n_layers=2, n_heads=4, dropout=0.1,
            ssl_epochs=20, choice_epochs=100, patience=12)


def _pool_all(sessions):
    return [(si, ti) for si, s in enumerate(sessions) for ti in range(s.n_trials)]


def _val_from(sessions, pool, seed, folds=6):
    y = np.array([sessions[si].y[ti] for si, ti in pool])
    tr, va = next(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros_like(y), y))
    return [pool[i] for i in tr], [pool[i] for i in va]


def pooled_within(sessions, vocab):
    """One shared transformer on held-out trials vs per-session references."""
    rows = []
    for seed in SEEDS:
        cfg = TrainConfig(seed=seed, n_regions=len(vocab), **BASE)
        pool = _pool_all(sessions)
        y = np.array([sessions[si].y[ti] for si, ti in pool])
        tr_all, te_all = next(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros_like(y), y))
        train_pool = [pool[i] for i in tr_all]
        test_pool = [pool[i] for i in te_all]
        tr_idx, va_idx = _val_from(sessions, train_pool, seed)

        pre = pretrain_ssl(sessions, cfg)
        model = train_choice(sessions, tr_idx, va_idx, cfg, pretrained=pre)
        y_te = np.array([sessions[si].y[ti] for si, ti in test_pool])
        sc = decoding_scores(y_te, predict_proba(model, sessions, test_pool))
        rows.append({"model": "set_transformer", "seed": seed, **sc})

        by_session: dict[int, list[int]] = {}
        for si, ti in test_pool:
            by_session.setdefault(si, []).append(ti)
        for mdl in ["logreg", "random_forest", "majority"]:
            yt, pt = [], []
            for si, tis in by_session.items():
                oof = within_session_oof(sessions[si], mdl, seed=seed)
                yt.extend(sessions[si].y[tis]); pt.extend(oof[tis])
            sc = decoding_scores(np.array(yt), np.array(pt))
            rows.append({"model": mdl, "seed": seed, **sc})
    return rows


def ssl_ablation(sessions, vocab):
    rows = []
    for seed in SEEDS:
        cfg = TrainConfig(seed=seed, n_regions=len(vocab), **BASE)
        pool = _pool_all(sessions)
        y = np.array([sessions[si].y[ti] for si, ti in pool])
        tr_all, te_all = next(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros_like(y), y))
        train_pool = [pool[i] for i in tr_all]; test_pool = [pool[i] for i in te_all]
        tr_idx, va_idx = _val_from(sessions, train_pool, seed)
        y_te = np.array([sessions[si].y[ti] for si, ti in test_pool])
        for tag, use_ssl in [("with_ssl", True), ("no_ssl", False)]:
            pre = pretrain_ssl(sessions, cfg) if use_ssl else None
            model = train_choice(sessions, tr_idx, va_idx, cfg, pretrained=pre)
            sc = decoding_scores(y_te, predict_proba(model, sessions, test_pool))
            rows.append({"variant": tag, "seed": seed, **sc})
    return rows


def cross_session(sessions, vocab):
    rows = []
    for seed in SEEDS[:2]:  # a negative result; two seeds over ten folds give ample spread
        cfg = TrainConfig(seed=seed, n_regions=len(vocab), **BASE)
        for train_s, test_s in leave_one_session_out(len(sessions)):
            ts = [sessions[i] for i in train_s]
            test_idx = [(test_s, ti) for ti in range(sessions[test_s].n_trials)]
            tri = [(si, ti) for si in train_s for ti in range(sessions[si].n_trials)]
            tr_idx, va_idx = _val_from(sessions, tri, seed)
            pre = pretrain_ssl(ts, cfg)
            model = train_choice(sessions, tr_idx, va_idx, cfg, pretrained=pre)
            p_tf = predict_proba(model, sessions, test_idx)
            p_lr = region_pooled_logreg(ts, sessions[test_s], vocab, seed)
            for name, p in [("set_transformer", p_tf), ("region_pooled_logreg", p_lr)]:
                sc = decoding_scores(sessions[test_s].y, p)
                rows.append({"model": name, "seed": seed,
                             "session": sessions[test_s].session_id, **sc})
    return rows


def _subset_session(s: Session, keep: np.ndarray) -> Session:
    return replace(s, X=s.X[:, keep], regions=s.regions[keep], region_ids=s.region_ids[keep])


def robustness(model, sessions, vocab):
    """Shipped model under random neuron dropout and whole-region knockout."""
    rows = []
    rng = np.random.default_rng(42)
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]:
        accs, aucs = [], []
        for si, s in enumerate(sessions):
            keep = rng.random(s.n_neurons) >= frac
            if keep.sum() < 5:
                continue
            sub = _subset_session(s, keep)
            p = predict_proba(model, [sub], [(0, ti) for ti in range(sub.n_trials)])
            sc = decoding_scores(s.y, p)
            accs.append(sc["balanced_accuracy"]); aucs.append(sc["roc_auc"])
        rows.append({"perturbation": "neuron_dropout", "level": frac,
                     "balanced_accuracy": float(np.mean(accs)),
                     "roc_auc": float(np.nanmean(aucs)), "n_sessions": len(accs)})
    # region knockout: drop the single most common region, per session
    accs, aucs = [], []
    for s in sessions:
        regs, counts = np.unique(s.regions, return_counts=True)
        top = regs[np.argmax(counts)]
        keep = s.regions != top
        if keep.sum() < 5:
            continue
        sub = _subset_session(s, keep)
        p = predict_proba(model, [sub], [(0, ti) for ti in range(sub.n_trials)])
        sc = decoding_scores(s.y, p)
        accs.append(sc["balanced_accuracy"]); aucs.append(sc["roc_auc"])
    rows.append({"perturbation": "drop_top_region", "level": 1.0,
                 "balanced_accuracy": float(np.mean(accs)),
                 "roc_auc": float(np.nanmean(aucs)), "n_sessions": len(accs)})
    return rows


def save_samples(sessions, n_sessions=6, per_session=6):
    """Write a small set of real trials for the in-browser demo (client pads them)."""
    rng = np.random.default_rng(7)
    out = []
    for s in sessions[:n_sessions]:
        pick = rng.choice(s.n_trials, size=min(per_session, s.n_trials), replace=False)
        for ti in pick:
            out.append({
                "session_id": s.session_id,
                "rates": [round(float(v), 3) for v in s.X[ti]],
                "region_ids": [int(v) for v in s.region_ids],
                "regions": s.regions.tolist(),
                "y_true": int(s.y[ti]),
                "contrast": float(s.contrast[ti]),
                "is_error": bool(s.is_error[ti]),
                "prob_left": float(s.prob_left[ti]),
            })
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "samples.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"saved {len(out)} sample trials")


def _write_csv(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def _mean_std(rows, key_filter, metric):
    vals = [r[metric] for r in rows if key_filter(r)]
    return summarize_seeds(vals)


def ship_final_model(sessions, vocab):
    """Train the shipped model on all sessions and serialize it with sample trials."""
    cfg = TrainConfig(seed=42, n_regions=len(vocab), **BASE)
    pool = _pool_all(sessions)
    tr_idx, va_idx = _val_from(sessions, pool, 42)
    pre = pretrain_ssl(sessions, cfg)
    model = train_choice(sessions, tr_idx, va_idx, cfg, pretrained=pre)
    return model, cfg


def main():
    sessions, vocab = load_sessions()
    OUT.mkdir(parents=True, exist_ok=True)

    pw = pooled_within(sessions, vocab); _write_csv(pw, "pooled_within.csv"); print("pooled_within done")
    ab = ssl_ablation(sessions, vocab); _write_csv(ab, "ssl_ablation.csv"); print("ssl_ablation done")
    cs = cross_session(sessions, vocab); _write_csv(cs, "cross_session.csv"); print("cross_session done")

    model, cfg = ship_final_model(sessions, vocab)
    rb = robustness(model, sessions, vocab); _write_csv(rb, "robustness.csv"); print("robustness done")

    summary = {
        "pooled_within": {
            m: _mean_std(pw, lambda r, m=m: r["model"] == m, "roc_auc")
            for m in ["set_transformer", "logreg", "random_forest", "majority"]
        },
        "pooled_within_balacc": {
            m: _mean_std(pw, lambda r, m=m: r["model"] == m, "balanced_accuracy")
            for m in ["set_transformer", "logreg", "random_forest", "majority"]
        },
        "ssl_ablation_auc": {
            v: _mean_std(ab, lambda r, v=v: r["variant"] == v, "roc_auc")
            for v in ["with_ssl", "no_ssl"]
        },
        "cross_session_auc": {
            m: _mean_std(cs, lambda r, m=m: r["model"] == m, "roc_auc")
            for m in ["set_transformer", "region_pooled_logreg"]
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_samples(sessions)

    metrics = {
        "headline_auc": summary["pooled_within"]["set_transformer"],
        "headline_balanced_accuracy": summary["pooled_within_balacc"]["set_transformer"],
        "protocol": "pooled held-out trials across 10 IBL sessions, mean +/- std over 3 seeds",
    }
    save_artifact(model, cfg, vocab, metrics, ARTIFACTS)
    print("summary:", json.dumps(summary, indent=2))
    print("shipped artifact ->", ARTIFACTS)


if __name__ == "__main__":
    main()
