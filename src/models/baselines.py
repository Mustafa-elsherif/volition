"""Reference decoders.

Two families are reported:

* Within-session baselines (``majority``, ``logreg``, ``random_forest``) are
  retrained inside each session with stratified k-fold. These match how most IBL
  single-session choice decoders are evaluated.
* The cross-session baseline (``region_pooled_logreg``) is a single logistic
  model over the shared region-averaged features, trained on a set of sessions
  and tested on a held-out one. It is the fair linear reference for the
  transformer, since both see only the shared region space across sessions.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.data import Session, region_pooled_matrix
from src.models.splits import within_session_folds


def _logreg() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def within_session_oof(
    session: Session, model: str, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    """Return out-of-fold probabilities for class 1, aligned to session.y.

    ``model`` is one of ``majority``, ``logreg``, ``random_forest``. Majority
    predicts the training-fold base rate for every test trial, which is the honest
    no-information reference under balanced accuracy.
    """
    X, y = session.X, session.y
    prob = np.zeros(session.n_trials, dtype=float)
    for train_idx, test_idx in within_session_folds(y, n_folds, seed):
        if model == "majority":
            prob[test_idx] = float(y[train_idx].mean())
            continue
        if model == "logreg":
            clf = make_pipeline(StandardScaler(), _logreg())
        elif model == "random_forest":
            clf = _random_forest(seed)
        else:
            raise ValueError(f"Unknown within-session model: {model}")
        clf.fit(X[train_idx], y[train_idx])
        prob[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return prob


def region_pooled_logreg(
    train_sessions: list[Session], test_session: Session, vocab: list[str], seed: int = 42
) -> np.ndarray:
    """Train one logistic model on pooled region features and score a held-out session.

    Returns class-1 probabilities for every trial of ``test_session``.
    """
    X_train = np.concatenate(
        [region_pooled_matrix(s, vocab) for s in train_sessions], axis=0
    )
    y_train = np.concatenate([s.y for s in train_sessions], axis=0)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=seed),
    )
    clf.fit(X_train, y_train)
    return clf.predict_proba(region_pooled_matrix(test_session, vocab))[:, 1]
