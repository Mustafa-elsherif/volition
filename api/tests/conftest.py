"""Test fixtures: build a small dummy artifact and a client bound to it.

The dummy artifact uses an untrained model, so these tests check the service
contract (shapes, validation, grounding) rather than decoding accuracy, which is
measured separately in the modelling code.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
API_DIR = REPO_ROOT / "api"
for p in (str(REPO_ROOT), str(API_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    from src.models.serialize import save_artifact
    from src.models.set_transformer import NeuronSetTransformer
    from src.models.train import TrainConfig

    out = tmp_path_factory.mktemp("artifacts")
    vocab = ["VISp", "CP", "MOs", "SCm", "LP", "root"]
    cfg = TrainConfig(n_regions=len(vocab))
    model = NeuronSetTransformer(len(vocab), cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout)
    metrics = {"headline_auc": {"mean": 0.79, "std": 0.01}}
    save_artifact(model, cfg, vocab, metrics, out)
    (out / "samples.json").write_text("[]", encoding="utf-8")
    return out


@pytest.fixture()
def client(artifact_dir, monkeypatch):
    monkeypatch.setenv("VOLITION_ARTIFACTS", str(artifact_dir))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import importlib

    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c
