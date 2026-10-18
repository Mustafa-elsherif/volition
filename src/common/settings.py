"""Shared settings and file helpers (owner: Mustafa, S0.1)."""
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config():
    """Return config.yaml as a dict."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def connect_ibl(local_only=False):
    """Return the IBL connection (public login, cache in data/raw/)."""
    from one.api import ONE
    cfg = load_config()["ibl"]
    user = os.environ.get("IBL_USERNAME") or cfg["public_username"]
    pwd = os.environ.get("IBL_PASSWORD") or cfg["public_password"]
    kwargs = dict(base_url=cfg["server"], username=user, password=pwd,
                  silent=True, cache_dir=str(ROOT / "data" / "raw"))
    if local_only:
        kwargs["mode"] = "local"
    return ONE(**kwargs)


def _sessions_yaml():
    with open(ROOT / "sessions.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["sessions"]


def load_session_list():
    """Return the session IDs of sessions.yaml, in order."""
    return [s["session_id"] for s in _sessions_yaml()]


def load_session_info(session_id):
    """Return one session's entry of sessions.yaml (with its pids)."""
    for s in _sessions_yaml():
        if s["session_id"] == session_id:
            return s
    raise KeyError(f"{session_id} is not in sessions.yaml")


def make_rng(*names):
    """Return a reproducible random generator built from the seed and the given names."""
    seed = load_config()["seed"]
    return np.random.default_rng([seed] + [zlib.crc32(str(n).encode()) for n in names])


def _processed_dir():
    return ROOT / "data" / "processed"


def save_session(data):
    """Check and save one ready session as data/processed/<session_id>.npz."""
    from src.common.checks import check_session
    check_session(data)
    out = {}
    for k, v in data.items():
        v = np.asarray(v)
        if v.dtype.kind in "OUS":
            v = v.astype(str)
        out[k] = v
    folder = _processed_dir()
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{str(np.asarray(data['session_id']))}.npz", **out)


def load_session(session_id):
    """Load one ready session as a dict of arrays."""
    with np.load(_processed_dir() / f"{session_id}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def list_processed_sessions():
    """Return the session IDs that have a ready file, in sessions.yaml order."""
    ready = {p.stem for p in _processed_dir().glob("*.npz")}
    return [s for s in load_session_list() if s in ready]


def _pred_dir(experiment):
    return ROOT / "results" / "predictions" / experiment


def save_predictions(df, experiment, session_id):
    """Check and save the predictions of one session and one experiment."""
    from src.common.checks import check_predictions
    check_predictions(df)
    folder = _pred_dir(experiment)
    folder.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["y_prob"] = df["y_prob"].round(4)
    df.to_csv(folder / f"{session_id}.csv.gz", index=False,
              compression={"method": "gzip", "mtime": 0})


def load_predictions(experiment=None):
    """Load results/predictions/<experiment>/*.csv.gz (all experiments when None)."""
    names = [experiment] if experiment else load_config()["experiments"]["names"]
    files = [p for e in names for p in sorted(_pred_dir(e).glob("*.csv.gz"))]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def save_table(df, name):
    """Check and save results/tables/<name>.csv."""
    from src.common.checks import check_table
    check_table(df, name)
    folder = ROOT / "results" / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)


def load_table(name):
    """Load results/tables/<name>.csv."""
    return pd.read_csv(ROOT / "results" / "tables" / f"{name}.csv")


def figure_path(name):
    """Return results/figures/<name>.png (folder created)."""
    folder = ROOT / "results" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{name}.png"