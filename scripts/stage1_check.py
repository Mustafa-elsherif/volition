"""Stage 1 gate check (owner: Mustafa). Read-only; writes no project files."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.common.checks import check_session
from src.common.settings import (
    connect_ibl,
    list_processed_sessions,
    load_config,
    load_session,
    load_session_list,
)
from src.data.clean_trials import clean_trials, load_trials

NOTEBOOKS = [
    "01_raw_data_eda",
    "02_sessions_and_dataset",
    "03_trial_cleaning",
    "04_neurons_and_regions",
]

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ((" - " + detail) if detail else ""))
    if not ok:
        failures.append(name)


config = load_config()
selected = load_session_list()
ready = list_processed_sessions()

print("== 1. sessions are built and load ==")
check("session list has the configured size",
      len(selected) == config["data"]["n_sessions"],
      str(len(selected)) + " selected")
check("every selected session is built",
      sorted(ready) == sorted(selected),
      str(len(ready)) + " built of " + str(len(selected)))

rows = []
for sid in selected:
    data = load_session(sid)
    check_session(data)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"])
    counts = np.bincount(y, minlength=2)
    rows.append({
        "session_id": sid,
        "n_trials": int(X.shape[0]),
        "n_neurons": int(X.shape[1]),
        "n_regions": int(len(set(map(str, data["neuron_region"])))),
        "mean_y": round(float(y.mean()), 3),
        "majority_rate": round(float(counts.max() / y.size), 3),
    })
shapes = pd.DataFrame(rows)
check("all sessions pass check_session", True, "format of the 12 fields is correct")

print()
print("== 2. shapes and y balance ==")
print(shapes.to_string(index=False))
check("every session reaches the trial minimum",
      bool((shapes["n_trials"] >= config["data"]["min_trials"]).all()),
      "minimum is " + str(config["data"]["min_trials"]))
check("every session reaches the neuron minimum",
      bool((shapes["n_neurons"] >= config["data"]["min_neurons"]).all()),
      "minimum is " + str(config["data"]["min_neurons"]))
check("every session reaches the region minimum",
      bool((shapes["n_regions"] >= config["data"]["min_regions"]).all()),
      "minimum is " + str(config["data"]["min_regions"]))
check("y uses both classes everywhere",
      bool((shapes["mean_y"] > 0).all() and (shapes["mean_y"] < 1).all()))
check("no session is more unbalanced than 75/25",
      bool((shapes["majority_rate"] <= 0.75).all()),
      "worst is " + str(float(shapes["majority_rate"].max())))

print()
print("== 3. IBL mask check and choice direction, per session ==")
one = connect_ibl()
audit = []
for sid in selected:
    trials, official_mask = load_trials(one, sid, config)
    _, report = clean_trials(trials, official_mask, config)
    direction = report["choice_direction"]
    audit.append({
        "session_id": sid,
        "input_trials": report["n_input_trials"],
        "kept_trials": report["n_kept_trials"],
        "ibl_check": report["ibl_check"]["passed"],
        "extra_ibl": report["ibl_check"]["n_extra_ibl_rejections"],
        "convention": direction["convention"],
        "agreement": round(direction["agreement_fraction"], 4),
        "checked_trials": direction["n_trials_checked"],
    })
audit = pd.DataFrame(audit)
print(audit.to_string(index=False))
check("IBL mask check passed for every session", bool(audit["ibl_check"].all()))
check("one choice-direction convention for all sessions",
      audit["convention"].nunique() == 1,
      "convention is " + str(audit["convention"].iloc[0]))
check("kept trials match the built sessions",
      bool((audit.set_index("session_id")["kept_trials"]
            == shapes.set_index("session_id")["n_trials"]).all()))

print()
print("== 4. notebooks 01-04 ran without errors ==")
for name in NOTEBOOKS:
    path = ROOT / "notebooks" / (name + ".ipynb")
    if not path.exists():
        check(name + " exists", False)
        continue
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    unrun = [c for c in code_cells if c.get("execution_count") is None]
    errors = [o for c in code_cells for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    text = "".join("".join(c["source"]) for c in cells)
    draft = "to be written" in text.lower()
    ok = (not unrun) and (not errors) and "## Plan" in text and "## Conclusions" in text and not draft
    detail = []
    if unrun:
        detail.append(str(len(unrun)) + " cells not run")
    if errors:
        detail.append(str(len(errors)) + " error outputs")
    if "## Plan" not in text:
        detail.append("no Plan cell")
    if "## Conclusions" not in text:
        detail.append("no Conclusions cell")
    if draft:
        detail.append("Conclusions still a draft")
    check(name, ok, ", ".join(detail))

print()
print("== 5. files match their task in the team guide ==")
import importlib

FIXED_NAMES = {
    "src.common.settings": ["load_config", "connect_ibl", "load_session_list", "load_session_info",
                            "make_rng", "save_session", "load_session", "list_processed_sessions",
                            "save_predictions", "load_predictions", "save_table", "load_table",
                            "figure_path"],
    "src.common.checks": ["check_session", "check_predictions", "check_table"],
    "src.data.sessions": ["main"],
    "src.data.clean_trials": ["load_trials", "clean_trials", "check_choice_direction"],
    "src.data.neurons": ["load_neurons"],
    "src.data.features": ["firing_rates"],
    "src.data.build_dataset": ["build_session", "assemble_session", "main"],
    "src.analysis.eda": ["trials_overview", "find_blocks", "sessions_overview"],
}
for module, names in FIXED_NAMES.items():
    mod = importlib.import_module(module)
    missing = [n for n in names if not callable(getattr(mod, n, None))]
    check(module + " has the guide's function names", not missing,
          ("missing: " + ", ".join(missing)) if missing else "")

card = (ROOT / "docs" / "data_card.md").read_text(encoding="utf-8")
for needle, label in [
    ("CC-BY 4.0", "licence"),
    ("data_release_brainwidemap", "link to the official release"),
    (config["ibl"]["release_freeze"], "release freeze"),
    ("Final cohort", "final cohort of 10 sessions"),
    ("Intended use and limits", "known limits"),
]:
    check("data card has the " + label, needle in card)

nb01 = json.loads((ROOT / "notebooks" / "01_raw_data_eda.ipynb").read_text(encoding="utf-8"))
nb01_code = "".join("".join(c["source"]) for c in nb01["cells"] if c["cell_type"] == "code")
check("notebook 01 loads the raw data with IBL's loaders",
      "load_trials(" in nb01_code and "load_neurons(" in nb01_code and "load_session(" not in nb01_code)

print()
if failures:
    print("STAGE 1 FAILED. Problems: " + ", ".join(failures))
    raise SystemExit(1)
print("STAGE 1 PASSED. All checks above are green.")

