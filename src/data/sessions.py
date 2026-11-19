
"""Select IBL Brain-Wide Map sessions for the project."""

from collections import Counter
from pathlib import Path

import yaml
import pandas as pd

from brainwidemap import bwm_query, load_trials_and_mask
from brainwidemap.bwm_loading import (
    download_aggregate_tables,
    filter_sessions,
    filter_units_region,
)

from src.common.settings import connect_ibl, load_config


def load_tables(one, config):
    """Return (bwm, units, valid_eids) from the frozen release and IBL's aggregate tables.

    units has one row per good neuron outside root/void (all release sessions);
    valid_eids are the sessions that pass IBL's trial rules and our trial minimums.
    """
    data_cfg = config["data"]
    ibl_cfg = config["ibl"]

    print("Loading BWM release...")
    bwm = bwm_query(freeze=ibl_cfg["release_freeze"])
    eids = bwm["eid"].unique()
    print("Available sessions:", len(eids))

    print("Downloading clusters table...")
    clusters_path = download_aggregate_tables(
        one,
        type="clusters",
        tag=ibl_cfg["aggregate_tag"],
    )

    print("Downloading trials table...")
    trials_path = download_aggregate_tables(
        one,
        type="trials",
        tag=ibl_cfg["aggregate_tag"],
    )

    print("Filtering good neurons...")
    units = filter_units_region(
        eids,
        clusters_table=clusters_path,
        mapping=data_cfg["region_mapping"],
        min_qc=data_cfg["good_neuron_qc"],
        min_units_sessions=None,
    )

    print("Filtering valid trials...")
    valid_eids = filter_sessions(
        eids,
        trials_path,
        bwm_include=True,
        min_errors=data_cfg["min_errors"],
        min_trials=data_cfg["min_trials"],
    )

    return bwm, units, valid_eids


def get_candidates(one, config):
    """Filter sessions using official IBL metadata."""
    bwm, units, valid_eids = load_tables(one, config)
    units = units[units["eid"].isin(valid_eids)].copy()

    return bwm, units


def selection_audit(bwm, units, valid_eids, config, selected, reserves):
    """Return how many sessions each selection rule removed, in the order the rules run.

    Read-only: it recomputes the counts from the same tables and rules as
    make_candidates and choose_sessions, and uses the saved selection
    (selected + reserves, in order) to tell a session whose trials failed to
    load apart from one removed by the mouse or lab rule. It never writes
    sessions.yaml.
    """
    cfg = config["data"]
    region_col = cfg["region_mapping"]
    release = set(bwm["eid"].astype(str))
    units = units.assign(eid=units["eid"].astype(str))
    valid = set(map(str, valid_eids))

    steps = [("sessions in the frozen release", len(release))]

    with_neurons = release & set(units["eid"])
    steps.append(("have good neurons outside root/void", len(with_neurons)))

    passing = with_neurons & valid
    steps.append((
        f"pass IBL trial rules, >= {cfg['min_trials']} trials, "
        f">= {cfg['min_errors']} errors",
        len(passing),
    ))

    kept = units[units["eid"].isin(passing)]
    n_neurons = kept.groupby("eid").size()
    enough_neurons = set(n_neurons[n_neurons >= cfg["min_neurons"]].index)
    steps.append((f">= {cfg['min_neurons']} good neurons", len(enough_neurons)))

    per_region = kept[kept["eid"].isin(enough_neurons)].groupby(["eid", region_col]).size()
    qualified = (per_region >= cfg["neurons_per_region"]).groupby(level="eid").sum()
    candidates_set = set(qualified[qualified >= cfg["min_regions"]].index)
    steps.append((
        f">= {cfg['min_regions']} regions with >= {cfg['neurons_per_region']} neurons",
        len(candidates_set),
    ))

    # walk the candidates in the same order as choose_sessions
    meta = bwm.drop_duplicates("eid").assign(eid=lambda d: d["eid"].astype(str)).set_index("eid")
    order = sorted(candidates_set, key=lambda e: (-int(n_neurons[e]), e))
    accepted_target = [str(e) for e in list(selected) + list(reserves)]
    accepted_set = set(accepted_target)
    used_subjects, lab_counts = set(), Counter()
    removed = Counter()
    accepted = []
    for eid in order:
        subject, lab = str(meta.loc[eid, "subject"]), str(meta.loc[eid, "lab"])
        if subject in used_subjects:
            removed["mouse"] += 1
        elif lab_counts[lab] >= cfg["max_sessions_per_lab"]:
            removed["lab"] += 1
        elif eid in accepted_set:
            used_subjects.add(subject)
            lab_counts[lab] += 1
            accepted.append(eid)
        else:
            removed["load"] += 1

    if accepted != accepted_target:
        raise ValueError(
            "The saved selection does not match the selection rules: "
            f"expected {len(accepted_target)} sessions in order, the rules give {len(accepted)}."
        )

    left = len(candidates_set)
    for rule, label in [
        ("mouse", "one session per mouse"),
        ("lab", f"at most {cfg['max_sessions_per_lab']} sessions per lab"),
        ("load", "official trials load without an error"),
    ]:
        left -= removed[rule]
        steps.append((label, left))

    rows, before = [], None
    for step, n in steps:
        rows.append({
            "step": step,
            "n_sessions": int(n),
            "n_removed": 0 if before is None else int(before - n),
        })
        before = n
    return pd.DataFrame(rows, columns=["step", "n_sessions", "n_removed"])


def run_selection_audit(one, config):
    """Load the tables and audit the saved sessions.yaml (selected and reserves)."""
    bwm, units, valid_eids = load_tables(one, config)
    with (Path(__file__).resolve().parents[2] / "sessions.yaml").open(encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    selected = [s["session_id"] for s in saved["sessions"]]
    return selection_audit(bwm, units, valid_eids, config, selected, saved.get("reserves", []))


def make_candidates(bwm, units, config, fallback=False):
    """Rank sessions by good neurons and region coverage."""
    cfg = config["data"]
    region_col = cfg["region_mapping"]

    min_neurons = (
        cfg["fallback_min_neurons"]
        if fallback else cfg["min_neurons"]
    )
    min_regions = (
        cfg["fallback_min_regions"]
        if fallback else cfg["min_regions"]
    )

    metadata = (
        bwm.drop_duplicates("eid")
        .set_index("eid")
    )

    rows = []

    for eid, group in units.groupby("eid"):
        region_counts = group[region_col].value_counts()

        qualified = region_counts[
            region_counts >= cfg["neurons_per_region"]
        ]

        if len(group) < min_neurons:
            continue
        if len(qualified) < min_regions:
            continue
        if eid not in metadata.index:
            continue

        info = metadata.loc[eid]

        rows.append({
            "session_id": str(eid),
            "subject": str(info["subject"]),
            "lab": str(info["lab"]),
            "pids": (
                bwm.loc[bwm["eid"] == eid, "pid"]
                .astype(str)
                .drop_duplicates()
                .tolist()
            ),
            "n_neurons": int(len(group)),
            "n_regions": int(len(region_counts)),
        })

    return sorted(
        rows,
        key=lambda row: (
            -row["n_neurons"],
            row["session_id"],
        ),
    )


def verify_trials(one, session_id, config):
    """Check that official trial loading succeeds."""
    cfg = config["data"]

    try:
        trials, mask = load_trials_and_mask(
            one,
            session_id,
            min_rt=cfg["min_reaction_time"],
            max_rt=cfg["max_reaction_time"],
            nan_exclude="default",
            exclude_nochoice=True,
        )
        return len(trials) > 0 and int(mask.sum()) > 0
    except Exception as exc:
        print(f"Skipping {session_id}: {exc}")
        return False


def choose_sessions(candidates, one, config):
    """Choose distinct mice while limiting sessions per lab."""
    cfg = config["data"]
    selected = []
    reserves = []
    used_subjects = set()
    lab_counts = Counter()

    target = cfg["n_sessions"]

    for candidate in candidates:
        subject = candidate["subject"]
        lab = candidate["lab"]

        if subject in used_subjects:
            continue

        if lab_counts[lab] >= cfg["max_sessions_per_lab"]:
            continue

        eid = candidate["session_id"]
        print("Checking:", eid)

        if not verify_trials(one, eid, config):
            continue

        used_subjects.add(subject)
        lab_counts[lab] += 1

        if len(selected) < target:
            selected.append(candidate)
            print(f"Selected {len(selected)}/{target}")
        else:
            reserves.append(eid)

    return selected, reserves


def main():
    """Select sessions and write sessions.yaml."""
    config = load_config()
    one = connect_ibl()

    bwm, units = get_candidates(one, config)

    candidates = make_candidates(
        bwm, units, config, fallback=False
    )

    print("Primary candidates:", len(candidates))

    selected, reserves = choose_sessions(
        candidates, one, config
    )

    if len(selected) < config["data"]["n_sessions"]:
        print("Insufficient primary sessions.")
        print("Fallback requires team-leader approval.")
        return

    result = {
        "sessions": selected,
        "reserves": reserves,
    }

    output = Path("sessions.yaml")
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            result,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\nSelection completed.")
    print("Selected sessions:", len(selected))
    print("Reserve sessions:", len(reserves))
    print("Saved to:", output.resolve())


if __name__ == "__main__":
    main()
