
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


def get_candidates(one, config):
    """Filter sessions using official IBL metadata."""
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

    units = units[units["eid"].isin(valid_eids)].copy()

    return bwm, units


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
