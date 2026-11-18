"""S1.3: load IBL trials, audit exclusions, and create decoding columns.

The official IBL mask decides which trials survive. The project rules are an
audit of that mask, not an alternative filter. No files are written here.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


# Fixed data-contract rules from the team guide, not tunable model settings.
REQUIRED_EVENTS = (
    "stimOn_times", "choice", "feedback_times", "probabilityLeft",
    "firstMovement_times", "feedbackType",
)
CONTRAST_COLUMNS = ("contrastLeft", "contrastRight")
CHOICE_DIRECTION_MIN_AGREEMENT = 0.99


def _reaction_time_limits(config):
    """Read and validate the configured inclusive reaction-time limits."""
    data_config = config["data"]
    minimum = float(data_config["min_reaction_time"])
    maximum = float(data_config["max_reaction_time"])
    if not np.isfinite([minimum, maximum]).all() or not 0 <= minimum <= maximum:
        raise ValueError("Reaction-time limits must be finite and 0 <= min <= max.")
    return minimum, maximum


def load_trials(one, session_id, config):
    """Return the IBL trials table and official inclusion mask for one session.

    IBL may truncate a session before returning its table. Audit counts below
    refer to this returned table, not to trials already removed by the loader.
    The import is local so the offline cleaning tests need no IBL connection.
    """
    from brainwidemap import load_trials_and_mask

    minimum, maximum = _reaction_time_limits(config)
    return load_trials_and_mask(
        one,
        session_id,
        min_rt=minimum,
        max_rt=maximum,
        nan_exclude="default",
        exclude_nochoice=True,
    )


def _validate_official_mask(trials, official_mask):
    """Return a Boolean array after checking mask shape, values and alignment."""
    if isinstance(official_mask, pd.Series):
        if not official_mask.index.equals(trials.index):
            raise ValueError("official_mask index must match the trials index in order.")
        if official_mask.isna().any() or not pd.api.types.is_bool_dtype(official_mask):
            raise ValueError("official_mask must contain only True/False, with no missing values.")
        mask = official_mask.to_numpy(dtype=bool)
    else:
        mask = np.asarray(official_mask)
        if mask.dtype.kind != "b":
            raise ValueError("official_mask must contain Boolean values, not numbers or strings.")
    if mask.ndim != 1 or len(mask) != len(trials):
        raise ValueError("official_mask must be one-dimensional with one value per trial.")
    return mask


def check_choice_direction(clean_table):
    """Check the guide's 99% sign-consistency rule on nonzero-contrast trials.

    Record the convention without flipping choice, y, stim_side or contrast.
    Zero-contrast trials provide no visible stimulus direction for this check.
    """
    visible = clean_table.loc[clean_table["contrast"].ne(0)]
    if visible.empty:
        raise ValueError("Cannot check choice direction: no kept nonzero-contrast trials.")
    choice_matches_side = visible["choice"].eq(visible["stim_side"])
    is_correct = visible["feedbackType"].eq(1)
    agreement = float(choice_matches_side.eq(is_correct).mean())
    if agreement >= CHOICE_DIRECTION_MIN_AGREEMENT:
        convention = "same"
    elif agreement <= 1 - CHOICE_DIRECTION_MIN_AGREEMENT:
        convention = "opposite"
    else:
        raise ValueError(
            "Inconsistent choice direction: agreement is "
            f"{agreement:.2%}; expected >= 99% or <= 1%."
        )
    return {
        "passed": True,
        "convention": convention,
        "n_trials_checked": int(len(visible)),
        "agreement_fraction": agreement,
        "minimum_consistency": CHOICE_DIRECTION_MIN_AGREEMENT,
    }


def clean_trials(trials, official_mask, config):
    """Return (clean_table, report), preserving the official IBL trial selection.

    Counts are sequential, so an excluded trial is counted only once. Original
    columns are preserved; trial_number is the zero-based row position in the
    unfiltered table returned by load_trials (not an arbitrary pandas label).
    Invalid retained data or a disagreement with IBL raises ValueError.
    """
    if not isinstance(trials, pd.DataFrame):
        raise TypeError("trials must be a pandas DataFrame from load_trials.")
    if not trials.columns.is_unique:
        raise ValueError("trials must not contain duplicate column names.")
    missing = sorted(set(REQUIRED_EVENTS + CONTRAST_COLUMNS) - set(trials.columns))
    if missing:
        raise ValueError(f"Missing required trial columns: {missing}")

    minimum, maximum = _reaction_time_limits(config)
    official_keep = _validate_official_mask(trials, official_mask)
    reaction_time = trials["firstMovement_times"] - trials["stimOn_times"]
    rules = (
        ("no_choice", trials["choice"].eq(0).fillna(False).to_numpy(dtype=bool)),
        ("missing_required_events", trials[list(REQUIRED_EVENTS)].isna().any(axis=1).to_numpy()),
        (
            "reaction_time_out_of_range",
            (reaction_time.lt(minimum) | reaction_time.gt(maximum)).fillna(False).to_numpy(dtype=bool),
        ),
    )

    our_keep = np.ones(len(trials), dtype=bool)
    steps = []
    for name, rejected_by_rule in rules:
        before = int(our_keep.sum())
        our_keep &= ~rejected_by_rule
        after = int(our_keep.sum())
        steps.append({"step": name, "n_before": before, "n_removed": before - after, "n_after": after})

    disagreement = (~our_keep) & official_keep
    if disagreement.any():
        positions = np.flatnonzero(disagreement).tolist()
        raise ValueError(
            "IBL mask disagreement: project rules reject trials kept by IBL. "
            f"Original row positions: {positions}"
        )

    extra_ibl = int((our_keep & ~official_keep).sum())
    steps.append({
        "step": "other IBL rules",
        "n_before": int(our_keep.sum()),
        "n_removed": extra_ibl,
        "n_after": int(official_keep.sum()),
    })
    if not official_keep.any():
        raise ValueError("No trials remain after applying official_mask.")

    clean_table = trials.iloc[np.flatnonzero(official_keep)].copy()
    clean_table["trial_number"] = np.flatnonzero(official_keep)
    left_present = clean_table["contrastLeft"].notna()
    right_present = clean_table["contrastRight"].notna()
    if not (left_present ^ right_present).all():
        raise ValueError("Each kept trial must have exactly one filled contrastLeft/contrastRight.")

    if not clean_table["choice"].isin([-1, 1]).all():
        raise ValueError("Kept choice values must be -1 or +1.")
    if not clean_table["feedbackType"].isin([-1, 1]).all():
        raise ValueError("Kept feedbackType values must be -1 or +1.")
    if not clean_table["probabilityLeft"].isin([0.2, 0.5, 0.8]).all():
        raise ValueError("Kept probabilityLeft values must be 0.2, 0.5 or 0.8.")
    if not np.isfinite(clean_table[list(REQUIRED_EVENTS)].to_numpy(dtype=float)).all():
        raise ValueError("Required events of kept trials must be finite numeric values.")
    magnitude = clean_table["contrastLeft"].where(left_present, clean_table["contrastRight"])
    if not np.isfinite(magnitude.to_numpy(dtype=float)).all() or magnitude.lt(0).any():
        raise ValueError("The filled contrast must be finite and nonnegative.")

    clean_table["y"] = clean_table["choice"].eq(1).astype(np.int8)
    clean_table["stim_side"] = np.where(left_present, 1, -1).astype(np.int8)
    clean_table["contrast"] = np.where(left_present, clean_table["contrastLeft"], -clean_table["contrastRight"])
    clean_table["is_error"] = clean_table["feedbackType"].eq(-1)
    clean_table["is_zero_contrast"] = clean_table["contrast"].eq(0)
    clean_table["prob_left"] = clean_table["probabilityLeft"]
    clean_table["reaction_time"] = reaction_time.to_numpy()[official_keep]
    clean_table = clean_table.reset_index(drop=True)

    report = {
        "n_input_trials": int(len(trials)),
        "n_kept_trials": int(len(clean_table)),
        "n_removed_trials": int(len(trials) - len(clean_table)),
        "reaction_time_limits": {"min": minimum, "max": maximum, "inclusive": True},
        "steps": steps,
        "ibl_check": {
            "passed": True,
            "n_project_rejections": int((~our_keep).sum()),
            "n_extra_ibl_rejections": extra_ibl,
        },
        "choice_direction": check_choice_direction(clean_table),
        "counts_scope": "The table returned by the IBL loader; earlier loader truncation is not included.",
    }
    return clean_table, report


def main():
    """Run a read-only smoke check on one real session using shared settings."""
    from brainwidemap import bwm_query
    from src.common.settings import connect_ibl, load_config

    parser = argparse.ArgumentParser(description="Check S1.3 on one real IBL session; writes no project files.")
    parser.add_argument("--session", help="Session UUID; default: first session in the configured frozen release.")
    args = parser.parse_args()
    config = load_config()
    session_id = args.session
    if session_id is None:
        sessions = bwm_query(freeze=config["ibl"]["release_freeze"])
        session_id = str(sessions["eid"].iloc[0])
    trials, official_mask = load_trials(connect_ibl(), session_id, config)
    clean_table, report = clean_trials(trials, official_mask, config)
    print(f"Session: {session_id}")
    print(json.dumps(report, indent=2, allow_nan=False))
    preview_columns = ["trial_number", "y", "stim_side", "contrast", "is_error", "is_zero_contrast", "reaction_time"]
    print(clean_table[preview_columns].head().to_string(index=False))


if __name__ == "__main__":
    main()
