
"""Build ready IBL sessions (Ahmed — S1.7)."""

import argparse
import numpy as np

from src.common.settings import (
    connect_ibl,
    load_config,
    load_session_info,
    load_session_list,
    list_processed_sessions,
    save_session,
)


def assemble_session(
    session_id,
    clean_table,
    neurons_table,
    firing_rate_matrix,
):
    """Assemble the 12 required fields for one session."""

    x = np.asarray(firing_rate_matrix, dtype=float)

    data = {
        "X": x,
        "y": clean_table["y"].to_numpy(dtype=int),
        "trial_number": clean_table[
            "trial_number"
        ].to_numpy(dtype=int),
        "stim_side": clean_table[
            "stim_side"
        ].to_numpy(dtype=int),
        "contrast": clean_table[
            "contrast"
        ].to_numpy(dtype=float),
        "is_error": clean_table[
            "is_error"
        ].to_numpy(dtype=bool),
        "is_zero_contrast": clean_table[
            "is_zero_contrast"
        ].to_numpy(dtype=bool),
        "prob_left": clean_table[
            "prob_left"
        ].to_numpy(dtype=float),
        "reaction_time": clean_table[
            "reaction_time"
        ].to_numpy(dtype=float),
        "neuron_id": neurons_table[
            "neuron_id"
        ].to_numpy(dtype=str),
        "neuron_region": neurons_table[
            "region"
        ].to_numpy(dtype=str),
        "session_id": np.asarray(str(session_id)),
    }

    if x.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    if x.shape != (
        len(clean_table),
        len(neurons_table),
    ):
        raise ValueError(
            "X dimensions do not match trials and neurons."
        )

    if not np.isfinite(x).all():
        raise ValueError("X contains NaN or infinity.")

    return data


def build_session(one, session_id, config):
    """Build and save one ready session."""

    # Import when needed; these files belong to teammates.
    from src.data.clean_trials import (
        load_trials,
        clean_trials,
    )
    from src.data.neurons import load_neurons
    from src.data.features import firing_rates

    info = load_session_info(session_id)
    pids = info["pids"]

    print(f"\nBuilding session: {session_id}")

    # Step 1: Load and clean trials.
    trials, official_mask = load_trials(
        one, session_id, config
    )

    clean_table, cleaning_report = clean_trials(
        trials, official_mask, config
    )

    if len(clean_table) == 0:
        raise ValueError("No valid trials remain.")

    # Step 2: Load neurons and spikes.
    neurons_table, spikes = load_neurons(
        one, pids, config
    )

    if len(neurons_table) == 0:
        raise ValueError("No valid neurons found.")

    # Step 3: Define pre-movement windows.
    window = config["data"]["window"]

    movement_times = clean_table[
        "firstMovement_times"
    ].to_numpy(dtype=float)

    starts = movement_times + float(window[0])
    ends = movement_times + float(window[1])

    if not np.isfinite(movement_times).all():
        raise ValueError("Invalid movement times.")

    if np.any(ends <= starts):
        raise ValueError("Invalid feature window.")

    # Step 4: Calculate firing rates in Hz.
    x = firing_rates(
        spikes["times"],
        spikes["neuron_index"],
        len(neurons_table),
        starts,
        ends,
    )

    # Step 5: Assemble and save.
    data = assemble_session(
        session_id,
        clean_table,
        neurons_table,
        x,
    )

    save_session(data)

    print(
        f"Saved: {len(clean_table)} trials, "
        f"{len(neurons_table)} neurons"
    )

    return data


def main():
    """Build selected sessions, skipping existing files."""

    parser = argparse.ArgumentParser(
        description="Build the selected IBL sessions."
    )

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Session IDs to build",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing session files",
    )

    args = parser.parse_args()

    config = load_config()
    selected = load_session_list()

    session_ids = (
        args.sessions
        if args.sessions is not None
        else selected
    )

    unknown = set(session_ids) - set(selected)

    if unknown:
        parser.error(
            f"Unknown session IDs: {sorted(unknown)}"
        )

    ready = set(list_processed_sessions())
    pending = []

    for session_id in session_ids:
        if session_id in ready and not args.force:
            print(f"Skipping existing: {session_id}")
        else:
            pending.append(session_id)

    if not pending:
        print("All requested sessions are ready.")
        return

    print(f"Sessions to build: {len(pending)}")

    try:
        from src.data.clean_trials import (
            load_trials,
            clean_trials,
        )
        from src.data.features import firing_rates
        from src.data.neurons import load_neurons
    except ImportError as exc:
        raise SystemExit(
            "Required teammate modules are not ready: "
            f"{exc}"
        ) from exc

    one = connect_ibl()

    failures = []

    for session_id in pending:
        try:
            build_session(one, session_id, config)
        except Exception as exc:
            failures.append((session_id, str(exc)))
            print(f"FAILED {session_id}: {exc}")

    completed = len(pending) - len(failures)

    print(f"\nBuilt successfully: {completed}")
    print(f"Failed: {len(failures)}")

    if failures:
        for session_id, error in failures:
            print(f"{session_id}: {error}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
