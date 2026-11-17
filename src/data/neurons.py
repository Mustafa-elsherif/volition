"""Good neurons and their brain regions (owner: Mustafa, S1.4)."""
import numpy as np
import pandas as pd
from brainwidemap import load_good_units
from iblatlas.atlas import BrainRegions


def probe_neurons(spikes, clusters, regions, excluded):
    """Keep the neurons of one probe that are outside the excluded regions.

    Returns (table, times, index). table has one row per kept neuron
    (neuron_id, region, firing_rate). index gives the row in table of each spike.
    """
    idx = np.asarray(spikes["clusters"])
    n = len(clusters)
    if len(idx) and (idx.min() < 0 or idx.max() > n - 1):
        raise ValueError(f"spikes['clusters'] must be row numbers 0..{n - 1} of clusters, "
                         f"got {idx.min()}..{idx.max()}")
    regions = np.asarray(regions)
    keep = ~np.isin(regions, excluded)
    new_row = np.full(n, -1)
    new_row[keep] = np.arange(keep.sum())
    spike_keep = keep[idx]
    times = np.asarray(spikes["times"])[spike_keep]
    index = new_row[idx[spike_keep]]
    counts = np.bincount(index, minlength=int(keep.sum()))
    duration = times.max() - times.min() if len(times) > 1 else np.nan
    table = pd.DataFrame({
        "neuron_id": np.asarray(clusters["uuids"]).astype(str)[keep],
        "region": regions[keep],
        "firing_rate": counts / duration,
    })
    return table, times, index


def load_neurons(one, pids, config):
    """Return (neurons_table, spikes) for all probes of a session, joined in order."""
    d = config["data"]
    br = BrainRegions()
    tables, all_times, all_index = [], [], []
    offset = 0
    for pid in pids:
        spikes, clusters = load_good_units(one, pid, qc=d["good_neuron_qc"])
        regions = br.id2acronym(clusters["atlas_id"].to_numpy(), mapping=d["region_mapping"])
        table, times, index = probe_neurons(spikes, clusters, regions, d["excluded_regions"])
        tables.append(table)
        all_times.append(times)
        all_index.append(index + offset)
        offset += len(table)
    neurons = pd.concat(tables, ignore_index=True)
    spikes = {"times": np.concatenate(all_times), "neuron_index": np.concatenate(all_index)}
    return neurons, spikes
