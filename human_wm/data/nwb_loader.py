"""
NWB Data Loader -- Schema-driven spike binning with MTL/Frontal separation

Loads NWB files, classifies neurons into MTL and Frontal groups using the
schema produced by ``nwb_explorer.generate_schema``, bins spikes into
fixed-width time bins, and produces train/val/test splits.

The key design principle is **schema-driven**: region classification uses
``schema['mtl_regions']`` and ``schema['frontal_regions']`` lists rather
than hardcoded patterns, so the pipeline adapts automatically to whatever
regions the schema discovered in the actual NWB files.

Usage (programmatic):
    from human_wm.data.nwb_loader import extract_patient_data, split_data
    from human_wm.config import load_nwb_schema

    schema = load_nwb_schema()
    X, Y, trial_info = extract_patient_data('sub-01.nwb', schema)
    splits = split_data(X, Y, trial_info, seed=42)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from human_wm.config import BIN_SIZE_S, TEST_FRAC, TRAIN_FRAC, VAL_FRAC


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_nwb(nwb_path: str | Path):
    """Open an NWB file via pynwb.

    Separated into its own function for testability -- tests mock this
    function to avoid requiring real NWB files on disk.

    Parameters
    ----------
    nwb_path : str or Path
        Path to the .nwb file.

    Returns
    -------
    pynwb.NWBFile
        The opened NWB file object.
    """
    from pynwb import NWBHDF5IO  # local import to keep module importable without pynwb

    io = NWBHDF5IO(str(nwb_path), mode='r')
    nwbfile = io.read()
    return nwbfile


def _classify_region(region_name: str, schema: dict) -> str:
    """Classify a brain region as 'mtl', 'frontal', or 'other' using schema.

    Uses case-insensitive matching against the schema's ``mtl_regions`` and
    ``frontal_regions`` lists. NO hardcoded patterns -- the schema is the
    single source of truth.

    Parameters
    ----------
    region_name : str
        Name of the brain region (e.g. 'hippocampus', 'dACC').
    schema : dict
        NWB schema dictionary with 'mtl_regions' and 'frontal_regions' keys.

    Returns
    -------
    str
        One of 'mtl', 'frontal', or 'other'.
    """
    name_lower = region_name.lower()

    for pattern in schema.get('mtl_regions', []):
        if pattern.lower() == name_lower:
            return 'mtl'

    for pattern in schema.get('frontal_regions', []):
        if pattern.lower() == name_lower:
            return 'frontal'

    return 'other'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_patient_data(
    nwb_path: str | Path,
    schema: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract binned spike data separated into MTL (X) and Frontal (Y).

    Parameters
    ----------
    nwb_path : str or Path
        Path to the .nwb file.
    schema : dict
        NWB schema dictionary (from ``generate_schema`` / ``load_nwb_schema``).
        Must contain keys: ``region_column``, ``mtl_regions``,
        ``frontal_regions``, ``trial_columns``.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_bins, n_mtl), dtype float32
        Binned spike counts for MTL neurons.
    Y : np.ndarray, shape (n_trials, n_bins, n_frontal), dtype float32
        Binned spike counts for Frontal neurons.
    trial_info : dict[str, np.ndarray]
        Dictionary mapping each trial column name to an array of values
        (length n_trials).
    """
    nwbfile = _open_nwb(nwb_path)

    # --- Identify region column ---
    region_column = schema.get('region_column', 'brain_region')

    # --- Classify neurons ---
    units = nwbfile.units
    n_units = len(units)

    mtl_indices = []
    frontal_indices = []

    for i in range(n_units):
        region = units[region_column][i]
        classification = _classify_region(region, schema)
        if classification == 'mtl':
            mtl_indices.append(i)
        elif classification == 'frontal':
            frontal_indices.append(i)

    n_mtl = len(mtl_indices)
    n_frontal = len(frontal_indices)

    # --- Extract trial timing ---
    trials = nwbfile.trials
    n_trials = len(trials)

    start_times = np.array([trials['start_time'][i] for i in range(n_trials)])
    stop_times = np.array([trials['stop_time'][i] for i in range(n_trials)])

    # --- Compute bin count from trial durations ---
    trial_durations = stop_times - start_times
    # Use the maximum duration to determine bin count (all trials same shape)
    max_duration = np.max(trial_durations)
    n_bins = int(np.ceil(max_duration / BIN_SIZE_S))

    # --- Bin spikes ---
    X = np.zeros((n_trials, n_bins, n_mtl), dtype=np.float32)
    Y = np.zeros((n_trials, n_bins, n_frontal), dtype=np.float32)

    for trial_idx in range(n_trials):
        t_start = start_times[trial_idx]
        t_stop = t_start + n_bins * BIN_SIZE_S  # use consistent bin count
        bin_edges = np.linspace(t_start, t_stop, n_bins + 1)

        # MTL neurons
        for neuron_idx, unit_idx in enumerate(mtl_indices):
            spike_times = np.asarray(units['spike_times'][unit_idx])
            # Select spikes within trial window
            mask = (spike_times >= t_start) & (spike_times < t_stop)
            trial_spikes = spike_times[mask]
            if len(trial_spikes) > 0:
                counts, _ = np.histogram(trial_spikes, bins=bin_edges)
                X[trial_idx, :, neuron_idx] = counts.astype(np.float32)

        # Frontal neurons
        for neuron_idx, unit_idx in enumerate(frontal_indices):
            spike_times = np.asarray(units['spike_times'][unit_idx])
            mask = (spike_times >= t_start) & (spike_times < t_stop)
            trial_spikes = spike_times[mask]
            if len(trial_spikes) > 0:
                counts, _ = np.histogram(trial_spikes, bins=bin_edges)
                Y[trial_idx, :, neuron_idx] = counts.astype(np.float32)

    # --- Extract trial info ---
    trial_columns = schema.get('trial_columns', [])
    trial_info: dict[str, Any] = {}
    for col in trial_columns:
        try:
            trial_info[col] = np.array([trials[col][i] for i in range(n_trials)])
        except (KeyError, TypeError):
            pass  # column not accessible; skip

    return X, Y, trial_info


def split_data(
    X: np.ndarray,
    Y: np.ndarray,
    trial_info: dict[str, Any],
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Split data into train / val / test sets.

    Uses random permutation with the given seed for reproducibility.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_bins, n_mtl)
    Y : np.ndarray, shape (n_trials, n_bins, n_frontal)
    trial_info : dict[str, np.ndarray]
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test', each containing
        a dict with keys 'X', 'Y', 'trial_info'.
    """
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)

    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # n_test gets the remainder to ensure exact sum
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def _subset(indices):
        sub_info = {}
        for k, v in trial_info.items():
            if isinstance(v, np.ndarray):
                sub_info[k] = v[indices]
            else:
                sub_info[k] = [v[i] for i in indices]
        return {
            'X': X[indices],
            'Y': Y[indices],
            'trial_info': sub_info,
        }

    return {
        'train': _subset(train_idx),
        'val': _subset(val_idx),
        'test': _subset(test_idx),
    }
