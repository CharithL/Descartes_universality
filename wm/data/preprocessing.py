"""
DESCARTES WM — Spike Extraction, Trial Alignment, and Binning

Converts raw NWB spike times into binned population activity tensors
suitable for LSTM training:
    X: (n_trials, n_timesteps, n_alm_neurons)  — ALM input during delay
    Y: (n_trials, n_timesteps, n_thal_neurons)  — Thalamus output during delay
"""

import logging
from pathlib import Path

import numpy as np
from numba import njit, prange

from wm.config import (
    BIN_SIZE_S,
    DELAY_START_S,
    N_DELAY_BINS,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
)

logger = logging.getLogger(__name__)


@njit(parallel=True, cache=True)
def _bin_spikes_numba(spike_times_list, spike_counts, output, t_starts, n_bins, bin_size):
    """Numba-accelerated spike binning.

    Parameters
    ----------
    spike_times_list : list of 1-D float64 arrays
        Spike times for each unit.
    spike_counts : 1-D int64 array
        Number of spikes per unit (used for bounds).
    output : 3-D float32 array, (n_trials, n_bins, n_units)
        Pre-allocated output to fill in-place.
    t_starts : 1-D float64 array, (n_trials,)
        Start time of the delay window for each trial.
    n_bins : int
    bin_size : float
    """
    n_trials = output.shape[0]
    n_units = output.shape[2]

    for t_idx in prange(n_trials):
        t_start = t_starts[t_idx]
        for j in range(n_units):
            spikes = spike_times_list[j]
            n_spk = len(spikes)
            for b in range(n_bins):
                bin_start = t_start + b * bin_size
                bin_end = bin_start + bin_size
                count = 0
                for s in range(n_spk):
                    if spikes[s] >= bin_start and spikes[s] < bin_end:
                        count += 1
                output[t_idx, b, j] = count


def extract_from_streamed(session_dir):
    """Extract binned spike tensors from stream-extracted data (NPZ+JSON).

    Parameters
    ----------
    session_dir : str or Path
        Directory containing spike_data.npz, unit_metadata.json, trial_metadata.json.

    Returns
    -------
    X, Y, trial_types, trial_outcomes, session_info : same as extract_session_data
    """
    import json as json_mod

    session_dir = Path(session_dir)

    # Load unit metadata
    with open(session_dir / 'unit_metadata.json') as f:
        unit_meta = json_mod.load(f)

    # Load trial metadata
    with open(session_dir / 'trial_metadata.json') as f:
        trial_meta = json_mod.load(f)

    # Load spike times
    spike_data = np.load(session_dir / 'spike_data.npz')

    classes = unit_meta['region_classes']
    n_units = unit_meta['n_units']

    # Separate ALM and thal units
    alm_indices = [i for i in range(n_units) if classes[i] == 'alm']
    thal_indices = [i for i in range(n_units) if classes[i] == 'thal']

    n_alm = len(alm_indices)
    n_thal = len(thal_indices)
    n_trials = trial_meta['n_trials']
    n_bins = N_DELAY_BINS

    logger.info("Streamed session %s: %d ALM, %d thal, %d trials, %d bins",
                session_dir.name, n_alm, n_thal, n_trials, n_bins)

    X = np.zeros((n_trials, n_bins, n_alm), dtype=np.float32)
    Y = np.zeros((n_trials, n_bins, n_thal), dtype=np.float32)
    trial_types = np.zeros(n_trials, dtype=int)
    trial_outcomes = []

    t_starts = np.array([
        t + DELAY_START_S for t in trial_meta['start_times']
    ], dtype=np.float64)

    for t_idx in range(n_trials):
        tt = trial_meta['trial_instructions'][t_idx]
        trial_types[t_idx] = 0 if tt.lower() == 'left' else 1
        trial_outcomes.append(trial_meta['outcomes'][t_idx])

    # Numba-accelerated spike binning
    if n_alm > 0:
        alm_spikes = [spike_data[f'unit_{i}'].astype(np.float64) for i in alm_indices]
        alm_counts = np.array([len(s) for s in alm_spikes], dtype=np.int64)
        _bin_spikes_numba(alm_spikes, alm_counts, X, t_starts, n_bins, BIN_SIZE_S)

    if n_thal > 0:
        thal_spikes = [spike_data[f'unit_{i}'].astype(np.float64) for i in thal_indices]
        thal_counts = np.array([len(s) for s in thal_spikes], dtype=np.int64)
        _bin_spikes_numba(thal_spikes, thal_counts, Y, t_starts, n_bins, BIN_SIZE_S)

    trial_outcomes = np.array(trial_outcomes)

    session_info = {
        'path': str(session_dir),
        'n_alm': n_alm,
        'n_thal': n_thal,
        'n_trials': n_trials,
        'n_bins': n_bins,
        'alm_regions': list(set(
            unit_meta['anno_names'][i] for i in alm_indices
        )),
        'thal_regions': list(set(
            unit_meta['anno_names'][i] for i in thal_indices
        )),
    }

    return X, Y, trial_types, trial_outcomes, session_info


def extract_session_data(nwb_path):
    """Extract binned spike counts for ALM (input) and thalamus (output).

    Parameters
    ----------
    nwb_path : str or Path

    Returns
    -------
    X : ndarray, (n_trials, n_timesteps, n_alm_neurons)
    Y : ndarray, (n_trials, n_timesteps, n_thal_neurons)
    trial_types : ndarray, (n_trials,) — integer trial type labels
    trial_outcomes : ndarray, (n_trials,) — string or int outcomes
    session_info : dict — metadata about the extraction
    """
    from wm.data.nwb_loader import load_session_units, load_session_trials

    units_data = load_session_units(nwb_path)
    trials_data, trial_meta = load_session_trials(nwb_path)

    # Separate ALM and thalamus units using region_class from nwb_loader
    alm_units = [u for u in units_data if u['region_class'] == 'alm']
    thal_units = [u for u in units_data if u['region_class'] == 'thal']

    n_alm = len(alm_units)
    n_thal = len(thal_units)
    n_trials = len(trials_data)
    n_bins = N_DELAY_BINS

    logger.info("Session %s: %d ALM, %d thal, %d trials, %d bins",
                Path(nwb_path).name, n_alm, n_thal, n_trials, n_bins)

    X = np.zeros((n_trials, n_bins, n_alm), dtype=np.float32)
    Y = np.zeros((n_trials, n_bins, n_thal), dtype=np.float32)
    trial_types = np.zeros(n_trials, dtype=int)
    trial_outcomes = []

    # Pre-compute trial start times and metadata
    t_starts = np.array([
        trial['start_time'] + DELAY_START_S for trial in trials_data
    ], dtype=np.float64)

    for t_idx, trial in enumerate(trials_data):
        tt = trial.get('trial_type', '0')
        if isinstance(tt, str):
            # Map left=0, right=1 for DANDI 000363
            trial_types[t_idx] = 0 if tt.lower() == 'left' else 1
        else:
            trial_types[t_idx] = int(tt)
        trial_outcomes.append(trial.get('outcome', 'unknown'))

    # Numba-accelerated spike binning
    if n_alm > 0:
        alm_spike_list = [u['spike_times'].astype(np.float64) for u in alm_units]
        alm_counts = np.array([len(s) for s in alm_spike_list], dtype=np.int64)
        _bin_spikes_numba(alm_spike_list, alm_counts, X, t_starts, n_bins, BIN_SIZE_S)

    if n_thal > 0:
        thal_spike_list = [u['spike_times'].astype(np.float64) for u in thal_units]
        thal_counts = np.array([len(s) for s in thal_spike_list], dtype=np.int64)
        _bin_spikes_numba(thal_spike_list, thal_counts, Y, t_starts, n_bins, BIN_SIZE_S)

    trial_outcomes = np.array(trial_outcomes)

    session_info = {
        'path': str(nwb_path),
        'n_alm': n_alm,
        'n_thal': n_thal,
        'n_trials': n_trials,
        'n_bins': n_bins,
        'alm_regions': list(set(u['region'] for u in alm_units)),
        'thal_regions': list(set(u['region'] for u in thal_units)),
    }

    return X, Y, trial_types, trial_outcomes, session_info


def filter_correct_trials(X, Y, trial_types, trial_outcomes):
    """Keep only correct trials.

    Parameters
    ----------
    X, Y, trial_types, trial_outcomes : arrays from extract_session_data

    Returns
    -------
    X_f, Y_f, trial_types_f : filtered arrays
    """
    # Try common outcome encodings
    if trial_outcomes.dtype.kind in ('U', 'S', 'O'):
        correct_mask = np.isin(trial_outcomes, ['correct', 'hit', 'Hit', '1'])
    else:
        correct_mask = trial_outcomes == 1

    n_correct = np.sum(correct_mask)
    logger.info("Filtered to %d correct trials out of %d total",
                n_correct, len(trial_outcomes))

    if n_correct == 0:
        logger.warning("No correct trials found! Using all trials instead.")
        return X, Y, trial_types

    return X[correct_mask], Y[correct_mask], trial_types[correct_mask]


def split_data(X, Y, trial_types, seed=42):
    """Split into train/val/test sets.

    Returns
    -------
    splits : dict with keys 'train', 'val', 'test'
        Each is a dict with 'X', 'Y', 'trial_types'.
    """
    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'train': {
            'X': X[train_idx], 'Y': Y[train_idx],
            'trial_types': trial_types[train_idx],
        },
        'val': {
            'X': X[val_idx], 'Y': Y[val_idx],
            'trial_types': trial_types[val_idx],
        },
        'test': {
            'X': X[test_idx], 'Y': Y[test_idx],
            'trial_types': trial_types[test_idx],
        },
    }


def save_processed_session(splits, session_info, output_dir):
    """Save processed session data to disk.

    Parameters
    ----------
    splits : dict from split_data
    session_info : dict from extract_session_data
    output_dir : str or Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data in splits.items():
        np.savez_compressed(
            output_dir / f'{split_name}.npz',
            X=data['X'],
            Y=data['Y'],
            trial_types=data['trial_types'],
        )

    # Save session metadata
    import json
    with open(output_dir / 'session_info.json', 'w') as f:
        json.dump(session_info, f, indent=2)

    logger.info("Saved processed session to %s", output_dir)


def load_processed_session(session_dir):
    """Load processed session data from disk.

    Returns
    -------
    splits : dict with 'train', 'val', 'test' splits
    session_info : dict
    """
    session_dir = Path(session_dir)
    splits = {}
    for split_name in ['train', 'val', 'test']:
        data = np.load(session_dir / f'{split_name}.npz')
        splits[split_name] = {
            'X': data['X'],
            'Y': data['Y'],
            'trial_types': data['trial_types'],
        }

    import json
    with open(session_dir / 'session_info.json', 'r') as f:
        session_info = json.load(f)

    return splits, session_info
