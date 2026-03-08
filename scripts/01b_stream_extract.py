"""
DESCARTES WM — Step 1b: Stream-Extract Spike Data from Large DANDI Files

The small NWB files in DANDI 000363 only contain ALM + Striatum.
Thalamic recordings are in the large (300-500 GB) files that include
raw imaging data. This script streams just the spike times and trial
metadata without downloading the full files.

Usage:
    python scripts/01b_stream_extract.py [--out-dir data/raw] [--max-sessions 2]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import h5py
import remfile
from dandi.dandiapi import DandiAPIClient

from wm.config import DANDISET_ID, MIN_ALM_NEURONS, MIN_THAL_NEURONS
from wm.data.nwb_loader import _classify_region

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Known thalamus-containing sessions (from our scan)
THALAMUS_SESSIONS = [
    'sub-440959/sub-440959_ses-20190223T173853_behavior+ecephys+image+ogen.nwb',
    'sub-440959/sub-440959_ses-20190219T121506_behavior+ecephys+image+ogen.nwb',
]


def stream_extract_session(asset_url, asset_path, out_dir):
    """Stream-extract spike times and trial data from a large NWB file.

    Only reads the units table (spike_times, anno_name) and trials table
    from the HDF5 file via HTTP range requests. Does NOT download imaging.

    Saves a compact local NWB-like NPZ file.
    """
    logger.info("Streaming: %s", asset_path)

    rfile = remfile.File(asset_url)
    with h5py.File(rfile, 'r') as f:
        # --- Units ---
        units = f['units']
        anno_names = units['anno_name'][:]
        anno_str = [v.decode() if isinstance(v, bytes) else str(v) for v in anno_names]

        # Classify regions
        n_units = len(anno_str)
        classes = [_classify_region(a) for a in anno_str]
        n_alm = sum(1 for c in classes if c == 'alm')
        n_thal = sum(1 for c in classes if c == 'thal')

        logger.info("  Units: %d total, %d ALM, %d thal", n_units, n_alm, n_thal)

        if n_alm < MIN_ALM_NEURONS or n_thal < MIN_THAL_NEURONS:
            logger.warning("  Insufficient coverage. Skipping.")
            return None

        # Read spike times (indexed format in HDF5)
        spike_times_data = units['spike_times'][:]
        spike_times_index = units['spike_times_index'][:]

        # Reconstruct per-unit spike times
        all_spike_times = []
        prev_idx = 0
        for i in range(n_units):
            end_idx = spike_times_index[i]
            unit_spikes = spike_times_data[prev_idx:end_idx]
            all_spike_times.append(unit_spikes.astype(np.float64))
            prev_idx = end_idx

        # --- Trials ---
        trials = f['intervals']['trials']
        trial_starts = trials['start_time'][:]
        trial_stops = trials['stop_time'][:]
        n_trials = len(trial_starts)

        # Trial instruction (left/right)
        trial_instructions = []
        if 'trial_instruction' in trials:
            raw = trials['trial_instruction'][:]
            trial_instructions = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in raw
            ]
        else:
            trial_instructions = ['unknown'] * n_trials

        # Outcome (hit/miss/ignore)
        outcomes = []
        if 'outcome' in trials:
            raw = trials['outcome'][:]
            outcomes = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in raw
            ]
        else:
            outcomes = ['unknown'] * n_trials

        # Photostim metadata (for photoinhibition validation)
        photostim_power = ['N/A'] * n_trials
        photostim_onset = ['N/A'] * n_trials
        photostim_duration = ['N/A'] * n_trials
        for col_name, target_list in [
            ('photostim_power', photostim_power),
            ('photostim_onset', photostim_onset),
            ('photostim_duration', photostim_duration),
        ]:
            if col_name in trials:
                raw = trials[col_name][:]
                for j, v in enumerate(raw):
                    target_list[j] = v.decode() if isinstance(v, bytes) else str(v)

        n_photostim = sum(1 for p in photostim_power if p != 'N/A')
        logger.info("  Trials: %d (%d photostim, %d control)",
                     n_trials, n_photostim, n_trials - n_photostim)

    # Save compact extracted data
    session_name = Path(asset_path).stem
    session_dir = Path(out_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save spike times per unit
    spike_file = session_dir / 'spike_data.npz'
    spike_dict = {}
    for i, spikes in enumerate(all_spike_times):
        spike_dict[f'unit_{i}'] = spikes
    np.savez_compressed(spike_file, **spike_dict)

    # Save unit metadata
    unit_meta = {
        'n_units': n_units,
        'n_alm': n_alm,
        'n_thal': n_thal,
        'anno_names': anno_str,
        'region_classes': classes,
    }
    with open(session_dir / 'unit_metadata.json', 'w') as fp:
        json.dump(unit_meta, fp, indent=2)

    # Save trial metadata
    trial_meta = {
        'n_trials': n_trials,
        'start_times': trial_starts.tolist(),
        'stop_times': trial_stops.tolist(),
        'trial_instructions': trial_instructions,
        'outcomes': outcomes,
        'photostim_power': photostim_power,
        'photostim_onset': photostim_onset,
        'photostim_duration': photostim_duration,
    }
    with open(session_dir / 'trial_metadata.json', 'w') as fp:
        json.dump(trial_meta, fp, indent=2)

    logger.info("  Saved extracted data to %s", session_dir)
    return session_dir


def main():
    parser = argparse.ArgumentParser(
        description="Stream-extract spike data from large DANDI files"
    )
    parser.add_argument('--out-dir', type=str, default='data/raw')
    parser.add_argument('--max-sessions', type=int, default=2)
    parser.add_argument('--scan-all', action='store_true',
                        help='Scan all large files instead of using known list')
    args = parser.parse_args()

    client = DandiAPIClient()
    ds = client.get_dandiset(DANDISET_ID)

    extracted = 0
    for asset in ds.get_assets():
        if extracted >= args.max_sessions:
            break

        # Only process known thalamus sessions (or scan all if requested)
        if not args.scan_all and asset.path not in THALAMUS_SESSIONS:
            continue

        if asset.size < 1e9:
            continue  # Skip small files (no thalamus)

        url = asset.get_content_url(follow_redirects=1, strip_query=True)

        session_name = Path(asset.path).stem
        session_dir = Path(args.out_dir) / session_name
        if (session_dir / 'spike_data.npz').exists():
            logger.info("Skipping %s (already extracted)", session_name)
            extracted += 1
            continue

        result = stream_extract_session(url, asset.path, args.out_dir)
        if result is not None:
            extracted += 1

    logger.info("Extracted %d sessions", extracted)


if __name__ == '__main__':
    main()
