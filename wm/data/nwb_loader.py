"""
DESCARTES WM — NWB File Parsing and Session Selection

Handles the Chen/Svoboda NWB format (DANDI 000363):
- Region via 'anno_name' column (CCF annotations)
- Trial type via 'trial_instruction' (left/right)
- Outcome via 'outcome' (hit/miss/ignore)
"""

import logging
from pathlib import Path

import numpy as np
import pynwb

from wm.config import (
    ALM_REGION_MARKERS,
    MIN_ALM_NEURONS,
    MIN_THAL_NEURONS,
    THAL_REGION_MARKERS,
)

logger = logging.getLogger(__name__)


def _classify_region(anno_name):
    """Classify CCF anno_name as 'alm', 'thal', or 'other'."""
    name_lower = anno_name.lower()
    for marker in ALM_REGION_MARKERS:
        if marker.lower() in name_lower:
            return 'alm'
    for marker in THAL_REGION_MARKERS:
        if marker.lower() in name_lower:
            return 'thal'
    return 'other'


def inspect_session(nwb_path):
    """Inspect an NWB file and report its structure."""
    nwb_path = Path(nwb_path)
    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        units = nwb.units
        n_units = len(units)

        region_counts = {}
        n_alm = 0
        n_thal = 0

        if 'anno_name' in units.colnames:
            for i in range(n_units):
                anno = str(units['anno_name'][i])
                region_counts[anno] = region_counts.get(anno, 0) + 1
                cls = _classify_region(anno)
                if cls == 'alm':
                    n_alm += 1
                elif cls == 'thal':
                    n_thal += 1
        else:
            logger.warning("No 'anno_name' column in %s", nwb_path.name)

        trials = nwb.trials
        n_trials = len(trials) if trials is not None else 0
        trial_cols = list(trials.colnames) if trials is not None else []

    qualifies = n_alm >= MIN_ALM_NEURONS and n_thal >= MIN_THAL_NEURONS

    return {
        'path': str(nwb_path),
        'n_units': n_units,
        'n_alm': n_alm,
        'n_thal': n_thal,
        'n_trials': n_trials,
        'region_counts': region_counts,
        'trial_columns': trial_cols,
        'unit_columns': list(units.colnames),
        'qualifies': qualifies,
    }


def find_qualifying_sessions(nwb_paths):
    """Find sessions with sufficient ALM + thalamus coverage."""
    qualifying = []

    for path in nwb_paths:
        try:
            info = inspect_session(path)
            if info['qualifies']:
                qualifying.append((path, info))
                logger.info(
                    "QUALIFIED: %s  (ALM=%d, Thal=%d, trials=%d)",
                    Path(path).name, info['n_alm'], info['n_thal'],
                    info['n_trials'],
                )
            else:
                logger.info(
                    "Skipped: %s  (ALM=%d, Thal=%d)",
                    Path(path).name, info['n_alm'], info['n_thal'],
                )
        except Exception as e:
            logger.warning("Error inspecting %s: %s", path, e)

    logger.info("Found %d qualifying sessions out of %d total",
                len(qualifying), len(nwb_paths))
    return qualifying


def load_session_units(nwb_path):
    """Load spike times and region assignments for all units.

    Uses 'anno_name' (CCF annotations) for region classification.

    Returns
    -------
    units_data : list of dict
        Each with 'spike_times', 'region', 'region_class', 'unit_idx'.
    """
    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        units = nwb.units
        n_units = len(units)
        has_anno = 'anno_name' in units.colnames

        units_data = []
        for i in range(n_units):
            spike_times = np.array(units['spike_times'][i])
            region = str(units['anno_name'][i]) if has_anno else 'unknown'
            region_class = _classify_region(region)

            units_data.append({
                'spike_times': spike_times,
                'region': region,
                'region_class': region_class,
                'unit_idx': i,
            })

    return units_data


def load_session_trials(nwb_path):
    """Load trial information.

    For DANDI 000363:
    - trial_instruction: 'left' or 'right'
    - outcome: 'hit', 'miss', or 'ignore'
    """
    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        trials = nwb.trials
        n_trials = len(trials)
        colnames = list(trials.colnames)

        # Trial type column
        type_col = None
        for col in ['trial_instruction', 'trial_type', 'response_direction']:
            if col in colnames:
                type_col = col
                break

        # Outcome column
        outcome_col = None
        for col in ['outcome', 'result', 'correct']:
            if col in colnames:
                outcome_col = col
                break

        trials_data = []
        for i in range(n_trials):
            entry = {
                'start_time': float(trials['start_time'][i]),
                'stop_time': float(trials['stop_time'][i]),
            }
            if type_col:
                entry['trial_type'] = str(trials[type_col][i])
            if outcome_col:
                entry['outcome'] = str(trials[outcome_col][i])
            trials_data.append(entry)

    metadata = {
        'type_col': type_col,
        'outcome_col': outcome_col,
        'n_trials': n_trials,
        'all_columns': colnames,
    }

    return trials_data, metadata
