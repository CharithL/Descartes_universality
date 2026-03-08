"""
NWB Explorer -- Inspect NWB files and generate nwb_schema.json

This is the HARD PREREQUISITE for the entire Human Universality pipeline.
All downstream code reads NWB column names from the schema JSON this module
produces. Without it, nothing else can run.

Usage (programmatic):
    from human_wm.data.nwb_explorer import explore_nwb, generate_schema
    info = explore_nwb('sub-01_ses-01.nwb')
    schema = generate_schema('sub-01_ses-01.nwb')

Usage (CLI):
    python scripts/10_human_explore_nwb.py --nwb-path data/raw/sub-01.nwb
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from human_wm.config import (
    FRONTAL_REGION_PATTERNS,
    MTL_REGION_PATTERNS,
    NWB_SCHEMA_PATH,
)

# ---------------------------------------------------------------------------
# Column name candidates -- tried in order; first match wins
# ---------------------------------------------------------------------------

_REGION_COLUMN_CANDIDATES: list[str] = [
    'brain_region',
    'location',
    'brain_area',
    'region',
    'area',
    'anno_name',
    'electrode_group',
]

_TIMING_COLUMN_CANDIDATES: list[str] = [
    'start_time',
    'stop_time',
    'stimulus_on_time',
    'stimulus_off_time',
    'delay_start',
    'delay_end',
    'probe_on_time',
    'probe_off_time',
    'response_time',
]


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


def _classify_region(region_name: str) -> str:
    """Classify a brain region as 'mtl', 'frontal', or 'other'.

    Uses case-insensitive substring matching against the pattern lists
    defined in human_wm.config.

    Parameters
    ----------
    region_name : str
        Name of the brain region (e.g. 'hippocampus', 'dACC').

    Returns
    -------
    str
        One of 'mtl', 'frontal', or 'other'.
    """
    name_lower = region_name.lower()

    for pattern in MTL_REGION_PATTERNS:
        if pattern.lower() in name_lower:
            return 'mtl'

    for pattern in FRONTAL_REGION_PATTERNS:
        if pattern.lower() in name_lower:
            return 'frontal'

    return 'other'


def _detect_region_column(colnames: list[str]) -> str | None:
    """Detect the region column from a list of column names.

    Tries each candidate in _REGION_COLUMN_CANDIDATES in priority order.

    Parameters
    ----------
    colnames : list[str]
        Column names from the units table.

    Returns
    -------
    str or None
        The detected region column name, or None if not found.
    """
    colnames_set = set(colnames)
    for candidate in _REGION_COLUMN_CANDIDATES:
        if candidate in colnames_set:
            return candidate
    return None


def _coerce_region_to_str(value) -> str:
    """Convert a region value to a plain string.

    NWB files may store region info as plain strings, or as PyNWB objects
    like ``ElectrodeGroup`` (which have ``.location`` and ``.name``
    attributes).  This helper normalises any variant to a simple string
    suitable for use as dict keys and in JSON output.
    """
    if isinstance(value, str):
        return value

    # PyNWB ElectrodeGroup — prefer .location (brain region) over .name
    location = getattr(value, 'location', None)
    if location and isinstance(location, str) and location.strip():
        return location.strip()

    name = getattr(value, 'name', None)
    if name and isinstance(name, str) and name.strip():
        return name.strip()

    return str(value)


def _get_unit_region_via_electrodes(nwbfile, unit_idx: int) -> str | None:
    """Get brain region for a unit by following the electrodes reference.

    NWB standard path:  units[i].electrodes -> electrodes table -> location
    This is the typical storage pattern for Rutishauser DANDI 000576.

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
    unit_idx : int
        Index of the unit in the units table.

    Returns
    -------
    str or None
        Brain region string, or None if lookup fails.
    """
    try:
        units = nwbfile.units
        # units['electrodes'] gives a DynamicTableRegion referencing the
        # electrodes table. For unit i it returns a pandas DataFrame (not
        # a simple index/array). The DataFrame's .index contains the
        # electrode row IDs.
        electrode_ref = units['electrodes'][unit_idx]
        # Handle pandas DataFrame (most common in modern pynwb)
        if hasattr(electrode_ref, 'index') and hasattr(electrode_ref.index, '__len__'):
            e_idx = int(electrode_ref.index[0])
        elif hasattr(electrode_ref, '__len__') and len(electrode_ref) > 0:
            e_idx = int(electrode_ref[0])
        else:
            e_idx = int(electrode_ref)

        electrodes = nwbfile.electrodes
        if electrodes is None:
            return None

        # Try 'location' column first (standard), then 'group_name'
        for col in ('location', 'group_name', 'brain_region', 'brain_area'):
            if col in electrodes.colnames:
                val = electrodes[col][e_idx]
                return _coerce_region_to_str(val)

        # Fall back to the electrode's group object
        if 'group' in electrodes.colnames:
            grp = electrodes['group'][e_idx]
            return _coerce_region_to_str(grp)
    except Exception:
        return None
    return None


def _extract_regions(nwbfile, region_column: str | None) -> dict[str, int]:
    """Extract brain region counts from the units table.

    Tries three strategies in order:
    1. Direct column on units table (brain_region, location, etc.)
    2. If that gives only generic labels (e.g. 'Intracranial'), try
       the indirect path: units → electrodes table → location
    3. Fall back to whatever the direct column gave

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The NWB file object.
    region_column : str or None
        Name of the column containing brain region labels (may be None).

    Returns
    -------
    dict[str, int]
        Mapping of region name (string) to unit count.
    """
    units = nwbfile.units
    n_units = len(units)

    # --- Strategy 1: direct column ---
    direct_regions = []
    if region_column is not None:
        for i in range(n_units):
            raw = units[region_column][i]
            direct_regions.append(_coerce_region_to_str(raw))

    # Check if direct regions are too generic (all the same, or known
    # generic labels like 'Intracranial', 'unknown')
    _GENERIC_LABELS = {'intracranial', 'unknown', 'brain', 'ieeg', 'ecog',
                       'seeg', 'n/a', ''}
    unique_direct = set(r.lower() for r in direct_regions) if direct_regions else set()
    direct_is_generic = (
        len(unique_direct) <= 1
        or unique_direct.issubset(_GENERIC_LABELS)
    )

    # --- Strategy 2: indirect via electrodes table ---
    if direct_is_generic and 'electrodes' in (list(units.colnames) if units.colnames else []):
        indirect_regions = []
        for i in range(n_units):
            region = _get_unit_region_via_electrodes(nwbfile, i)
            indirect_regions.append(region or 'unknown')
        # Only use indirect if it's richer
        unique_indirect = set(r.lower() for r in indirect_regions)
        if len(unique_indirect) > len(unique_direct):
            return dict(Counter(indirect_regions))

    # Fall back to direct
    if direct_regions:
        return dict(Counter(direct_regions))

    return {}


def _detect_timing_columns(trial_colnames: list[str]) -> list[str]:
    """Detect which timing columns are present in the trials table.

    Parameters
    ----------
    trial_colnames : list[str]
        Column names from the trials table.

    Returns
    -------
    list[str]
        Timing column names that were found.
    """
    colnames_set = set(trial_colnames)
    return [c for c in _TIMING_COLUMN_CANDIDATES if c in colnames_set]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explore_nwb(nwb_path: str | Path) -> dict[str, Any]:
    """Explore an NWB file and return a summary dictionary.

    Parameters
    ----------
    nwb_path : str or Path
        Path to the .nwb file.

    Returns
    -------
    dict
        Summary with keys:
        - units_columns : list[str]
        - trial_columns : list[str]
        - brain_regions : dict[str, int]  (region name -> count)
        - region_column_detected : str or None
        - electrode_groups : list[str]
        - n_units : int
        - n_trials : int
    """
    nwbfile = _open_nwb(nwb_path)

    # --- Units ---
    units_columns = list(nwbfile.units.colnames) if nwbfile.units is not None else []
    n_units = len(nwbfile.units) if nwbfile.units is not None else 0

    # --- Electrodes table ---
    electrodes_columns = []
    if nwbfile.electrodes is not None:
        electrodes_columns = list(nwbfile.electrodes.colnames)

    # --- Region detection ---
    # Try direct column on units table first
    region_column = _detect_region_column(units_columns)
    region_source = 'direct'  # track whether we used direct or indirect

    if n_units > 0:
        brain_regions = _extract_regions(nwbfile, region_column)
        # Determine if regions came from indirect electrode table lookup
        if region_column is None:
            # No direct region column on units table at all — if we got any
            # regions, they came from the electrodes table
            if brain_regions:
                region_source = 'electrodes_table'
        else:
            # Direct column exists — check if it gave only generic labels
            _GENERIC = {'intracranial', 'unknown', 'brain', 'ieeg', 'ecog',
                        'seeg', 'n/a', ''}
            direct_unique = set()
            for i in range(n_units):
                raw = nwbfile.units[region_column][i]
                direct_unique.add(_coerce_region_to_str(raw).lower())
            if direct_unique.issubset(_GENERIC) and len(brain_regions) > 1:
                region_source = 'electrodes_table'
    else:
        brain_regions = {}

    # --- Trials ---
    trial_columns = list(nwbfile.trials.colnames) if nwbfile.trials is not None else []
    n_trials = len(nwbfile.trials) if nwbfile.trials is not None else 0

    # --- Electrode groups ---
    electrode_groups = list(nwbfile.electrode_groups.keys()) if nwbfile.electrode_groups else []

    return {
        'units_columns': units_columns,
        'trial_columns': trial_columns,
        'brain_regions': brain_regions,
        'region_column_detected': region_column,
        'region_source': region_source,
        'electrodes_columns': electrodes_columns,
        'electrode_groups': electrode_groups,
        'n_units': n_units,
        'n_trials': n_trials,
    }


def generate_schema(
    nwb_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate nwb_schema.json from an NWB file.

    This is the HARD PREREQUISITE for the entire pipeline. The output JSON
    contains every column name, region list, and count that downstream
    modules need to process NWB data consistently.

    Parameters
    ----------
    nwb_path : str or Path
        Path to the .nwb file.
    output_path : str or Path or None
        Where to save the JSON schema. Defaults to NWB_SCHEMA_PATH from config.

    Returns
    -------
    dict
        The schema dictionary with keys:
        - units_columns : list[str]
        - region_column : str or None
        - trial_columns : list[str]
        - mtl_regions : list[str]
        - frontal_regions : list[str]
        - all_regions : list[str]
        - region_counts : dict[str, int]
        - timing_columns : list[str]
        - n_units : int
        - n_trials : int
        - sample_values : dict
    """
    if output_path is None:
        output_path = NWB_SCHEMA_PATH
    output_path = Path(output_path)

    # Run exploration
    info = explore_nwb(nwb_path)

    # Classify regions
    all_regions = sorted(info['brain_regions'].keys())
    mtl_regions = [r for r in all_regions if _classify_region(r) == 'mtl']
    frontal_regions = [r for r in all_regions if _classify_region(r) == 'frontal']

    # Detect timing columns
    timing_columns = _detect_timing_columns(info['trial_columns'])

    # Build sample values (empty for now; populated when real data is read)
    sample_values: dict[str, Any] = {}

    schema = {
        'units_columns': info['units_columns'],
        'region_column': info['region_column_detected'],
        'region_source': info.get('region_source', 'direct'),
        'electrodes_columns': info.get('electrodes_columns', []),
        'trial_columns': info['trial_columns'],
        'mtl_regions': mtl_regions,
        'frontal_regions': frontal_regions,
        'all_regions': all_regions,
        'region_counts': info['brain_regions'],
        'timing_columns': timing_columns,
        'n_units': info['n_units'],
        'n_trials': info['n_trials'],
        'sample_values': sample_values,
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2, sort_keys=False)

    return schema
