"""
Patient Inventory -- Discover usable NWB sessions for the pipeline

Scans all NWB files, extracts neuron counts and trial counts via the
NWB loader, and determines which sessions meet the minimum thresholds
for inclusion in the DESCARTES Human Universality pipeline.

Usage (programmatic):
    from human_wm.data.patient_inventory import (
        build_inventory, get_best_patient, get_usable_patients,
    )
    from human_wm.config import load_nwb_schema, RAW_NWB_DIR

    schema = load_nwb_schema()
    nwb_paths = sorted(RAW_NWB_DIR.glob('*.nwb'))
    inventory = build_inventory(nwb_paths, schema)
    best = get_best_patient(inventory)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from human_wm.config import MIN_FRONTAL_NEURONS, MIN_MTL_NEURONS, MIN_TRIALS
from human_wm.data.nwb_loader import extract_patient_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_inventory(
    nwb_paths: list[str | Path],
    schema: dict,
) -> list[dict[str, Any]]:
    """Build an inventory of all NWB sessions with usability flags.

    For each NWB file, extracts patient data and records neuron counts,
    trial count, and whether the session meets minimum thresholds.

    Parameters
    ----------
    nwb_paths : list of str or Path
        Paths to NWB files.
    schema : dict
        NWB schema dictionary (from ``generate_schema`` / ``load_nwb_schema``).

    Returns
    -------
    list[dict]
        List of inventory entries, each with keys:
        - patient_id : str  (stem of the NWB filename)
        - path : str  (absolute path to the NWB file)
        - n_mtl : int
        - n_frontal : int
        - n_trials : int
        - usable : bool
    """
    inventory = []

    for nwb_path in nwb_paths:
        nwb_path = Path(nwb_path)
        patient_id = nwb_path.stem

        try:
            X, Y, trial_info = extract_patient_data(nwb_path, schema)
        except Exception as e:
            print(f'  WARNING: Failed to load {nwb_path.name}: {e}')
            inventory.append({
                'patient_id': patient_id,
                'path': str(nwb_path),
                'n_mtl': 0,
                'n_frontal': 0,
                'n_trials': 0,
                'usable': False,
            })
            continue

        n_mtl = X.shape[2]
        n_frontal = Y.shape[2]
        n_trials = X.shape[0]

        usable = (
            n_mtl >= MIN_MTL_NEURONS
            and n_frontal >= MIN_FRONTAL_NEURONS
            and n_trials >= MIN_TRIALS
        )

        inventory.append({
            'patient_id': patient_id,
            'path': str(nwb_path),
            'n_mtl': n_mtl,
            'n_frontal': n_frontal,
            'n_trials': n_trials,
            'usable': usable,
        })

    return inventory


def get_best_patient(inventory: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the usable patient with the most neurons.

    "Best" is defined as the patient with the highest ``min(n_mtl, n_frontal)``,
    among usable sessions only.

    Parameters
    ----------
    inventory : list[dict]
        Output of ``build_inventory``.

    Returns
    -------
    dict or None
        The best patient entry, or None if no usable patients exist.
    """
    usable = get_usable_patients(inventory)
    if not usable:
        return None

    return max(usable, key=lambda p: min(p['n_mtl'], p['n_frontal']))


def get_usable_patients(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only usable patients from the inventory.

    Parameters
    ----------
    inventory : list[dict]
        Output of ``build_inventory``.

    Returns
    -------
    list[dict]
        Filtered list of patient entries where ``usable`` is True.
    """
    return [p for p in inventory if p['usable']]
