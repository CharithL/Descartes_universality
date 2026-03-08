#!/usr/bin/env python
"""
12 -- Build Patient Inventory

Scans all NWB files in RAW_NWB_DIR, extracts neuron/trial counts, and
produces a summary table showing which sessions are usable for the
DESCARTES Human Universality pipeline.

Optionally saves the inventory as JSON for downstream consumption.

Usage
-----
    # Print inventory table
    python scripts/12_human_inventory.py

    # Save to JSON
    python scripts/12_human_inventory.py --output data/patient_inventory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path so human_wm is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from human_wm.config import (
    MIN_FRONTAL_NEURONS,
    MIN_MTL_NEURONS,
    MIN_TRIALS,
    RAW_NWB_DIR,
    load_nwb_schema,
)
from human_wm.data.patient_inventory import (
    build_inventory,
    get_best_patient,
    get_usable_patients,
)


def _print_table(inventory: list[dict]) -> None:
    """Print a formatted summary table of the inventory."""
    header = f'{"Patient ID":<40s} {"MTL":>5s} {"Front":>5s} {"Trials":>6s} {"Usable":>6s}'
    sep = '-' * len(header)
    print(header)
    print(sep)
    for entry in inventory:
        usable_str = 'YES' if entry['usable'] else 'no'
        print(
            f'{entry["patient_id"]:<40s} '
            f'{entry["n_mtl"]:>5d} '
            f'{entry["n_frontal"]:>5d} '
            f'{entry["n_trials"]:>6d} '
            f'{usable_str:>6s}'
        )
    print(sep)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Build patient inventory from NWB files',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save inventory JSON (optional)',
    )
    parser.add_argument(
        '--nwb-dir',
        type=str,
        default=None,
        help=f'Directory containing NWB files (default: {RAW_NWB_DIR})',
    )
    args = parser.parse_args(argv)

    # Resolve NWB directory
    nwb_dir = Path(args.nwb_dir) if args.nwb_dir else RAW_NWB_DIR

    # Load schema
    schema = load_nwb_schema()
    if schema is None:
        print(
            'ERROR: nwb_schema.json not found. '
            'Run scripts/10_human_explore_nwb.py first.',
            file=sys.stderr,
        )
        sys.exit(1)

    # Find NWB files
    nwb_paths = sorted(nwb_dir.glob('*.nwb'))
    if not nwb_paths:
        # Also check sub-directories one level deep
        nwb_paths = sorted(nwb_dir.glob('*/*.nwb'))
    if not nwb_paths:
        print(f'ERROR: No .nwb files found in {nwb_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'\n=== Patient Inventory: {len(nwb_paths)} NWB files ===')
    print(f'  Thresholds: MTL >= {MIN_MTL_NEURONS}, '
          f'Frontal >= {MIN_FRONTAL_NEURONS}, '
          f'Trials >= {MIN_TRIALS}\n')

    # Build inventory
    inventory = build_inventory(nwb_paths, schema)

    # Print table
    _print_table(inventory)

    # Summary
    usable = get_usable_patients(inventory)
    best = get_best_patient(inventory)

    print(f'\n  Total sessions:  {len(inventory)}')
    print(f'  Usable sessions: {len(usable)}')

    if best is not None:
        print(f'  Best patient:    {best["patient_id"]} '
              f'(MTL={best["n_mtl"]}, Frontal={best["n_frontal"]}, '
              f'Trials={best["n_trials"]})')
    else:
        print('  Best patient:    None (no usable sessions)')

    # Save JSON if requested
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(inventory, f, indent=2)
        print(f'\n  Inventory saved to {output_path}')

    print()


if __name__ == '__main__':
    main()
