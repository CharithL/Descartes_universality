#!/usr/bin/env python
"""
10 -- Explore NWB and Generate Schema (HARD PREREQUISITE)

Inspects an NWB file from the Rutishauser Human Single-Neuron WM dataset
(DANDI 000576) and produces ``nwb_schema.json``.  This schema is the HARD
PREREQUISITE for every downstream step in the Human Universality pipeline.

Usage
-----
    # Auto-detect first .nwb in data/raw/
    python scripts/10_human_explore_nwb.py

    # Explicit paths
    python scripts/10_human_explore_nwb.py \
        --nwb-path data/raw/sub-01_ses-01.nwb \
        --output data/nwb_schema.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path so human_wm is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from human_wm.config import NWB_SCHEMA_PATH, RAW_NWB_DIR
from human_wm.data.nwb_explorer import explore_nwb, generate_schema


def _find_first_nwb(directory: Path) -> Path | None:
    """Return the first .nwb file found in *directory*, or None."""
    nwb_files = sorted(directory.glob('*.nwb'))
    if nwb_files:
        return nwb_files[0]
    # dandi download creates nested dirs: data/raw/000576/sub-XXX/file.nwb
    # Use recursive glob to find NWB files at any depth
    nwb_files = sorted(directory.rglob('*.nwb'))
    return nwb_files[0] if nwb_files else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Explore NWB file and generate nwb_schema.json',
    )
    parser.add_argument(
        '--nwb-path',
        type=str,
        default=None,
        help='Path to the .nwb file. If omitted, finds the first .nwb in RAW_NWB_DIR.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output path for nwb_schema.json (default: {NWB_SCHEMA_PATH})',
    )
    args = parser.parse_args(argv)

    # Resolve NWB path
    if args.nwb_path is not None:
        nwb_path = Path(args.nwb_path)
    else:
        print(f'No --nwb-path given. Searching {RAW_NWB_DIR} ...')
        nwb_path = _find_first_nwb(RAW_NWB_DIR)
        if nwb_path is None:
            print(
                f'ERROR: No .nwb files found in {RAW_NWB_DIR}.\n'
                'Download data first (see scripts/01_download_data.py) or '
                'specify --nwb-path explicitly.',
                file=sys.stderr,
            )
            sys.exit(1)
        print(f'Found: {nwb_path}')

    if not nwb_path.exists():
        print(f'ERROR: NWB file not found: {nwb_path}', file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    output_path = Path(args.output) if args.output else NWB_SCHEMA_PATH

    # --- Explore ---
    print(f'\n=== Exploring NWB: {nwb_path.name} ===')
    info = explore_nwb(nwb_path)

    print(f'  Units:            {info["n_units"]}')
    print(f'  Trials:           {info["n_trials"]}')
    print(f'  Region column:    {info["region_column_detected"]}')
    print(f'  Units columns:    {info["units_columns"]}')
    print(f'  Trial columns:    {info["trial_columns"]}')
    print(f'  Electrode groups: {info["electrode_groups"]}')

    if info['brain_regions']:
        print('\n  Brain region counts:')
        for region, count in sorted(info['brain_regions'].items()):
            print(f'    {region:30s} {count:4d}')

    # --- Generate schema ---
    print(f'\n=== Generating schema: {output_path} ===')
    schema = generate_schema(nwb_path, output_path=output_path)

    print(f'  MTL regions:     {schema["mtl_regions"]}')
    print(f'  Frontal regions: {schema["frontal_regions"]}')
    print(f'  Timing columns:  {schema["timing_columns"]}')
    print(f'\nSchema saved to {output_path}')
    print('Done. This schema is the HARD PREREQUISITE for all downstream tasks.')


if __name__ == '__main__':
    main()
