#!/usr/bin/env python
"""
10 -- Explore NWB and Generate Schema (HARD PREREQUISITE)

Inspects NWB files from the Rutishauser Human Single-Neuron WM dataset
(DANDI 000576) and produces ``nwb_schema.json``.  This schema is the HARD
PREREQUISITE for every downstream step in the Human Universality pipeline.

When no --nwb-path is given, scans up to 10 NWB files and picks the one
with the most brain regions and units -- important because some files
(e.g. sub-01) have only 4 generic units.

Usage
-----
    # Auto-detect best .nwb in data/raw/
    python scripts/10_human_explore_nwb.py

    # Explicit path
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


def _find_all_nwb(directory: Path) -> list[Path]:
    """Return all .nwb files found recursively in *directory*."""
    nwb_files = sorted(directory.glob('*.nwb'))
    if not nwb_files:
        nwb_files = sorted(directory.rglob('*.nwb'))
    return nwb_files


def _find_best_nwb(directory: Path, max_scan: int = 10) -> Path | None:
    """Scan up to *max_scan* NWB files and pick the one with the richest
    brain region info and most units.

    Returns the best file path, or None if no NWB files exist.
    """
    all_nwb = _find_all_nwb(directory)
    if not all_nwb:
        return None

    best_path = None
    best_score = -1

    scan_count = min(len(all_nwb), max_scan)
    print(f'Scanning {scan_count} of {len(all_nwb)} NWB files to find best candidate...')

    for i, nwb_path in enumerate(all_nwb[:max_scan]):
        try:
            info = explore_nwb(nwb_path)
            n_regions = len(info['brain_regions'])
            n_units = info['n_units']
            n_trials = info['n_trials']
            # Score: prioritise region diversity, then unit count, then trials
            score = n_regions * 1000 + n_units * 10 + n_trials
            region_list = list(info['brain_regions'].keys())
            print(f'  [{i+1}/{scan_count}] {nwb_path.name}: '
                  f'{n_units} units, {n_trials} trials, '
                  f'{n_regions} regions {region_list[:5]}  '
                  f'(score={score})')
            if score > best_score:
                best_score = score
                best_path = nwb_path
        except Exception as e:
            print(f'  [{i+1}/{scan_count}] {nwb_path.name}: FAILED ({e})')

    if best_path:
        print(f'\nBest candidate: {best_path.name} (score={best_score})')
    return best_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description='Explore NWB file and generate nwb_schema.json',
    )
    parser.add_argument(
        '--nwb-path',
        type=str,
        default=None,
        help='Path to the .nwb file. If omitted, scans RAW_NWB_DIR for best file.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output path for nwb_schema.json (default: {NWB_SCHEMA_PATH})',
    )
    parser.add_argument(
        '--max-scan',
        type=int,
        default=10,
        help='Max NWB files to scan when auto-detecting (default: 10)',
    )
    args = parser.parse_args(argv)

    # Resolve NWB path
    if args.nwb_path is not None:
        nwb_path = Path(args.nwb_path)
    else:
        print(f'No --nwb-path given. Searching {RAW_NWB_DIR} ...')
        nwb_path = _find_best_nwb(RAW_NWB_DIR, max_scan=args.max_scan)
        if nwb_path is None:
            print(
                f'ERROR: No .nwb files found in {RAW_NWB_DIR}.\n'
                'Download data first (see scripts/11_human_download.py) or '
                'specify --nwb-path explicitly.',
                file=sys.stderr,
            )
            sys.exit(1)

    if not nwb_path.exists():
        print(f'ERROR: NWB file not found: {nwb_path}', file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    output_path = Path(args.output) if args.output else NWB_SCHEMA_PATH

    # --- Explore ---
    print(f'\n=== Exploring NWB: {nwb_path.name} ===')
    info = explore_nwb(nwb_path)

    print(f'  Units:              {info["n_units"]}')
    print(f'  Trials:             {info["n_trials"]}')
    print(f'  Region column:      {info["region_column_detected"]}')
    print(f'  Region source:      {info.get("region_source", "direct")}')
    print(f'  Units columns:      {info["units_columns"]}')
    print(f'  Electrodes columns: {info.get("electrodes_columns", [])}')
    print(f'  Trial columns:      {info["trial_columns"]}')
    print(f'  Electrode groups:   {info["electrode_groups"]}')

    if info['brain_regions']:
        print('\n  Brain region counts:')
        for region, count in sorted(info['brain_regions'].items()):
            print(f'    {str(region):30s} {count:4d}')

    # --- Generate schema ---
    print(f'\n=== Generating schema: {output_path} ===')
    schema = generate_schema(nwb_path, output_path=output_path)

    print(f'  Region source:   {schema.get("region_source", "direct")}')
    print(f'  All regions:     {schema["all_regions"]}')
    print(f'  MTL regions:     {schema["mtl_regions"]}')
    print(f'  Frontal regions: {schema["frontal_regions"]}')
    print(f'  Timing columns:  {schema["timing_columns"]}')
    print(f'\nSchema saved to {output_path}')

    if not schema['mtl_regions'] or not schema['frontal_regions']:
        print(
            '\nWARNING: No MTL or Frontal regions detected!\n'
            'The pipeline needs both MTL and Frontal neurons.\n'
            'Try: python scripts/10_human_explore_nwb.py --max-scan 30\n'
            'Or specify a different file with --nwb-path.',
            file=sys.stderr,
        )
    else:
        print('Done. This schema is the HARD PREREQUISITE for all downstream tasks.')


if __name__ == '__main__':
    main()
