"""
DESCARTES WM — Step 0: Inspect NWB Column Names

Run this BEFORE writing any extraction logic. Prints
nwb.trials.colnames and nwb.units.colnames from the first
downloaded file so you know exactly what fields exist.

Usage:
    python scripts/00_inspect_nwb.py [--data-dir DATA_DIR] [--max-files 3]
"""

import argparse
import sys
from pathlib import Path

import pynwb


def inspect_nwb(nwb_path):
    """Print column names and basic metadata from an NWB file."""
    print(f"\n{'='*70}")
    print(f"FILE: {Path(nwb_path).name}")
    print(f"{'='*70}")

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()

        # --- Session metadata ---
        print(f"\nSession ID:    {getattr(nwb, 'session_id', 'N/A')}")
        print(f"Description:   {getattr(nwb, 'session_description', 'N/A')[:100]}")
        print(f"Identifier:    {getattr(nwb, 'identifier', 'N/A')}")

        # --- Units (neurons) ---
        if nwb.units is not None:
            print(f"\n--- units ({len(nwb.units)} rows) ---")
            colnames = list(nwb.units.colnames) if nwb.units.colnames else []
            print(f"  colnames ({len(colnames)}): {colnames}")

            # Show a few values from region-like columns
            region_candidates = [
                c for c in colnames
                if any(k in c.lower() for k in ['region', 'location', 'area', 'structure'])
            ]
            for col in region_candidates:
                vals = nwb.units[col][:]
                unique_vals = list(set(str(v) for v in vals))
                print(f"  {col} unique values ({len(unique_vals)}): {unique_vals[:15]}")
        else:
            print("\n--- units: NOT FOUND ---")

        # --- Trials ---
        if nwb.trials is not None:
            print(f"\n--- trials ({len(nwb.trials)} rows) ---")
            colnames = list(nwb.trials.colnames) if nwb.trials.colnames else []
            print(f"  colnames ({len(colnames)}): {colnames}")

            # Show a few values from trial-type and outcome columns
            type_candidates = [
                c for c in colnames
                if any(k in c.lower() for k in ['type', 'condition', 'stim', 'instruction'])
            ]
            for col in type_candidates:
                vals = nwb.trials[col][:]
                unique_vals = list(set(str(v) for v in vals))
                print(f"  {col} unique values ({len(unique_vals)}): {unique_vals[:15]}")

            outcome_candidates = [
                c for c in colnames
                if any(k in c.lower() for k in ['outcome', 'correct', 'hit', 'result', 'response'])
            ]
            for col in outcome_candidates:
                vals = nwb.trials[col][:]
                unique_vals = list(set(str(v) for v in vals))
                print(f"  {col} unique values ({len(unique_vals)}): {unique_vals[:15]}")
        else:
            print("\n--- trials: NOT FOUND ---")

        # --- Electrodes (if present) ---
        if hasattr(nwb, 'electrodes') and nwb.electrodes is not None:
            colnames = list(nwb.electrodes.colnames) if nwb.electrodes.colnames else []
            print(f"\n--- electrodes ({len(nwb.electrodes)} rows) ---")
            print(f"  colnames ({len(colnames)}): {colnames}")
        else:
            print("\n--- electrodes: NOT FOUND ---")

    print()


def main():
    parser = argparse.ArgumentParser(description="Inspect NWB files for column names")
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing NWB files')
    parser.add_argument('--max-files', type=int, default=3,
                        help='Maximum number of files to inspect')
    parser.add_argument('--file', type=str, default=None,
                        help='Inspect a specific NWB file')
    args = parser.parse_args()

    if args.file:
        nwb_files = [Path(args.file)]
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            print("Run 01_download_data.py first, or pass --file <path.nwb>")
            sys.exit(1)

        nwb_files = sorted(data_dir.glob('**/*.nwb'))
        if not nwb_files:
            print(f"No NWB files found in {data_dir}")
            sys.exit(1)

    print(f"Found {len(nwb_files)} NWB file(s). Inspecting up to {args.max_files}.")

    for f in nwb_files[:args.max_files]:
        try:
            inspect_nwb(f)
        except Exception as e:
            print(f"\nERROR reading {f.name}: {e}")

    print("Done. Use the column names above to verify nwb_loader.py mappings.")


if __name__ == '__main__':
    main()
