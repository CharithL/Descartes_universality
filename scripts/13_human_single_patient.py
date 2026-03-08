#!/usr/bin/env python
"""
Script 13 -- Single-Patient Proof of Concept

Runs the full DESCARTES pipeline on the BEST patient (most neurons in both
regions, most trials). This is the first validation step before scaling to
cross-seed, cross-patient, and cross-architecture tests.

Usage:
    python scripts/13_human_single_patient.py
    python scripts/13_human_single_patient.py --arch lstm --hidden 128
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import (
    HIDDEN_SIZES,
    PROCESSED_DIR,
    RAW_NWB_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)
from human_wm.data.nwb_loader import extract_patient_data, split_data
from human_wm.data.patient_inventory import (
    build_inventory,
    get_best_patient,
)
from human_wm.analysis.universality import run_single_patient_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def estimate_task_timing(trial_info, n_bins):
    """Estimate task phase boundaries from trial metadata.

    The Sternberg task has three phases: encoding, delay, probe.
    We attempt to infer bin boundaries from timing columns in trial_info.
    If specific timing columns are not available, we use reasonable defaults
    based on typical Sternberg task structure:
      - Encoding: first 40% of bins
      - Delay:    middle 30% of bins
      - Probe:    final 30% of bins

    Parameters
    ----------
    trial_info : dict
        Trial metadata.
    n_bins : int
        Total number of time bins per trial.

    Returns
    -------
    dict
        Task timing with 'encoding_bins', 'delay_bins', 'probe_bins' as slices.
    """
    # Try to detect timing from columns
    # (Will be refined once NWB schema is explored)
    encoding_end = int(0.4 * n_bins)
    delay_end = int(0.7 * n_bins)

    return {
        'encoding_bins': slice(0, encoding_end),
        'delay_bins': slice(encoding_end, delay_end),
        'probe_bins': slice(delay_end, n_bins),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run single-patient proof of concept',
    )
    parser.add_argument(
        '--arch', type=str, default='lstm',
        choices=['lstm', 'gru', 'transformer', 'linear'],
        help='Surrogate architecture (default: lstm)',
    )
    parser.add_argument(
        '--hidden', type=int, default=128,
        help='Hidden size (default: 128)',
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed (default: 0)',
    )
    parser.add_argument(
        '--nwb-path', type=str, default=None,
        help='Path to a specific NWB file (overrides auto-detection)',
    )
    args = parser.parse_args()

    # --- Load schema ---
    schema = load_nwb_schema()
    if schema is None:
        logger.error(
            'NWB schema not found. Run script 10_human_explore_nwb.py first '
            'to generate data/nwb_schema.json'
        )
        sys.exit(1)

    # --- Find best patient ---
    if args.nwb_path:
        nwb_path = Path(args.nwb_path)
        logger.info('Using specified NWB file: %s', nwb_path)
    else:
        nwb_paths = sorted(RAW_NWB_DIR.glob('**/*.nwb'))
        if not nwb_paths:
            logger.error('No NWB files found in %s', RAW_NWB_DIR)
            sys.exit(1)

        logger.info('Building patient inventory from %d NWB files...', len(nwb_paths))
        inventory = build_inventory(nwb_paths, schema)
        best = get_best_patient(inventory)

        if best is None:
            logger.error('No usable patients found in inventory')
            sys.exit(1)

        nwb_path = Path(best['path'])
        logger.info(
            'Best patient: %s (MTL=%d, Frontal=%d, Trials=%d)',
            best['patient_id'], best['n_mtl'], best['n_frontal'], best['n_trials'],
        )

    # --- Extract and split data ---
    logger.info('Extracting data from %s...', nwb_path.name)
    X, Y, trial_info = extract_patient_data(nwb_path, schema)
    logger.info(
        'X: %s  Y: %s  trials: %d',
        X.shape, Y.shape, X.shape[0],
    )

    splits = split_data(X, Y, trial_info, seed=args.seed)
    logger.info(
        'Split: train=%d  val=%d  test=%d',
        splits['train']['X'].shape[0],
        splits['val']['X'].shape[0],
        splits['test']['X'].shape[0],
    )

    # --- Estimate task timing ---
    n_bins = X.shape[1]
    task_timing = estimate_task_timing(trial_info, n_bins)
    logger.info(
        'Task timing: encoding=%s  delay=%s  probe=%s',
        task_timing['encoding_bins'],
        task_timing['delay_bins'],
        task_timing['probe_bins'],
    )

    # --- Run pipeline ---
    output_dir = RESULTS_DIR / 'single_patient'
    logger.info('Running single-patient pipeline...')

    results = run_single_patient_pipeline(
        splits=splits,
        task_timing=task_timing,
        arch_name=args.arch,
        hidden_size=args.hidden,
        seed=args.seed,
        output_dir=output_dir,
    )

    if results is None:
        logger.error('Pipeline failed (model quality too low)')
        sys.exit(1)

    # --- Print results ---
    print('\n' + '=' * 60)
    print('SINGLE-PATIENT PROOF OF CONCEPT RESULTS')
    print('=' * 60)
    print(f'Architecture: {args.arch}')
    print(f'Hidden size:  {args.hidden}')
    print(f'CC:           {results["cc"]:.3f}')
    print()
    print(f'{"Variable":<25} {"dR2":<8} {"Classification"}')
    print('-' * 50)

    for var_name, var_result in results['variables'].items():
        dr2 = var_result.get('delta_R2', 0.0)
        cls = var_result['classification']
        print(f'{var_name:<25} {dr2:<8.3f} {cls}')

    print()
    logger.info('Results saved to %s', output_dir)


if __name__ == '__main__':
    main()
