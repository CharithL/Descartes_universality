#!/usr/bin/env python
"""
Script 15 -- Cross-Patient Universality Test

Train a separate model per patient (all usable patients from the inventory).
Reports which variables are MANDATORY across patients -- THE universality
test for human cognition.

Threshold: ≥80% patients = UNIVERSAL, <50% = NOT UNIVERSAL

Usage:
    python scripts/15_human_cross_patient.py
    python scripts/15_human_cross_patient.py --arch lstm --hidden 128
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import (
    RAW_NWB_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)
from human_wm.data.nwb_loader import extract_patient_data, split_data
from human_wm.data.patient_inventory import (
    build_inventory,
    get_usable_patients,
)
from human_wm.analysis.universality import cross_patient_test

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
import _timing_helper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Cross-patient universality test')
    parser.add_argument('--arch', type=str, default='lstm')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--patients', type=str, default=None,
        help='Comma-separated patient IDs (for Vast.ai parallelisation)',
    )
    args = parser.parse_args()

    schema = load_nwb_schema()
    if schema is None:
        logger.error('NWB schema not found. Run script 10 first.')
        sys.exit(1)

    # Build inventory
    nwb_paths = sorted(RAW_NWB_DIR.glob('**/*.nwb'))
    if not nwb_paths:
        logger.error('No NWB files found in %s', RAW_NWB_DIR)
        sys.exit(1)

    inventory = build_inventory(nwb_paths, schema)
    usable = get_usable_patients(inventory)
    logger.info('Found %d usable patients out of %d total', len(usable), len(inventory))

    if not usable:
        logger.error('No usable patients found')
        sys.exit(1)

    # Filter to requested patients if specified
    if args.patients:
        requested = set(args.patients.split(','))
        usable = [p for p in usable if p['patient_id'] in requested]
        logger.info('Filtered to %d requested patients', len(usable))

    # Extract and split data for each patient
    patient_data = []
    for p in usable:
        logger.info('Loading patient %s...', p['patient_id'])
        try:
            X, Y, trial_info = extract_patient_data(p['path'], schema)
            splits = split_data(X, Y, trial_info, seed=args.seed)
            patient_data.append({
                'patient_id': p['patient_id'],
                'splits': splits,
            })
        except Exception as e:
            logger.warning('Failed to load patient %s: %s', p['patient_id'], e)

    if not patient_data:
        logger.error('No patients loaded successfully')
        sys.exit(1)

    # Estimate timing from first patient
    first_X = patient_data[0]['splits']['test']['X']
    first_info = patient_data[0]['splits']['test']['trial_info']
    task_timing = _timing_helper.estimate_task_timing(first_info, first_X.shape[1])

    # Run cross-patient test
    output_dir = RESULTS_DIR / 'cross_patient'
    summary = cross_patient_test(
        patient_data=patient_data,
        task_timing=task_timing,
        arch_name=args.arch,
        hidden_size=args.hidden,
        seed=args.seed,
        output_dir=output_dir,
    )

    # Print summary
    print('\n' + '=' * 65)
    print('CROSS-PATIENT UNIVERSALITY SUMMARY')
    print('=' * 65)
    print(f'Patients tested: {summary["n_patients"]}')
    print()
    print(f'{"Variable":<25} {"Mandatory":<15} {"Universal?"}')
    print('-' * 55)

    for var_name, var_info in summary['variables'].items():
        mand_str = f'{var_info["n_mandatory"]}/{var_info["n_patients"]} ({var_info["pct"]:.0f}%)'
        print(f'{var_name:<25} {mand_str:<15} {var_info["verdict"]}')

    print()
    logger.info('Results saved to %s', output_dir)


if __name__ == '__main__':
    main()
