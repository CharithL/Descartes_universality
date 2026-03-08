#!/usr/bin/env python
"""
Script 14 -- Cross-Seed Consistency Test

Train 10 LSTMs with different random seeds on the SAME patient's data.
Reports how many seeds find each variable as MANDATORY.

Threshold: ≥8/10 = ROBUST, <5/10 = FRAGILE (seed-dependent artefact)

Usage:
    python scripts/14_human_cross_seed.py
    python scripts/14_human_cross_seed.py --n-seeds 10 --hidden 128
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import (
    N_SEEDS,
    RAW_NWB_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)
from human_wm.data.nwb_loader import extract_patient_data, split_data
from human_wm.data.patient_inventory import build_inventory, get_best_patient
from human_wm.analysis.universality import cross_seed_test
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
import _timing_helper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Cross-seed consistency test')
    parser.add_argument('--arch', type=str, default='lstm')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS)
    parser.add_argument('--nwb-path', type=str, default=None)
    args = parser.parse_args()

    schema = load_nwb_schema()
    if schema is None:
        logger.error('NWB schema not found. Run script 10 first.')
        sys.exit(1)

    # Find best patient
    if args.nwb_path:
        nwb_path = Path(args.nwb_path)
    else:
        nwb_paths = sorted(RAW_NWB_DIR.glob('**/*.nwb'))
        inventory = build_inventory(nwb_paths, schema)
        best = get_best_patient(inventory)
        if best is None:
            logger.error('No usable patients found')
            sys.exit(1)
        nwb_path = Path(best['path'])
        logger.info('Best patient: %s', best['patient_id'])

    # Extract and split
    X, Y, trial_info = extract_patient_data(nwb_path, schema)
    splits = split_data(X, Y, trial_info, seed=42)
    task_timing = _timing_helper.estimate_task_timing(trial_info, X.shape[1])

    # Run cross-seed test
    # Note: cross_seed_test() internally appends '/cross_seed' to output_dir
    output_dir = RESULTS_DIR
    summary = cross_seed_test(
        splits=splits,
        task_timing=task_timing,
        arch_name=args.arch,
        hidden_size=args.hidden,
        n_seeds=args.n_seeds,
        output_dir=output_dir,
    )

    # Print summary
    print('\n' + '=' * 60)
    print('CROSS-SEED CONSISTENCY SUMMARY')
    print('=' * 60)
    print(f'Seeds: {summary["successful_seeds"]}/{summary["n_seeds"]}')
    print()
    print(f'{"Variable":<25} {"Mandatory":<12} {"Verdict"}')
    print('-' * 50)

    for var_name, var_info in summary['variables'].items():
        mand_str = f'{var_info["n_mandatory"]}/{var_info["n_total"]}'
        print(f'{var_name:<25} {mand_str:<12} {var_info["verdict"]}')

    print()
    logger.info('Results saved to %s', RESULTS_DIR / 'cross_seed')


if __name__ == '__main__':
    main()
