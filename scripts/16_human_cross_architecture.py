#!/usr/bin/env python
"""
Script 16 -- Cross-Architecture Universality Test

Train LSTM, GRU, Transformer, and Linear models on the best patient.
Reports which variables are MANDATORY across architectures.

    4/4 architectures = mathematical necessity
    1/4 architectures = architecture artefact

Usage:
    python scripts/16_human_cross_architecture.py
    python scripts/16_human_cross_architecture.py --hidden 128
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import (
    ARCHITECTURES,
    RAW_NWB_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)
from human_wm.data.nwb_loader import extract_patient_data, split_data
from human_wm.data.patient_inventory import build_inventory, get_best_patient
from human_wm.analysis.universality import (
    cross_architecture_test,
    format_universality_table,
)

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
import _timing_helper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-architecture universality test',
    )
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
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
    splits = split_data(X, Y, trial_info, seed=args.seed)
    task_timing = _timing_helper.estimate_task_timing(trial_info, X.shape[1])

    # Run cross-architecture test
    output_dir = RESULTS_DIR / 'cross_architecture'
    summary = cross_architecture_test(
        splits=splits,
        task_timing=task_timing,
        hidden_size=args.hidden,
        seed=args.seed,
        output_dir=output_dir,
    )

    # Print summary
    print('\n' + '=' * 85)
    print('CROSS-ARCHITECTURE UNIVERSALITY SUMMARY')
    print('=' * 85)
    print()
    print(f'{"Variable":<25}', end='')
    for arch in ARCHITECTURES:
        print(f' {arch:<14}', end='')
    print('  Verdict')
    print('-' * 85)

    for var_name, var_info in summary['variables'].items():
        print(f'{var_name:<25}', end='')

        for arch in ARCHITECTURES:
            if arch in var_info.get('mandatory_in', []):
                symbol = 'MAND'
            else:
                symbol = 'zombie'
            print(f' {symbol:<14}', end='')

        print(f'  {var_info["verdict"]}')

    print()
    logger.info('Results saved to %s', output_dir)


if __name__ == '__main__':
    main()
