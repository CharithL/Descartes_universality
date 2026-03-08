#!/usr/bin/env python
"""
Script 17 -- Generate Universality Report

Combines results from cross-seed, cross-patient, and cross-architecture
tests into the final Universality Table.

Usage:
    python scripts/17_human_universality_report.py

Reads JSON summaries from data/results/{cross_seed,cross_patient,cross_architecture}/
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import RESULTS_DIR
from human_wm.analysis.universality import format_universality_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def load_json_if_exists(path):
    """Load a JSON file if it exists, otherwise return None."""
    path = Path(path)
    if not path.exists():
        logger.warning('Summary not found: %s', path)
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # Load summaries from each test
    cross_seed = load_json_if_exists(
        RESULTS_DIR / 'cross_seed' / 'cross_seed_summary.json'
    )
    cross_patient = load_json_if_exists(
        RESULTS_DIR / 'cross_patient' / 'cross_patient_summary.json'
    )
    cross_arch = load_json_if_exists(
        RESULTS_DIR / 'cross_architecture' / 'cross_arch_summary.json'
    )

    if cross_seed is None and cross_patient is None and cross_arch is None:
        logger.error(
            'No test summaries found. Run scripts 14, 15, and/or 16 first.'
        )
        sys.exit(1)

    # Generate and print the universality table
    table = format_universality_table(
        cross_seed_summary=cross_seed,
        cross_patient_summary=cross_patient,
        cross_arch_summary=cross_arch,
    )
    print(table)

    # Save to file
    report_path = RESULTS_DIR / 'universality_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(table)

    logger.info('Report saved to %s', report_path)


if __name__ == '__main__':
    main()
