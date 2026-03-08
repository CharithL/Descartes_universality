"""
DESCARTES WM — Step 5: Ridge ΔR² Probing

Runs Ridge regression probing on all Level B and C targets
across hidden sizes. This is the core zombie diagnostic.

Usage:
    python scripts/05_run_probing.py [--data-dir data/processed] \
        [--hidden-dir hidden_states] [--out-dir results/probing]
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from wm.config import HIDDEN_SIZES
from wm.data.preprocessing import load_processed_session
from wm.analysis.run_probing import compute_all_targets, run_probing_all

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Ridge probing analysis")
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--hidden-dir', type=str, default='hidden_states')
    parser.add_argument('--out-dir', type=str, default='results/probing')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    hidden_dir = Path(args.hidden_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for session_dir in session_dirs:
        session_name = session_dir.name
        logger.info("=== Probing %s ===", session_name)

        splits, session_info = load_processed_session(session_dir)
        Y_test = splits['test']['Y']
        trial_types_test = splits['test']['trial_types']

        # Compute probe targets
        targets = compute_all_targets(Y_test, trial_types_test)

        # Load hidden states
        hidden_states_dict = {}
        for hs in HIDDEN_SIZES:
            trained_path = hidden_dir / session_name / f'wm_h{hs}_trained.npz'
            untrained_path = hidden_dir / session_name / f'wm_h{hs}_untrained.npz'

            if not trained_path.exists() or not untrained_path.exists():
                logger.warning("Missing hidden states for h=%d, skipping", hs)
                continue

            trained_data = np.load(trained_path)
            untrained_data = np.load(untrained_path)
            hidden_states_dict[hs] = (
                trained_data['hidden_states'],
                untrained_data['hidden_states'],
            )

        if not hidden_states_dict:
            logger.warning("No hidden states for %s", session_name)
            continue

        # Run probing
        session_out = out_dir / session_name
        results = run_probing_all(hidden_states_dict, targets, save_dir=session_out)

        # Summary
        for level in results:
            for hs in results[level]:
                for r in results[level][hs]:
                    logger.info(
                        "%s | h=%d | %s: ΔR²=%.3f → %s",
                        session_name, hs, r['var_name'],
                        r['delta_R2'], r['category'],
                    )


if __name__ == '__main__':
    main()
