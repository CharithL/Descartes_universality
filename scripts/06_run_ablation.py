"""
DESCARTES WM — Step 6: Resample Ablation + Classification

For all variables classified as LEARNED (delta_R2 >= 0.1), run
resample ablation with progressive k-sweep to determine
whether they are MANDATORY or LEARNED_BYPRODUCT.

Usage:
    python scripts/06_run_ablation.py [--data-dir data/processed] \
        [--model-dir models] [--hidden-dir hidden_states] \
        [--probe-dir results/probing] [--out-dir results/ablation]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from wm.config import HIDDEN_SIZES
from wm.data.preprocessing import load_processed_session
from wm.analysis.run_probing import compute_all_targets
from wm.analysis.run_ablation import run_ablation_on_learned

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ablation analysis")
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--hidden-dir', type=str, default='hidden_states')
    parser.add_argument('--probe-dir', type=str, default='results/probing')
    parser.add_argument('--out-dir', type=str, default='results/ablation')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    hidden_dir = Path(args.hidden_dir)
    probe_dir = Path(args.probe_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for session_dir in session_dirs:
        session_name = session_dir.name
        logger.info("=== Ablation for %s ===", session_name)

        splits, session_info = load_processed_session(session_dir)
        Y_test = splits['test']['Y']
        trial_types_test = splits['test']['trial_types']

        # Compute targets (same as probing)
        targets = compute_all_targets(Y_test, trial_types_test)

        # Load probe results and hidden states
        probe_results = {}
        hidden_states_dict = {}

        for hs in HIDDEN_SIZES:
            # Load hidden states
            trained_path = hidden_dir / session_name / f'wm_h{hs}_trained.npz'
            untrained_path = hidden_dir / session_name / f'wm_h{hs}_untrained.npz'
            if not trained_path.exists():
                continue
            trained_data = np.load(trained_path)
            untrained_data = np.load(untrained_path)
            hidden_states_dict[hs] = (
                trained_data['hidden_states'],
                untrained_data['hidden_states'],
            )

            # Load probe results for this hidden size
            for level in ['B', 'C']:
                probe_path = probe_dir / session_name / f'probe_level{level}_h{hs}.json'
                if probe_path.exists():
                    with open(probe_path) as f:
                        results_list = json.load(f)
                    if level not in probe_results:
                        probe_results[level] = {}
                    probe_results[level][hs] = results_list

        if not probe_results or not hidden_states_dict:
            logger.warning("No probe/hidden data for %s, skipping", session_name)
            continue

        session_model_dir = model_dir / session_name
        session_out = out_dir / session_name

        run_ablation_on_learned(
            probe_results=probe_results,
            splits=splits,
            hidden_states_dict=hidden_states_dict,
            model_dir=session_model_dir,
            targets=targets,
            save_dir=session_out,
        )


if __name__ == '__main__':
    main()
