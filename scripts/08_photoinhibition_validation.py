"""
DESCARTES WM -- Step 8: Photoinhibition Validation

Feed the trained model ALM activity from photoinhibition trials
and check whether mandatory variables (theta_mod, delay_stability,
choice_signal) collapse -- matching biological ground truth.

If they do, the model captured the causal architecture.
If they persist, the model found shortcuts not present in biology.

Usage:
    python scripts/08_photoinhibition_validation.py \
        [--data-dir data/processed] [--raw-dir data/raw] \
        [--model-dir models] [--hidden-dir hidden_states] \
        [--out-dir results/photoinhibition]
"""

import argparse
import logging
from pathlib import Path

from wm.data.preprocessing import load_processed_session
from wm.analysis.photoinhibition import (
    identify_photostim_in_processed,
    identify_photostim_from_nwb,
    run_photoinhibition_validation,
    print_validation_summary,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Photoinhibition validation for DESCARTES WM"
    )
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--raw-dir', type=str, default='data/raw')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--hidden-dir', type=str, default='hidden_states')
    parser.add_argument('--out-dir', type=str, default='results/photoinhibition')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir)
    model_dir = Path(args.model_dir)
    hidden_dir = Path(args.hidden_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    all_results = {}

    for session_dir in session_dirs:
        session_name = session_dir.name
        logger.info("=== Photoinhibition: %s ===", session_name)

        splits, session_info = load_processed_session(session_dir)

        # Find photostim labels for test trials
        is_photostim = None

        # Try 1: Stream-extracted session with trial_metadata.json
        for raw_subdir in raw_dir.iterdir():
            if raw_subdir.is_dir() and raw_subdir.name == session_name:
                meta_path = raw_subdir / 'trial_metadata.json'
                if meta_path.exists():
                    import json
                    with open(meta_path) as f:
                        meta = json.load(f)
                    if 'photostim_power' in meta:
                        logger.info("  Using trial_metadata.json from %s",
                                    raw_subdir.name)
                        is_photostim = identify_photostim_in_processed(
                            session_dir, meta_path
                        )
                        break

        # Try 2: NWB file (exact stem match)
        if is_photostim is None:
            for nwb_path in raw_dir.rglob('*.nwb'):
                if nwb_path.stem == session_name:
                    logger.info("  Using NWB file: %s", nwb_path.name)
                    n_test = len(splits['test']['trial_types'])
                    is_photostim = identify_photostim_from_nwb(
                        nwb_path,
                        n_correct_trials=n_test,
                    )
                    break

        if is_photostim is None:
            logger.warning("  Could not find photostim labels for %s",
                           session_name)
            continue

        n_stim = int(is_photostim.sum())
        n_ctrl = len(is_photostim) - n_stim
        logger.info("  Test trials: %d control, %d photostim", n_ctrl, n_stim)

        if n_stim < 3:
            logger.warning("  Too few photostim test trials. Skipping.")
            continue

        # Run validation
        session_model_dir = model_dir / session_name
        session_hidden_dir = hidden_dir / session_name
        session_out = out_dir / session_name

        results = run_photoinhibition_validation(
            splits=splits,
            session_info=session_info,
            model_dir=session_model_dir,
            hidden_dir=session_hidden_dir,
            is_photostim_test=is_photostim,
            save_dir=session_out,
        )

        if results:
            all_results[session_name] = results

    # Print combined summary
    if all_results:
        for session_name, results in all_results.items():
            print(f"\n{'='*70}")
            print(f"  Session: {session_name}")
            print_validation_summary(results)
    else:
        logger.warning("No photoinhibition results generated.")


if __name__ == '__main__':
    main()
