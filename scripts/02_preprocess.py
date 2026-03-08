"""
DESCARTES WM — Step 2: Preprocess NWB → Binned Spike Tensors

Reads downloaded NWB files, extracts ALM/thalamus spike trains,
bins into (n_trials, n_timesteps, n_neurons) tensors, filters
correct trials, and splits into train/val/test.

Usage:
    python scripts/02_preprocess.py [--raw-dir data/raw] [--out-dir data/processed]
"""

import argparse
import logging
from pathlib import Path

from wm.data.nwb_loader import find_qualifying_sessions
from wm.data.preprocessing import (
    extract_session_data,
    extract_from_streamed,
    filter_correct_trials,
    split_data,
    save_processed_session,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess NWB to spike tensors")
    parser.add_argument('--raw-dir', type=str, default='data/raw')
    parser.add_argument('--out-dir', type=str, default='data/processed')
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    processed_any = False

    # --- Phase 1: Process stream-extracted sessions (NPZ+JSON from 01b) ---
    streamed_dirs = []
    for spike_file in sorted(raw_dir.glob('**/spike_data.npz')):
        streamed_dirs.append(spike_file.parent)

    if streamed_dirs:
        logger.info("Found %d stream-extracted sessions", len(streamed_dirs))
        for session_dir in streamed_dirs:
            session_name = session_dir.name
            session_out = out_dir / session_name
            if session_out.exists():
                logger.info("Skipping %s (already processed)", session_name)
                continue

            logger.info("Processing streamed session %s ...", session_name)
            try:
                X, Y, trial_types, trial_outcomes, session_info = extract_from_streamed(session_dir)
                X, Y, trial_types = filter_correct_trials(X, Y, trial_types, trial_outcomes)
                splits = split_data(X, Y, trial_types)
                save_processed_session(splits, session_info, session_out)
                logger.info("Saved %s (%d trials)", session_name, len(X))
                processed_any = True
            except Exception as e:
                logger.error("Failed to process %s: %s", session_name, e)

    # --- Phase 2: Process NWB files (from 01_download) ---
    nwb_files = sorted(raw_dir.glob('**/*.nwb'))
    if nwb_files:
        qualifying = find_qualifying_sessions(nwb_files)
        logger.info("Found %d qualifying NWB sessions out of %d total", len(qualifying), len(nwb_files))

        for nwb_path, info in qualifying:
            session_name = Path(nwb_path).stem
            session_out = out_dir / session_name
            if session_out.exists():
                logger.info("Skipping %s (already processed)", session_name)
                continue

            logger.info("Processing NWB session %s ...", session_name)
            try:
                X, Y, trial_types, trial_outcomes, session_info = extract_session_data(nwb_path)
                X, Y, trial_types = filter_correct_trials(X, Y, trial_types, trial_outcomes)
                splits = split_data(X, Y, trial_types)
                save_processed_session(splits, session_info, session_out)
                logger.info("Saved %s (%d trials)", session_name, len(X))
                processed_any = True
            except Exception as e:
                logger.error("Failed to process %s: %s", session_name, e)

    if not processed_any and not streamed_dirs and not nwb_files:
        logger.error("No data found in %s. Run 01_download_data.py or 01b_stream_extract.py first.", raw_dir)


if __name__ == '__main__':
    main()
