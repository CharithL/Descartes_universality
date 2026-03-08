"""
DESCARTES WM — Step 3: Train LSTM Surrogates

Trains WMSurrogate LSTMs at multiple hidden sizes
(64, 128, 256) on each processed session.

Usage:
    python scripts/03_train_surrogates.py [--data-dir data/processed] [--out-dir models]
"""

import argparse
import logging
from pathlib import Path

from wm.config import HIDDEN_SIZES
from wm.data.preprocessing import load_processed_session
from wm.surrogate.train import train_all_sizes

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train WM surrogate LSTMs")
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--out-dir', type=str, default='models')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not session_dirs:
        logger.error("No processed sessions found in %s", data_dir)
        return

    for session_dir in session_dirs:
        session_name = session_dir.name
        logger.info("=== Training surrogates for %s ===", session_name)

        splits, session_info = load_processed_session(session_dir)
        model_dir = out_dir / session_name
        model_dir.mkdir(parents=True, exist_ok=True)

        results = train_all_sizes(
            splits,
            session_info=session_info,
            output_dir=model_dir,
        )

        # Save training summary
        import json
        summary = {
            hs: {'cc': r['cc'], 'n_params': r['n_params']}
            for hs, r in results.items()
        }
        with open(model_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info("Training summary: %s", summary)


if __name__ == '__main__':
    main()
