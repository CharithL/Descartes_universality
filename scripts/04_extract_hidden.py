"""
DESCARTES WM — Step 4: Extract Hidden States

Extracts hidden state activations from trained and
randomly-initialised LSTMs for probing analysis.

Usage:
    python scripts/04_extract_hidden.py [--data-dir data/processed] \
        [--model-dir models] [--out-dir hidden_states]
"""

import argparse
import logging
from pathlib import Path

from wm.config import HIDDEN_SIZES
from wm.data.preprocessing import load_processed_session
from wm.surrogate.extract_hidden import extract_all_sizes

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from surrogates")
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--out-dir', type=str, default='hidden_states')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for session_dir in session_dirs:
        session_name = session_dir.name
        logger.info("=== Extracting hidden states for %s ===", session_name)

        splits, session_info = load_processed_session(session_dir)
        session_model_dir = model_dir / session_name
        session_out = out_dir / session_name

        if not session_model_dir.exists():
            logger.warning("No models found for %s, skipping", session_name)
            continue

        extract_all_sizes(
            splits=splits,
            model_dir=session_model_dir,
            save_dir=session_out,
        )


if __name__ == '__main__':
    main()
