"""
DESCARTES WM — Step 1: Download NWB Data from DANDI

Downloads sessions from the Chen & Svoboda (2024) dataset
(DANDI archive 000363).

Usage:
    python scripts/01_download_data.py [--output-dir data/raw] [--max-sessions 5]
"""

import argparse
import logging

from wm.data.download import download_sessions

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(description="Download NWB sessions from DANDI")
    parser.add_argument('--output-dir', type=str, default='data/raw',
                        help='Where to save downloaded NWB files')
    parser.add_argument('--max-sessions', type=int, default=5,
                        help='Maximum number of sessions to download')
    args = parser.parse_args()

    download_sessions(
        output_dir=args.output_dir,
        n_sessions=args.max_sessions,
    )


if __name__ == '__main__':
    main()
