#!/usr/bin/env python
"""
11 -- Download NWB files from DANDI 000469

Provides two download modes:
  1. CLI mode (default): uses ``dandi download`` for robust, resumable downloads
  2. API mode (--use-api): uses the ``dandi`` Python API for limited downloads

Without --download, simply lists available NWB assets.

Usage
-----
    # List available NWB assets
    python scripts/11_human_download.py

    # Download all sessions via CLI
    python scripts/11_human_download.py --download

    # Download first 5 sessions via Python API
    python scripts/11_human_download.py --download --use-api -n 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure the project root is on sys.path so human_wm is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from human_wm.config import DANDISET_ID, RAW_NWB_DIR


def list_assets() -> list[str]:
    """List available NWB assets in the DANDI dataset.

    Uses the ``dandi`` Python API to enumerate assets in the latest version
    of the dandiset.

    Returns
    -------
    list[str]
        List of asset paths (e.g. 'sub-01/sub-01_ses-01.nwb').
    """
    from dandi.dandiapi import DandiAPIClient

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID, 'draft')
        assets = list(dandiset.get_assets())

    nwb_assets = [a.path for a in assets if a.path.endswith('.nwb')]
    nwb_assets.sort()

    print(f'\n=== DANDI:{DANDISET_ID} -- {len(nwb_assets)} NWB assets ===')
    for i, path in enumerate(nwb_assets, 1):
        print(f'  {i:3d}. {path}')
    print()

    return nwb_assets


def download_via_cli() -> None:
    """Download the full dandiset using ``dandi download`` CLI.

    Uses ``--existing skip`` so re-runs are safe and resumable.
    Files are saved to RAW_NWB_DIR.
    """
    RAW_NWB_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        'dandi', 'download',
        f'DANDI:{DANDISET_ID}',
        '-o', str(RAW_NWB_DIR),
        '--existing', 'skip',
    ]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
    print(f'\nDownload complete. Files saved to {RAW_NWB_DIR}')


def download_via_api(n_sessions: int | None = None) -> list[Path]:
    """Download NWB files using the dandi Python API.

    Parameters
    ----------
    n_sessions : int or None
        Maximum number of NWB sessions to download. If None, downloads all.

    Returns
    -------
    list[Path]
        List of paths to downloaded files.
    """
    from dandi.dandiapi import DandiAPIClient

    RAW_NWB_DIR.mkdir(parents=True, exist_ok=True)

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID, 'draft')
        assets = list(dandiset.get_assets())

    # Filter to NWB files only
    nwb_assets = [a for a in assets if a.path.endswith('.nwb')]
    nwb_assets.sort(key=lambda a: a.path)

    if n_sessions is not None:
        nwb_assets = nwb_assets[:n_sessions]

    downloaded = []
    for i, asset in enumerate(nwb_assets, 1):
        # Preserve original directory structure (e.g. sub-1/file.nwb)
        dest = RAW_NWB_DIR / asset.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            print(f'  [{i}/{len(nwb_assets)}] SKIP (exists): {asset.path}')
            downloaded.append(dest)
            continue
        print(f'  [{i}/{len(nwb_assets)}] Downloading: {asset.path} '
              f'({asset.size/(1024*1024):.1f} MB) ...')
        asset.download(dest)
        downloaded.append(dest)
        print(f'    -> {dest}')

    print(f'\nDownloaded {len(downloaded)} files to {RAW_NWB_DIR}')
    return downloaded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f'Download NWB files from DANDI:{DANDISET_ID}',
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Actually download files (without this flag, only lists assets)',
    )
    parser.add_argument(
        '--use-api',
        action='store_true',
        help='Use Python API instead of CLI for download',
    )
    parser.add_argument(
        '-n',
        type=int,
        default=None,
        help='Number of sessions to download (API mode only)',
    )
    args = parser.parse_args(argv)

    if not args.download:
        list_assets()
        print('To download, re-run with --download')
        return

    if args.use_api:
        download_via_api(n_sessions=args.n)
    else:
        if args.n is not None:
            print(
                'WARNING: -n is ignored in CLI mode. '
                'Use --use-api to limit download count.',
            )
        download_via_cli()


if __name__ == '__main__':
    main()
