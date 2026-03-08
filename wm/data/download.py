"""
DESCARTES WM — DANDI Download Utilities

Download and manage NWB files from DANDI archive (dandiset 000363).
Supports both full download and streaming access.
"""

import logging
from pathlib import Path

from wm.config import DANDISET_ID, RAW_NWB_DIR

logger = logging.getLogger(__name__)


def list_assets(limit=None):
    """List all assets in the dandiset.

    Returns
    -------
    assets : list of dict
        Each with 'path', 'size_gb', 'asset_id'.
    """
    from dandi.dandiapi import DandiAPIClient

    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)
    assets = list(dandiset.get_assets())

    result = []
    for i, asset in enumerate(assets):
        if limit is not None and i >= limit:
            break
        result.append({
            'path': asset.path,
            'size_gb': asset.size / 1e9,
            'asset_id': asset.identifier,
        })
    return result


def download_sessions(n_sessions=5, output_dir=None):
    """Download a limited number of NWB sessions via Python API.

    Parameters
    ----------
    n_sessions : int
        Number of sessions to download.
    output_dir : str or Path, optional
        Download directory. Defaults to RAW_NWB_DIR.
    """
    from dandi.dandiapi import DandiAPIClient

    if output_dir is None:
        output_dir = RAW_NWB_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)

    # Filter to NWB files only
    nwb_assets = []
    for asset in dandiset.get_assets():
        if asset.path.endswith('.nwb'):
            nwb_assets.append(asset)
        if len(nwb_assets) >= n_sessions:
            break

    logger.info("Downloading %d sessions to %s", len(nwb_assets), output_dir)

    for i, asset in enumerate(nwb_assets):
        dest = output_dir / asset.path
        if dest.exists():
            logger.info("[%d/%d] Skipping %s (already exists)",
                        i + 1, len(nwb_assets), asset.path)
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[%d/%d] Downloading %s (%.2f GB) ...",
                    i + 1, len(nwb_assets), asset.path,
                    asset.size / 1e9)
        asset.download(dest)
        logger.info("  Saved to %s", dest)


def get_streaming_url(asset_path):
    """Get a streaming URL for an NWB asset (no download required).

    Parameters
    ----------
    asset_path : str
        Path within the dandiset (e.g., 'sub-XXX/sub-XXX_ses-YYY_ecephys.nwb').

    Returns
    -------
    url : str
        HTTP URL for streaming access via fsspec.
    """
    from dandi.dandiapi import DandiAPIClient

    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)

    for asset in dandiset.get_assets():
        if asset.path == asset_path:
            return asset.get_content_url(follow_redirects=1, strip_query=True)

    raise FileNotFoundError(f"Asset not found: {asset_path}")


def find_nwb_files(data_dir=None):
    """Find all downloaded NWB files.

    Returns
    -------
    paths : list of Path
    """
    if data_dir is None:
        data_dir = RAW_NWB_DIR
    data_dir = Path(data_dir)

    paths = sorted(data_dir.rglob('*.nwb'))
    logger.info("Found %d NWB files in %s", len(paths), data_dir)
    return paths
