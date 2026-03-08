"""Tests for human_wm.data.nwb_explorer -- NWB exploration and schema generation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from human_wm.data import nwb_explorer


# ---------------------------------------------------------------------------
# Fixtures: mock NWB file structure
# ---------------------------------------------------------------------------

def _make_mock_nwb():
    """Create a MagicMock NWB file matching Rutishauser dataset structure.

    Structure:
    - units: 15 units across 5 brain regions
    - trials: 50 trials with timing and stimulus columns
    """
    nwbfile = MagicMock()

    # --- Units table ---
    units = MagicMock()
    units.colnames = [
        'spike_times', 'electrodes', 'brain_region',
        'origClusterID', 'IsolationDist', 'SNR',
    ]

    # 15 units: hippocampus(4), amygdala(3), dACC(3), pre-SMA(3), vmPFC(2)
    regions = (
        ['hippocampus'] * 4
        + ['amygdala'] * 3
        + ['dACC'] * 3
        + ['pre-SMA'] * 3
        + ['vmPFC'] * 2
    )
    region_col = MagicMock()
    region_col.__getitem__ = lambda self, idx: regions[idx]
    region_col.__iter__ = lambda self: iter(regions)
    region_col.__len__ = lambda self: len(regions)
    region_col.data = regions

    units.__len__ = lambda self: 15
    units.__getitem__ = lambda self, key: region_col if key == 'brain_region' else MagicMock()
    units.__contains__ = lambda self, key: key in units.colnames

    nwbfile.units = units

    # --- Trials table ---
    trials = MagicMock()
    trials.colnames = [
        'start_time', 'stop_time', 'stimulus_image',
        'response_value', 'loads', 'in_out',
    ]
    trials.__len__ = lambda self: 50

    nwbfile.trials = trials

    # --- Electrode groups ---
    eg1 = MagicMock()
    eg1.name = 'LA'
    eg1.description = 'Left Amygdala'
    eg1.location = 'amygdala'

    eg2 = MagicMock()
    eg2.name = 'RAH'
    eg2.description = 'Right Anterior Hippocampus'
    eg2.location = 'hippocampus'

    eg3 = MagicMock()
    eg3.name = 'dACC'
    eg3.description = 'dorsal Anterior Cingulate Cortex'
    eg3.location = 'dACC'

    nwbfile.electrode_groups = {'LA': eg1, 'RAH': eg2, 'dACC': eg3}

    return nwbfile


@pytest.fixture
def mock_nwb():
    """Provide a mock NWB file object."""
    return _make_mock_nwb()


@pytest.fixture
def patch_open_nwb(mock_nwb):
    """Patch _open_nwb to return the mock NWB file."""
    with patch.object(nwb_explorer, '_open_nwb', return_value=mock_nwb) as m:
        yield m


# ---------------------------------------------------------------------------
# explore_nwb tests
# ---------------------------------------------------------------------------

class TestExploreNwb:
    """Tests for the explore_nwb function."""

    def test_returns_dict_with_required_keys(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        required_keys = {
            'units_columns', 'trial_columns', 'brain_regions',
            'region_column_detected', 'electrode_groups',
            'n_units', 'n_trials',
        }
        assert isinstance(result, dict)
        assert required_keys.issubset(result.keys())

    def test_detects_brain_regions(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        regions = result['brain_regions']
        assert 'hippocampus' in regions
        assert 'amygdala' in regions
        assert 'dACC' in regions
        assert 'pre-SMA' in regions
        assert 'vmPFC' in regions

    def test_region_counts(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        regions = result['brain_regions']
        assert regions['hippocampus'] == 4
        assert regions['amygdala'] == 3
        assert regions['dACC'] == 3
        assert regions['pre-SMA'] == 3
        assert regions['vmPFC'] == 2

    def test_n_units(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        assert result['n_units'] == 15

    def test_n_trials(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        assert result['n_trials'] == 50

    def test_auto_detects_region_column(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        assert result['region_column_detected'] == 'brain_region'

    def test_units_columns(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        assert 'spike_times' in result['units_columns']
        assert 'brain_region' in result['units_columns']

    def test_trial_columns(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        assert 'start_time' in result['trial_columns']
        assert 'stop_time' in result['trial_columns']

    def test_electrode_groups(self, patch_open_nwb):
        result = nwb_explorer.explore_nwb('fake.nwb')
        assert len(result['electrode_groups']) == 3


# ---------------------------------------------------------------------------
# _classify_region tests
# ---------------------------------------------------------------------------

class TestClassifyRegion:
    """Tests for region classification logic."""

    def test_hippocampus_is_mtl(self):
        assert nwb_explorer._classify_region('hippocampus') == 'mtl'

    def test_amygdala_is_mtl(self):
        assert nwb_explorer._classify_region('amygdala') == 'mtl'

    def test_dacc_is_frontal(self):
        assert nwb_explorer._classify_region('dACC') == 'frontal'

    def test_presma_is_frontal(self):
        assert nwb_explorer._classify_region('pre-SMA') == 'frontal'

    def test_vmpfc_is_frontal(self):
        assert nwb_explorer._classify_region('vmPFC') == 'frontal'

    def test_case_insensitive_mtl(self):
        assert nwb_explorer._classify_region('Hippocampus') == 'mtl'
        assert nwb_explorer._classify_region('AMYGDALA') == 'mtl'

    def test_case_insensitive_frontal(self):
        assert nwb_explorer._classify_region('DACC') == 'frontal'
        assert nwb_explorer._classify_region('PRE-SMA') == 'frontal'

    def test_unknown_region_is_other(self):
        assert nwb_explorer._classify_region('thalamus') == 'other'
        assert nwb_explorer._classify_region('cerebellum') == 'other'

    def test_entorhinal_is_mtl(self):
        assert nwb_explorer._classify_region('entorhinal') == 'mtl'

    def test_prefrontal_is_frontal(self):
        assert nwb_explorer._classify_region('prefrontal') == 'frontal'

    def test_anterior_cingulate_is_frontal(self):
        assert nwb_explorer._classify_region('anterior cingulate') == 'frontal'


# ---------------------------------------------------------------------------
# generate_schema tests
# ---------------------------------------------------------------------------

class TestGenerateSchema:
    """Tests for schema generation and JSON output."""

    def test_generates_valid_json(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            # Verify it wrote valid JSON
            with open(tmp_path, 'r') as f:
                loaded = json.load(f)
            assert isinstance(loaded, dict)
            assert loaded == schema
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_has_required_keys(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            required_keys = {
                'units_columns', 'region_column', 'trial_columns',
                'mtl_regions', 'frontal_regions', 'all_regions',
                'region_counts', 'timing_columns', 'n_units',
                'n_trials', 'sample_values',
            }
            assert required_keys.issubset(schema.keys())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_mtl_regions(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            assert 'hippocampus' in schema['mtl_regions']
            assert 'amygdala' in schema['mtl_regions']
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_frontal_regions(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            assert 'dACC' in schema['frontal_regions']
            assert 'pre-SMA' in schema['frontal_regions']
            assert 'vmPFC' in schema['frontal_regions']
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_timing_columns(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            # start_time and stop_time are in trial_columns and should be detected
            assert 'start_time' in schema['timing_columns']
            assert 'stop_time' in schema['timing_columns']
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_region_counts(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            assert schema['region_counts']['hippocampus'] == 4
            assert schema['region_counts']['amygdala'] == 3
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_n_units_and_trials(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            assert schema['n_units'] == 15
            assert schema['n_trials'] == 50
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_schema_region_column(self, patch_open_nwb):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            tmp_path = f.name
        try:
            schema = nwb_explorer.generate_schema('fake.nwb', output_path=tmp_path)
            assert schema['region_column'] == 'brain_region'
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Candidate lists
# ---------------------------------------------------------------------------

class TestCandidateLists:
    """Verify candidate column lists are defined and reasonable."""

    def test_region_column_candidates_exist(self):
        assert hasattr(nwb_explorer, '_REGION_COLUMN_CANDIDATES')
        assert 'brain_region' in nwb_explorer._REGION_COLUMN_CANDIDATES
        assert 'location' in nwb_explorer._REGION_COLUMN_CANDIDATES

    def test_timing_column_candidates_exist(self):
        assert hasattr(nwb_explorer, '_TIMING_COLUMN_CANDIDATES')
        assert 'start_time' in nwb_explorer._TIMING_COLUMN_CANDIDATES
        assert 'stop_time' in nwb_explorer._TIMING_COLUMN_CANDIDATES
