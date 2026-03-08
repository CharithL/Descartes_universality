"""Tests for human_wm.data.nwb_loader -- schema-driven NWB data loading."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from human_wm.data import nwb_loader


# ---------------------------------------------------------------------------
# Test schema matching the explorer output structure
# ---------------------------------------------------------------------------

_TEST_SCHEMA = {
    'region_column': 'brain_region',
    'mtl_regions': ['hippocampus', 'amygdala'],
    'frontal_regions': ['dACC', 'pre-SMA', 'vmPFC'],
    'trial_columns': [
        'start_time', 'stop_time', 'stimulus_image',
        'response_value', 'loads', 'in_out',
    ],
    'timing_columns': ['start_time', 'stop_time'],
    'units_columns': ['spike_times', 'brain_region'],
}

# Neuron layout: 8 MTL (5 hippocampus + 3 amygdala),
#                6 frontal (2 dACC + 2 pre-SMA + 2 vmPFC),
#                2 other (thalamus)
_REGIONS = (
    ['hippocampus'] * 5
    + ['amygdala'] * 3
    + ['dACC'] * 2
    + ['pre-SMA'] * 2
    + ['vmPFC'] * 2
    + ['thalamus'] * 2
)

_N_UNITS = len(_REGIONS)  # 16
_N_TRIALS = 60
_TRIAL_DURATION = 2.0  # seconds


def _make_mock_nwb():
    """Create a MagicMock NWB file with 16 neurons and 60 trials.

    8 MTL neurons, 6 frontal neurons, 2 other neurons.
    Each trial is 2.0 s long, producing 40 bins at 50 ms bin size.
    """
    nwbfile = MagicMock()

    # --- Units table ---
    units = MagicMock()
    units.colnames = ['spike_times', 'brain_region']

    # Build region column mock
    region_col = MagicMock()
    region_col.__getitem__ = lambda self, idx: _REGIONS[idx]
    region_col.__len__ = lambda self: _N_UNITS

    # Build spike_times column: each neuron gets random spikes in [0, 120]
    rng = np.random.RandomState(0)
    spike_data = []
    for _ in range(_N_UNITS):
        n_spikes = rng.randint(100, 500)
        spikes = np.sort(rng.uniform(0, _N_TRIALS * _TRIAL_DURATION + 10, n_spikes))
        spike_data.append(spikes)

    spike_col = MagicMock()
    spike_col.__getitem__ = lambda self, idx: spike_data[idx]

    def units_getitem(self, key):
        if key == 'brain_region':
            return region_col
        elif key == 'spike_times':
            return spike_col
        return MagicMock()

    units.__getitem__ = units_getitem
    units.__len__ = lambda self: _N_UNITS
    nwbfile.units = units

    # --- Trials table ---
    trials = MagicMock()
    trials.colnames = _TEST_SCHEMA['trial_columns']

    # Each trial: start_time = i*2.5, stop_time = start + 2.0
    start_times = [i * 2.5 for i in range(_N_TRIALS)]
    stop_times = [s + _TRIAL_DURATION for s in start_times]
    stimulus_images = [f'img_{i:03d}.png' for i in range(_N_TRIALS)]
    response_values = [rng.randint(0, 2) for _ in range(_N_TRIALS)]
    loads_values = [rng.choice([1, 2, 3]) for _ in range(_N_TRIALS)]
    in_out_values = [rng.choice([0, 1]) for _ in range(_N_TRIALS)]

    trial_data = {
        'start_time': start_times,
        'stop_time': stop_times,
        'stimulus_image': stimulus_images,
        'response_value': response_values,
        'loads': loads_values,
        'in_out': in_out_values,
    }

    def trial_col_getitem(self, key):
        col = MagicMock()
        data = trial_data[key]
        col.__getitem__ = lambda self, idx: data[idx]
        col.__len__ = lambda self: len(data)
        return col

    trials.__getitem__ = trial_col_getitem
    trials.__len__ = lambda self: _N_TRIALS
    nwbfile.trials = trials

    return nwbfile


@pytest.fixture
def mock_nwb():
    return _make_mock_nwb()


@pytest.fixture
def patch_open_nwb(mock_nwb):
    with patch('human_wm.data.nwb_loader._open_nwb', return_value=mock_nwb):
        yield


# ---------------------------------------------------------------------------
# _classify_region tests
# ---------------------------------------------------------------------------

class TestClassifyRegion:
    """Test schema-driven region classification (NO hardcoded patterns)."""

    def test_hippocampus_is_mtl(self):
        assert nwb_loader._classify_region('hippocampus', _TEST_SCHEMA) == 'mtl'

    def test_amygdala_is_mtl(self):
        assert nwb_loader._classify_region('amygdala', _TEST_SCHEMA) == 'mtl'

    def test_dacc_is_frontal(self):
        assert nwb_loader._classify_region('dACC', _TEST_SCHEMA) == 'frontal'

    def test_presma_is_frontal(self):
        assert nwb_loader._classify_region('pre-SMA', _TEST_SCHEMA) == 'frontal'

    def test_vmpfc_is_frontal(self):
        assert nwb_loader._classify_region('vmPFC', _TEST_SCHEMA) == 'frontal'

    def test_unknown_is_other(self):
        assert nwb_loader._classify_region('thalamus', _TEST_SCHEMA) == 'other'
        assert nwb_loader._classify_region('cerebellum', _TEST_SCHEMA) == 'other'

    def test_case_insensitive(self):
        assert nwb_loader._classify_region('Hippocampus', _TEST_SCHEMA) == 'mtl'
        assert nwb_loader._classify_region('AMYGDALA', _TEST_SCHEMA) == 'mtl'
        assert nwb_loader._classify_region('dacc', _TEST_SCHEMA) == 'frontal'
        assert nwb_loader._classify_region('PRE-SMA', _TEST_SCHEMA) == 'frontal'
        assert nwb_loader._classify_region('VMPFC', _TEST_SCHEMA) == 'frontal'


# ---------------------------------------------------------------------------
# extract_patient_data tests
# ---------------------------------------------------------------------------

class TestExtractPatientData:
    """Test binned spike extraction with MTL/Frontal separation."""

    def test_returns_correct_shapes(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)

        assert X.ndim == 3
        assert Y.ndim == 3

        # n_trials
        assert X.shape[0] == _N_TRIALS
        assert Y.shape[0] == _N_TRIALS

        # n_mtl == 8 (5 hippocampus + 3 amygdala)
        assert X.shape[2] == 8

        # n_frontal == 6 (2 dACC + 2 pre-SMA + 2 vmPFC)
        assert Y.shape[2] == 6

    def test_bin_count(self, patch_open_nwb):
        X, Y, _ = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)

        # 2.0 s / 0.05 s = 40 bins
        assert X.shape[1] == 40
        assert Y.shape[1] == 40

    def test_dtype_float32(self, patch_open_nwb):
        X, Y, _ = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_trial_info_has_keys(self, patch_open_nwb):
        _, _, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        for col in _TEST_SCHEMA['trial_columns']:
            assert col in trial_info, f'Missing trial_info key: {col}'

    def test_trial_info_lengths(self, patch_open_nwb):
        _, _, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        for col in _TEST_SCHEMA['trial_columns']:
            assert len(trial_info[col]) == _N_TRIALS

    def test_spike_counts_non_negative(self, patch_open_nwb):
        X, Y, _ = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        assert np.all(X >= 0)
        assert np.all(Y >= 0)


# ---------------------------------------------------------------------------
# split_data tests
# ---------------------------------------------------------------------------

class TestSplitData:
    """Test train/val/test splitting."""

    def test_split_sizes_sum_to_n(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        splits = nwb_loader.split_data(X, Y, trial_info, seed=42)

        n_train = splits['train']['X'].shape[0]
        n_val = splits['val']['X'].shape[0]
        n_test = splits['test']['X'].shape[0]

        assert n_train + n_val + n_test == _N_TRIALS

    def test_has_correct_keys(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        splits = nwb_loader.split_data(X, Y, trial_info, seed=42)

        assert set(splits.keys()) == {'train', 'val', 'test'}
        for split_name in ('train', 'val', 'test'):
            assert 'X' in splits[split_name]
            assert 'Y' in splits[split_name]
            assert 'trial_info' in splits[split_name]

    def test_split_shapes_consistent(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        splits = nwb_loader.split_data(X, Y, trial_info, seed=42)

        for split_name in ('train', 'val', 'test'):
            s = splits[split_name]
            n = s['X'].shape[0]
            assert s['Y'].shape[0] == n
            assert s['X'].shape[1:] == X.shape[1:]
            assert s['Y'].shape[1:] == Y.shape[1:]

    def test_reproducible_with_same_seed(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        s1 = nwb_loader.split_data(X, Y, trial_info, seed=42)
        s2 = nwb_loader.split_data(X, Y, trial_info, seed=42)

        np.testing.assert_array_equal(s1['train']['X'], s2['train']['X'])
        np.testing.assert_array_equal(s1['val']['X'], s2['val']['X'])
        np.testing.assert_array_equal(s1['test']['X'], s2['test']['X'])

    def test_different_seed_gives_different_split(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        s1 = nwb_loader.split_data(X, Y, trial_info, seed=42)
        s2 = nwb_loader.split_data(X, Y, trial_info, seed=99)

        # Extremely unlikely to be identical with different seeds
        assert not np.array_equal(s1['train']['X'], s2['train']['X'])

    def test_approximate_split_fractions(self, patch_open_nwb):
        X, Y, trial_info = nwb_loader.extract_patient_data('fake.nwb', _TEST_SCHEMA)
        splits = nwb_loader.split_data(X, Y, trial_info, seed=42)

        n_train = splits['train']['X'].shape[0]
        n_val = splits['val']['X'].shape[0]
        n_test = splits['test']['X'].shape[0]

        # Check approximate fractions (within rounding tolerance)
        assert abs(n_train / _N_TRIALS - 0.7) < 0.05
        assert abs(n_val / _N_TRIALS - 0.15) < 0.05
        assert abs(n_test / _N_TRIALS - 0.15) < 0.05
