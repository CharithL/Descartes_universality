"""Tests for human_wm.data.patient_inventory -- session discovery and filtering."""

import numpy as np
import pytest
from unittest.mock import patch

from human_wm.data.patient_inventory import (
    build_inventory,
    get_best_patient,
    get_usable_patients,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA = {
    'region_column': 'brain_region',
    'mtl_regions': ['hippocampus', 'amygdala'],
    'frontal_regions': ['dACC', 'pre-SMA', 'vmPFC'],
    'trial_columns': ['start_time', 'stop_time'],
}


def _make_mock_extract(n_mtl, n_frontal, n_trials):
    """Create mock return values for extract_patient_data."""
    n_bins = 40
    X = np.zeros((n_trials, n_bins, n_mtl), dtype=np.float32)
    Y = np.zeros((n_trials, n_bins, n_frontal), dtype=np.float32)
    trial_info = {
        'start_time': np.arange(n_trials, dtype=float),
        'stop_time': np.arange(n_trials, dtype=float) + 2.0,
    }
    return X, Y, trial_info


# ---------------------------------------------------------------------------
# build_inventory tests
# ---------------------------------------------------------------------------

class TestBuildInventory:
    """Test inventory construction from NWB paths."""

    def test_returns_list_of_dicts(self):
        def mock_extract(path, schema):
            return _make_mock_extract(12, 15, 60)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['p1.nwb', 'p2.nwb'], _SCHEMA)

        assert isinstance(inventory, list)
        assert len(inventory) == 2
        assert all(isinstance(e, dict) for e in inventory)

    def test_has_correct_keys(self):
        def mock_extract(path, schema):
            return _make_mock_extract(12, 15, 60)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['p1.nwb'], _SCHEMA)

        required_keys = {'patient_id', 'path', 'n_mtl', 'n_frontal', 'n_trials', 'usable'}
        assert required_keys == set(inventory[0].keys())

    def test_usable_true_when_above_thresholds(self):
        """n_mtl=12 >= 10, n_frontal=15 >= 10, n_trials=60 >= 50 -> usable."""
        def mock_extract(path, schema):
            return _make_mock_extract(12, 15, 60)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['good.nwb'], _SCHEMA)

        assert inventory[0]['usable'] is True
        assert inventory[0]['n_mtl'] == 12
        assert inventory[0]['n_frontal'] == 15
        assert inventory[0]['n_trials'] == 60

    def test_usable_false_when_mtl_below_threshold(self):
        """n_mtl=5 < 10 -> not usable."""
        def mock_extract(path, schema):
            return _make_mock_extract(5, 15, 60)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['bad_mtl.nwb'], _SCHEMA)

        assert inventory[0]['usable'] is False

    def test_usable_false_when_frontal_below_threshold(self):
        """n_frontal=3 < 10 -> not usable."""
        def mock_extract(path, schema):
            return _make_mock_extract(12, 3, 60)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['bad_frontal.nwb'], _SCHEMA)

        assert inventory[0]['usable'] is False

    def test_usable_false_when_trials_below_threshold(self):
        """n_trials=30 < 50 -> not usable."""
        def mock_extract(path, schema):
            return _make_mock_extract(12, 15, 30)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['few_trials.nwb'], _SCHEMA)

        assert inventory[0]['usable'] is False

    def test_usable_true_at_exact_thresholds(self):
        """n_mtl=10, n_frontal=10, n_trials=50 -> usable (>= not >)."""
        def mock_extract(path, schema):
            return _make_mock_extract(10, 10, 50)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['edge.nwb'], _SCHEMA)

        assert inventory[0]['usable'] is True

    def test_handles_extraction_failure_gracefully(self):
        """If extract_patient_data raises, entry is marked not usable."""
        def mock_extract(path, schema):
            raise RuntimeError('corrupt file')

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['corrupt.nwb'], _SCHEMA)

        assert len(inventory) == 1
        assert inventory[0]['usable'] is False
        assert inventory[0]['n_mtl'] == 0


# ---------------------------------------------------------------------------
# get_best_patient tests
# ---------------------------------------------------------------------------

class TestGetBestPatient:
    """Test best patient selection (max min(n_mtl, n_frontal))."""

    def test_returns_patient_with_most_neurons(self):
        # patient_a: min(12, 15) = 12
        # patient_b: min(20, 10) = 10
        # patient_c: min(15, 18) = 15  <- best
        configs = [
            ('a.nwb', 12, 15, 60),
            ('b.nwb', 20, 10, 60),
            ('c.nwb', 15, 18, 60),
        ]
        call_idx = [0]

        def mock_extract(path, schema):
            idx = call_idx[0]
            call_idx[0] += 1
            _, n_mtl, n_frontal, n_trials = configs[idx]
            return _make_mock_extract(n_mtl, n_frontal, n_trials)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory([c[0] for c in configs], _SCHEMA)

        best = get_best_patient(inventory)
        assert best is not None
        assert best['patient_id'] == 'c'

    def test_returns_none_when_none_usable(self):
        def mock_extract(path, schema):
            return _make_mock_extract(3, 4, 20)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['bad.nwb'], _SCHEMA)

        assert get_best_patient(inventory) is None

    def test_returns_none_for_empty_inventory(self):
        assert get_best_patient([]) is None


# ---------------------------------------------------------------------------
# get_usable_patients tests
# ---------------------------------------------------------------------------

class TestGetUsablePatients:
    """Test filtering to only usable patients."""

    def test_filters_correctly(self):
        configs = [
            ('good.nwb', 12, 15, 60),    # usable
            ('bad.nwb', 5, 3, 20),        # not usable
            ('also_good.nwb', 10, 10, 50), # usable (at threshold)
        ]
        call_idx = [0]

        def mock_extract(path, schema):
            idx = call_idx[0]
            call_idx[0] += 1
            _, n_mtl, n_frontal, n_trials = configs[idx]
            return _make_mock_extract(n_mtl, n_frontal, n_trials)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory([c[0] for c in configs], _SCHEMA)

        usable = get_usable_patients(inventory)
        assert len(usable) == 2
        ids = {p['patient_id'] for p in usable}
        assert ids == {'good', 'also_good'}

    def test_returns_empty_when_none_usable(self):
        def mock_extract(path, schema):
            return _make_mock_extract(3, 4, 20)

        with patch(
            'human_wm.data.patient_inventory.extract_patient_data',
            side_effect=mock_extract,
        ):
            inventory = build_inventory(['bad.nwb'], _SCHEMA)

        assert get_usable_patients(inventory) == []

    def test_returns_empty_for_empty_inventory(self):
        assert get_usable_patients([]) == []
