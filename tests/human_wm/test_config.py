"""Tests for human_wm.config -- constants, paths, and schema loader."""

import json
import tempfile
from pathlib import Path

import pytest

from human_wm import config


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

class TestPaths:
    """Verify all directory path constants are well-formed."""

    def test_project_root_is_path(self):
        assert isinstance(config.PROJECT_ROOT, Path)

    def test_data_dir(self):
        assert config.DATA_DIR == config.PROJECT_ROOT / 'data'

    def test_raw_nwb_dir(self):
        assert config.RAW_NWB_DIR == config.DATA_DIR / 'raw'

    def test_processed_dir(self):
        assert config.PROCESSED_DIR == config.DATA_DIR / 'processed'

    def test_surrogate_dir(self):
        assert config.SURROGATE_DIR == config.DATA_DIR / 'surrogates'

    def test_results_dir(self):
        assert config.RESULTS_DIR == config.DATA_DIR / 'results'

    def test_hidden_dir(self):
        assert config.HIDDEN_DIR == config.SURROGATE_DIR / 'hidden'

    def test_nwb_schema_path(self):
        assert config.NWB_SCHEMA_PATH == config.DATA_DIR / 'nwb_schema.json'


# ---------------------------------------------------------------------------
# DANDI constants
# ---------------------------------------------------------------------------

class TestDandi:
    def test_dandiset_id(self):
        assert config.DANDISET_ID == '000576'


# ---------------------------------------------------------------------------
# Brain region patterns
# ---------------------------------------------------------------------------

class TestRegionPatterns:
    def test_mtl_patterns_is_list(self):
        assert isinstance(config.MTL_REGION_PATTERNS, list)
        assert len(config.MTL_REGION_PATTERNS) > 0

    def test_mtl_contains_hippocampus(self):
        assert 'hippocampus' in config.MTL_REGION_PATTERNS

    def test_mtl_contains_amygdala(self):
        assert 'amygdala' in config.MTL_REGION_PATTERNS

    def test_frontal_patterns_is_list(self):
        assert isinstance(config.FRONTAL_REGION_PATTERNS, list)
        assert len(config.FRONTAL_REGION_PATTERNS) > 0

    def test_frontal_contains_prefrontal(self):
        assert 'prefrontal' in config.FRONTAL_REGION_PATTERNS


# ---------------------------------------------------------------------------
# Neuron / trial thresholds
# ---------------------------------------------------------------------------

class TestThresholds:
    def test_min_mtl_neurons(self):
        assert config.MIN_MTL_NEURONS == 10

    def test_min_frontal_neurons(self):
        assert config.MIN_FRONTAL_NEURONS == 10

    def test_min_trials(self):
        assert config.MIN_TRIALS == 50


# ---------------------------------------------------------------------------
# Spike binning
# ---------------------------------------------------------------------------

class TestBinning:
    def test_bin_size_ms(self):
        assert config.BIN_SIZE_MS == 50

    def test_bin_size_s(self):
        assert config.BIN_SIZE_S == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

class TestSplits:
    def test_train_frac(self):
        assert config.TRAIN_FRAC == pytest.approx(0.7)

    def test_val_frac(self):
        assert config.VAL_FRAC == pytest.approx(0.15)

    def test_test_frac(self):
        assert config.TEST_FRAC == pytest.approx(0.15)

    def test_splits_sum_to_one(self):
        total = config.TRAIN_FRAC + config.VAL_FRAC + config.TEST_FRAC
        assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Surrogate training hyperparameters
# ---------------------------------------------------------------------------

class TestTrainingHyperparams:
    def test_hidden_sizes(self):
        assert config.HIDDEN_SIZES == [64, 128, 256]

    def test_n_lstm_layers(self):
        assert config.N_LSTM_LAYERS == 2

    def test_learning_rate(self):
        assert config.LEARNING_RATE == pytest.approx(1e-3)

    def test_weight_decay(self):
        assert config.WEIGHT_DECAY == pytest.approx(1e-5)

    def test_early_stop_patience(self):
        assert config.EARLY_STOP_PATIENCE == 20

    def test_max_epochs(self):
        assert config.MAX_EPOCHS == 200

    def test_batch_size(self):
        assert config.BATCH_SIZE == 32

    def test_grad_clip_norm(self):
        assert config.GRAD_CLIP_NORM == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_n_seeds(self):
        assert config.N_SEEDS == 10

    def test_architectures(self):
        assert config.ARCHITECTURES == ['lstm', 'gru', 'transformer', 'linear']

    def test_architectures_length(self):
        assert len(config.ARCHITECTURES) == 4


# ---------------------------------------------------------------------------
# Oscillation bands
# ---------------------------------------------------------------------------

class TestOscillationBands:
    def test_theta_band(self):
        assert config.THETA_BAND == (4, 8)

    def test_gamma_band(self):
        assert config.GAMMA_BAND == (30, 80)


# ---------------------------------------------------------------------------
# Probe targets
# ---------------------------------------------------------------------------

class TestTargets:
    def test_level_b_targets(self):
        assert config.LEVEL_B_TARGETS == [
            'persistent_delay',
            'memory_load',
            'delay_stability',
            'recognition_decision',
        ]

    def test_level_c_targets(self):
        assert config.LEVEL_C_TARGETS == [
            'theta_modulation',
            'gamma_modulation',
            'population_synchrony',
        ]

    def test_level_a_targets(self):
        assert config.LEVEL_A_TARGETS == [
            'concept_selectivity',
            'mean_firing_rate',
        ]

    def test_all_targets_composition(self):
        expected = (
            config.LEVEL_B_TARGETS
            + config.LEVEL_C_TARGETS
            + config.LEVEL_A_TARGETS
        )
        assert config.ALL_TARGETS == expected

    def test_all_targets_length(self):
        assert len(config.ALL_TARGETS) == 9


# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

class TestQualityThresholds:
    def test_min_cc(self):
        assert config.MIN_CC_THRESHOLD == pytest.approx(0.3)

    def test_good_cc(self):
        assert config.GOOD_CC_THRESHOLD == pytest.approx(0.5)

    def test_strong_cc(self):
        assert config.STRONG_CC_THRESHOLD == pytest.approx(0.7)

    def test_threshold_ordering(self):
        assert (
            config.MIN_CC_THRESHOLD
            < config.GOOD_CC_THRESHOLD
            < config.STRONG_CC_THRESHOLD
        )


# ---------------------------------------------------------------------------
# NWB schema loader
# ---------------------------------------------------------------------------

class TestLoadNwbSchema:
    def test_missing_file_returns_none(self):
        result = config.load_nwb_schema('/nonexistent/path/schema.json')
        assert result is None

    def test_valid_json_returns_dict(self):
        schema = {'fields': ['spike_times', 'electrode_group'], 'version': 1}
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(schema, f)
            tmp_path = f.name
        try:
            result = config.load_nwb_schema(tmp_path)
            assert isinstance(result, dict)
            assert result == schema
        finally:
            Path(tmp_path).unlink()

    def test_invalid_json_returns_none(self):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            f.write('{not valid json!!!}')
            tmp_path = f.name
        try:
            result = config.load_nwb_schema(tmp_path)
            assert result is None
        finally:
            Path(tmp_path).unlink()

    def test_default_schema_path(self):
        # When no arg is given, should use NWB_SCHEMA_PATH (likely missing)
        # Should return None gracefully, not raise
        result = config.load_nwb_schema()
        assert result is None or isinstance(result, dict)
