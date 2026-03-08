"""
Tests for human_wm.targets.probe_targets

Covers all nine probe-target functions plus the master ``compute_all_targets``
dispatcher.  Uses small synthetic tensors (n_trials=20, n_timesteps=40,
n_neurons=8) with deterministic seeds so tests are fast and reproducible.

Test categories:
    1. Output shapes for every target function.
    2. Directional / magnitude checks (persistent_delay, delay_stability,
       mean_firing_rate, population_synchrony).
    3. Master function key coverage and graceful fallback on missing keys.
    4. Edge cases: zero-variance input, single-neuron tensor, missing
       trial_info keys.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from human_wm.targets.probe_targets import (
    compute_all_targets,
    compute_concept_selectivity,
    compute_delay_stability,
    compute_gamma_modulation,
    compute_mean_firing_rate,
    compute_memory_load_signal,
    compute_persistent_delay,
    compute_population_synchrony,
    compute_recognition_decision,
    compute_theta_modulation,
)

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

N_TRIALS = 20
N_TIMESTEPS = 40
N_NEURONS = 8
DELAY_BINS = slice(16, 28)
PROBE_BINS = slice(28, 40)
ENCODING_BINS = slice(0, 16)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture()
def Y(rng):
    """Standard synthetic activity tensor (n_trials, n_timesteps, n_neurons)."""
    return rng.standard_normal((N_TRIALS, N_TIMESTEPS, N_NEURONS)).astype(
        np.float64
    )


@pytest.fixture()
def trial_info(rng):
    """Minimal trial_info dict with all required keys.

    - set_size: integer memory load labels (1, 2, or 3).
    - in_set: binary match labels (0 or 1).
    - stimulus_id: categorical condition labels (0..4).
    """
    return {
        "set_size": rng.choice([1, 2, 3], size=N_TRIALS),
        "in_set": rng.choice([0, 1], size=N_TRIALS),
        "stimulus_id": rng.choice(5, size=N_TRIALS),
    }


@pytest.fixture()
def task_timing():
    """Standard task_timing dict with encoding / delay / probe slices."""
    return {
        "encoding_bins": ENCODING_BINS,
        "delay_bins": DELAY_BINS,
        "probe_bins": PROBE_BINS,
    }


@pytest.fixture()
def Y_zeros():
    """All-zeros activity tensor (zero-variance edge case)."""
    return np.zeros((N_TRIALS, N_TIMESTEPS, N_NEURONS), dtype=np.float64)


@pytest.fixture()
def Y_single_neuron(rng):
    """Activity tensor with exactly one neuron."""
    return rng.standard_normal((N_TRIALS, N_TIMESTEPS, 1)).astype(np.float64)


# ===================================================================
# 1. Output shape tests
# ===================================================================


class TestOutputShapes:
    """Every target function must return arrays with shape (n_trials,)."""

    def test_persistent_delay_shape(self, Y):
        """compute_persistent_delay returns (n_trials,)."""
        result = compute_persistent_delay(Y, DELAY_BINS)
        assert result.shape == (N_TRIALS,)

    def test_memory_load_signal_shape(self, Y, trial_info):
        """compute_memory_load_signal returns (n_trials,) signal and
        (n_neurons,) axis."""
        signal, axis = compute_memory_load_signal(Y, trial_info, DELAY_BINS)
        assert signal.shape == (N_TRIALS,)
        assert axis.shape == (N_NEURONS,)

    def test_delay_stability_shape(self, Y):
        """compute_delay_stability returns (n_trials,)."""
        result = compute_delay_stability(Y, DELAY_BINS)
        assert result.shape == (N_TRIALS,)

    def test_recognition_decision_shape(self, Y, trial_info):
        """compute_recognition_decision returns (n_trials,) signal and
        (n_neurons,) axis."""
        signal, axis = compute_recognition_decision(Y, trial_info, PROBE_BINS)
        assert signal.shape == (N_TRIALS,)
        assert axis.shape == (N_NEURONS,)

    def test_theta_modulation_shape(self, Y):
        """compute_theta_modulation returns (n_trials,)."""
        result = compute_theta_modulation(Y)
        assert result.shape == (N_TRIALS,)

    def test_gamma_modulation_shape(self, Y):
        """compute_gamma_modulation returns (n_trials,)."""
        result = compute_gamma_modulation(Y)
        assert result.shape == (N_TRIALS,)

    def test_population_synchrony_shape(self, Y):
        """compute_population_synchrony returns (n_trials,)."""
        result = compute_population_synchrony(Y)
        assert result.shape == (N_TRIALS,)

    def test_concept_selectivity_shape(self, Y, trial_info):
        """compute_concept_selectivity returns (n_trials,)."""
        result = compute_concept_selectivity(Y, trial_info, ENCODING_BINS)
        assert result.shape == (N_TRIALS,)

    def test_mean_firing_rate_shape(self, Y):
        """compute_mean_firing_rate returns (n_trials,)."""
        result = compute_mean_firing_rate(Y)
        assert result.shape == (N_TRIALS,)


# ===================================================================
# 2. Directional / magnitude checks
# ===================================================================


class TestPersistentDelay:
    """Trials with higher delay-period activity should yield higher values."""

    def test_higher_activity_gives_higher_value(self, rng):
        """Doubling the activity during the delay should roughly double the
        persistent-delay measure."""
        Y_low = rng.uniform(0, 1, (N_TRIALS, N_TIMESTEPS, N_NEURONS))
        Y_high = Y_low.copy()
        # Multiply delay period by 5
        Y_high[:, DELAY_BINS, :] *= 5.0

        low = compute_persistent_delay(Y_low, DELAY_BINS)
        high = compute_persistent_delay(Y_high, DELAY_BINS)

        # Every trial in the high condition should exceed the low condition
        assert np.all(high > low)

    def test_zero_input_returns_zero(self, Y_zeros):
        """Zero activity should give exactly zero persistent delay."""
        result = compute_persistent_delay(Y_zeros, DELAY_BINS)
        np.testing.assert_array_equal(result, 0.0)


class TestDelayStability:
    """A constant signal through the delay should yield high stability,
    whereas a random signal should yield lower stability on average."""

    def test_constant_signal_high_stability(self):
        """If every time-bin in the delay is identical across time, the
        first-half / second-half correlation should be 1.0 for each trial."""
        # Build a tensor where delay activity is constant over time
        Y = np.zeros((N_TRIALS, N_TIMESTEPS, N_NEURONS), dtype=np.float64)
        rng = np.random.default_rng(99)
        # Each trial gets a fixed spatial pattern repeated across delay bins
        patterns = rng.standard_normal((N_TRIALS, N_NEURONS))
        n_delay = DELAY_BINS.stop - DELAY_BINS.start
        for t in range(n_delay):
            Y[:, DELAY_BINS.start + t, :] = patterns

        stability = compute_delay_stability(Y, DELAY_BINS)

        # All trials should have correlation ~1.0
        np.testing.assert_allclose(stability, 1.0, atol=1e-8)

    def test_random_signal_lower_stability(self, rng):
        """Random noise during the delay should produce stability values
        well below 1.0 on average across trials."""
        Y = rng.standard_normal((N_TRIALS, N_TIMESTEPS, N_NEURONS))
        stability = compute_delay_stability(Y, DELAY_BINS)

        mean_stability = stability.mean()
        assert mean_stability < 0.9, (
            f"Expected mean stability < 0.9 for random noise, got {mean_stability}"
        )


class TestMeanFiringRate:
    """Mean firing rate should scale linearly with input magnitude."""

    def test_scaling(self, rng):
        """Multiplying Y by a constant should multiply the rate by the
        same constant."""
        Y_base = rng.uniform(1, 5, (N_TRIALS, N_TIMESTEPS, N_NEURONS))
        scale = 3.0

        rate_base = compute_mean_firing_rate(Y_base)
        rate_scaled = compute_mean_firing_rate(Y_base * scale)

        np.testing.assert_allclose(rate_scaled, rate_base * scale, rtol=1e-10)

    def test_nonneg_input_nonneg_output(self, rng):
        """Non-negative input should yield non-negative mean rates."""
        Y = rng.uniform(0, 10, (N_TRIALS, N_TIMESTEPS, N_NEURONS))
        rate = compute_mean_firing_rate(Y)
        assert np.all(rate >= 0)


class TestPopulationSynchrony:
    """Perfectly correlated neurons should yield high synchrony."""

    def test_identical_neurons_high_synchrony(self):
        """If all neurons have identical time-courses, pairwise |r| = 1."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal((N_TRIALS, N_TIMESTEPS, 1))
        # Tile so every neuron is identical
        Y = np.tile(base, (1, 1, N_NEURONS))

        sync = compute_population_synchrony(Y)

        np.testing.assert_allclose(sync, 1.0, atol=1e-8)

    def test_synchrony_nonneg(self, Y):
        """Synchrony (mean |r|) is always non-negative."""
        sync = compute_population_synchrony(Y)
        assert np.all(sync >= 0.0)

    def test_synchrony_bounded_by_one(self, Y):
        """Synchrony (mean |r|) should not exceed 1.0."""
        sync = compute_population_synchrony(Y)
        assert np.all(sync <= 1.0 + 1e-10)


# ===================================================================
# 3. compute_all_targets
# ===================================================================


class TestComputeAllTargets:
    """Master function should return a dict with the expected keys."""

    EXPECTED_ALWAYS_PRESENT = {
        "persistent_delay",
        "delay_stability",
        "theta_modulation",
        "gamma_modulation",
        "population_synchrony",
        "mean_firing_rate",
    }

    EXPECTED_WITH_FULL_INFO = EXPECTED_ALWAYS_PRESENT | {
        "memory_load",
        "recognition_decision",
        "concept_selectivity",
    }

    def test_all_keys_with_full_trial_info(
        self, Y, trial_info, task_timing
    ):
        """When trial_info has all needed keys, every target is present."""
        targets = compute_all_targets(Y, trial_info, task_timing)

        assert isinstance(targets, dict)
        assert self.EXPECTED_WITH_FULL_INFO.issubset(targets.keys()), (
            f"Missing keys: "
            f"{self.EXPECTED_WITH_FULL_INFO - targets.keys()}"
        )

    def test_all_values_have_correct_shape(
        self, Y, trial_info, task_timing
    ):
        """Every value in the targets dict should be (n_trials,)."""
        targets = compute_all_targets(Y, trial_info, task_timing)

        for name, arr in targets.items():
            assert arr.shape == (N_TRIALS,), (
                f"Target '{name}' has shape {arr.shape}, expected ({N_TRIALS},)"
            )

    def test_missing_trial_info_keys_still_returns_core(
        self, Y, task_timing
    ):
        """When trial_info is empty, targets that need it are skipped but
        core targets are still present."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            targets = compute_all_targets(Y, {}, task_timing)

        assert self.EXPECTED_ALWAYS_PRESENT.issubset(targets.keys())
        # These should be absent because their keys are missing
        assert "memory_load" not in targets
        assert "recognition_decision" not in targets
        assert "concept_selectivity" not in targets


# ===================================================================
# 4. Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge-case inputs: zero-variance, single neuron, missing keys."""

    # ---- zero-variance input ----

    def test_persistent_delay_zero_variance(self, Y_zeros):
        """Zero-variance tensor should return zeros without error."""
        result = compute_persistent_delay(Y_zeros, DELAY_BINS)
        np.testing.assert_array_equal(result, 0.0)

    def test_delay_stability_zero_variance(self, Y_zeros):
        """Zero-variance tensor should return zeros (safe_corrcoef path)."""
        result = compute_delay_stability(Y_zeros, DELAY_BINS)
        np.testing.assert_array_equal(result, 0.0)

    def test_population_synchrony_zero_variance(self, Y_zeros):
        """Zero-variance tensor should return zeros (excluded neurons)."""
        result = compute_population_synchrony(Y_zeros)
        np.testing.assert_array_equal(result, 0.0)

    def test_mean_firing_rate_zero_variance(self, Y_zeros):
        """Zero-variance tensor should return zeros."""
        result = compute_mean_firing_rate(Y_zeros)
        np.testing.assert_array_equal(result, 0.0)

    # ---- single neuron ----

    def test_persistent_delay_single_neuron(self, Y_single_neuron):
        """Single-neuron tensor should still produce (n_trials,)."""
        result = compute_persistent_delay(Y_single_neuron, DELAY_BINS)
        assert result.shape == (N_TRIALS,)

    def test_delay_stability_single_neuron(self, Y_single_neuron):
        """Single-neuron tensor: correlation is not well-defined, so
        _safe_corrcoef returns 0.0 (only one element in vector)."""
        result = compute_delay_stability(Y_single_neuron, DELAY_BINS)
        assert result.shape == (N_TRIALS,)
        # With one neuron, first/second half vectors have length 1, so
        # std is 0 -> safe_corrcoef returns 0
        np.testing.assert_array_equal(result, 0.0)

    def test_population_synchrony_single_neuron(self, Y_single_neuron):
        """Single-neuron tensor: cannot compute pairwise correlation,
        so synchrony should be 0."""
        result = compute_population_synchrony(Y_single_neuron)
        np.testing.assert_array_equal(result, 0.0)

    def test_mean_firing_rate_single_neuron(self, Y_single_neuron):
        """Single-neuron tensor should return the mean of that neuron."""
        result = compute_mean_firing_rate(Y_single_neuron)
        assert result.shape == (N_TRIALS,)

    # ---- missing trial_info keys ----

    def test_memory_load_missing_key_raises(self, Y):
        """compute_memory_load_signal should raise KeyError when trial_info
        has none of the expected load keys."""
        with pytest.raises(KeyError, match="set_size"):
            compute_memory_load_signal(Y, {"unrelated": [0]}, DELAY_BINS)

    def test_recognition_decision_missing_key_raises(self, Y):
        """compute_recognition_decision should raise KeyError when
        trial_info has none of the expected match keys."""
        with pytest.raises(KeyError, match="in_set"):
            compute_recognition_decision(Y, {"unrelated": [0]}, PROBE_BINS)

    def test_concept_selectivity_missing_key_raises(self, Y):
        """compute_concept_selectivity should raise KeyError when
        trial_info has none of the expected condition keys."""
        with pytest.raises(KeyError, match="stimulus_id"):
            compute_concept_selectivity(Y, {"unrelated": [0]}, ENCODING_BINS)


# ===================================================================
# 5. Memory load: trial_info key aliases
# ===================================================================


class TestMemoryLoad:
    """compute_memory_load_signal should accept multiple key aliases."""

    def test_set_size_key(self, Y, rng):
        """'set_size' key is accepted."""
        info = {"set_size": rng.choice([1, 2, 3], size=N_TRIALS)}
        signal, axis = compute_memory_load_signal(Y, info, DELAY_BINS)
        assert signal.shape == (N_TRIALS,)
        assert axis.shape == (N_NEURONS,)

    def test_load_key(self, Y, rng):
        """'load' key is accepted as an alias for set_size."""
        info = {"load": rng.choice([1, 2, 3], size=N_TRIALS)}
        signal, axis = compute_memory_load_signal(Y, info, DELAY_BINS)
        assert signal.shape == (N_TRIALS,)

    def test_n_items_key(self, Y, rng):
        """'n_items' key is accepted as an alias for set_size."""
        info = {"n_items": rng.choice([1, 2, 3], size=N_TRIALS)}
        signal, axis = compute_memory_load_signal(Y, info, DELAY_BINS)
        assert signal.shape == (N_TRIALS,)

    def test_load_axis_is_unit_norm(self, Y, trial_info):
        """The returned load axis should have unit L2 norm (unless zero)."""
        _, axis = compute_memory_load_signal(Y, trial_info, DELAY_BINS)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)


# ===================================================================
# 6. Recognition decision: trial_info key aliases
# ===================================================================


class TestRecognitionDecision:
    """compute_recognition_decision should accept multiple key aliases."""

    def test_in_set_key(self, Y, rng):
        """'in_set' key is accepted."""
        info = {"in_set": rng.choice([0, 1], size=N_TRIALS)}
        signal, axis = compute_recognition_decision(Y, info, PROBE_BINS)
        assert signal.shape == (N_TRIALS,)
        assert axis.shape == (N_NEURONS,)

    def test_match_key(self, Y, rng):
        """'match' key is accepted as an alias."""
        info = {"match": rng.choice([0, 1], size=N_TRIALS)}
        signal, axis = compute_recognition_decision(Y, info, PROBE_BINS)
        assert signal.shape == (N_TRIALS,)

    def test_probe_match_key(self, Y, rng):
        """'probe_match' key is accepted as an alias."""
        info = {"probe_match": rng.choice([0, 1], size=N_TRIALS)}
        signal, axis = compute_recognition_decision(Y, info, PROBE_BINS)
        assert signal.shape == (N_TRIALS,)

    def test_decision_axis_is_unit_norm(self, Y, trial_info):
        """The returned decision axis should have unit L2 norm
        (unless zero)."""
        _, axis = compute_recognition_decision(Y, trial_info, PROBE_BINS)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_single_condition_returns_zeros(self, Y):
        """When all trials have the same label, the function should warn
        and return zeros."""
        info = {"in_set": np.ones(N_TRIALS, dtype=int)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            signal, axis = compute_recognition_decision(Y, info, PROBE_BINS)
        np.testing.assert_array_equal(signal, 0.0)
        np.testing.assert_array_equal(axis, 0.0)


# ===================================================================
# 7. Oscillatory modulation
# ===================================================================


class TestOscillatoryModulation:
    """Theta and gamma modulation edge cases."""

    def test_theta_nonneg(self, Y):
        """Theta amplitude should be non-negative."""
        amp = compute_theta_modulation(Y)
        assert np.all(amp >= 0.0)

    def test_gamma_default_bins_returns_zeros_with_warning(self, Y):
        """With default 50 ms bins (20 Hz sampling), gamma (30-80 Hz) is
        above Nyquist.  Should return zeros and emit a RuntimeWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)
            amp = compute_gamma_modulation(Y, bin_size_s=0.05)

        np.testing.assert_array_equal(amp, 0.0)
        # Check that a Nyquist-related warning was raised
        nyquist_warnings = [
            x for x in w
            if issubclass(x.category, RuntimeWarning) and "Nyquist" in str(x.message)
        ]
        assert len(nyquist_warnings) > 0, (
            "Expected a RuntimeWarning about Nyquist frequency"
        )

    def test_gamma_fine_bins_returns_nonzero(self, rng):
        """With sufficiently fine bins, gamma modulation should be
        computable (non-zero for non-trivial input)."""
        # Use bin_size_s=0.001 (1 kHz sampling) so Nyquist = 500 Hz
        n_timesteps_fine = 500
        Y_fine = rng.standard_normal(
            (N_TRIALS, n_timesteps_fine, N_NEURONS)
        )
        amp = compute_gamma_modulation(
            Y_fine, bin_size_s=0.001, gamma_band=(30, 80)
        )
        assert amp.shape == (N_TRIALS,)
        # At least some trials should have non-zero amplitude
        assert np.any(amp > 0.0)

    def test_theta_zero_input(self, Y_zeros):
        """Zero input should give zero theta amplitude."""
        amp = compute_theta_modulation(Y_zeros)
        np.testing.assert_array_equal(amp, 0.0)


# ===================================================================
# 8. Concept selectivity
# ===================================================================


class TestConceptSelectivity:
    """Tests for compute_concept_selectivity."""

    def test_shape(self, Y, trial_info):
        """Output shape should be (n_trials,)."""
        result = compute_concept_selectivity(Y, trial_info, ENCODING_BINS)
        assert result.shape == (N_TRIALS,)

    def test_single_condition_returns_zeros(self, Y):
        """With only one stimulus condition, selectivity should be zero."""
        info = {"stimulus_id": np.zeros(N_TRIALS, dtype=int)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = compute_concept_selectivity(Y, info, ENCODING_BINS)
        np.testing.assert_array_equal(result, 0.0)

    def test_category_key_alias(self, Y, rng):
        """'category' should be accepted as an alias for stimulus_id."""
        info = {"category": rng.choice(3, size=N_TRIALS)}
        result = compute_concept_selectivity(Y, info, ENCODING_BINS)
        assert result.shape == (N_TRIALS,)

    def test_nonneg_output(self, Y, trial_info):
        """F-statistics are non-negative, so selectivity should be too."""
        result = compute_concept_selectivity(Y, trial_info, ENCODING_BINS)
        assert np.all(result >= 0.0)
