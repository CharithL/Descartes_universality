"""Tests for human_wm.ablation.resample_ablation -- resample ablation protocol.

Tests cover the three public functions used in the architecture-agnostic
resample ablation pipeline:

    1. identify_top_k_dims   -- dimension-level correlation ranking
    2. resample_hidden_dims  -- marginal-preserving resampling
    3. classify_variable     -- DESCARTES zombie taxonomy classification

All tests use small synthetic numpy arrays (n_trials=30, hidden_size=16)
so they run in <1 second with no GPU or model dependencies.
"""

import numpy as np
import pytest

from human_wm.ablation.resample_ablation import (
    classify_variable,
    identify_top_k_dims,
    resample_hidden_dims,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_TRIALS = 30
HIDDEN_SIZE = 16


@pytest.fixture
def rng():
    """Deterministic random state for reproducible tests."""
    return np.random.RandomState(42)


@pytest.fixture
def hidden_states(rng):
    """Random hidden-state matrix (n_trials, hidden_size)."""
    return rng.randn(N_TRIALS, HIDDEN_SIZE).astype(np.float64)


# ---------------------------------------------------------------------------
# identify_top_k_dims
# ---------------------------------------------------------------------------

class TestIdentifyTopKDims:
    """Tests for identify_top_k_dims: finds the k dimensions most
    correlated with a target variable."""

    def test_returns_correct_number_of_dims(self, hidden_states):
        """Should return exactly k indices when k < hidden_size."""
        target = np.random.RandomState(0).randn(N_TRIALS)
        k = 5
        result = identify_top_k_dims(hidden_states, target, k)
        assert result.shape == (k,)

    def test_identifies_planted_signal(self):
        """When one dimension is perfectly correlated with the target,
        that dimension must appear first in the returned indices."""
        rng = np.random.RandomState(7)
        states = rng.randn(N_TRIALS, HIDDEN_SIZE)

        # Plant a strong linear signal in dimension 3
        target = rng.randn(N_TRIALS)
        states[:, 3] = target * 10.0  # near-perfect correlation

        top_dims = identify_top_k_dims(states, target, k=3)

        assert top_dims[0] == 3, (
            f"Planted signal dim (3) should be ranked first, got {top_dims}"
        )

    def test_identifies_multiple_planted_signals(self):
        """When two dimensions carry strong signal, both should be in
        the top-k result (order may vary between them)."""
        rng = np.random.RandomState(11)
        states = rng.randn(N_TRIALS, HIDDEN_SIZE)

        target = rng.randn(N_TRIALS)
        # Plant strong signals in dims 5 and 12
        states[:, 5] = target * 8.0 + rng.randn(N_TRIALS) * 0.1
        states[:, 12] = -target * 6.0 + rng.randn(N_TRIALS) * 0.1

        top_dims = identify_top_k_dims(states, target, k=4)
        top_set = set(top_dims.tolist())

        assert 5 in top_set, f"Planted dim 5 should be in top-4, got {top_dims}"
        assert 12 in top_set, f"Planted dim 12 should be in top-4, got {top_dims}"

    def test_k_clamped_to_hidden_size(self, hidden_states):
        """Requesting k larger than hidden_size should return all dims."""
        target = np.random.RandomState(0).randn(N_TRIALS)
        result = identify_top_k_dims(hidden_states, target, k=HIDDEN_SIZE + 100)
        assert result.shape == (HIDDEN_SIZE,)

    def test_k_clamped_to_one(self, hidden_states):
        """Requesting k=0 should be clamped to 1."""
        target = np.random.RandomState(0).randn(N_TRIALS)
        result = identify_top_k_dims(hidden_states, target, k=0)
        assert result.shape == (1,)

    def test_returns_integer_indices(self, hidden_states):
        """All returned indices must be valid integer dimension indices."""
        target = np.random.RandomState(0).randn(N_TRIALS)
        result = identify_top_k_dims(hidden_states, target, k=5)

        assert result.dtype in (np.int32, np.int64, np.intp)
        assert np.all(result >= 0)
        assert np.all(result < HIDDEN_SIZE)

    def test_constant_dimension_gets_zero_correlation(self):
        """Dimensions with zero variance should not appear among the
        top-ranked dimensions (they receive correlation = 0.0)."""
        rng = np.random.RandomState(19)
        states = rng.randn(N_TRIALS, HIDDEN_SIZE)

        # Make dimension 7 constant
        states[:, 7] = 5.0

        # Plant a signal in dim 0
        target = rng.randn(N_TRIALS)
        states[:, 0] = target * 10.0

        top_dims = identify_top_k_dims(states, target, k=3)

        assert 7 not in top_dims, (
            "Constant dimension should not appear in top dims"
        )


# ---------------------------------------------------------------------------
# resample_hidden_dims
# ---------------------------------------------------------------------------

class TestResampleHiddenDims:
    """Tests for resample_hidden_dims: replaces specified dimensions
    with values drawn from their empirical marginal distribution."""

    def test_output_shape_matches_input(self, hidden_states, rng):
        """Ablated output must have the same shape as the input."""
        dims = [0, 3, 7]
        result = resample_hidden_dims(hidden_states, dims, rng)
        assert result.shape == hidden_states.shape

    def test_non_ablated_dims_unchanged(self, hidden_states, rng):
        """Dimensions NOT in dims_to_ablate must be identical to input."""
        dims_to_ablate = [2, 5, 10]
        result = resample_hidden_dims(hidden_states, dims_to_ablate, rng)

        all_dims = set(range(HIDDEN_SIZE))
        kept_dims = sorted(all_dims - set(dims_to_ablate))

        np.testing.assert_array_equal(
            result[:, kept_dims],
            hidden_states[:, kept_dims],
            err_msg="Non-ablated dimensions should be unchanged",
        )

    def test_ablated_dims_differ(self, hidden_states, rng):
        """Ablated dimensions should (almost certainly) differ from
        the original values due to random resampling."""
        dims_to_ablate = [1, 4, 9]
        result = resample_hidden_dims(hidden_states, dims_to_ablate, rng)

        # At least one ablated column should differ (overwhelmingly likely
        # with 30 trials and random permutation).
        any_changed = False
        for d in dims_to_ablate:
            if not np.array_equal(result[:, d], hidden_states[:, d]):
                any_changed = True
                break

        assert any_changed, "At least one ablated dim should differ from original"

    def test_does_not_modify_original(self, hidden_states, rng):
        """The original hidden_states array must not be mutated."""
        original_copy = hidden_states.copy()
        dims = [0, 3]
        _ = resample_hidden_dims(hidden_states, dims, rng)

        np.testing.assert_array_equal(
            hidden_states, original_copy,
            err_msg="Original array should not be modified in-place",
        )

    def test_marginal_distribution_preserved(self):
        """The resampled values for each ablated dimension should come
        from the same empirical distribution as the original column.

        With enough samples, the set of unique values in the resampled
        column must be a subset of the original column's values.
        """
        rng_data = np.random.RandomState(55)
        states = rng_data.randn(N_TRIALS, HIDDEN_SIZE)
        rng_resample = np.random.RandomState(88)

        dims_to_ablate = [6, 11]
        result = resample_hidden_dims(states, dims_to_ablate, rng_resample)

        for d in dims_to_ablate:
            original_values = set(states[:, d].tolist())
            resampled_values = set(result[:, d].tolist())
            assert resampled_values.issubset(original_values), (
                f"Resampled dim {d} contains values not present in the "
                f"original column -- resampling should draw from the "
                f"empirical marginal distribution."
            )

    def test_empty_dims_returns_copy(self, hidden_states, rng):
        """When dims_to_ablate is empty, result should equal input."""
        result = resample_hidden_dims(hidden_states, [], rng)
        np.testing.assert_array_equal(result, hidden_states)

    def test_all_dims_ablated(self, hidden_states, rng):
        """Ablating all dimensions should still produce valid output
        with the correct shape."""
        all_dims = list(range(HIDDEN_SIZE))
        result = resample_hidden_dims(hidden_states, all_dims, rng)
        assert result.shape == hidden_states.shape

    def test_reproducibility_with_same_seed(self, hidden_states):
        """Two calls with the same RandomState seed should produce
        identical results."""
        dims = [2, 8, 14]
        result_a = resample_hidden_dims(
            hidden_states, dims, np.random.RandomState(99)
        )
        result_b = resample_hidden_dims(
            hidden_states, dims, np.random.RandomState(99)
        )
        np.testing.assert_array_equal(result_a, result_b)


# ---------------------------------------------------------------------------
# classify_variable
# ---------------------------------------------------------------------------

class TestClassifyVariable:
    """Tests for classify_variable: three-tier DESCARTES zombie taxonomy.

    Classification logic:
        ZOMBIE:         delta_r2 < 0.1  (DELTA_THRESHOLD_LEARNED)
        LEARNED_ZOMBIE: delta_r2 >= 0.1  AND  (no ablation or not causal)
        MANDATORY:      delta_r2 >= 0.1  AND  ablation shows causality
    """

    # -- ZOMBIE --------------------------------------------------------

    def test_zombie_low_delta_r2(self):
        """delta_r2 below threshold -> ZOMBIE regardless of ablation."""
        probe = {'delta_r2': 0.05}
        result = classify_variable(probe, None)
        assert result == 'ZOMBIE'

    def test_zombie_zero_delta_r2(self):
        """delta_r2 of exactly 0.0 -> ZOMBIE."""
        probe = {'delta_r2': 0.0}
        result = classify_variable(probe, None)
        assert result == 'ZOMBIE'

    def test_zombie_just_below_threshold(self):
        """delta_r2 just below 0.1 -> ZOMBIE."""
        probe = {'delta_r2': 0.099}
        result = classify_variable(probe, None)
        assert result == 'ZOMBIE'

    def test_zombie_ignores_causal_ablation(self):
        """Even if ablation says causal, a sub-threshold delta_r2
        should still produce ZOMBIE (probing gate comes first)."""
        probe = {'delta_r2': 0.05}
        ablation = {'any_causal': True}
        result = classify_variable(probe, ablation)
        assert result == 'ZOMBIE'

    def test_zombie_missing_delta_r2_key(self):
        """If the probe_result dict has no delta_r2 key, the function
        defaults to 0.0, which is below threshold -> ZOMBIE."""
        probe = {}
        result = classify_variable(probe, None)
        assert result == 'ZOMBIE'

    # -- LEARNED_ZOMBIE ------------------------------------------------

    def test_learned_zombie_no_ablation_result(self):
        """delta_r2 above threshold but ablation_result is None
        -> LEARNED_ZOMBIE."""
        probe = {'delta_r2': 0.25}
        result = classify_variable(probe, None)
        assert result == 'LEARNED_ZOMBIE'

    def test_learned_zombie_not_causal(self):
        """delta_r2 above threshold but any_causal=False
        -> LEARNED_ZOMBIE."""
        probe = {'delta_r2': 0.30}
        ablation = {'any_causal': False}
        result = classify_variable(probe, ablation)
        assert result == 'LEARNED_ZOMBIE'

    def test_learned_zombie_at_threshold(self):
        """delta_r2 exactly at threshold (0.1) with no causality
        -> LEARNED_ZOMBIE (threshold is strict less-than)."""
        probe = {'delta_r2': 0.1}
        ablation = {'any_causal': False}
        result = classify_variable(probe, ablation)
        assert result == 'LEARNED_ZOMBIE'

    def test_learned_zombie_missing_any_causal_key(self):
        """If ablation_result exists but lacks any_causal key,
        the .get default is False -> LEARNED_ZOMBIE."""
        probe = {'delta_r2': 0.5}
        ablation = {'min_z_score': -3.0}  # no 'any_causal' key
        result = classify_variable(probe, ablation)
        assert result == 'LEARNED_ZOMBIE'

    # -- MANDATORY -----------------------------------------------------

    def test_mandatory_causal(self):
        """delta_r2 above threshold AND any_causal=True -> MANDATORY."""
        probe = {'delta_r2': 0.25}
        ablation = {'any_causal': True}
        result = classify_variable(probe, ablation)
        assert result == 'MANDATORY'

    def test_mandatory_high_delta_r2(self):
        """Very high delta_r2 with causality -> MANDATORY."""
        probe = {'delta_r2': 0.85}
        ablation = {'any_causal': True}
        result = classify_variable(probe, ablation)
        assert result == 'MANDATORY'

    def test_mandatory_at_threshold(self):
        """delta_r2 exactly at threshold (0.1) with causality
        -> MANDATORY (0.1 is NOT less than 0.1, so it passes gate 1)."""
        probe = {'delta_r2': 0.1}
        ablation = {'any_causal': True}
        result = classify_variable(probe, ablation)
        assert result == 'MANDATORY'

    # -- Return type ---------------------------------------------------

    def test_return_type_is_string(self):
        """classify_variable should always return a string."""
        for probe, ablation in [
            ({'delta_r2': 0.01}, None),
            ({'delta_r2': 0.5}, None),
            ({'delta_r2': 0.5}, {'any_causal': True}),
            ({'delta_r2': 0.5}, {'any_causal': False}),
        ]:
            result = classify_variable(probe, ablation)
            assert isinstance(result, str)
            assert result in ('ZOMBIE', 'LEARNED_ZOMBIE', 'MANDATORY')
