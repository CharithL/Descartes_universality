"""
Tests for human_wm.surrogate.train -- training loop and related utilities.

Organised into two sections:
    1. Pure-Python tests (no torch import) for _detect_condition_column
    2. Torch-dependent tests for create_dataloader, train_surrogate,
       compute_cross_condition_cc, extract_hidden_states

All torch-dependent tests are grouped under a single class and
skip gracefully if torch is not importable (e.g. DLL issue on Windows
pytest collection).  To run torch tests directly:

    python -c "import tests.human_wm.test_train"

or:

    python tests/human_wm/test_train.py
"""

from __future__ import annotations

import numpy as np
import pytest


# ===================================================================
# 1. _detect_condition_column (no torch dependency)
# ===================================================================


class TestDetectConditionColumn:
    """Tests for the condition-column auto-detection heuristic.

    _detect_condition_column searches trial_info keys against a
    priority-ordered list: 'in_set', 'match', 'correct', 'set_size',
    'category', 'condition'. Returns the first one found with
    2..20 unique values.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        """Import the private function once."""
        from human_wm.surrogate.train import _detect_condition_column
        self._detect = _detect_condition_column

    def test_prefers_in_set_over_set_size(self):
        """'in_set' has higher priority than 'set_size'."""
        info = {
            'in_set': np.array([0, 1, 0, 1, 0]),
            'set_size': np.array([1, 2, 3, 1, 2]),
        }
        assert self._detect(info) == 'in_set'

    def test_falls_back_to_set_size(self):
        """When no higher-priority column exists, 'set_size' is used."""
        info = {
            'set_size': np.array([1, 2, 3, 1, 2]),
            'other': np.arange(5),
        }
        assert self._detect(info) == 'set_size'

    def test_skips_single_value_column(self):
        """A column with only one unique value is not categorical."""
        info = {
            'in_set': np.ones(10, dtype=int),  # only 1 unique
            'set_size': np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1]),
        }
        assert self._detect(info) == 'set_size'

    def test_skips_too_many_unique_values(self):
        """A column with >20 unique values is too fine-grained."""
        info = {
            'in_set': np.arange(25),  # 25 unique = too many
            'match': np.array([0, 1] * 12 + [0]),  # 2 unique
        }
        assert self._detect(info) == 'match'

    def test_returns_none_when_no_suitable_column(self):
        """Returns None when no column has 2..20 unique values."""
        info = {
            'unique_id': np.arange(100),  # 100 unique
            'constant': np.zeros(100, dtype=int),  # 1 unique
        }
        assert self._detect(info) is None

    def test_empty_trial_info(self):
        """Empty dict returns None."""
        assert self._detect({}) is None

    def test_fallback_to_non_priority_column(self):
        """When no priority column matches, falls back to any column
        with 2..20 unique values."""
        info = {
            'my_custom_condition': np.array([0, 1, 2, 0, 1, 2]),
        }
        assert self._detect(info) == 'my_custom_condition'

    def test_priority_column_with_exactly_two_values(self):
        """Binary column (2 unique values) at the boundary is accepted."""
        info = {
            'correct': np.array([0, 1, 0, 1, 1, 0]),
        }
        assert self._detect(info) == 'correct'

    def test_priority_column_with_twenty_values(self):
        """Column with exactly 20 unique values is accepted."""
        info = {
            'category': np.tile(np.arange(20), 3),  # 20 unique
        }
        assert self._detect(info) == 'category'

    def test_priority_column_with_twentyone_values(self):
        """Column with exactly 21 unique values is rejected."""
        info = {
            'category': np.tile(np.arange(21), 3),  # 21 unique
        }
        assert self._detect(info) is None


# ===================================================================
# 2. Torch-dependent tests
# ===================================================================

# Guard: these tests require torch to import successfully.
# On Windows machines where pytorch's DLL loading fails under pytest's
# import-rewriting, they'll be skipped.

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(
    not HAS_TORCH,
    reason="torch not available (DLL error or not installed)",
)


@needs_torch
class TestCreateDataloader:
    """Tests for create_dataloader: numpy → DataLoader conversion."""

    def test_returns_dataloader(self):
        from human_wm.surrogate.train import create_dataloader
        X = np.random.randn(20, 30, 5).astype(np.float32)
        Y = np.random.randn(20, 30, 3).astype(np.float32)
        dl = create_dataloader(X, Y, batch_size=8)
        assert hasattr(dl, '__iter__')

    def test_batch_shapes(self):
        from human_wm.surrogate.train import create_dataloader
        X = np.random.randn(20, 30, 5).astype(np.float32)
        Y = np.random.randn(20, 30, 3).astype(np.float32)
        dl = create_dataloader(X, Y, batch_size=8, shuffle=False)
        batch_x, batch_y = next(iter(dl))
        assert batch_x.shape == (8, 30, 5)
        assert batch_y.shape == (8, 30, 3)

    def test_total_samples_correct(self):
        from human_wm.surrogate.train import create_dataloader
        n = 17
        X = np.random.randn(n, 10, 4).astype(np.float32)
        Y = np.random.randn(n, 10, 2).astype(np.float32)
        dl = create_dataloader(X, Y, batch_size=5, shuffle=False)
        total = sum(bx.shape[0] for bx, _ in dl)
        assert total == n


@needs_torch
class TestTrainSurrogate:
    """Quick smoke tests for train_surrogate (very short training)."""

    def _make_splits(self, n_train=40, n_val=10, n_test=10,
                     n_bins=15, n_in=5, n_out=3):
        """Create synthetic split data."""
        rng = np.random.default_rng(42)
        def _make(n):
            return {
                'X': rng.standard_normal((n, n_bins, n_in)).astype(np.float32),
                'Y': rng.standard_normal((n, n_bins, n_out)).astype(np.float32),
                'trial_info': {
                    'in_set': rng.choice([0, 1], size=n),
                    'set_size': rng.choice([1, 2, 3], size=n),
                },
            }
        return {
            'train': _make(n_train),
            'val': _make(n_val),
            'test': _make(n_test),
        }

    def test_training_reduces_loss(self, tmp_path):
        """Training for a few epochs should reduce the loss."""
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import (
            create_dataloader, train_surrogate,
        )

        splits = self._make_splits()
        model = create_surrogate('lstm', 5, 3, 16)

        train_dl = create_dataloader(
            splits['train']['X'], splits['train']['Y'], batch_size=8,
        )
        val_dl = create_dataloader(
            splits['val']['X'], splits['val']['Y'],
            batch_size=8, shuffle=False,
        )

        model, history = train_surrogate(
            model, train_dl, val_dl,
            n_epochs=5, patience=10,
            save_path=tmp_path / 'test_model.pt',
        )

        assert len(history['train_loss']) >= 2
        assert len(history['val_loss']) >= 2
        # Training loss should decrease
        assert history['train_loss'][-1] < history['train_loss'][0]

    def test_returns_model_and_history(self, tmp_path):
        """train_surrogate returns (model, history)."""
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import (
            create_dataloader, train_surrogate,
        )

        splits = self._make_splits()
        model = create_surrogate('gru', 5, 3, 16)

        train_dl = create_dataloader(
            splits['train']['X'], splits['train']['Y'], batch_size=16,
        )
        val_dl = create_dataloader(
            splits['val']['X'], splits['val']['Y'],
            batch_size=16, shuffle=False,
        )

        result = train_surrogate(
            model, train_dl, val_dl,
            n_epochs=3, patience=10,
            save_path=tmp_path / 'model.pt',
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        model_out, history = result
        assert hasattr(model_out, 'forward')
        assert 'train_loss' in history
        assert 'val_loss' in history


@needs_torch
class TestExtractHiddenStates:
    """Tests for extract_hidden_states."""

    def test_output_shape(self):
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import extract_hidden_states

        model = create_surrogate('lstm', 5, 3, 16)
        X = np.random.randn(10, 20, 5).astype(np.float32)
        h = extract_hidden_states(model, X)

        assert h.shape == (10, 20, 16)
        assert h.dtype == np.float32 or h.dtype == np.float64

    def test_works_for_all_architectures(self):
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import extract_hidden_states

        X = np.random.randn(8, 15, 5).astype(np.float32)
        for arch in ['lstm', 'gru', 'transformer', 'linear']:
            model = create_surrogate(arch, 5, 3, 16)
            h = extract_hidden_states(model, X)
            assert h.shape == (8, 15, 16), f"Failed for {arch}"


@needs_torch
class TestComputeCrossConditionCC:
    """Tests for compute_cross_condition_cc."""

    def test_returns_tuple(self):
        """Should return (cc_float, column_name_or_none)."""
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import compute_cross_condition_cc

        model = create_surrogate('linear', 5, 3, 16)
        X = np.random.randn(20, 10, 5).astype(np.float32)
        Y = np.random.randn(20, 10, 3).astype(np.float32)
        info = {'in_set': np.random.choice([0, 1], size=20)}

        result = compute_cross_condition_cc(model, X, Y, info)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cc, col = result
        assert isinstance(cc, float)
        assert col == 'in_set'

    def test_nan_when_no_condition_column(self):
        """Returns NaN when no condition column is found."""
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import compute_cross_condition_cc

        model = create_surrogate('linear', 5, 3, 16)
        X = np.random.randn(20, 10, 5).astype(np.float32)
        Y = np.random.randn(20, 10, 3).astype(np.float32)
        info = {}

        cc, col = compute_cross_condition_cc(model, X, Y, info)
        assert np.isnan(cc)
        assert col is None

    def test_cc_is_finite(self):
        """With valid data and condition column, CC should be finite."""
        from human_wm.surrogate.models import create_surrogate
        from human_wm.surrogate.train import compute_cross_condition_cc

        model = create_surrogate('linear', 5, 3, 16)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 10, 5)).astype(np.float32)
        Y = rng.standard_normal((30, 10, 3)).astype(np.float32)
        info = {'set_size': rng.choice([1, 2, 3], size=30)}

        cc, col = compute_cross_condition_cc(model, X, Y, info)
        assert np.isfinite(cc)
        assert -1.0 <= cc <= 1.0


# ===================================================================
# 3. Run all tests as a script (workaround for Windows DLL issue)
# ===================================================================

if __name__ == '__main__':
    # Run with pytest if possible, otherwise print a message
    import sys
    sys.exit(pytest.main([__file__, '-v', '--tb=short']))
