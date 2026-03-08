# Human Universality Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `human_wm/` package to test whether mandatory variables from mouse WM are universal across 21 human patients, 10 random seeds, and 4 model architectures.

**Architecture:** LSTM/GRU/Transformer/Linear surrogates trained on MTL→Frontal transformation from Rutishauser dataset (DANDI 000576). Architecture-specific ablation dispatch. All NWB column names dynamically loaded from schema JSON.

**Tech Stack:** PyTorch, pynwb, dandi, scikit-learn, scipy, numpy

---

### Task 1: Package Scaffolding + Config

**Files:**
- `human_wm/__init__.py`
- `human_wm/config.py`
- `human_wm/data/__init__.py`
- `human_wm/surrogate/__init__.py`
- `human_wm/targets/__init__.py`
- `human_wm/analysis/__init__.py`
- `human_wm/ablation/__init__.py`
- `tests/human_wm/__init__.py`
- `tests/human_wm/test_config.py`

**Steps:**

1. Write `tests/human_wm/test_config.py`:

```python
"""Tests for human_wm.config -- constants, paths, and schema loader."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from human_wm.config import (
    ARCHITECTURES,
    BIN_SIZE_MS,
    BATCH_SIZE,
    DANDISET_ID,
    DATA_DIR,
    EARLY_STOP_PATIENCE,
    GAMMA_BAND,
    GRAD_CLIP_NORM,
    HIDDEN_SIZES,
    LEARNING_RATE,
    LEVEL_A_TARGETS,
    LEVEL_B_TARGETS,
    LEVEL_C_TARGETS,
    MAX_EPOCHS,
    MIN_FRONTAL_NEURONS,
    MIN_MTL_NEURONS,
    MIN_TRIALS,
    N_LSTM_LAYERS,
    N_SEEDS,
    NWB_SCHEMA_PATH,
    PROJECT_ROOT,
    THETA_BAND,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
    WEIGHT_DECAY,
    load_nwb_schema,
)


class TestPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists()

    def test_data_dir_is_under_project(self):
        assert str(DATA_DIR).startswith(str(PROJECT_ROOT))

    def test_schema_path_is_path(self):
        assert isinstance(NWB_SCHEMA_PATH, Path)


class TestConstants:
    def test_dandiset_id(self):
        assert DANDISET_ID == '000576'

    def test_hidden_sizes(self):
        assert HIDDEN_SIZES == [64, 128, 256]

    def test_n_lstm_layers(self):
        assert N_LSTM_LAYERS == 2

    def test_learning_rate(self):
        assert LEARNING_RATE == 1e-3

    def test_weight_decay(self):
        assert WEIGHT_DECAY == 1e-5

    def test_early_stop_patience(self):
        assert EARLY_STOP_PATIENCE == 20

    def test_max_epochs(self):
        assert MAX_EPOCHS == 200

    def test_batch_size(self):
        assert BATCH_SIZE == 32

    def test_grad_clip_norm(self):
        assert GRAD_CLIP_NORM == 1.0

    def test_min_mtl_neurons(self):
        assert MIN_MTL_NEURONS == 10

    def test_min_frontal_neurons(self):
        assert MIN_FRONTAL_NEURONS == 10

    def test_min_trials(self):
        assert MIN_TRIALS == 50

    def test_bin_size_ms(self):
        assert BIN_SIZE_MS == 50

    def test_theta_band(self):
        assert THETA_BAND == (4, 8)

    def test_gamma_band(self):
        assert GAMMA_BAND == (30, 80)

    def test_n_seeds(self):
        assert N_SEEDS == 10

    def test_architectures(self):
        assert ARCHITECTURES == ['lstm', 'gru', 'transformer', 'linear']

    def test_splits_sum_to_one(self):
        assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9


class TestLevelTargets:
    def test_level_b_targets(self):
        assert 'persistent_delay' in LEVEL_B_TARGETS
        assert 'memory_load' in LEVEL_B_TARGETS
        assert 'delay_stability' in LEVEL_B_TARGETS
        assert 'recognition_decision' in LEVEL_B_TARGETS

    def test_level_c_targets(self):
        assert 'theta_modulation' in LEVEL_C_TARGETS
        assert 'gamma_modulation' in LEVEL_C_TARGETS
        assert 'population_synchrony' in LEVEL_C_TARGETS

    def test_level_a_targets(self):
        assert 'concept_selectivity' in LEVEL_A_TARGETS
        assert 'mean_firing_rate' in LEVEL_A_TARGETS


class TestSchemaLoader:
    def test_missing_file_returns_none(self):
        result = load_nwb_schema(Path('/nonexistent/path/schema.json'))
        assert result is None

    def test_valid_file_returns_dict(self):
        schema = {
            'units_columns': ['spike_times', 'electrodes'],
            'region_column': 'brain_region',
            'trial_columns': ['start_time', 'stop_time'],
            'mtl_regions': ['hippocampus', 'amygdala'],
            'frontal_regions': ['dACC', 'pre-SMA', 'vmPFC'],
            'timing_columns': ['start_time', 'stop_time'],
            'n_units': 100,
            'n_trials': 200,
            'sample_values': {},
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(schema, f)
            tmp_path = Path(f.name)
        try:
            result = load_nwb_schema(tmp_path)
            assert isinstance(result, dict)
            assert result['region_column'] == 'brain_region'
            assert 'hippocampus' in result['mtl_regions']
            assert 'dACC' in result['frontal_regions']
        finally:
            os.unlink(tmp_path)
```

2. Run test -- expect import errors (module not created yet):
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_config.py -x 2>&1 | head -30
```

3. Implement all files:

`human_wm/__init__.py`:
```python
"""
DESCARTES Human Universality -- MTL->Frontal Zombie Test

Applies the DESCARTES methodology to the Rutishauser Human Single-Neuron
WM dataset (DANDI 000576) to test whether mandatory variables from mouse
working memory are universal across 21 human patients, 10 random seeds,
and 4 model architectures (LSTM, GRU, Transformer, Linear).
"""
```

`human_wm/data/__init__.py`:
```python
```

`human_wm/surrogate/__init__.py`:
```python
```

`human_wm/targets/__init__.py`:
```python
```

`human_wm/analysis/__init__.py`:
```python
```

`human_wm/ablation/__init__.py`:
```python
```

`tests/human_wm/__init__.py`:
```python
```

`human_wm/config.py`:
```python
"""
DESCARTES Human Universality -- Experiment-Specific Configuration

All hyperparameters, paths, and constants specific to the Rutishauser
Human Single-Neuron WM experiment (DANDI 000576). Methodology constants
(probing thresholds, ablation k-fractions) are inherited from
descartes_core.config.

Key difference from wm/config.py:
- MTL (hippocampus+amygdala) -> Frontal (dACC+pre-SMA+vmPFC)
- 4 architectures (LSTM, GRU, Transformer, Linear)
- 10 random seeds for cross-seed universality
- NWB column names loaded from schema JSON (NEVER hardcoded)
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_NWB_DIR = DATA_DIR / 'human_raw'
PROCESSED_DIR = DATA_DIR / 'human_processed'
SURROGATE_DIR = DATA_DIR / 'human_surrogates'
RESULTS_DIR = DATA_DIR / 'human_results'
HIDDEN_DIR = SURROGATE_DIR / 'hidden'

# === DANDI Dataset ===
DANDISET_ID = '000576'
DANDISET_URL = 'https://dandiarchive.org/dandiset/000576'

# === NWB Schema (populated by 10_human_explore_nwb.py) ===
NWB_SCHEMA_PATH = DATA_DIR / 'nwb_schema.json'


def load_nwb_schema(schema_path=None):
    """Load NWB column name schema from JSON.

    Parameters
    ----------
    schema_path : Path, optional
        Path to nwb_schema.json. Defaults to NWB_SCHEMA_PATH.

    Returns
    -------
    schema : dict or None
        Schema dict with keys: units_columns, region_column, trial_columns,
        mtl_regions, frontal_regions, timing_columns, n_units, n_trials,
        sample_values. Returns None if file not found.
    """
    if schema_path is None:
        schema_path = NWB_SCHEMA_PATH
    schema_path = Path(schema_path)

    if not schema_path.exists():
        logger.warning("NWB schema not found at %s. Run 10_human_explore_nwb.py first.", schema_path)
        return None

    with open(schema_path, 'r') as f:
        schema = json.load(f)

    logger.info("Loaded NWB schema from %s", schema_path)
    return schema


# === Brain Region Identifiers ===
# These are FALLBACK patterns only -- actual region names come from schema
MTL_REGION_PATTERNS = ['hippocampus', 'amygdala']
FRONTAL_REGION_PATTERNS = ['dACC', 'pre-SMA', 'vmPFC',
                           'dorsal anterior cingulate',
                           'presupplementary motor area',
                           'ventromedial prefrontal']

# Minimum neuron counts per region for a patient to qualify
MIN_MTL_NEURONS = 10
MIN_FRONTAL_NEURONS = 10
MIN_TRIALS = 50

# === Spike Binning ===
BIN_SIZE_MS = 50
BIN_SIZE_S = BIN_SIZE_MS / 1000.0

# === Data Splits ===
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# === Surrogate Training ===
HIDDEN_SIZES = [64, 128, 256]
N_LSTM_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20
MAX_EPOCHS = 200
BATCH_SIZE = 32
GRAD_CLIP_NORM = 1.0

# === Universality ===
N_SEEDS = 10
ARCHITECTURES = ['lstm', 'gru', 'transformer', 'linear']

# === Oscillation Bands (Hz) ===
THETA_BAND = (4, 8)
GAMMA_BAND = (30, 80)

# === Probe Target Levels ===
# Level B: Population-level cognitive variables (THE KEY LEVEL)
LEVEL_B_TARGETS = [
    'persistent_delay',        # Mean frontal firing during delay
    'memory_load',             # Load-dependent activity (1/2/3 items)
    'delay_stability',         # Temporal autocorrelation through delay
    'recognition_decision',    # Match vs non-match divergence at probe
]

# Level C: Emergent dynamics
LEVEL_C_TARGETS = [
    'theta_modulation',        # 4-8 Hz population rate amplitude
    'gamma_modulation',        # 30-80 Hz population rate amplitude
    'population_synchrony',    # Pairwise correlation magnitude
]

# Level A: Individual neuron selectivity (expected mostly zombie)
LEVEL_A_TARGETS = [
    'concept_selectivity',     # Image-specific neuron tuning (F-stat)
    'mean_firing_rate',        # Overall mean firing rate
]

# All targets combined
ALL_TARGETS = LEVEL_B_TARGETS + LEVEL_C_TARGETS + LEVEL_A_TARGETS

# === Quality Thresholds ===
MIN_CC_THRESHOLD = 0.3
GOOD_CC_THRESHOLD = 0.5
STRONG_CC_THRESHOLD = 0.7
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_config.py -v 2>&1 | tail -30
```

5. Commit:
```
git add human_wm/__init__.py human_wm/config.py \
  human_wm/data/__init__.py human_wm/surrogate/__init__.py \
  human_wm/targets/__init__.py human_wm/analysis/__init__.py \
  human_wm/ablation/__init__.py \
  tests/human_wm/__init__.py tests/human_wm/test_config.py
git commit -m "human_wm: scaffold package and config with schema loader"
```

---

### Task 2: NWB Explorer (HARD PREREQUISITE)

**Files:**
- `human_wm/data/nwb_explorer.py`
- `scripts/10_human_explore_nwb.py`
- `tests/human_wm/test_nwb_explorer.py`

**Steps:**

1. Write `tests/human_wm/test_nwb_explorer.py`:

```python
"""Tests for human_wm.data.nwb_explorer -- NWB structure discovery."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from human_wm.data.nwb_explorer import explore_nwb, generate_schema


def _make_mock_nwb():
    """Create a mock NWB file object with realistic Rutishauser structure."""
    nwb = MagicMock()

    # Mock units table
    units = MagicMock()
    units.colnames = [
        'spike_times', 'electrodes', 'brain_region',
        'origClusterID', 'IsolationDist', 'SNR',
    ]
    n_units = 15

    # Brain regions: mix of MTL and frontal
    regions = (
        ['hippocampus'] * 4 +
        ['amygdala'] * 3 +
        ['dACC'] * 3 +
        ['pre-SMA'] * 3 +
        ['vmPFC'] * 2
    )
    units.__len__ = MagicMock(return_value=n_units)

    def units_getitem(key):
        if key == 'brain_region':
            return regions
        if key == 'spike_times':
            return [np.array([0.1, 0.2, 0.3])] * n_units
        return [f'val_{i}' for i in range(n_units)]

    units.__getitem__ = MagicMock(side_effect=units_getitem)
    nwb.units = units

    # Mock electrode groups
    eg1 = MagicMock()
    eg1.name = 'group0'
    eg1.location = 'hippocampus'
    eg2 = MagicMock()
    eg2.name = 'group1'
    eg2.location = 'dACC'
    nwb.electrode_groups = {'group0': eg1, 'group1': eg2}

    # Mock trials table
    trials = MagicMock()
    trials.colnames = [
        'start_time', 'stop_time', 'stimulus_image',
        'response_value', 'loads', 'in_out',
    ]
    n_trials = 50
    trials.__len__ = MagicMock(return_value=n_trials)

    def trials_getitem(key):
        if key == 'start_time':
            return np.arange(n_trials, dtype=float)
        if key == 'stop_time':
            return np.arange(n_trials, dtype=float) + 2.0
        if key == 'loads':
            return np.random.choice([1, 2, 3], size=n_trials)
        if key == 'in_out':
            return np.random.choice([0, 1], size=n_trials)
        return [f'{key}_{i}' for i in range(n_trials)]

    trials.__getitem__ = MagicMock(side_effect=trials_getitem)
    nwb.trials = trials

    return nwb


class TestExploreNwb:
    def test_returns_dict(self):
        mock_nwb = _make_mock_nwb()
        with patch('human_wm.data.nwb_explorer._open_nwb', return_value=mock_nwb):
            result = explore_nwb('fake_path.nwb')

        assert isinstance(result, dict)
        assert 'units_columns' in result
        assert 'trial_columns' in result
        assert 'brain_regions' in result
        assert 'n_units' in result
        assert 'n_trials' in result

    def test_detects_brain_regions(self):
        mock_nwb = _make_mock_nwb()
        with patch('human_wm.data.nwb_explorer._open_nwb', return_value=mock_nwb):
            result = explore_nwb('fake_path.nwb')

        assert 'hippocampus' in result['brain_regions']
        assert 'amygdala' in result['brain_regions']
        assert 'dACC' in result['brain_regions']

    def test_counts_correct(self):
        mock_nwb = _make_mock_nwb()
        with patch('human_wm.data.nwb_explorer._open_nwb', return_value=mock_nwb):
            result = explore_nwb('fake_path.nwb')

        assert result['n_units'] == 15
        assert result['n_trials'] == 50


class TestGenerateSchema:
    def test_outputs_valid_json(self):
        mock_nwb = _make_mock_nwb()
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            with patch('human_wm.data.nwb_explorer._open_nwb', return_value=mock_nwb):
                schema = generate_schema('fake_path.nwb', output_path)

            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)

            assert 'units_columns' in loaded
            assert 'region_column' in loaded
            assert 'trial_columns' in loaded
            assert 'mtl_regions' in loaded
            assert 'frontal_regions' in loaded
            assert 'timing_columns' in loaded
            assert 'n_units' in loaded
            assert 'n_trials' in loaded
            assert 'sample_values' in loaded
        finally:
            output_path.unlink(missing_ok=True)

    def test_auto_detects_region_column(self):
        mock_nwb = _make_mock_nwb()
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            with patch('human_wm.data.nwb_explorer._open_nwb', return_value=mock_nwb):
                schema = generate_schema('fake_path.nwb', output_path)

            assert schema['region_column'] == 'brain_region'
        finally:
            output_path.unlink(missing_ok=True)

    def test_classifies_mtl_and_frontal(self):
        mock_nwb = _make_mock_nwb()
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_path = Path(f.name)

        try:
            with patch('human_wm.data.nwb_explorer._open_nwb', return_value=mock_nwb):
                schema = generate_schema('fake_path.nwb', output_path)

            assert len(schema['mtl_regions']) > 0
            assert len(schema['frontal_regions']) > 0
            # hippocampus and amygdala should be MTL
            mtl_lower = [r.lower() for r in schema['mtl_regions']]
            assert any('hippocampus' in r for r in mtl_lower)
            assert any('amygdala' in r for r in mtl_lower)
        finally:
            output_path.unlink(missing_ok=True)
```

2. Run test -- expect import errors:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_nwb_explorer.py -x 2>&1 | head -20
```

3. Implement `human_wm/data/nwb_explorer.py`:

```python
"""
DESCARTES Human WM -- NWB File Structure Explorer

HARD PREREQUISITE: This module must be run before ANY other data loading.
It discovers the actual column names, brain region labels, and trial
structure from the Rutishauser NWB files and outputs a JSON schema.

All downstream code reads from this schema -- ZERO hardcoded column names.
"""

import json
import logging
from pathlib import Path

import numpy as np

from human_wm.config import (
    FRONTAL_REGION_PATTERNS,
    MTL_REGION_PATTERNS,
    NWB_SCHEMA_PATH,
)

logger = logging.getLogger(__name__)

# Candidate column names for the brain region field in units table
_REGION_COLUMN_CANDIDATES = [
    'brain_region',
    'location',
    'brain_area',
    'region',
    'area',
    'anno_name',
    'electrode_group',
]

# Candidate column names for timing in trials table
_TIMING_COLUMN_CANDIDATES = [
    'start_time', 'stop_time',
    'stimulus_on_time', 'stimulus_off_time',
    'delay_start', 'delay_end',
    'probe_on_time', 'probe_off_time',
    'response_time',
]


def _open_nwb(nwb_path):
    """Open an NWB file and return the NWBFile object.

    Separated for testability (mocked in tests).
    """
    import pynwb

    io = pynwb.NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()
    return nwb


def explore_nwb(nwb_path):
    """Explore an NWB file and report its structure.

    Parameters
    ----------
    nwb_path : str or Path
        Path to NWB file.

    Returns
    -------
    info : dict
        Keys: units_columns, trial_columns, brain_regions, region_counts,
        electrode_groups, n_units, n_trials.
    """
    nwb = _open_nwb(nwb_path)

    # --- Units table ---
    units = nwb.units
    n_units = len(units)
    units_columns = list(units.colnames)

    # Detect region column
    region_col = None
    for candidate in _REGION_COLUMN_CANDIDATES:
        if candidate in units_columns:
            region_col = candidate
            break

    # Collect brain regions
    brain_regions = {}
    if region_col is not None:
        for i in range(n_units):
            region = str(units[region_col][i])
            brain_regions[region] = brain_regions.get(region, 0) + 1

    # --- Electrode groups ---
    electrode_groups = {}
    if hasattr(nwb, 'electrode_groups') and nwb.electrode_groups:
        for name, eg in nwb.electrode_groups.items():
            loc = getattr(eg, 'location', 'unknown')
            electrode_groups[name] = str(loc)

    # --- Trials table ---
    trials = nwb.trials
    n_trials = 0
    trial_columns = []
    if trials is not None:
        n_trials = len(trials)
        trial_columns = list(trials.colnames)

    info = {
        'units_columns': units_columns,
        'trial_columns': trial_columns,
        'brain_regions': brain_regions,
        'region_column_detected': region_col,
        'electrode_groups': electrode_groups,
        'n_units': n_units,
        'n_trials': n_trials,
    }

    # Print summary
    logger.info("=== NWB Structure: %s ===", Path(nwb_path).name)
    logger.info("Units: %d units, columns: %s", n_units, units_columns)
    logger.info("Region column: %s", region_col)
    logger.info("Brain regions: %s", brain_regions)
    logger.info("Electrode groups: %s", electrode_groups)
    logger.info("Trials: %d trials, columns: %s", n_trials, trial_columns)

    return info


def _classify_region(region_name):
    """Classify a brain region as 'mtl', 'frontal', or 'other'.

    Uses case-insensitive substring matching against known patterns.
    """
    name_lower = region_name.lower()

    for pattern in MTL_REGION_PATTERNS:
        if pattern.lower() in name_lower:
            return 'mtl'

    for pattern in FRONTAL_REGION_PATTERNS:
        if pattern.lower() in name_lower:
            return 'frontal'

    return 'other'


def generate_schema(nwb_path, output_path=None):
    """Generate NWB schema JSON from a representative NWB file.

    This is the HARD PREREQUISITE for all downstream processing.
    The schema defines every column name, region label, and timing
    field used by the pipeline. Nothing is hardcoded.

    Parameters
    ----------
    nwb_path : str or Path
        Path to a representative NWB file.
    output_path : str or Path, optional
        Where to save the JSON schema. Defaults to NWB_SCHEMA_PATH.

    Returns
    -------
    schema : dict
        The generated schema.
    """
    if output_path is None:
        output_path = NWB_SCHEMA_PATH
    output_path = Path(output_path)

    nwb = _open_nwb(nwb_path)

    # --- Units table ---
    units = nwb.units
    n_units = len(units)
    units_columns = list(units.colnames)

    # Auto-detect region column
    region_column = None
    for candidate in _REGION_COLUMN_CANDIDATES:
        if candidate in units_columns:
            region_column = candidate
            break

    if region_column is None:
        logger.warning(
            "Could not auto-detect region column. "
            "Tried: %s. Available: %s",
            _REGION_COLUMN_CANDIDATES, units_columns,
        )

    # Collect unique brain regions and classify them
    all_regions = set()
    mtl_regions = set()
    frontal_regions = set()
    region_counts = {}

    if region_column is not None:
        for i in range(n_units):
            region = str(units[region_column][i])
            all_regions.add(region)
            region_counts[region] = region_counts.get(region, 0) + 1
            cls = _classify_region(region)
            if cls == 'mtl':
                mtl_regions.add(region)
            elif cls == 'frontal':
                frontal_regions.add(region)

    # --- Trials table ---
    trials = nwb.trials
    n_trials = 0
    trial_columns = []
    timing_columns = []
    sample_values = {}

    if trials is not None:
        n_trials = len(trials)
        trial_columns = list(trials.colnames)

        # Identify timing columns
        for col in trial_columns:
            if col in _TIMING_COLUMN_CANDIDATES:
                timing_columns.append(col)

        # Collect sample values (first 3 entries) for key columns
        for col in trial_columns:
            try:
                vals = []
                for i in range(min(3, n_trials)):
                    v = trials[col][i]
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        v = v.item()
                    vals.append(v)
                sample_values[col] = vals
            except Exception:
                sample_values[col] = ['<error reading>']

    # --- Build schema ---
    schema = {
        'units_columns': units_columns,
        'region_column': region_column,
        'trial_columns': trial_columns,
        'mtl_regions': sorted(mtl_regions),
        'frontal_regions': sorted(frontal_regions),
        'all_regions': sorted(all_regions),
        'region_counts': region_counts,
        'timing_columns': timing_columns,
        'n_units': n_units,
        'n_trials': n_trials,
        'sample_values': sample_values,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2, default=str)

    logger.info("Saved NWB schema to %s", output_path)
    logger.info("Region column: %s", region_column)
    logger.info("MTL regions: %s", sorted(mtl_regions))
    logger.info("Frontal regions: %s", sorted(frontal_regions))
    logger.info("Timing columns: %s", timing_columns)

    return schema
```

Implement `scripts/10_human_explore_nwb.py`:

```python
#!/usr/bin/env python3
"""
DESCARTES Human WM -- Script 10: Explore NWB Structure

HARD PREREQUISITE: Run this FIRST before any other human_wm script.
Discovers column names, brain regions, and trial structure from
the Rutishauser NWB files and outputs data/nwb_schema.json.

Usage:
    python scripts/10_human_explore_nwb.py
    python scripts/10_human_explore_nwb.py --nwb-path path/to/file.nwb
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import NWB_SCHEMA_PATH, RAW_NWB_DIR
from human_wm.data.nwb_explorer import explore_nwb, generate_schema

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Explore NWB file structure and generate schema JSON'
    )
    parser.add_argument(
        '--nwb-path', type=str, default=None,
        help='Path to a specific NWB file. If not given, uses first file in data/human_raw/'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for schema JSON. Defaults to data/nwb_schema.json'
    )
    args = parser.parse_args()

    # Find NWB file
    if args.nwb_path:
        nwb_path = Path(args.nwb_path)
    else:
        nwb_files = sorted(RAW_NWB_DIR.rglob('*.nwb'))
        if not nwb_files:
            logger.error(
                "No NWB files found in %s. "
                "Run 11_human_download.py first, or specify --nwb-path.",
                RAW_NWB_DIR,
            )
            sys.exit(1)
        nwb_path = nwb_files[0]
        logger.info("Using first NWB file: %s", nwb_path)

    output_path = Path(args.output) if args.output else NWB_SCHEMA_PATH

    # Step 1: Explore and print structure
    logger.info("=" * 60)
    logger.info("STEP 1: Exploring NWB structure")
    logger.info("=" * 60)
    info = explore_nwb(nwb_path)

    print("\n=== Units Table Columns ===")
    for col in info['units_columns']:
        print(f"  {col}")

    print(f"\n=== Region Column: {info['region_column_detected']} ===")

    print("\n=== Brain Regions ===")
    for region, count in sorted(info['brain_regions'].items()):
        print(f"  {region}: {count} units")

    print("\n=== Electrode Groups ===")
    for name, loc in info['electrode_groups'].items():
        print(f"  {name}: {loc}")

    print("\n=== Trial Table Columns ===")
    for col in info['trial_columns']:
        print(f"  {col}")

    print(f"\nTotal: {info['n_units']} units, {info['n_trials']} trials")

    # Step 2: Generate schema
    logger.info("=" * 60)
    logger.info("STEP 2: Generating NWB schema")
    logger.info("=" * 60)
    schema = generate_schema(nwb_path, output_path)

    print(f"\n=== Schema saved to {output_path} ===")
    print(json.dumps(schema, indent=2, default=str))


if __name__ == '__main__':
    main()
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_nwb_explorer.py -v 2>&1 | tail -20
```

5. Commit:
```
git add human_wm/data/nwb_explorer.py scripts/10_human_explore_nwb.py \
  tests/human_wm/test_nwb_explorer.py
git commit -m "human_wm: NWB explorer with schema generation (hard prerequisite)"
```

---

### Task 3: Download Script

**Files:**
- `scripts/11_human_download.py`

**Steps:**

1. No test file needed (thin CLI wrapper around dandi CLI).

2. N/A.

3. Implement `scripts/11_human_download.py`:

```python
#!/usr/bin/env python3
"""
DESCARTES Human WM -- Script 11: Download from DANDI 000576

Downloads Rutishauser Human Single-Neuron WM dataset NWB files
to data/human_raw/ using the dandi CLI.

Usage:
    python scripts/11_human_download.py                  # list sessions
    python scripts/11_human_download.py --download       # download all
    python scripts/11_human_download.py --download -n 5  # download first 5
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import DANDISET_ID, RAW_NWB_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def list_assets():
    """List available NWB files in the dandiset via Python API."""
    from dandi.dandiapi import DandiAPIClient

    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)

    print(f"\n=== DANDI {DANDISET_ID} Assets ===\n")
    nwb_count = 0
    total_size_gb = 0.0

    for asset in dandiset.get_assets():
        if asset.path.endswith('.nwb'):
            nwb_count += 1
            size_gb = asset.size / 1e9
            total_size_gb += size_gb
            print(f"  [{nwb_count:3d}] {asset.path}  ({size_gb:.2f} GB)")

    print(f"\nTotal: {nwb_count} NWB files, {total_size_gb:.1f} GB")
    return nwb_count


def download_via_cli(n_sessions=None):
    """Download NWB files using dandi CLI command."""
    RAW_NWB_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        'dandi', 'download',
        f'DANDI:{DANDISET_ID}',
        '-o', str(RAW_NWB_DIR),
        '--existing', 'skip',
    ]

    logger.info("Running: %s", ' '.join(cmd))
    logger.info("Output directory: %s", RAW_NWB_DIR)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("Download complete.")
    except subprocess.CalledProcessError as e:
        logger.error("dandi download failed with exit code %d", e.returncode)
        sys.exit(1)
    except FileNotFoundError:
        logger.error(
            "dandi CLI not found. Install with: pip install dandi"
        )
        sys.exit(1)

    # Report what we have
    nwb_files = sorted(RAW_NWB_DIR.rglob('*.nwb'))
    logger.info("Found %d NWB files in %s", len(nwb_files), RAW_NWB_DIR)
    for f in nwb_files:
        size_mb = f.stat().st_size / 1e6
        logger.info("  %s (%.1f MB)", f.name, size_mb)


def download_via_api(n_sessions=5):
    """Download a limited number of sessions via Python API."""
    from dandi.dandiapi import DandiAPIClient

    RAW_NWB_DIR.mkdir(parents=True, exist_ok=True)

    client = DandiAPIClient()
    dandiset = client.get_dandiset(DANDISET_ID)

    nwb_assets = []
    for asset in dandiset.get_assets():
        if asset.path.endswith('.nwb'):
            nwb_assets.append(asset)
        if n_sessions and len(nwb_assets) >= n_sessions:
            break

    logger.info("Downloading %d sessions to %s", len(nwb_assets), RAW_NWB_DIR)

    for i, asset in enumerate(nwb_assets):
        dest = RAW_NWB_DIR / Path(asset.path).name
        if dest.exists():
            logger.info(
                "[%d/%d] Skipping %s (already exists)",
                i + 1, len(nwb_assets), asset.path,
            )
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[%d/%d] Downloading %s (%.2f GB) ...",
            i + 1, len(nwb_assets), asset.path, asset.size / 1e9,
        )
        asset.download(dest)
        logger.info("  Saved to %s", dest)


def main():
    parser = argparse.ArgumentParser(
        description=f'Download NWB files from DANDI {DANDISET_ID}'
    )
    parser.add_argument(
        '--download', action='store_true',
        help='Download files (default: just list available sessions)'
    )
    parser.add_argument(
        '-n', type=int, default=None,
        help='Number of sessions to download (default: all). Uses Python API.'
    )
    parser.add_argument(
        '--use-api', action='store_true',
        help='Use Python API instead of dandi CLI for download'
    )
    args = parser.parse_args()

    if not args.download:
        list_assets()
        print("\nTo download, run with --download flag.")
        return

    if args.n or args.use_api:
        n = args.n or 5
        download_via_api(n_sessions=n)
    else:
        download_via_cli()


if __name__ == '__main__':
    main()
```

4. No test to run (CLI script).

5. Commit:
```
git add scripts/11_human_download.py
git commit -m "human_wm: download script for DANDI 000576"
```

---

### Task 4: NWB Data Loader

**Files:**
- `human_wm/data/nwb_loader.py`
- `tests/human_wm/test_nwb_loader.py`

**Steps:**

1. Write `tests/human_wm/test_nwb_loader.py`:

```python
"""Tests for human_wm.data.nwb_loader -- schema-driven NWB data extraction."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from human_wm.data.nwb_loader import (
    _classify_region,
    extract_patient_data,
    split_data,
)


def _make_schema():
    """Create a test schema matching generate_schema output."""
    return {
        'units_columns': ['spike_times', 'brain_region', 'electrodes'],
        'region_column': 'brain_region',
        'trial_columns': ['start_time', 'stop_time', 'loads', 'in_out',
                          'stimulus_image', 'response_value'],
        'mtl_regions': ['amygdala', 'hippocampus'],
        'frontal_regions': ['dACC', 'pre-SMA', 'vmPFC'],
        'timing_columns': ['start_time', 'stop_time'],
        'n_units': 20,
        'n_trials': 60,
        'sample_values': {},
    }


def _make_mock_nwb(n_mtl=8, n_frontal=6, n_other=2, n_trials=60):
    """Create a mock NWB with controllable region counts."""
    nwb = MagicMock()
    n_units = n_mtl + n_frontal + n_other

    regions = (
        ['hippocampus'] * (n_mtl // 2) +
        ['amygdala'] * (n_mtl - n_mtl // 2) +
        ['dACC'] * (n_frontal // 3) +
        ['pre-SMA'] * (n_frontal // 3) +
        ['vmPFC'] * (n_frontal - 2 * (n_frontal // 3)) +
        ['other_region'] * n_other
    )

    units = MagicMock()
    units.colnames = ['spike_times', 'brain_region', 'electrodes']
    units.__len__ = MagicMock(return_value=n_units)

    # Generate realistic spike times
    rng = np.random.RandomState(42)

    def units_getitem(key):
        if key == 'brain_region':
            return regions
        if key == 'spike_times':
            return [
                np.sort(rng.uniform(0, n_trials * 3.0, size=rng.randint(50, 200)))
                for _ in range(n_units)
            ]
        return [f'val_{i}' for i in range(n_units)]

    units.__getitem__ = MagicMock(side_effect=units_getitem)
    nwb.units = units

    trials = MagicMock()
    trials.colnames = ['start_time', 'stop_time', 'loads', 'in_out',
                       'stimulus_image', 'response_value']
    trials.__len__ = MagicMock(return_value=n_trials)

    start_times = np.arange(n_trials, dtype=float) * 3.0
    stop_times = start_times + 2.5

    def trials_getitem(key):
        if key == 'start_time':
            return start_times
        if key == 'stop_time':
            return stop_times
        if key == 'loads':
            return rng.choice([1, 2, 3], size=n_trials)
        if key == 'in_out':
            return rng.choice([0, 1], size=n_trials)
        if key == 'response_value':
            return rng.choice([0, 1], size=n_trials)
        return [f'{key}_{i}' for i in range(n_trials)]

    trials.__getitem__ = MagicMock(side_effect=trials_getitem)
    nwb.trials = trials

    return nwb


class TestClassifyRegion:
    def test_hippocampus_is_mtl(self):
        schema = _make_schema()
        assert _classify_region('hippocampus', schema) == 'mtl'

    def test_amygdala_is_mtl(self):
        schema = _make_schema()
        assert _classify_region('amygdala', schema) == 'mtl'

    def test_dacc_is_frontal(self):
        schema = _make_schema()
        assert _classify_region('dACC', schema) == 'frontal'

    def test_presma_is_frontal(self):
        schema = _make_schema()
        assert _classify_region('pre-SMA', schema) == 'frontal'

    def test_vmpfc_is_frontal(self):
        schema = _make_schema()
        assert _classify_region('vmPFC', schema) == 'frontal'

    def test_unknown_is_other(self):
        schema = _make_schema()
        assert _classify_region('entorhinal', schema) == 'other'


class TestExtractPatientData:
    def test_returns_correct_shapes(self):
        schema = _make_schema()
        mock_nwb = _make_mock_nwb(n_mtl=8, n_frontal=6, n_trials=60)

        with patch('human_wm.data.nwb_loader._open_nwb', return_value=mock_nwb):
            X, Y, trial_info = extract_patient_data('fake.nwb', schema)

        assert X.ndim == 3
        assert Y.ndim == 3
        assert X.shape[0] == Y.shape[0]  # same n_trials
        assert X.shape[1] == Y.shape[1]  # same n_bins
        assert X.shape[2] == 8           # n_mtl
        assert Y.shape[2] == 6           # n_frontal
        assert isinstance(trial_info, dict)

    def test_trial_info_has_keys(self):
        schema = _make_schema()
        mock_nwb = _make_mock_nwb()

        with patch('human_wm.data.nwb_loader._open_nwb', return_value=mock_nwb):
            X, Y, trial_info = extract_patient_data('fake.nwb', schema)

        assert 'n_mtl' in trial_info
        assert 'n_frontal' in trial_info
        assert 'n_trials' in trial_info
        assert 'n_bins' in trial_info

    def test_x_y_are_float32(self):
        schema = _make_schema()
        mock_nwb = _make_mock_nwb()

        with patch('human_wm.data.nwb_loader._open_nwb', return_value=mock_nwb):
            X, Y, trial_info = extract_patient_data('fake.nwb', schema)

        assert X.dtype == np.float32
        assert Y.dtype == np.float32


class TestSplitData:
    def test_split_sizes(self):
        rng = np.random.RandomState(0)
        n, t, d_in, d_out = 100, 20, 8, 6
        X = rng.randn(n, t, d_in).astype(np.float32)
        Y = rng.randn(n, t, d_out).astype(np.float32)
        trial_info = {'loads': rng.choice([1, 2, 3], size=n)}

        splits = split_data(X, Y, trial_info)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        total = (
            splits['train']['X'].shape[0] +
            splits['val']['X'].shape[0] +
            splits['test']['X'].shape[0]
        )
        assert total == n

    def test_split_keys(self):
        rng = np.random.RandomState(0)
        n = 50
        X = rng.randn(n, 10, 5).astype(np.float32)
        Y = rng.randn(n, 10, 3).astype(np.float32)
        trial_info = {'loads': rng.choice([1, 2, 3], size=n)}

        splits = split_data(X, Y, trial_info)

        for split_name in ['train', 'val', 'test']:
            assert 'X' in splits[split_name]
            assert 'Y' in splits[split_name]
            assert 'trial_info' in splits[split_name]
```

2. Run test -- expect import errors:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_nwb_loader.py -x 2>&1 | head -20
```

3. Implement `human_wm/data/nwb_loader.py`:

```python
"""
DESCARTES Human WM -- Schema-Driven NWB Data Loader

Extracts binned spike count tensors from Rutishauser NWB files:
    X: (n_trials, n_bins, n_mtl_neurons)     -- hippocampus + amygdala
    Y: (n_trials, n_bins, n_frontal_neurons)  -- dACC + pre-SMA + vmPFC

ALL column names come from the schema JSON produced by nwb_explorer.py.
ZERO hardcoded column names.
"""

import logging
from pathlib import Path

import numpy as np

from human_wm.config import (
    BIN_SIZE_S,
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
)

logger = logging.getLogger(__name__)


def _open_nwb(nwb_path):
    """Open an NWB file and return the NWBFile object.

    Separated for testability (mocked in tests).
    """
    import pynwb

    io = pynwb.NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()
    return nwb


def _classify_region(region_name, schema):
    """Classify a brain region as 'mtl', 'frontal', or 'other'.

    Uses the schema-defined region lists -- NO hardcoded patterns.

    Parameters
    ----------
    region_name : str
    schema : dict from nwb_schema.json

    Returns
    -------
    classification : str, one of 'mtl', 'frontal', 'other'
    """
    name_lower = region_name.lower()

    for mtl_region in schema.get('mtl_regions', []):
        if mtl_region.lower() in name_lower or name_lower in mtl_region.lower():
            return 'mtl'

    for frontal_region in schema.get('frontal_regions', []):
        if frontal_region.lower() in name_lower or name_lower in frontal_region.lower():
            return 'frontal'

    return 'other'


def extract_patient_data(nwb_path, schema):
    """Extract binned spike counts for MTL (input) and Frontal (output).

    Uses schema-defined column names for everything. Never hardcodes
    column names.

    Parameters
    ----------
    nwb_path : str or Path
    schema : dict
        From nwb_schema.json (produced by nwb_explorer.generate_schema).

    Returns
    -------
    X : ndarray, (n_trials, n_bins, n_mtl_neurons)
    Y : ndarray, (n_trials, n_bins, n_frontal_neurons)
    trial_info : dict
        Contains per-trial metadata (loads, in_out, etc.) plus session info.
    """
    nwb = _open_nwb(nwb_path)

    # --- Extract units ---
    units = nwb.units
    n_units = len(units)
    region_col = schema['region_column']

    mtl_indices = []
    frontal_indices = []
    mtl_spike_times = []
    frontal_spike_times = []

    for i in range(n_units):
        region = str(units[region_col][i])
        cls = _classify_region(region, schema)
        spike_times = np.array(units['spike_times'][i], dtype=np.float64)

        if cls == 'mtl':
            mtl_indices.append(i)
            mtl_spike_times.append(spike_times)
        elif cls == 'frontal':
            frontal_indices.append(i)
            frontal_spike_times.append(spike_times)

    n_mtl = len(mtl_indices)
    n_frontal = len(frontal_indices)

    # --- Extract trials ---
    trials = nwb.trials
    n_trials = len(trials)

    start_times = np.array(
        [float(trials['start_time'][i]) for i in range(n_trials)],
        dtype=np.float64,
    )
    stop_times = np.array(
        [float(trials['stop_time'][i]) for i in range(n_trials)],
        dtype=np.float64,
    )

    # Compute trial duration and number of bins
    trial_durations = stop_times - start_times
    median_duration = float(np.median(trial_durations))
    n_bins = int(median_duration / BIN_SIZE_S)

    logger.info(
        "Patient %s: %d MTL, %d frontal, %d trials, %d bins (%.0f ms bins, %.2f s median trial)",
        Path(nwb_path).name, n_mtl, n_frontal, n_trials, n_bins,
        BIN_SIZE_S * 1000, median_duration,
    )

    # --- Bin spikes ---
    X = np.zeros((n_trials, n_bins, n_mtl), dtype=np.float32)
    Y = np.zeros((n_trials, n_bins, n_frontal), dtype=np.float32)

    for t_idx in range(n_trials):
        t_start = start_times[t_idx]

        for j, spikes in enumerate(mtl_spike_times):
            for b in range(n_bins):
                bin_start = t_start + b * BIN_SIZE_S
                bin_end = bin_start + BIN_SIZE_S
                X[t_idx, b, j] = np.sum(
                    (spikes >= bin_start) & (spikes < bin_end)
                )

        for j, spikes in enumerate(frontal_spike_times):
            for b in range(n_bins):
                bin_start = t_start + b * BIN_SIZE_S
                bin_end = bin_start + BIN_SIZE_S
                Y[t_idx, b, j] = np.sum(
                    (spikes >= bin_start) & (spikes < bin_end)
                )

    # --- Build trial_info dict ---
    trial_info = {
        'n_mtl': n_mtl,
        'n_frontal': n_frontal,
        'n_trials': n_trials,
        'n_bins': n_bins,
        'path': str(nwb_path),
        'mtl_indices': mtl_indices,
        'frontal_indices': frontal_indices,
    }

    # Extract all trial-level metadata columns from schema
    for col in schema.get('trial_columns', []):
        if col in ('start_time', 'stop_time'):
            continue
        try:
            vals = []
            for i in range(n_trials):
                v = trials[col][i]
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, (np.integer, np.floating)):
                    v = v.item()
                vals.append(v)
            trial_info[col] = np.array(vals) if all(
                isinstance(v, (int, float)) for v in vals
            ) else vals
        except Exception as e:
            logger.warning("Could not extract trial column '%s': %s", col, e)

    return X, Y, trial_info


def split_data(X, Y, trial_info, seed=42):
    """Split into train/val/test sets.

    Parameters
    ----------
    X : ndarray, (n_trials, n_bins, n_mtl)
    Y : ndarray, (n_trials, n_bins, n_frontal)
    trial_info : dict with per-trial metadata arrays
    seed : int

    Returns
    -------
    splits : dict with keys 'train', 'val', 'test'
        Each is a dict with 'X', 'Y', 'trial_info' (subset of trial_info).
    """
    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def _subset_trial_info(idx):
        """Subset trial_info arrays by index."""
        subset = {}
        for key, val in trial_info.items():
            if isinstance(val, np.ndarray) and val.shape[0] == n:
                subset[key] = val[idx]
            elif isinstance(val, list) and len(val) == n:
                subset[key] = [val[i] for i in idx]
            else:
                # Scalar or session-level metadata -- copy as-is
                subset[key] = val
        return subset

    return {
        'train': {
            'X': X[train_idx],
            'Y': Y[train_idx],
            'trial_info': _subset_trial_info(train_idx),
        },
        'val': {
            'X': X[val_idx],
            'Y': Y[val_idx],
            'trial_info': _subset_trial_info(val_idx),
        },
        'test': {
            'X': X[test_idx],
            'Y': Y[test_idx],
            'trial_info': _subset_trial_info(test_idx),
        },
    }
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_nwb_loader.py -v 2>&1 | tail -20
```

5. Commit:
```
git add human_wm/data/nwb_loader.py tests/human_wm/test_nwb_loader.py
git commit -m "human_wm: schema-driven NWB data loader with MTL/Frontal binning"
```

---

### Task 5: Patient Inventory

**Files:**
- `human_wm/data/patient_inventory.py`
- `scripts/12_human_inventory.py`
- `tests/human_wm/test_patient_inventory.py`

**Steps:**

1. Write `tests/human_wm/test_patient_inventory.py`:

```python
"""Tests for human_wm.data.patient_inventory -- patient usability inventory."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from human_wm.data.patient_inventory import (
    build_inventory,
    get_best_patient,
    get_usable_patients,
)


def _make_schema():
    return {
        'units_columns': ['spike_times', 'brain_region'],
        'region_column': 'brain_region',
        'trial_columns': ['start_time', 'stop_time', 'loads'],
        'mtl_regions': ['hippocampus', 'amygdala'],
        'frontal_regions': ['dACC', 'pre-SMA', 'vmPFC'],
        'timing_columns': ['start_time', 'stop_time'],
        'n_units': 20,
        'n_trials': 60,
        'sample_values': {},
    }


def _make_mock_extract(n_mtl, n_frontal, n_trials):
    """Create mock return value for extract_patient_data."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_trials, 10, n_mtl).astype(np.float32)
    Y = rng.randn(n_trials, 10, n_frontal).astype(np.float32)
    trial_info = {
        'n_mtl': n_mtl,
        'n_frontal': n_frontal,
        'n_trials': n_trials,
        'n_bins': 10,
    }
    return X, Y, trial_info


class TestBuildInventory:
    @patch('human_wm.data.patient_inventory.extract_patient_data')
    def test_returns_list_of_dicts(self, mock_extract):
        mock_extract.return_value = _make_mock_extract(15, 12, 80)
        schema = _make_schema()

        inventory = build_inventory(
            ['patient1.nwb', 'patient2.nwb'], schema
        )

        assert isinstance(inventory, list)
        assert len(inventory) == 2
        assert 'patient_id' in inventory[0]
        assert 'n_mtl' in inventory[0]
        assert 'n_frontal' in inventory[0]
        assert 'n_trials' in inventory[0]
        assert 'usable' in inventory[0]

    @patch('human_wm.data.patient_inventory.extract_patient_data')
    def test_usable_flag_correct(self, mock_extract):
        schema = _make_schema()

        # Patient with enough neurons and trials
        mock_extract.return_value = _make_mock_extract(15, 12, 80)
        inventory = build_inventory(['good.nwb'], schema)
        assert inventory[0]['usable'] is True

        # Patient with too few MTL neurons
        mock_extract.return_value = _make_mock_extract(3, 12, 80)
        inventory = build_inventory(['bad_mtl.nwb'], schema)
        assert inventory[0]['usable'] is False

        # Patient with too few trials
        mock_extract.return_value = _make_mock_extract(15, 12, 20)
        inventory = build_inventory(['bad_trials.nwb'], schema)
        assert inventory[0]['usable'] is False


class TestGetBestPatient:
    def test_returns_patient_with_most_neurons(self):
        inventory = [
            {'patient_id': 'p1', 'n_mtl': 10, 'n_frontal': 8,
             'n_trials': 60, 'usable': True, 'path': 'p1.nwb'},
            {'patient_id': 'p2', 'n_mtl': 20, 'n_frontal': 15,
             'n_trials': 80, 'usable': True, 'path': 'p2.nwb'},
            {'patient_id': 'p3', 'n_mtl': 5, 'n_frontal': 3,
             'n_trials': 40, 'usable': False, 'path': 'p3.nwb'},
        ]

        best = get_best_patient(inventory)
        assert best['patient_id'] == 'p2'

    def test_returns_none_if_no_usable(self):
        inventory = [
            {'patient_id': 'p1', 'n_mtl': 3, 'n_frontal': 2,
             'n_trials': 20, 'usable': False, 'path': 'p1.nwb'},
        ]
        best = get_best_patient(inventory)
        assert best is None


class TestGetUsablePatients:
    def test_filters_correctly(self):
        inventory = [
            {'patient_id': 'p1', 'usable': True, 'path': 'p1.nwb',
             'n_mtl': 10, 'n_frontal': 10, 'n_trials': 60},
            {'patient_id': 'p2', 'usable': False, 'path': 'p2.nwb',
             'n_mtl': 3, 'n_frontal': 3, 'n_trials': 20},
            {'patient_id': 'p3', 'usable': True, 'path': 'p3.nwb',
             'n_mtl': 15, 'n_frontal': 12, 'n_trials': 80},
        ]

        usable = get_usable_patients(inventory)
        assert len(usable) == 2
        assert all(p['usable'] for p in usable)
```

2. Run test -- expect import errors:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_patient_inventory.py -x 2>&1 | head -20
```

3. Implement `human_wm/data/patient_inventory.py`:

```python
"""
DESCARTES Human WM -- Patient Inventory

Scans all NWB files and builds an inventory of which patients are usable
for the universality test (sufficient MTL neurons, frontal neurons, and trials).
"""

import logging
from pathlib import Path

from human_wm.config import MIN_FRONTAL_NEURONS, MIN_MTL_NEURONS, MIN_TRIALS
from human_wm.data.nwb_loader import extract_patient_data

logger = logging.getLogger(__name__)


def build_inventory(nwb_paths, schema):
    """Build an inventory of all patients and their data quality.

    Parameters
    ----------
    nwb_paths : list of str or Path
        Paths to NWB files.
    schema : dict
        From nwb_schema.json.

    Returns
    -------
    inventory : list of dict
        Each dict has: patient_id, path, n_mtl, n_frontal, n_trials, usable.
    """
    inventory = []

    for nwb_path in nwb_paths:
        nwb_path = Path(nwb_path)
        patient_id = nwb_path.stem

        try:
            X, Y, trial_info = extract_patient_data(str(nwb_path), schema)

            n_mtl = trial_info['n_mtl']
            n_frontal = trial_info['n_frontal']
            n_trials = trial_info['n_trials']

            usable = (
                n_mtl >= MIN_MTL_NEURONS
                and n_frontal >= MIN_FRONTAL_NEURONS
                and n_trials >= MIN_TRIALS
            )

            entry = {
                'patient_id': patient_id,
                'path': str(nwb_path),
                'n_mtl': n_mtl,
                'n_frontal': n_frontal,
                'n_trials': n_trials,
                'usable': usable,
            }
            inventory.append(entry)

            status = 'USABLE' if usable else 'SKIP'
            logger.info(
                "[%s] %s: MTL=%d, Frontal=%d, trials=%d",
                status, patient_id, n_mtl, n_frontal, n_trials,
            )

        except Exception as e:
            logger.warning("Error processing %s: %s", nwb_path.name, e)
            inventory.append({
                'patient_id': patient_id,
                'path': str(nwb_path),
                'n_mtl': 0,
                'n_frontal': 0,
                'n_trials': 0,
                'usable': False,
                'error': str(e),
            })

    n_usable = sum(1 for p in inventory if p['usable'])
    logger.info(
        "Inventory complete: %d/%d patients usable",
        n_usable, len(inventory),
    )

    return inventory


def get_best_patient(inventory):
    """Return the usable patient with the most neurons in both regions.

    Ranks by min(n_mtl, n_frontal) to find the best-balanced patient.

    Parameters
    ----------
    inventory : list of dict

    Returns
    -------
    best : dict or None
    """
    usable = [p for p in inventory if p['usable']]
    if not usable:
        logger.warning("No usable patients found in inventory.")
        return None

    best = max(usable, key=lambda p: min(p['n_mtl'], p['n_frontal']))
    logger.info(
        "Best patient: %s (MTL=%d, Frontal=%d, trials=%d)",
        best['patient_id'], best['n_mtl'], best['n_frontal'],
        best['n_trials'],
    )
    return best


def get_usable_patients(inventory):
    """Return list of all usable patients.

    Parameters
    ----------
    inventory : list of dict

    Returns
    -------
    usable : list of dict
    """
    usable = [p for p in inventory if p['usable']]
    logger.info("%d usable patients out of %d total", len(usable), len(inventory))
    return usable
```

Implement `scripts/12_human_inventory.py`:

```python
#!/usr/bin/env python3
"""
DESCARTES Human WM -- Script 12: Build Patient Inventory

Scans all downloaded NWB files and reports which patients have enough
MTL neurons, frontal neurons, and trials for the universality test.

Usage:
    python scripts/12_human_inventory.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import DATA_DIR, RAW_NWB_DIR, load_nwb_schema
from human_wm.data.patient_inventory import (
    build_inventory,
    get_best_patient,
    get_usable_patients,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    # Load schema
    schema = load_nwb_schema()
    if schema is None:
        logger.error(
            "NWB schema not found. Run 10_human_explore_nwb.py first."
        )
        sys.exit(1)

    # Find NWB files
    nwb_files = sorted(RAW_NWB_DIR.rglob('*.nwb'))
    if not nwb_files:
        logger.error("No NWB files found in %s", RAW_NWB_DIR)
        sys.exit(1)

    logger.info("Found %d NWB files", len(nwb_files))

    # Build inventory
    inventory = build_inventory(nwb_files, schema)

    # Print summary
    print("\n=== Patient Inventory ===\n")
    print(f"{'Patient':<40} {'MTL':>5} {'Front':>5} {'Trials':>6} {'Status':<8}")
    print("-" * 70)
    for p in inventory:
        status = 'USABLE' if p['usable'] else 'SKIP'
        print(
            f"{p['patient_id']:<40} {p['n_mtl']:>5} "
            f"{p['n_frontal']:>5} {p['n_trials']:>6} {status:<8}"
        )

    usable = get_usable_patients(inventory)
    best = get_best_patient(inventory)

    print(f"\nUsable: {len(usable)}/{len(inventory)} patients")
    if best:
        print(
            f"Best patient: {best['patient_id']} "
            f"(MTL={best['n_mtl']}, Frontal={best['n_frontal']}, "
            f"trials={best['n_trials']})"
        )

    # Save inventory
    inventory_path = DATA_DIR / 'patient_inventory.json'
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(inventory_path, 'w') as f:
        json.dump(inventory, f, indent=2, default=str)
    print(f"\nInventory saved to {inventory_path}")


if __name__ == '__main__':
    main()
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_patient_inventory.py -v 2>&1 | tail -20
```

5. Commit:
```
git add human_wm/data/patient_inventory.py scripts/12_human_inventory.py \
  tests/human_wm/test_patient_inventory.py
git commit -m "human_wm: patient inventory with usability filtering"
```

---

### Task 6: Model Architectures (all 4)

**Files:**
- `human_wm/surrogate/architectures.py`
- `tests/human_wm/test_architectures.py`

**Steps:**

1. Write `tests/human_wm/test_architectures.py`:

```python
"""Tests for human_wm.surrogate.architectures -- all 4 model architectures."""

import torch
import pytest

from human_wm.surrogate.architectures import (
    HumanGRUSurrogate,
    HumanLinearSurrogate,
    HumanLSTMSurrogate,
    HumanTransformerSurrogate,
    create_model,
)


INPUT_DIM = 12
OUTPUT_DIM = 8
HIDDEN_SIZE = 64
BATCH_SIZE = 4
SEQ_LEN = 20


class TestHumanLSTMSurrogate:
    def test_forward_shape(self):
        model = HumanLSTMSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        (y_pred,) = model(x)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

    def test_return_hidden(self):
        model = HumanLSTMSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        y_pred, h_seq = model(x, return_hidden=True)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
        assert h_seq.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_attributes(self):
        model = HumanLSTMSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'lstm')
        assert model.hidden_size == HIDDEN_SIZE
        assert model.n_layers == 2
        assert hasattr(model, 'proj')

    def test_count_parameters(self):
        model = HumanLSTMSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        n_params = model.count_parameters()
        assert n_params > 0
        assert isinstance(n_params, int)


class TestHumanGRUSurrogate:
    def test_forward_shape(self):
        model = HumanGRUSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        (y_pred,) = model(x)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

    def test_return_hidden(self):
        model = HumanGRUSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        y_pred, h_seq = model(x, return_hidden=True)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
        assert h_seq.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_attributes(self):
        model = HumanGRUSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'gru')
        assert model.hidden_size == HIDDEN_SIZE
        assert model.n_layers == 2
        assert hasattr(model, 'proj')


class TestHumanTransformerSurrogate:
    def test_forward_shape(self):
        model = HumanTransformerSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        (y_pred,) = model(x)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

    def test_return_hidden(self):
        model = HumanTransformerSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        y_pred, h_seq = model(x, return_hidden=True)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
        assert h_seq.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_attributes(self):
        model = HumanTransformerSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'input_proj')
        assert hasattr(model, 'pos_enc')
        assert model.hidden_size == HIDDEN_SIZE
        assert hasattr(model, 'proj')


class TestHumanLinearSurrogate:
    def test_forward_shape(self):
        model = HumanLinearSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        (y_pred,) = model(x)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

    def test_return_hidden(self):
        model = HumanLinearSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        y_pred, h_seq = model(x, return_hidden=True)
        assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)
        assert h_seq.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_attributes(self):
        model = HumanLinearSurrogate(INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'proj_in')
        assert hasattr(model, 'proj_out')
        assert model.hidden_size == HIDDEN_SIZE
        # .proj is alias for .proj_out
        assert model.proj is model.proj_out


class TestFactory:
    def test_create_lstm(self):
        model = create_model('lstm', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, HumanLSTMSurrogate)

    def test_create_gru(self):
        model = create_model('gru', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, HumanGRUSurrogate)

    def test_create_transformer(self):
        model = create_model('transformer', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, HumanTransformerSurrogate)

    def test_create_linear(self):
        model = create_model('linear', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, HumanLinearSurrogate)

    def test_invalid_arch_raises(self):
        with pytest.raises(ValueError, match='Unknown architecture'):
            create_model('rnn', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)

    def test_all_architectures_forward_compatible(self):
        """All architectures produce same output shape and support return_hidden."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

        for arch in ['lstm', 'gru', 'transformer', 'linear']:
            model = create_model(arch, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
            (y_pred,) = model(x)
            assert y_pred.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM), (
                f"{arch} y_pred shape mismatch"
            )

            y_pred2, h_seq = model(x, return_hidden=True)
            assert h_seq.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE), (
                f"{arch} h_seq shape mismatch"
            )
```

2. Run test -- expect import errors:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_architectures.py -x 2>&1 | head -20
```

3. Implement `human_wm/surrogate/architectures.py`:

```python
"""
DESCARTES Human WM -- Surrogate Model Architectures

Four architectures for the MTL -> Frontal transformation:
    1. HumanLSTMSurrogate   -- LSTM + linear projection
    2. HumanGRUSurrogate    -- GRU + linear projection
    3. HumanTransformerSurrogate -- Transformer encoder + projections
    4. HumanLinearSurrogate -- Linear proj_in + proj_out (no recurrence)

ALL share the same interface:
    forward(x, return_hidden=False) -> (y_pred,) or (y_pred, h_seq)
    count_parameters() -> int

Hidden states (batch, T, hidden_size) are the substrate for probing.
"""

import math

import torch
import torch.nn as nn

from human_wm.config import N_LSTM_LAYERS


# ---------------------------------------------------------------------------
# LSTM Surrogate
# ---------------------------------------------------------------------------

class HumanLSTMSurrogate(nn.Module):
    """LSTM surrogate for MTL -> Frontal transformation.

    Attributes: .lstm, .hidden_size, .n_layers, .proj
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_layers=N_LSTM_LAYERS, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, return_hidden=False):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        y_pred : Tensor, (batch, T, output_dim)
        h_seq : Tensor, (batch, T, hidden_size) -- only if return_hidden
        """
        h_seq, (h_n, c_n) = self.lstm(x)
        y_pred = self.proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanLSTMSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# GRU Surrogate
# ---------------------------------------------------------------------------

class HumanGRUSurrogate(nn.Module):
    """GRU surrogate for MTL -> Frontal transformation.

    Attributes: .gru, .hidden_size, .n_layers, .proj
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_layers=N_LSTM_LAYERS, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, return_hidden=False):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        y_pred : Tensor, (batch, T, output_dim)
        h_seq : Tensor, (batch, T, hidden_size) -- only if return_hidden
        """
        h_seq, h_n = self.gru(x)
        y_pred = self.proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanGRUSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Transformer Surrogate
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer input."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input.

        Parameters
        ----------
        x : Tensor, (batch, T, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class HumanTransformerSurrogate(nn.Module):
    """Transformer encoder surrogate for MTL -> Frontal transformation.

    Architecture: input_proj -> pos_enc -> TransformerEncoder -> proj

    Attributes: .transformer, .input_proj, .pos_enc, .hidden_size, .proj
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_enc = _PositionalEncoding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, return_hidden=False):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        y_pred : Tensor, (batch, T, output_dim)
        h_seq : Tensor, (batch, T, hidden_size) -- only if return_hidden
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)

        # Causal mask to prevent looking ahead
        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T)
        causal_mask = causal_mask.to(x.device)

        h_seq = self.transformer(h, mask=causal_mask)
        y_pred = self.proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanTransformerSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Linear Surrogate
# ---------------------------------------------------------------------------

class HumanLinearSurrogate(nn.Module):
    """Linear surrogate for MTL -> Frontal transformation.

    Architecture: proj_in -> proj_out (no recurrence, no temporal state).
    Acts as a lower bound / negative control for recurrent architectures.

    Attributes: .proj_in, .proj_out, .hidden_size, .proj (alias for proj_out)
    """

    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = 1  # For compatibility with ablation code

        self.proj_in = nn.Linear(input_dim, hidden_size)
        self.proj_out = nn.Linear(hidden_size, output_dim)

    @property
    def proj(self):
        """Alias for proj_out (compatibility with other architectures)."""
        return self.proj_out

    def forward(self, x, return_hidden=False):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        y_pred : Tensor, (batch, T, output_dim)
        h_seq : Tensor, (batch, T, hidden_size) -- only if return_hidden
        """
        h_seq = self.proj_in(x)
        y_pred = self.proj_out(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanLinearSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ARCHITECTURE_MAP = {
    'lstm': HumanLSTMSurrogate,
    'gru': HumanGRUSurrogate,
    'transformer': HumanTransformerSurrogate,
    'linear': HumanLinearSurrogate,
}


def create_model(arch_name, input_dim, output_dim, hidden_size=128):
    """Factory function to create a model by architecture name.

    Parameters
    ----------
    arch_name : str
        One of 'lstm', 'gru', 'transformer', 'linear'.
    input_dim : int
    output_dim : int
    hidden_size : int

    Returns
    -------
    model : nn.Module
    """
    arch_name = arch_name.lower()
    if arch_name not in _ARCHITECTURE_MAP:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Choose from: {list(_ARCHITECTURE_MAP.keys())}"
        )

    cls = _ARCHITECTURE_MAP[arch_name]
    return cls(input_dim=input_dim, output_dim=output_dim,
               hidden_size=hidden_size)
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_architectures.py -v 2>&1 | tail -30
```

5. Commit:
```
git add human_wm/surrogate/architectures.py tests/human_wm/test_architectures.py
git commit -m "human_wm: 4 surrogate architectures (LSTM/GRU/Transformer/Linear) with factory"
```

---

### Task 7: Training Loop + Hidden Extraction

**Files:**
- `human_wm/surrogate/train.py`
- `human_wm/surrogate/extract_hidden.py`
- `tests/human_wm/test_train.py`

**Steps:**

1. Write `tests/human_wm/test_train.py`:

```python
"""Tests for human_wm.surrogate.train and extract_hidden."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from human_wm.surrogate.architectures import create_model
from human_wm.surrogate.train import create_dataloader, train_surrogate
from human_wm.surrogate.extract_hidden import (
    extract_hidden_states,
    extract_trained_and_untrained,
)


INPUT_DIM = 8
OUTPUT_DIM = 5
HIDDEN_SIZE = 16
BATCH_SIZE = 4
SEQ_LEN = 10
N_TRIALS = 20


def _make_dummy_data():
    """Create small random data for training tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(N_TRIALS, SEQ_LEN, INPUT_DIM).astype(np.float32)
    Y = rng.randn(N_TRIALS, SEQ_LEN, OUTPUT_DIM).astype(np.float32)
    return X, Y


class TestCreateDataloader:
    def test_creates_dataloader(self):
        X, Y = _make_dummy_data()
        loader = create_dataloader(X, Y, batch_size=BATCH_SIZE)
        batch_X, batch_Y = next(iter(loader))
        assert batch_X.shape == (BATCH_SIZE, SEQ_LEN, INPUT_DIM)
        assert batch_Y.shape == (BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)

    def test_correct_n_batches(self):
        X, Y = _make_dummy_data()
        loader = create_dataloader(X, Y, batch_size=BATCH_SIZE)
        n_batches = len(loader)
        expected = (N_TRIALS + BATCH_SIZE - 1) // BATCH_SIZE
        assert n_batches == expected


class TestTrainSurrogate:
    @pytest.mark.parametrize('arch', ['lstm', 'gru', 'transformer', 'linear'])
    def test_train_2_epochs(self, arch):
        X, Y = _make_dummy_data()
        train_loader = create_dataloader(X[:15], Y[:15], batch_size=BATCH_SIZE)
        val_loader = create_dataloader(X[15:], Y[15:], batch_size=BATCH_SIZE,
                                        shuffle=False)

        model = create_model(arch, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / f'{arch}_best.pt'
            model, history = train_surrogate(
                model, train_loader, val_loader,
                n_epochs=2, save_path=save_path,
            )

            assert save_path.exists()
            assert len(history['train_loss']) == 2
            assert len(history['val_loss']) == 2
            assert all(isinstance(v, float) for v in history['train_loss'])

    def test_train_loss_decreases(self):
        """Loss should decrease over a few epochs on random data."""
        X, Y = _make_dummy_data()
        train_loader = create_dataloader(X[:15], Y[:15], batch_size=BATCH_SIZE)
        val_loader = create_dataloader(X[15:], Y[15:], batch_size=BATCH_SIZE,
                                        shuffle=False)

        model = create_model('lstm', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'lstm_best.pt'
            model, history = train_surrogate(
                model, train_loader, val_loader,
                n_epochs=10, save_path=save_path,
            )

            # Training loss should generally decrease
            assert history['train_loss'][-1] <= history['train_loss'][0]


class TestExtractHiddenStates:
    @pytest.mark.parametrize('arch', ['lstm', 'gru', 'transformer', 'linear'])
    def test_shapes(self, arch):
        model = create_model(arch, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        X = np.random.randn(N_TRIALS, SEQ_LEN, INPUT_DIM).astype(np.float32)

        hidden_avg, hidden_full = extract_hidden_states(model, X)

        assert hidden_avg.shape == (N_TRIALS, HIDDEN_SIZE)
        assert hidden_full.shape == (N_TRIALS, SEQ_LEN, HIDDEN_SIZE)

    def test_avg_is_mean_of_full(self):
        model = create_model('lstm', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        X = np.random.randn(N_TRIALS, SEQ_LEN, INPUT_DIM).astype(np.float32)

        hidden_avg, hidden_full = extract_hidden_states(model, X)
        expected_avg = hidden_full.mean(axis=1)

        np.testing.assert_allclose(hidden_avg, expected_avg, atol=1e-5)


class TestExtractTrainedAndUntrained:
    @pytest.mark.parametrize('arch', ['lstm', 'gru', 'transformer', 'linear'])
    def test_returns_different_states(self, arch):
        X, Y = _make_dummy_data()

        model = create_model(arch, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a "trained" model (just save current random weights)
            model_path = Path(tmpdir) / f'{arch}_best.pt'
            torch.save(model.state_dict(), model_path)

            trained_H, untrained_H = extract_trained_and_untrained(
                model_path, X,
                arch_name=arch,
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                hidden_size=HIDDEN_SIZE,
            )

            assert trained_H.shape == (N_TRIALS, HIDDEN_SIZE)
            assert untrained_H.shape == (N_TRIALS, HIDDEN_SIZE)

            # Trained and untrained should differ (different random init)
            assert not np.allclose(trained_H, untrained_H)
```

2. Run test -- expect import errors:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_train.py -x 2>&1 | head -20
```

3. Implement `human_wm/surrogate/train.py`:

```python
"""
DESCARTES Human WM -- Architecture-Agnostic Surrogate Training Loop

Standard training with early stopping, cosine annealing, and gradient
clipping. Trains any of the 4 architectures (LSTM/GRU/Transformer/Linear)
to predict frontal population activity from MTL population activity.

Identical interface to wm/surrogate/train.py but imports from human_wm.config.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from human_wm.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    HIDDEN_SIZES,
    LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
)
from human_wm.surrogate.architectures import create_model

logger = logging.getLogger(__name__)


def create_dataloader(X, Y, batch_size=BATCH_SIZE, shuffle=True):
    """Create a DataLoader from numpy arrays.

    Parameters
    ----------
    X : ndarray, (n_trials, T, input_dim)
    Y : ndarray, (n_trials, T, output_dim)
    batch_size : int
    shuffle : bool

    Returns
    -------
    loader : DataLoader
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_surrogate(model, train_loader, val_loader, n_epochs=MAX_EPOCHS,
                    lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE,
                    save_path=None):
    """Train surrogate with early stopping and cosine annealing.

    Architecture-agnostic: works with LSTM, GRU, Transformer, and Linear.
    All models return (y_pred,) from forward(), so model(x)[0] works.

    Parameters
    ----------
    model : nn.Module
        Any of the 4 HumanXxxSurrogate architectures.
    train_loader : DataLoader
    val_loader : DataLoader
    n_epochs : int
    lr : float
    patience : int
    save_path : str or Path, optional

    Returns
    -------
    model : nn.Module (best weights loaded)
    history : dict with 'train_loss', 'val_loss' lists
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    if save_path is None:
        save_path = f'human_surrogate_h{model.hidden_size}_best.pt'
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model(X_batch)[0]
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_train += X_batch.size(0)

        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                Y_pred = model(X_batch)[0]
                loss = criterion(Y_pred, Y_batch)
                val_loss += loss.item() * X_batch.size(0)
                n_val += X_batch.size(0)

        val_loss /= max(n_val, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= patience:
            logger.info(
                "Epoch %d/%d  train=%.6f  val=%.6f  best=%.6f  patience=%d/%d",
                epoch, n_epochs, train_loss, val_loss, best_val_loss,
                patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    logger.info("Loaded best model from %s (val_loss=%.6f)",
                save_path, best_val_loss)
    return model, history


def compute_cross_condition_correlation(model, X_test, Y_test, trial_types):
    """Compute CC between predicted and actual frontal activity.

    Parameters
    ----------
    model : nn.Module
    X_test : ndarray, (n_trials, T, input_dim)
    Y_test : ndarray, (n_trials, T, output_dim)
    trial_types : ndarray, (n_trials,) -- integer condition labels

    Returns
    -------
    cc : float
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_pred = model(X_tensor)[0].cpu().numpy()

    conditions = np.unique(trial_types)
    pred_means = []
    true_means = []
    for c in conditions:
        mask = trial_types == c
        pred_means.append(Y_pred[mask].mean(axis=0).flatten())
        true_means.append(Y_test[mask].mean(axis=0).flatten())

    pred_concat = np.concatenate(pred_means)
    true_concat = np.concatenate(true_means)

    cc = np.corrcoef(pred_concat, true_concat)[0, 1]
    return float(cc)


def train_all_sizes(splits, session_info, output_dir, arch_name='lstm'):
    """Train surrogates at all hidden sizes for one patient + architecture.

    Parameters
    ----------
    splits : dict from split_data
    session_info : dict
    output_dir : str or Path
    arch_name : str

    Returns
    -------
    results : dict mapping hidden_size -> {model_path, cc, history}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = splits['train']['X'].shape[2]
    output_dim = splits['train']['Y'].shape[2]

    train_loader = create_dataloader(splits['train']['X'], splits['train']['Y'])
    val_loader = create_dataloader(splits['val']['X'], splits['val']['Y'],
                                   shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for hs in HIDDEN_SIZES:
        logger.info("=== Training %s h=%d (in=%d, out=%d) ===",
                     arch_name, hs, input_dim, output_dim)

        model = create_model(
            arch_name=arch_name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hs,
        ).to(device)

        save_path = output_dir / f'human_{arch_name}_h{hs}_best.pt'
        model, history = train_surrogate(model, train_loader, val_loader,
                                         save_path=save_path)

        # Get trial_types for CC if available
        trial_types = splits['test'].get('trial_info', {}).get('loads', None)
        if trial_types is None:
            trial_types = np.zeros(splits['test']['X'].shape[0], dtype=int)

        cc = compute_cross_condition_correlation(
            model,
            splits['test']['X'],
            splits['test']['Y'],
            trial_types,
        )
        logger.info("%s h=%d  CC=%.3f", arch_name, hs, cc)

        results[hs] = {
            'model_path': str(save_path),
            'cc': cc,
            'n_params': model.count_parameters(),
            'history': history,
        }

    return results
```

Implement `human_wm/surrogate/extract_hidden.py`:

```python
"""
DESCARTES Human WM -- Hidden State Extraction

Extract hidden representations from trained AND untrained (random-init)
surrogates on the test set. Works with all 4 architectures via the
create_model factory.

The untrained baseline is critical:
    delta_R2 = R2_trained - R2_untrained
isolates learned representations from random-projection artifacts.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from human_wm.config import BATCH_SIZE, HIDDEN_SIZES
from human_wm.surrogate.architectures import create_model

logger = logging.getLogger(__name__)


def extract_hidden_states(model, test_X, batch_size=BATCH_SIZE):
    """Run model on test data and collect hidden states.

    Works with any architecture that supports forward(x, return_hidden=True).

    Parameters
    ----------
    model : nn.Module
    test_X : ndarray, (n_trials, T, input_dim)
    batch_size : int

    Returns
    -------
    hidden_avg : ndarray, (n_trials, hidden_dim)
        Trial-averaged hidden states (averaged over timesteps).
    hidden_full : ndarray, (n_trials, T, hidden_dim)
        Full timestep-level hidden states.
    """
    device = next(model.parameters()).device
    model.eval()

    all_hidden = []
    n_trials = test_X.shape[0]

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            end = min(start + batch_size, n_trials)
            x_batch = torch.tensor(
                test_X[start:end], dtype=torch.float32
            ).to(device)

            _, h_seq = model(x_batch, return_hidden=True)
            # h_seq: (batch, T, hidden_dim)
            all_hidden.append(h_seq.cpu().numpy())

    hidden_full = np.concatenate(all_hidden, axis=0)  # (n_trials, T, hidden)
    hidden_avg = hidden_full.mean(axis=1)  # (n_trials, hidden)

    logger.info(
        "Extracted hidden states: full=%s  avg=%s",
        hidden_full.shape, hidden_avg.shape,
    )
    return hidden_avg, hidden_full


def extract_trained_and_untrained(model_path, test_X, arch_name,
                                  input_dim, output_dim, hidden_size,
                                  save_dir=None):
    """Extract hidden states from trained model and untrained baseline.

    Uses the create_model factory to instantiate the correct architecture
    for both trained (loaded from checkpoint) and untrained (random init).

    Parameters
    ----------
    model_path : str or Path
    test_X : ndarray, (n_trials, T, input_dim)
    arch_name : str
        One of 'lstm', 'gru', 'transformer', 'linear'.
    input_dim : int
    output_dim : int
    hidden_size : int
    save_dir : str or Path, optional

    Returns
    -------
    trained_H : ndarray, (n_trials, hidden_dim)
    untrained_H : ndarray, (n_trials, hidden_dim)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Trained model
    trained_model = create_model(
        arch_name=arch_name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    trained_model.load_state_dict(state_dict)
    trained_H, _ = extract_hidden_states(trained_model, test_X)

    # Untrained baseline (random init, no weight loading)
    untrained_model = create_model(
        arch_name=arch_name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
    ).to(device)
    untrained_H, _ = extract_hidden_states(untrained_model, test_X)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_dir / f'human_{arch_name}_h{hidden_size}_trained.npz',
            hidden_states=trained_H,
        )
        np.savez_compressed(
            save_dir / f'human_{arch_name}_h{hidden_size}_untrained.npz',
            hidden_states=untrained_H,
        )
        logger.info("Saved hidden states to %s", save_dir)

    return trained_H, untrained_H


def extract_all_sizes(splits, model_dir, save_dir, arch_name='lstm'):
    """Extract hidden states for all hidden sizes and one architecture.

    Parameters
    ----------
    splits : dict from split_data
    model_dir : str or Path
    save_dir : str or Path
    arch_name : str

    Returns
    -------
    all_hidden : dict mapping hidden_size -> (trained_H, untrained_H)
    """
    model_dir = Path(model_dir)
    input_dim = splits['test']['X'].shape[2]
    output_dim = splits['test']['Y'].shape[2]

    all_hidden = {}
    for hs in HIDDEN_SIZES:
        model_path = model_dir / f'human_{arch_name}_h{hs}_best.pt'
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            continue

        logger.info("=== Extracting %s h=%d ===", arch_name, hs)
        trained_H, untrained_H = extract_trained_and_untrained(
            model_path, splits['test']['X'],
            arch_name=arch_name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hs,
            save_dir=save_dir,
        )
        all_hidden[hs] = (trained_H, untrained_H)

    return all_hidden
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_train.py -v 2>&1 | tail -30
```

5. Commit:
```
git add human_wm/surrogate/train.py human_wm/surrogate/extract_hidden.py \
  tests/human_wm/test_train.py
git commit -m "human_wm: architecture-agnostic training loop and hidden extraction"
```

---

### Task 8: Probe Targets (7 modules + __init__)

**Files:**
- `human_wm/targets/persistent_delay.py`
- `human_wm/targets/memory_load.py`
- `human_wm/targets/delay_stability.py`
- `human_wm/targets/recognition_decision.py`
- `human_wm/targets/concept_selectivity.py`
- `human_wm/targets/oscillatory.py`
- `human_wm/targets/population_sync.py`
- `human_wm/targets/__init__.py` (modify)
- `tests/human_wm/test_targets.py`

**Steps:**

1. Write `tests/human_wm/test_targets.py`:

```python
"""Tests for human_wm.targets -- all probe target computations."""

import numpy as np
import pandas as pd
import pytest

from human_wm.targets import compute_all_targets
from human_wm.targets.persistent_delay import compute as compute_persistent_delay
from human_wm.targets.memory_load import compute as compute_memory_load
from human_wm.targets.delay_stability import compute as compute_delay_stability
from human_wm.targets.recognition_decision import compute as compute_recognition_decision
from human_wm.targets.concept_selectivity import compute as compute_concept_selectivity
from human_wm.targets.oscillatory import compute_theta, compute_gamma
from human_wm.targets.population_sync import compute as compute_population_sync


def _make_synthetic_data(n_trials=60, n_bins=40, n_neurons=20, seed=42):
    """Create synthetic Y, trial_info, schema for target testing."""
    rng = np.random.RandomState(seed)

    Y = rng.randn(n_trials, n_bins, n_neurons).astype(np.float32)

    trial_info = pd.DataFrame({
        'start_time': np.arange(n_trials, dtype=float),
        'stop_time': np.arange(n_trials, dtype=float) + 3.0,
        'loads': rng.choice([1, 2, 3], size=n_trials),
        'in_out': rng.choice([0, 1], size=n_trials),
        'stimulus_image': rng.choice(
            ['img_A', 'img_B', 'img_C', 'img_D', 'img_E'],
            size=n_trials,
        ),
    })

    schema = {
        'delay_bins': (10, 30),       # bins 10-29 are delay period
        'probe_bins': (30, 40),       # bins 30-39 are probe period
        'encoding_bins': (0, 10),     # bins 0-9 are encoding period
        'bin_size_s': 0.05,
        'stimulus_column': 'stimulus_image',
        'load_column': 'loads',
        'match_column': 'in_out',
    }

    return Y, trial_info, schema


class TestPersistentDelay:
    def test_output_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_persistent_delay(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_finite_values(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_persistent_delay(Y, trial_info, schema)
        assert np.all(np.isfinite(result))


class TestMemoryLoad:
    def test_output_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_memory_load(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_finite_values(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_memory_load(Y, trial_info, schema)
        assert np.all(np.isfinite(result))


class TestDelayStability:
    def test_output_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_delay_stability(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_finite_values(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_delay_stability(Y, trial_info, schema)
        assert np.all(np.isfinite(result))


class TestRecognitionDecision:
    def test_output_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_recognition_decision(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_finite_values(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_recognition_decision(Y, trial_info, schema)
        assert np.all(np.isfinite(result))


class TestConceptSelectivity:
    def test_output_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_concept_selectivity(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_finite_values(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_concept_selectivity(Y, trial_info, schema)
        assert np.all(np.isfinite(result))


class TestOscillatory:
    def test_theta_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_theta(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_gamma_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_gamma(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_theta_nonnegative(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_theta(Y, trial_info, schema)
        assert np.all(result >= 0)

    def test_gamma_nonnegative(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_gamma(Y, trial_info, schema)
        assert np.all(result >= 0)


class TestPopulationSync:
    def test_output_shape(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_population_sync(Y, trial_info, schema)
        assert result.shape == (Y.shape[0],)

    def test_finite_values(self):
        Y, trial_info, schema = _make_synthetic_data()
        result = compute_population_sync(Y, trial_info, schema)
        assert np.all(np.isfinite(result))


class TestComputeAllTargets:
    def test_returns_dict(self):
        Y, trial_info, schema = _make_synthetic_data()
        targets = compute_all_targets(Y, trial_info, schema)
        assert isinstance(targets, dict)

    def test_expected_keys(self):
        Y, trial_info, schema = _make_synthetic_data()
        targets = compute_all_targets(Y, trial_info, schema)
        expected = {
            'persistent_delay', 'memory_load', 'delay_stability',
            'recognition_decision', 'concept_selectivity',
            'theta_modulation', 'gamma_modulation', 'population_synchrony',
        }
        assert expected.issubset(set(targets.keys()))

    def test_all_shapes_match(self):
        Y, trial_info, schema = _make_synthetic_data()
        targets = compute_all_targets(Y, trial_info, schema)
        for name, arr in targets.items():
            assert arr.shape == (Y.shape[0],), f"{name} has wrong shape: {arr.shape}"
```

2. Run test -- expect import errors (modules not created yet):
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_targets.py -x 2>&1 | head -20
```

3. Implement all target modules:

`human_wm/targets/persistent_delay.py`:
```python
"""
Probe target: Persistent delay activity.

Returns (n_trials,) mean frontal firing rate during the delay period.
This captures sustained activity that may maintain working memory content.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute(Y, trial_info, schema):
    """Compute mean frontal firing during delay bins.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
        Frontal population activity (output of surrogate).
    trial_info : DataFrame
        Trial metadata (not directly used here, but kept for API consistency).
    schema : dict
        Must contain 'delay_bins': (start_bin, end_bin).

    Returns
    -------
    target : ndarray, (n_trials,)
        Mean firing rate across neurons and delay bins per trial.
    """
    delay_start, delay_end = schema['delay_bins']
    delay_activity = Y[:, delay_start:delay_end, :]  # (n_trials, delay_len, n_neurons)
    target = np.mean(delay_activity, axis=(1, 2))     # (n_trials,)

    logger.debug(
        "persistent_delay: delay_bins=(%d,%d), mean=%.4f, std=%.4f",
        delay_start, delay_end, np.mean(target), np.std(target),
    )
    return target.astype(np.float64)
```

`human_wm/targets/memory_load.py`:
```python
"""
Probe target: Memory load projection.

Returns (n_trials,) projection of delay activity onto the load axis.
Uses LinearRegression of mean delay activity onto load (1/2/3 items)
to define the axis, then projects each trial onto it.
"""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def compute(Y, trial_info, schema):
    """Compute projection onto the memory load axis.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
        Must contain the load column (e.g., 'loads') with values 1/2/3.
    schema : dict
        Must contain 'delay_bins', 'load_column'.

    Returns
    -------
    target : ndarray, (n_trials,)
        Projection of each trial's delay activity onto load axis.
    """
    delay_start, delay_end = schema['delay_bins']
    load_col = schema.get('load_column', 'loads')

    # Mean delay activity per trial: (n_trials, n_neurons)
    delay_mean = np.mean(Y[:, delay_start:delay_end, :], axis=1)

    # Load labels
    loads = np.array(trial_info[load_col], dtype=np.float64)

    # Fit linear regression: delay_mean -> load
    # The regression coefficients define the "load axis" in neural space
    reg = LinearRegression()
    reg.fit(delay_mean, loads)

    # Project each trial onto load axis (dot product with coefficients)
    load_axis = reg.coef_  # (n_neurons,)
    load_axis_norm = load_axis / (np.linalg.norm(load_axis) + 1e-10)
    target = delay_mean @ load_axis_norm  # (n_trials,)

    logger.debug(
        "memory_load: R2=%.4f, axis_norm=%.4f",
        reg.score(delay_mean, loads), np.linalg.norm(load_axis),
    )
    return target.astype(np.float64)
```

`human_wm/targets/delay_stability.py`:
```python
"""
Probe target: Delay stability (temporal autocorrelation).

Returns (n_trials,) correlation between first-half and second-half
of delay period activity. High stability = persistent coding; low = drifting.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute(Y, trial_info, schema):
    """Compute delay stability as first/second half correlation.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
        Not directly used.
    schema : dict
        Must contain 'delay_bins': (start_bin, end_bin).

    Returns
    -------
    target : ndarray, (n_trials,)
        Pearson correlation between first-half and second-half delay
        activity (across neurons) for each trial.
    """
    delay_start, delay_end = schema['delay_bins']
    delay_activity = Y[:, delay_start:delay_end, :]  # (n_trials, delay_len, n_neurons)

    delay_len = delay_end - delay_start
    midpoint = delay_len // 2

    # Mean activity in each half: (n_trials, n_neurons)
    first_half = np.mean(delay_activity[:, :midpoint, :], axis=1)
    second_half = np.mean(delay_activity[:, midpoint:, :], axis=1)

    n_trials = Y.shape[0]
    target = np.zeros(n_trials, dtype=np.float64)

    for i in range(n_trials):
        v1 = first_half[i]
        v2 = second_half[i]
        std1 = np.std(v1)
        std2 = np.std(v2)

        if std1 < 1e-10 or std2 < 1e-10:
            target[i] = 0.0
        else:
            target[i] = float(np.corrcoef(v1, v2)[0, 1])

    logger.debug(
        "delay_stability: mean_corr=%.4f, std=%.4f",
        np.mean(target), np.std(target),
    )
    return target
```

`human_wm/targets/recognition_decision.py`:
```python
"""
Probe target: Recognition decision signal.

Returns (n_trials,) projection onto the match-vs-nonmatch axis
at probe time. Captures the neural divergence between in-set (match)
and out-of-set (nonmatch) trials.
"""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


def compute(Y, trial_info, schema):
    """Compute projection onto match-vs-nonmatch axis at probe.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
        Must contain the match column (e.g., 'in_out') with 0/1 values.
    schema : dict
        Must contain 'probe_bins', 'match_column'.

    Returns
    -------
    target : ndarray, (n_trials,)
        Projection of each trial's probe activity onto decision axis.
    """
    probe_start, probe_end = schema['probe_bins']
    match_col = schema.get('match_column', 'in_out')

    # Mean probe activity per trial: (n_trials, n_neurons)
    probe_mean = np.mean(Y[:, probe_start:probe_end, :], axis=1)

    # Match labels (0=nonmatch, 1=match)
    match_labels = np.array(trial_info[match_col], dtype=np.float64)

    # Fit linear regression: probe_mean -> match_label
    reg = LinearRegression()
    reg.fit(probe_mean, match_labels)

    # Project onto decision axis
    decision_axis = reg.coef_  # (n_neurons,)
    decision_axis_norm = decision_axis / (np.linalg.norm(decision_axis) + 1e-10)
    target = probe_mean @ decision_axis_norm  # (n_trials,)

    logger.debug(
        "recognition_decision: R2=%.4f, axis_norm=%.4f",
        reg.score(probe_mean, match_labels), np.linalg.norm(decision_axis),
    )
    return target.astype(np.float64)
```

`human_wm/targets/concept_selectivity.py`:
```python
"""
Probe target: Concept/stimulus selectivity.

Returns (n_trials,) F-statistic based selectivity using image_id or
stimulus_id from trial_info. Measures how much each trial's neural
pattern differentiates between stimulus identities.
"""

import logging

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def compute(Y, trial_info, schema):
    """Compute per-trial concept selectivity using leave-one-out F-stat.

    For each trial, compute how well the population activity during
    the encoding period discriminates stimulus identity. Uses the
    F-statistic from a one-way ANOVA across stimulus categories,
    then assigns each trial the F-stat computed over all other trials
    with the same neuron pattern.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
        Must contain stimulus identity column.
    schema : dict
        Must contain 'encoding_bins', 'stimulus_column'.

    Returns
    -------
    target : ndarray, (n_trials,)
        Per-trial selectivity score (F-stat contribution).
    """
    enc_start, enc_end = schema['encoding_bins']
    stim_col = schema.get('stimulus_column', 'stimulus_image')

    # Mean encoding activity per trial: (n_trials, n_neurons)
    enc_mean = np.mean(Y[:, enc_start:enc_end, :], axis=1)

    # Stimulus labels
    stim_labels = np.array(trial_info[stim_col])
    unique_stims = np.unique(stim_labels)
    n_stims = len(unique_stims)
    n_trials = Y.shape[0]
    n_neurons = Y.shape[2]

    if n_stims < 2:
        logger.warning("concept_selectivity: fewer than 2 unique stimuli, returning zeros")
        return np.zeros(n_trials, dtype=np.float64)

    # Compute per-neuron F-stat across stimulus categories
    # Then assign each trial the mean F-stat across neurons
    neuron_f_stats = np.zeros(n_neurons, dtype=np.float64)
    for j in range(n_neurons):
        groups = [enc_mean[stim_labels == s, j] for s in unique_stims]
        # Filter out groups with < 2 samples
        groups = [g for g in groups if len(g) >= 1]
        if len(groups) < 2:
            continue
        try:
            f_val, _ = sp_stats.f_oneway(*groups)
            if np.isfinite(f_val):
                neuron_f_stats[j] = f_val
        except Exception:
            pass

    # Per-trial selectivity: project each trial's activity onto the
    # selectivity axis (neurons weighted by their F-stat)
    f_weights = neuron_f_stats / (np.sum(neuron_f_stats) + 1e-10)
    target = enc_mean @ f_weights  # (n_trials,)

    logger.debug(
        "concept_selectivity: n_stims=%d, mean_F=%.4f, max_F=%.4f",
        n_stims, np.mean(neuron_f_stats), np.max(neuron_f_stats),
    )
    return target.astype(np.float64)
```

`human_wm/targets/oscillatory.py`:
```python
"""
Probe targets: Oscillatory modulation (theta and gamma).

Returns (n_trials,) bandpass amplitude of the population rate signal.
- Theta: 4-8 Hz
- Gamma: 30-80 Hz

Uses scipy.signal.butter + filtfilt for zero-phase bandpass filtering.
"""

import logging

import numpy as np
from scipy.signal import butter, filtfilt

from human_wm.config import GAMMA_BAND, THETA_BAND

logger = logging.getLogger(__name__)


def _bandpass_amplitude(signal, fs, low, high, order=3):
    """Bandpass filter a signal and return the analytic amplitude.

    Parameters
    ----------
    signal : ndarray, (n_samples,)
    fs : float
        Sampling frequency in Hz.
    low : float
        Low cutoff in Hz.
    high : float
        High cutoff in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    amplitude : float
        Mean envelope amplitude of the bandpass-filtered signal.
    """
    nyq = fs / 2.0

    # Guard against impossible filter specs
    if high >= nyq:
        high = nyq - 1.0
    if low >= high:
        return 0.0
    if len(signal) < (3 * order + 1):
        return 0.0

    b, a = butter(order, [low / nyq, high / nyq], btype='band')

    try:
        filtered = filtfilt(b, a, signal)
    except ValueError:
        return 0.0

    # Envelope via absolute value of analytic signal approximation
    amplitude = float(np.mean(np.abs(filtered)))
    return amplitude


def _compute_oscillation(Y, trial_info, schema, band):
    """Compute oscillatory amplitude for a given frequency band.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
    schema : dict
        Must contain 'delay_bins', 'bin_size_s'.
    band : tuple of (low_hz, high_hz)

    Returns
    -------
    target : ndarray, (n_trials,)
        Mean bandpass amplitude of population rate during delay.
    """
    delay_start, delay_end = schema['delay_bins']
    bin_size_s = schema.get('bin_size_s', 0.05)
    fs = 1.0 / bin_size_s  # Sampling frequency

    low_hz, high_hz = band

    # Population rate: mean across neurons at each time bin
    # (n_trials, n_bins)
    pop_rate = np.mean(Y, axis=2)

    # Extract delay period
    delay_rate = pop_rate[:, delay_start:delay_end]  # (n_trials, delay_len)

    n_trials = Y.shape[0]
    target = np.zeros(n_trials, dtype=np.float64)

    for i in range(n_trials):
        signal = delay_rate[i]
        target[i] = _bandpass_amplitude(signal, fs, low_hz, high_hz)

    return target


def compute_theta(Y, trial_info, schema):
    """Compute theta-band (4-8 Hz) oscillatory amplitude.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
    schema : dict

    Returns
    -------
    target : ndarray, (n_trials,)
    """
    result = _compute_oscillation(Y, trial_info, schema, THETA_BAND)
    logger.debug("theta_modulation: mean=%.6f, std=%.6f",
                 np.mean(result), np.std(result))
    return result


def compute_gamma(Y, trial_info, schema):
    """Compute gamma-band (30-80 Hz) oscillatory amplitude.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
    schema : dict

    Returns
    -------
    target : ndarray, (n_trials,)
    """
    result = _compute_oscillation(Y, trial_info, schema, GAMMA_BAND)
    logger.debug("gamma_modulation: mean=%.6f, std=%.6f",
                 np.mean(result), np.std(result))
    return result
```

`human_wm/targets/population_sync.py`:
```python
"""
Probe target: Population synchrony.

Returns (n_trials,) mean pairwise correlation during the delay period.
Measures how synchronized the population of frontal neurons is during
working memory maintenance.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute(Y, trial_info, schema):
    """Compute mean pairwise correlation during delay.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
    trial_info : DataFrame
        Not directly used.
    schema : dict
        Must contain 'delay_bins': (start_bin, end_bin).

    Returns
    -------
    target : ndarray, (n_trials,)
        Mean absolute pairwise correlation among neurons during delay.
    """
    delay_start, delay_end = schema['delay_bins']
    delay_activity = Y[:, delay_start:delay_end, :]  # (n_trials, delay_len, n_neurons)

    n_trials, delay_len, n_neurons = delay_activity.shape
    target = np.zeros(n_trials, dtype=np.float64)

    if n_neurons < 2:
        logger.warning("population_sync: fewer than 2 neurons, returning zeros")
        return target

    for i in range(n_trials):
        # (delay_len, n_neurons) -- time series for each neuron in this trial
        trial_data = delay_activity[i]

        # Check for constant neurons
        stds = np.std(trial_data, axis=0)
        valid_mask = stds > 1e-10
        n_valid = np.sum(valid_mask)

        if n_valid < 2:
            target[i] = 0.0
            continue

        # Correlation matrix among valid neurons
        corr_matrix = np.corrcoef(trial_data[:, valid_mask].T)

        # Mean of upper triangle (excluding diagonal)
        n = corr_matrix.shape[0]
        upper_mask = np.triu_indices(n, k=1)
        pairwise_corrs = corr_matrix[upper_mask]

        # Use absolute correlation (synchrony regardless of sign)
        target[i] = float(np.mean(np.abs(pairwise_corrs)))

    logger.debug(
        "population_synchrony: mean=%.4f, std=%.4f",
        np.mean(target), np.std(target),
    )
    return target
```

4. Modify `human_wm/targets/__init__.py`:

```python
"""
DESCARTES Human Universality -- Probe Targets

Each target module is a pure function:
    compute(Y, trial_info, schema) -> ndarray (n_trials,)

compute_all_targets() runs all targets and returns a dict.
"""

import logging

import numpy as np

from human_wm.targets.persistent_delay import compute as _persistent_delay
from human_wm.targets.memory_load import compute as _memory_load
from human_wm.targets.delay_stability import compute as _delay_stability
from human_wm.targets.recognition_decision import compute as _recognition_decision
from human_wm.targets.concept_selectivity import compute as _concept_selectivity
from human_wm.targets.oscillatory import compute_theta as _theta
from human_wm.targets.oscillatory import compute_gamma as _gamma
from human_wm.targets.population_sync import compute as _population_sync

logger = logging.getLogger(__name__)


# Registry mapping target_name -> compute function
_TARGET_REGISTRY = {
    'persistent_delay': _persistent_delay,
    'memory_load': _memory_load,
    'delay_stability': _delay_stability,
    'recognition_decision': _recognition_decision,
    'concept_selectivity': _concept_selectivity,
    'theta_modulation': _theta,
    'gamma_modulation': _gamma,
    'population_synchrony': _population_sync,
}


def compute_all_targets(Y, trial_info, schema, target_names=None):
    """Compute all (or selected) probe targets.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_bins, n_neurons)
        Frontal population activity.
    trial_info : DataFrame
        Trial metadata.
    schema : dict
        Timing and column name configuration.
    target_names : list of str, optional
        Subset of targets to compute. If None, computes all.

    Returns
    -------
    targets : dict
        Mapping target_name -> ndarray (n_trials,).
    """
    if target_names is None:
        target_names = list(_TARGET_REGISTRY.keys())

    targets = {}
    for name in target_names:
        if name not in _TARGET_REGISTRY:
            logger.warning("Unknown target '%s', skipping", name)
            continue
        try:
            result = _TARGET_REGISTRY[name](Y, trial_info, schema)
            assert result.shape == (Y.shape[0],), (
                f"Target '{name}' returned shape {result.shape}, "
                f"expected ({Y.shape[0]},)"
            )
            targets[name] = result
            logger.info("Computed target '%s': mean=%.4f, std=%.4f",
                        name, np.mean(result), np.std(result))
        except Exception as e:
            logger.error("Failed to compute target '%s': %s", name, e)
            targets[name] = np.zeros(Y.shape[0], dtype=np.float64)

    return targets
```

5. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_targets.py -v 2>&1 | tail -30
```

6. Commit:
```
git add human_wm/targets/persistent_delay.py human_wm/targets/memory_load.py \
  human_wm/targets/delay_stability.py human_wm/targets/recognition_decision.py \
  human_wm/targets/concept_selectivity.py human_wm/targets/oscillatory.py \
  human_wm/targets/population_sync.py human_wm/targets/__init__.py \
  tests/human_wm/test_targets.py
git commit -m "human_wm: 8 probe target computations with registry and tests"
```

---

### Task 9: Architecture-Specific Ablation (4 modules)

**Files:**
- `human_wm/ablation/recurrent.py`
- `human_wm/ablation/transformer.py`
- `human_wm/ablation/linear.py`
- `human_wm/ablation/dispatch.py`
- `tests/human_wm/test_ablation.py`

**Steps:**

1. Write `tests/human_wm/test_ablation.py`:

```python
"""Tests for human_wm.ablation -- architecture-specific ablation dispatch."""

import numpy as np
import pytest
import torch

from human_wm.surrogate.architectures import create_model
from human_wm.ablation.recurrent import forward_with_resample_recurrent
from human_wm.ablation.transformer import forward_with_resample_transformer
from human_wm.ablation.linear import forward_with_resample_linear
from human_wm.ablation.dispatch import (
    forward_with_resample,
    resample_ablation_multiarch,
)


def _make_model_and_data(arch_name, input_dim=8, output_dim=5,
                          hidden_size=16, n_trials=20, T=10):
    """Create a model and synthetic data for ablation testing."""
    model = create_model(arch_name, input_dim, output_dim, hidden_size)
    model.eval()

    rng = np.random.RandomState(42)
    test_inputs = torch.randn(n_trials, T, input_dim)
    hidden_states = rng.randn(n_trials, hidden_size).astype(np.float32)

    return model, test_inputs, hidden_states


class TestRecurrentAblation:
    def test_lstm_output_shape(self):
        model, inputs, hs = _make_model_and_data('lstm')
        clamp_dims = [0, 1, 2]
        output = forward_with_resample_recurrent(
            model, inputs, clamp_dims, hs, rng=np.random.RandomState(0),
            arch='lstm',
        )
        assert output.shape == (inputs.shape[0], inputs.shape[1], model.output_dim)

    def test_gru_output_shape(self):
        model, inputs, hs = _make_model_and_data('gru')
        clamp_dims = [0, 1, 2]
        output = forward_with_resample_recurrent(
            model, inputs, clamp_dims, hs, rng=np.random.RandomState(0),
            arch='gru',
        )
        assert output.shape == (inputs.shape[0], inputs.shape[1], model.output_dim)

    def test_empty_clamp_dims(self):
        model, inputs, hs = _make_model_and_data('lstm')
        output = forward_with_resample_recurrent(
            model, inputs, [], hs, rng=np.random.RandomState(0),
            arch='lstm',
        )
        assert output.shape[0] == inputs.shape[0]


class TestTransformerAblation:
    def test_output_shape(self):
        model, inputs, hs = _make_model_and_data('transformer')
        clamp_dims = [0, 1, 2]
        output = forward_with_resample_transformer(
            model, inputs, clamp_dims, hs, rng=np.random.RandomState(0),
        )
        assert output.shape == (inputs.shape[0], inputs.shape[1], model.output_dim)

    def test_empty_clamp_dims(self):
        model, inputs, hs = _make_model_and_data('transformer')
        output = forward_with_resample_transformer(
            model, inputs, [], hs, rng=np.random.RandomState(0),
        )
        assert output.shape[0] == inputs.shape[0]


class TestLinearAblation:
    def test_output_shape(self):
        model, inputs, hs = _make_model_and_data('linear')
        clamp_dims = [0, 1, 2]
        output = forward_with_resample_linear(
            model, inputs, clamp_dims, hs, rng=np.random.RandomState(0),
        )
        assert output.shape == (inputs.shape[0], inputs.shape[1], model.output_dim)

    def test_empty_clamp_dims(self):
        model, inputs, hs = _make_model_and_data('linear')
        output = forward_with_resample_linear(
            model, inputs, [], hs, rng=np.random.RandomState(0),
        )
        assert output.shape[0] == inputs.shape[0]


class TestDispatch:
    @pytest.mark.parametrize("arch", ['lstm', 'gru', 'transformer', 'linear'])
    def test_dispatch_routes_correctly(self, arch):
        model, inputs, hs = _make_model_and_data(arch)
        clamp_dims = [0, 1]
        output = forward_with_resample(
            model, inputs, clamp_dims, hs, rng=np.random.RandomState(0),
        )
        assert output.shape[0] == inputs.shape[0]

    @pytest.mark.parametrize("arch", ['lstm', 'gru', 'transformer', 'linear'])
    def test_resample_ablation_multiarch_runs(self, arch):
        model, inputs, hs = _make_model_and_data(arch, n_trials=30)
        model.eval()

        rng = np.random.RandomState(42)
        test_outputs = rng.randn(30, 10, model.output_dim).astype(np.float32)
        target_y = rng.randn(30).astype(np.float32)

        results, baseline_cc = resample_ablation_multiarch(
            model, inputs, test_outputs, target_y, hs,
            k_fractions=[0.10, 0.20], n_random_repeats=2,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert isinstance(baseline_cc, float)
        for r in results:
            assert 'k_frac' in r
            assert 'z_score' in r
            assert 'verdict' in r
```

2. Run test -- expect import errors:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_ablation.py -x 2>&1 | head -20
```

3. Implement all ablation modules:

`human_wm/ablation/recurrent.py`:
```python
"""
Architecture-specific ablation: Recurrent models (LSTM and GRU).

Manual timestep unrolling with hidden-state intervention at each step.
Resampled values replace clamped dims to preserve marginal statistics
(OOD-robust ablation per Grant et al. 2025).
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def forward_with_resample_recurrent(model, test_inputs, clamp_dims,
                                     hidden_states, rng, arch='lstm'):
    """Forward pass with resample ablation for LSTM or GRU.

    Manually unrolls one timestep at a time. After each step, replaces
    clamp_dims in h_t with resampled values from the empirical
    distribution. Modified h_t propagates to t+1.

    Parameters
    ----------
    model : nn.Module
        HumanLSTMSurrogate or HumanGRUSurrogate.
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    clamp_dims : list or ndarray of int
        Hidden dimensions to resample.
    hidden_states : ndarray, (n_samples, hidden_size)
        Empirical distribution source (trial-averaged hidden states).
    rng : np.random.RandomState
        Random number generator for resampling.
    arch : str
        'lstm' or 'gru'.

    Returns
    -------
    output : ndarray, (n_trials, T, output_dim)
    """
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)
    clamp_dims = list(clamp_dims)

    if len(clamp_dims) == 0:
        with torch.no_grad():
            result = model(test_inputs)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

    batch_size, T, _ = test_inputs.shape
    hidden_size = model.hidden_size
    n_layers = model.n_layers

    # Initialize hidden states
    h = torch.zeros(n_layers, batch_size, hidden_size, device=device)
    if arch == 'lstm':
        c = torch.zeros(n_layers, batch_size, hidden_size, device=device)

    # Pre-compute resample values (consistent across timesteps)
    resample_values = np.zeros((batch_size, len(clamp_dims)), dtype=np.float32)
    for j, d in enumerate(clamp_dims):
        col = hidden_states[:, d]
        resample_values[:, j] = rng.choice(col, size=batch_size, replace=True)
    resample_tensor = torch.tensor(resample_values, dtype=torch.float32,
                                    device=device)

    rnn_module = model.lstm if arch == 'lstm' else model.gru

    outputs = []
    with torch.no_grad():
        for t in range(T):
            x_t = test_inputs[:, t:t+1, :]  # (batch, 1, input_dim)

            if arch == 'lstm':
                _, (h, c) = rnn_module(x_t, (h, c))
            else:
                _, h = rnn_module(x_t, h)

            # Intervene on last layer's hidden state
            h_last = h[-1].clone()
            for j, d in enumerate(clamp_dims):
                h_last[:, d] = resample_tensor[:, j]
            h[-1] = h_last

            # Readout from last layer
            out_t = model.proj(h_last)  # (batch, output_dim)
            outputs.append(out_t)

    output = torch.stack(outputs, dim=1)  # (batch, T, output_dim)
    return output.cpu().numpy()
```

`human_wm/ablation/transformer.py`:
```python
"""
Architecture-specific ablation: Transformer model.

Runs full sequence through input projection and positional encoding,
then processes through each transformer layer with intervention between
layers (replacing clamp_dims with resampled values).
"""

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def forward_with_resample_transformer(model, test_inputs, clamp_dims,
                                       hidden_states, rng):
    """Forward pass with resample ablation for Transformer.

    Processes through input_proj + pos_enc, then intervenes between
    each transformer encoder layer by replacing clamp_dims in the
    hidden representation with resampled values.

    Parameters
    ----------
    model : nn.Module
        HumanTransformerSurrogate.
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    clamp_dims : list or ndarray of int
    hidden_states : ndarray, (n_samples, hidden_size)
        Empirical distribution source.
    rng : np.random.RandomState

    Returns
    -------
    output : ndarray, (n_trials, T, output_dim)
    """
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)
    clamp_dims = list(clamp_dims)

    if len(clamp_dims) == 0:
        with torch.no_grad():
            result = model(test_inputs)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

    batch_size, T, _ = test_inputs.shape

    # Pre-compute resample values: (batch_size, len(clamp_dims))
    resample_values = np.zeros((batch_size, len(clamp_dims)), dtype=np.float32)
    for j, d in enumerate(clamp_dims):
        col = hidden_states[:, d]
        resample_values[:, j] = rng.choice(col, size=batch_size, replace=True)
    resample_tensor = torch.tensor(resample_values, dtype=torch.float32,
                                    device=device)

    with torch.no_grad():
        # Input projection + positional encoding
        h = model.input_proj(test_inputs)
        h = model.pos_enc(h)  # (batch, T, hidden_size)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T)
        causal_mask = causal_mask.to(device)

        # Process through each transformer encoder layer with intervention
        for layer in model.transformer.layers:
            h = layer(h, src_mask=causal_mask)

            # Intervene: replace clamp_dims across all timesteps
            for j, d in enumerate(clamp_dims):
                h[:, :, d] = resample_tensor[:, j:j+1]  # broadcast over T

        # Output projection
        y = model.proj(h)  # (batch, T, output_dim)

    return y.cpu().numpy()
```

`human_wm/ablation/linear.py`:
```python
"""
Architecture-specific ablation: Linear model.

Direct intervention on the hidden representation between proj_in and proj_out.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def forward_with_resample_linear(model, test_inputs, clamp_dims,
                                  hidden_states, rng):
    """Forward pass with resample ablation for Linear model.

    Computes h = proj_in(x), replaces clamp_dims with resampled values,
    then computes y = proj_out(h).

    Parameters
    ----------
    model : nn.Module
        HumanLinearSurrogate.
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    clamp_dims : list or ndarray of int
    hidden_states : ndarray, (n_samples, hidden_size)
        Empirical distribution source.
    rng : np.random.RandomState

    Returns
    -------
    output : ndarray, (n_trials, T, output_dim)
    """
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)
    clamp_dims = list(clamp_dims)

    if len(clamp_dims) == 0:
        with torch.no_grad():
            result = model(test_inputs)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

    batch_size, T, _ = test_inputs.shape

    # Pre-compute resample values: (batch_size, len(clamp_dims))
    resample_values = np.zeros((batch_size, len(clamp_dims)), dtype=np.float32)
    for j, d in enumerate(clamp_dims):
        col = hidden_states[:, d]
        resample_values[:, j] = rng.choice(col, size=batch_size, replace=True)
    resample_tensor = torch.tensor(resample_values, dtype=torch.float32,
                                    device=device)

    with torch.no_grad():
        # Hidden representation
        h = model.proj_in(test_inputs)  # (batch, T, hidden_size)

        # Intervene: replace clamp_dims across all timesteps
        for j, d in enumerate(clamp_dims):
            h[:, :, d] = resample_tensor[:, j:j+1]  # broadcast over T

        # Output projection
        y = model.proj_out(h)  # (batch, T, output_dim)

    return y.cpu().numpy()
```

`human_wm/ablation/dispatch.py`:
```python
"""
Architecture dispatch for multi-architecture ablation.

Routes ablation calls to the correct architecture-specific forward function
based on model attributes. Wraps the full resample ablation protocol
(progressive clamping with random vs targeted comparison).
"""

import logging

import numpy as np
import torch
from scipy import stats

from descartes_core.config import (
    ABLATION_K_FRACTIONS,
    ABLATION_N_RANDOM,
    CAUSAL_Z_THRESHOLD,
)
from descartes_core.metrics import cross_condition_correlation

from human_wm.ablation.recurrent import forward_with_resample_recurrent
from human_wm.ablation.transformer import forward_with_resample_transformer
from human_wm.ablation.linear import forward_with_resample_linear

logger = logging.getLogger(__name__)


def forward_with_resample(model, test_inputs, clamp_dims, hidden_states,
                           rng=None):
    """Dispatch resample ablation to the correct architecture.

    Routes by attribute detection:
      - hasattr(model, 'lstm') -> recurrent (LSTM)
      - hasattr(model, 'gru')  -> recurrent (GRU)
      - hasattr(model, 'transformer') -> transformer
      - else -> linear

    Parameters
    ----------
    model : nn.Module
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    clamp_dims : list or ndarray of int
    hidden_states : ndarray, (n_samples, hidden_size)
    rng : np.random.RandomState, optional

    Returns
    -------
    output : ndarray, (n_trials, T, output_dim) or (n_trials, T)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    if hasattr(model, 'lstm'):
        return forward_with_resample_recurrent(
            model, test_inputs, clamp_dims, hidden_states, rng, arch='lstm',
        )
    elif hasattr(model, 'gru'):
        return forward_with_resample_recurrent(
            model, test_inputs, clamp_dims, hidden_states, rng, arch='gru',
        )
    elif hasattr(model, 'transformer'):
        return forward_with_resample_transformer(
            model, test_inputs, clamp_dims, hidden_states, rng,
        )
    else:
        return forward_with_resample_linear(
            model, test_inputs, clamp_dims, hidden_states, rng,
        )


def resample_ablation_multiarch(model, test_inputs, test_outputs, target_y,
                                 hidden_states, trial_types=None,
                                 cc_fn=None, k_fractions=None,
                                 n_random_repeats=None):
    """Resample ablation protocol using architecture-dispatched forward.

    Same protocol as descartes_core.ablation.resample_ablation but uses
    the multi-architecture dispatch forward function.

    Parameters
    ----------
    model : nn.Module
        Any of the 4 HumanXxxSurrogate architectures.
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    test_outputs : ndarray, (n_trials, T) or (n_trials, T, output_dim)
    target_y : ndarray, (n_trials,)
    hidden_states : ndarray, (n_trials, hidden_size)
    trial_types : ndarray, optional
    cc_fn : callable, optional
    k_fractions : list of float, optional
    n_random_repeats : int, optional

    Returns
    -------
    results : list of dict
    baseline_cc : float
    """
    if k_fractions is None:
        k_fractions = ABLATION_K_FRACTIONS
    if n_random_repeats is None:
        n_random_repeats = ABLATION_N_RANDOM

    if cc_fn is None:
        if trial_types is not None:
            from descartes_core.metrics import cross_condition_correlation_grouped
            def cc_fn(pred, actual):
                return cross_condition_correlation_grouped(
                    pred, actual, trial_types
                )
        else:
            cc_fn = cross_condition_correlation

    hidden_size = hidden_states.shape[1]

    # Rank hidden dims by correlation with target
    correlations = np.array([
        abs(float(stats.pearsonr(hidden_states[:, d], target_y)[0]))
        if np.std(hidden_states[:, d]) > 1e-10 else 0.0
        for d in range(hidden_size)
    ])
    sorted_dims = np.argsort(correlations)[::-1]

    # Baseline: intact model (no clamping)
    intact_output = forward_with_resample(
        model, test_inputs, [], hidden_states, rng=np.random.RandomState(99),
    )
    baseline_cc = cc_fn(intact_output, test_outputs)

    rng = np.random.RandomState(99)
    results = []

    for k_frac in k_fractions:
        n_clamp = max(1, int(round(k_frac * hidden_size)))
        target_dims = sorted_dims[:n_clamp]

        # Targeted ablation: clamp most-correlated dims
        target_output = forward_with_resample(
            model, test_inputs, target_dims, hidden_states, rng=rng,
        )
        target_cc = cc_fn(target_output, test_outputs)

        # Random ablation: clamp random dims (repeated)
        random_ccs = []
        for _ in range(n_random_repeats):
            rand_dims = rng.choice(hidden_size, size=n_clamp, replace=False)
            rand_output = forward_with_resample(
                model, test_inputs, rand_dims, hidden_states, rng=rng,
            )
            rand_cc = cc_fn(rand_output, test_outputs)
            random_ccs.append(rand_cc)

        random_mean = float(np.mean(random_ccs))
        random_std = float(np.std(random_ccs))

        if random_std > 1e-10:
            z_score = (target_cc - random_mean) / random_std
        else:
            z_score = -10.0 if target_cc < random_mean else 0.0

        verdict = 'CAUSAL' if z_score < CAUSAL_Z_THRESHOLD else 'NON_CAUSAL'

        results.append({
            'k_frac': float(k_frac),
            'n_clamped': int(n_clamp),
            'target_cc': float(target_cc),
            'target_cc_drop': float(baseline_cc - target_cc),
            'random_cc_mean': random_mean,
            'random_cc_std': random_std,
            'z_score': float(z_score),
            'verdict': verdict,
        })

        logger.info(
            "  [multiarch-resample] k=%.0f%%: target_cc=%.3f  z=%.2f  [%s]",
            k_frac * 100, target_cc, z_score, verdict,
        )

    return results, float(baseline_cc)
```

4. Run test -- expect all to pass:
```bash
cd "Working memory" && python -m pytest tests/human_wm/test_ablation.py -v 2>&1 | tail -30
```

5. Commit:
```
git add human_wm/ablation/recurrent.py human_wm/ablation/transformer.py \
  human_wm/ablation/linear.py human_wm/ablation/dispatch.py \
  tests/human_wm/test_ablation.py
git commit -m "human_wm: architecture-specific ablation with dispatch for LSTM/GRU/Transformer/Linear"
```

---

### Task 10: Single Patient Pipeline

**Files:**
- `human_wm/analysis/single_patient.py`
- `scripts/13_human_single_patient.py`

**Steps:**

1. Implement `human_wm/analysis/single_patient.py`:

```python
"""
DESCARTES Human Universality -- Single Patient Pipeline

Runs the full DESCARTES zombie test for one patient:
  1. Split data
  2. Create + train surrogate
  3. Check quality (CC)
  4. Extract hidden states (trained + untrained)
  5. Compute all probe targets
  6. For each target: probe -> ablation -> classify
  7. Return results dict
"""

import logging

import numpy as np
import torch

from descartes_core.ablation import classify_mandatory_type
from descartes_core.classify import classify_variable
from descartes_core.ridge_probe import probe_single_variable

from human_wm.ablation.dispatch import (
    forward_with_resample,
    resample_ablation_multiarch,
)
from human_wm.config import (
    BATCH_SIZE,
    HIDDEN_SIZES,
    MIN_CC_THRESHOLD,
    N_SEEDS,
)
from human_wm.data.loader import split_data
from human_wm.surrogate.architectures import create_model
from human_wm.surrogate.extract_hidden import extract_hidden_states
from human_wm.surrogate.train import (
    compute_cross_condition_correlation,
    create_dataloader,
    train_surrogate,
)
from human_wm.targets import compute_all_targets

logger = logging.getLogger(__name__)


def run_single_patient(X, Y, trial_info, schema, arch_name='lstm',
                       hidden_size=128, seed=42, save_dir=None):
    """Run full DESCARTES pipeline for one patient.

    Parameters
    ----------
    X : ndarray, (n_trials, n_bins, n_mtl)
        MTL input activity.
    Y : ndarray, (n_trials, n_bins, n_frontal)
        Frontal output activity (ground truth).
    trial_info : DataFrame
        Trial metadata with load, match, stimulus columns.
    schema : dict
        NWB schema with timing info and column names.
    arch_name : str
        One of 'lstm', 'gru', 'transformer', 'linear'.
    hidden_size : int
        Hidden dimension for the surrogate.
    seed : int
        Random seed for reproducibility.
    save_dir : str or Path, optional
        Directory to save model checkpoints and results.

    Returns
    -------
    results : dict
        Per-target classifications, quality metrics, and metadata.
        Returns None if quality check fails.
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("=== Single patient pipeline: arch=%s h=%d seed=%d ===",
                arch_name, hidden_size, seed)

    # --- Step 1: Split data ---
    splits = split_data(X, Y, trial_info, seed=seed)
    input_dim = X.shape[2]
    output_dim = Y.shape[2]

    logger.info("Data split: train=%d val=%d test=%d",
                splits['train']['X'].shape[0],
                splits['val']['X'].shape[0],
                splits['test']['X'].shape[0])

    # --- Step 2: Create and train model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(arch_name, input_dim, output_dim, hidden_size)
    model = model.to(device)

    train_loader = create_dataloader(splits['train']['X'], splits['train']['Y'])
    val_loader = create_dataloader(splits['val']['X'], splits['val']['Y'],
                                    shuffle=False)

    save_path = None
    if save_dir is not None:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'human_{arch_name}_h{hidden_size}_s{seed}_best.pt'

    model, history = train_surrogate(model, train_loader, val_loader,
                                      save_path=save_path)

    # --- Step 3: Check quality ---
    # Build trial types for CC computation (use load as condition)
    load_col = schema.get('load_column', 'loads')
    test_trial_info = splits['test']['trial_info']
    trial_types = np.array(test_trial_info[load_col], dtype=int)

    cc = compute_cross_condition_correlation(
        model, splits['test']['X'], splits['test']['Y'], trial_types,
    )
    logger.info("Surrogate quality CC = %.4f (threshold = %.2f)",
                cc, MIN_CC_THRESHOLD)

    if cc < MIN_CC_THRESHOLD:
        logger.warning("CC %.4f below threshold %.2f -- skipping patient",
                       cc, MIN_CC_THRESHOLD)
        return {
            'status': 'QUALITY_FAIL',
            'cc': cc,
            'arch': arch_name,
            'hidden_size': hidden_size,
            'seed': seed,
        }

    # --- Step 4: Extract hidden states ---
    trained_H, _ = extract_hidden_states(model, splits['test']['X'])

    # Untrained baseline
    untrained_model = create_model(arch_name, input_dim, output_dim,
                                    hidden_size).to(device)
    untrained_H, _ = extract_hidden_states(untrained_model, splits['test']['X'])

    # --- Step 5: Compute all targets ---
    targets = compute_all_targets(
        splits['test']['Y'], test_trial_info, schema,
    )

    # --- Step 6: Probe + ablate + classify each target ---
    test_inputs_tensor = torch.tensor(
        splits['test']['X'], dtype=torch.float32,
    )
    test_Y = splits['test']['Y']

    classifications = {}
    for target_name, target_y in targets.items():
        logger.info("--- Probing target: %s ---", target_name)

        # Ridge probe
        ridge_result = probe_single_variable(
            trained_H, untrained_H, target_y, target_name,
        )

        if ridge_result['category'] == 'LEARNED':
            # Run resample ablation (multi-arch)
            ablation_steps, baseline_cc = resample_ablation_multiarch(
                model, test_inputs_tensor, test_Y, target_y, trained_H,
                trial_types=trial_types,
            )
            abl_class, breaking_point = classify_mandatory_type(
                ablation_steps, baseline_cc,
            )
            ablation_result = {
                'classification': abl_class,
                'breaking_point': breaking_point,
                'baseline_cc': baseline_cc,
                'ablation_steps': ablation_steps,
            }
        else:
            ablation_result = None

        # Final classification
        final = classify_variable(ridge_result, ablation_result)
        classifications[target_name] = final

        logger.info(
            "  %s -> %s (delta_R2=%.4f)",
            target_name, final['final_category'],
            ridge_result.get('delta_R2', 0.0),
        )

    # --- Step 7: Package results ---
    results = {
        'status': 'OK',
        'cc': cc,
        'arch': arch_name,
        'hidden_size': hidden_size,
        'seed': seed,
        'n_test_trials': splits['test']['X'].shape[0],
        'classifications': classifications,
        'summary': {
            cat: sum(
                1 for c in classifications.values()
                if c['final_category'] == cat
            )
            for cat in [
                'ZOMBIE', 'LEARNED_BYPRODUCT',
                'MANDATORY_CONCENTRATED', 'MANDATORY_DISTRIBUTED',
                'MANDATORY_REDUNDANT',
            ]
        },
    }

    logger.info("Pipeline complete: %s", results['summary'])
    return results
```

2. Implement `scripts/13_human_single_patient.py`:

```python
#!/usr/bin/env python
"""
Script 13: Run DESCARTES single-patient pipeline on best human patient.

Usage:
    cd "Working memory"
    python scripts/13_human_single_patient.py

Loads the best patient from the inventory, runs the full pipeline,
and saves results to human_results/.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from human_wm.analysis.single_patient import run_single_patient
from human_wm.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    # Load schema
    schema = load_nwb_schema()
    if schema is None:
        logger.error("NWB schema not found. Run 10_human_explore_nwb.py first.")
        sys.exit(1)

    # Load inventory to find best patient
    inventory_path = PROCESSED_DIR / 'patient_inventory.json'
    if not inventory_path.exists():
        logger.error("Patient inventory not found at %s. "
                     "Run 11_human_build_inventory.py first.", inventory_path)
        sys.exit(1)

    with open(inventory_path) as f:
        inventory = json.load(f)

    # Find best patient (most trials * neurons product)
    best_patient = None
    best_score = 0
    for patient in inventory:
        if not patient.get('usable', False):
            continue
        score = patient.get('n_trials', 0) * patient.get('n_frontal', 0)
        if score > best_score:
            best_score = score
            best_patient = patient

    if best_patient is None:
        logger.error("No usable patients in inventory.")
        sys.exit(1)

    patient_id = best_patient['patient_id']
    logger.info("Selected best patient: %s (score=%d)", patient_id, best_score)

    # Load processed data
    data_path = PROCESSED_DIR / f'{patient_id}_processed.npz'
    if not data_path.exists():
        logger.error("Processed data not found: %s", data_path)
        sys.exit(1)

    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    Y = data['Y']

    import pandas as pd
    trial_info = pd.DataFrame(data['trial_info'].item())

    logger.info("Loaded data: X=%s  Y=%s  trials=%d", X.shape, Y.shape,
                len(trial_info))

    # Run pipeline
    results = run_single_patient(X, Y, trial_info, schema)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f'{patient_id}_single_patient_results.json'

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(_convert(results), f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Print summary
    if results['status'] == 'OK':
        print(f"\n{'='*60}")
        print(f"  Single Patient Results: {patient_id}")
        print(f"  Architecture: {results['arch']}  Hidden: {results['hidden_size']}")
        print(f"  Quality CC: {results['cc']:.4f}")
        print(f"{'='*60}")
        for name, cls in results['classifications'].items():
            print(f"  {name:<25s}  {cls['final_category']}")
        print(f"{'='*60}\n")
    else:
        print(f"\nQuality check failed: CC={results['cc']:.4f}\n")


if __name__ == '__main__':
    main()
```

3. Commit:
```
git add human_wm/analysis/single_patient.py scripts/13_human_single_patient.py
git commit -m "human_wm: single patient DESCARTES pipeline with script"
```

---

### Task 11: Cross-Seed Test

**Files:**
- `human_wm/analysis/cross_seed.py`
- `scripts/14_human_cross_seed.py`

**Steps:**

1. Implement `human_wm/analysis/cross_seed.py`:

```python
"""
DESCARTES Human Universality -- Cross-Seed Test

Runs the single-patient pipeline across N_SEEDS random seeds
to test whether mandatory classifications are robust to
training initialization.

Verdict thresholds:
  >= 8/10 seeds mandatory -> ROBUST
  >= 5/10 seeds mandatory -> MODERATE
  <  5/10 seeds mandatory -> FRAGILE
"""

import logging

import numpy as np
import torch

from human_wm.analysis.single_patient import run_single_patient
from human_wm.config import N_SEEDS

logger = logging.getLogger(__name__)

# Classification verdicts
ROBUST_THRESHOLD = 0.8     # >= 80% of seeds -> ROBUST
MODERATE_THRESHOLD = 0.5   # >= 50% of seeds -> MODERATE


def cross_seed_test(X, Y, trial_info, schema, n_seeds=None,
                    arch_name='lstm', hidden_size=128):
    """Run cross-seed universality test for one patient.

    Parameters
    ----------
    X : ndarray, (n_trials, n_bins, n_mtl)
    Y : ndarray, (n_trials, n_bins, n_frontal)
    trial_info : DataFrame
    schema : dict
    n_seeds : int, optional
        Number of seeds to test. Defaults to N_SEEDS from config.
    arch_name : str
    hidden_size : int

    Returns
    -------
    summary : dict
        Per-target summary with {n_mandatory, n_total, verdict, per_seed}.
    """
    if n_seeds is None:
        n_seeds = N_SEEDS

    logger.info("=== Cross-seed test: %d seeds, arch=%s, h=%d ===",
                n_seeds, arch_name, hidden_size)

    # Collect results across seeds
    all_seed_results = []
    for seed in range(n_seeds):
        logger.info("--- Seed %d/%d ---", seed + 1, n_seeds)
        result = run_single_patient(
            X, Y, trial_info, schema,
            arch_name=arch_name,
            hidden_size=hidden_size,
            seed=seed,
        )
        all_seed_results.append(result)

    # Aggregate per target
    # Collect all target names from successful runs
    target_names = set()
    for result in all_seed_results:
        if result['status'] == 'OK':
            target_names.update(result['classifications'].keys())

    summary = {}
    for target_name in sorted(target_names):
        n_mandatory = 0
        n_zombie = 0
        n_total = 0
        per_seed = []

        for seed_idx, result in enumerate(all_seed_results):
            if result['status'] != 'OK':
                per_seed.append({
                    'seed': seed_idx,
                    'status': 'QUALITY_FAIL',
                    'category': None,
                })
                continue

            n_total += 1
            cls = result['classifications'].get(target_name, {})
            category = cls.get('final_category', 'ZOMBIE')

            is_mandatory = category.startswith('MANDATORY')
            if is_mandatory:
                n_mandatory += 1
            if category == 'ZOMBIE':
                n_zombie += 1

            per_seed.append({
                'seed': seed_idx,
                'status': 'OK',
                'category': category,
            })

        # Determine verdict
        if n_total == 0:
            verdict = 'NO_DATA'
        else:
            frac = n_mandatory / n_total
            if frac >= ROBUST_THRESHOLD:
                verdict = 'ROBUST'
            elif frac >= MODERATE_THRESHOLD:
                verdict = 'MODERATE'
            else:
                verdict = 'FRAGILE'

        summary[target_name] = {
            'n_mandatory': n_mandatory,
            'n_zombie': n_zombie,
            'n_total': n_total,
            'frac_mandatory': n_mandatory / max(n_total, 1),
            'verdict': verdict,
            'per_seed': per_seed,
        }

        logger.info(
            "  %s: %d/%d mandatory -> %s",
            target_name, n_mandatory, n_total, verdict,
        )

    return summary
```

2. Implement `scripts/14_human_cross_seed.py`:

```python
#!/usr/bin/env python
"""
Script 14: Run cross-seed universality test.

Usage:
    cd "Working memory"
    python scripts/14_human_cross_seed.py

Tests whether mandatory classifications are robust across 10 random seeds.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from human_wm.analysis.cross_seed import cross_seed_test
from human_wm.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    schema = load_nwb_schema()
    if schema is None:
        logger.error("NWB schema not found.")
        sys.exit(1)

    # Load best patient (same logic as script 13)
    inventory_path = PROCESSED_DIR / 'patient_inventory.json'
    with open(inventory_path) as f:
        inventory = json.load(f)

    best_patient = max(
        (p for p in inventory if p.get('usable', False)),
        key=lambda p: p.get('n_trials', 0) * p.get('n_frontal', 0),
        default=None,
    )
    if best_patient is None:
        logger.error("No usable patients.")
        sys.exit(1)

    patient_id = best_patient['patient_id']
    data = np.load(PROCESSED_DIR / f'{patient_id}_processed.npz',
                   allow_pickle=True)
    X = data['X']
    Y = data['Y']
    trial_info = pd.DataFrame(data['trial_info'].item())

    # Run cross-seed test
    summary = cross_seed_test(X, Y, trial_info, schema)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f'{patient_id}_cross_seed_results.json'

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(_convert(summary), f, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Cross-Seed Results: {patient_id}  (10 seeds)")
    print(f"{'='*60}")
    print(f"  {'Target':<25s}  {'Mandatory':>9s}  {'Total':>5s}  {'Verdict':<10s}")
    print(f"  {'-'*55}")
    for name, info in sorted(summary.items()):
        print(f"  {name:<25s}  {info['n_mandatory']:>5d}/{info['n_total']:<3d}  "
              f"       {info['verdict']:<10s}")
    print(f"{'='*60}\n")

    logger.info("Saved to %s", output_path)


if __name__ == '__main__':
    main()
```

3. Commit:
```
git add human_wm/analysis/cross_seed.py scripts/14_human_cross_seed.py
git commit -m "human_wm: cross-seed universality test (10 seeds per patient)"
```

---

### Task 12: Cross-Patient + Cross-Architecture Tests

**Files:**
- `human_wm/analysis/cross_patient.py`
- `human_wm/analysis/cross_architecture.py`
- `scripts/15_human_cross_patient.py`
- `scripts/16_human_cross_architecture.py`

**Steps:**

1. Implement `human_wm/analysis/cross_patient.py`:

```python
"""
DESCARTES Human Universality -- Cross-Patient Test

Runs the single-patient pipeline across all usable patients in the
inventory to test whether mandatory variables generalize.

Verdict threshold:
  >= 80% of patients mandatory -> UNIVERSAL
  <  80% -> NOT_UNIVERSAL
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from human_wm.analysis.single_patient import run_single_patient
from human_wm.config import PROCESSED_DIR

logger = logging.getLogger(__name__)

UNIVERSAL_THRESHOLD = 0.80  # >= 80% of patients -> UNIVERSAL


def cross_patient_test(inventory, schema, arch_name='lstm', hidden_size=128,
                       seed=42):
    """Run cross-patient universality test.

    Parameters
    ----------
    inventory : list of dict
        Patient inventory from patient_inventory.json.
        Each dict must have 'patient_id', 'usable', and data file info.
    schema : dict
        NWB schema.
    arch_name : str
    hidden_size : int
    seed : int

    Returns
    -------
    summary : dict
        Per-target summary with {n_mandatory, n_patients, frac, verdict,
        per_patient}.
    """
    usable_patients = [p for p in inventory if p.get('usable', False)]
    n_patients = len(usable_patients)

    logger.info("=== Cross-patient test: %d usable patients, arch=%s ===",
                n_patients, arch_name)

    all_patient_results = {}
    for patient in usable_patients:
        patient_id = patient['patient_id']
        logger.info("--- Patient: %s ---", patient_id)

        data_path = PROCESSED_DIR / f'{patient_id}_processed.npz'
        if not data_path.exists():
            logger.warning("Data not found for %s, skipping", patient_id)
            all_patient_results[patient_id] = None
            continue

        try:
            data = np.load(data_path, allow_pickle=True)
            X = data['X']
            Y = data['Y']
            trial_info = pd.DataFrame(data['trial_info'].item())

            result = run_single_patient(
                X, Y, trial_info, schema,
                arch_name=arch_name,
                hidden_size=hidden_size,
                seed=seed,
            )
            all_patient_results[patient_id] = result
        except Exception as e:
            logger.error("Failed for patient %s: %s", patient_id, e)
            all_patient_results[patient_id] = None

    # Aggregate per target
    target_names = set()
    for result in all_patient_results.values():
        if result is not None and result.get('status') == 'OK':
            target_names.update(result['classifications'].keys())

    summary = {}
    for target_name in sorted(target_names):
        n_mandatory = 0
        n_tested = 0
        per_patient = []

        for patient_id, result in all_patient_results.items():
            if result is None or result.get('status') != 'OK':
                per_patient.append({
                    'patient_id': patient_id,
                    'status': 'SKIP',
                    'category': None,
                })
                continue

            n_tested += 1
            cls = result['classifications'].get(target_name, {})
            category = cls.get('final_category', 'ZOMBIE')

            if category.startswith('MANDATORY'):
                n_mandatory += 1

            per_patient.append({
                'patient_id': patient_id,
                'status': 'OK',
                'category': category,
            })

        frac = n_mandatory / max(n_tested, 1)
        verdict = 'UNIVERSAL' if frac >= UNIVERSAL_THRESHOLD else 'NOT_UNIVERSAL'

        summary[target_name] = {
            'n_mandatory': n_mandatory,
            'n_tested': n_tested,
            'n_total_patients': n_patients,
            'frac_mandatory': frac,
            'verdict': verdict,
            'per_patient': per_patient,
        }

        logger.info("  %s: %d/%d mandatory (%.0f%%) -> %s",
                     target_name, n_mandatory, n_tested, frac * 100, verdict)

    return summary
```

2. Implement `human_wm/analysis/cross_architecture.py`:

```python
"""
DESCARTES Human Universality -- Cross-Architecture Test

Runs the single-patient pipeline across all 4 architectures
(LSTM, GRU, Transformer, Linear) to test whether mandatory
variables are architecture-independent.

Verdict threshold:
  >= 3/4 architectures mandatory -> UNIVERSAL
  <  3/4 -> NOT_UNIVERSAL
"""

import logging

import numpy as np

from human_wm.analysis.single_patient import run_single_patient
from human_wm.config import ARCHITECTURES

logger = logging.getLogger(__name__)

ARCH_UNIVERSAL_THRESHOLD = 3  # >= 3 out of 4 architectures -> UNIVERSAL


def cross_architecture_test(X, Y, trial_info, schema, hidden_size=128,
                             seed=42, architectures=None):
    """Run cross-architecture universality test for one patient.

    Parameters
    ----------
    X : ndarray, (n_trials, n_bins, n_mtl)
    Y : ndarray, (n_trials, n_bins, n_frontal)
    trial_info : DataFrame
    schema : dict
    hidden_size : int
    seed : int
    architectures : list of str, optional
        Architectures to test. Defaults to ARCHITECTURES from config.

    Returns
    -------
    summary : dict
        Per-target summary with {n_mandatory, n_archs, verdict, per_arch}.
    """
    if architectures is None:
        architectures = ARCHITECTURES

    n_archs = len(architectures)
    logger.info("=== Cross-architecture test: %d architectures ===", n_archs)

    all_arch_results = {}
    for arch_name in architectures:
        logger.info("--- Architecture: %s ---", arch_name)
        try:
            result = run_single_patient(
                X, Y, trial_info, schema,
                arch_name=arch_name,
                hidden_size=hidden_size,
                seed=seed,
            )
            all_arch_results[arch_name] = result
        except Exception as e:
            logger.error("Failed for architecture %s: %s", arch_name, e)
            all_arch_results[arch_name] = None

    # Aggregate per target
    target_names = set()
    for result in all_arch_results.values():
        if result is not None and result.get('status') == 'OK':
            target_names.update(result['classifications'].keys())

    summary = {}
    for target_name in sorted(target_names):
        n_mandatory = 0
        n_tested = 0
        per_arch = []

        for arch_name in architectures:
            result = all_arch_results.get(arch_name)
            if result is None or result.get('status') != 'OK':
                per_arch.append({
                    'arch': arch_name,
                    'status': 'FAIL',
                    'category': None,
                })
                continue

            n_tested += 1
            cls = result['classifications'].get(target_name, {})
            category = cls.get('final_category', 'ZOMBIE')

            if category.startswith('MANDATORY'):
                n_mandatory += 1

            per_arch.append({
                'arch': arch_name,
                'status': 'OK',
                'category': category,
                'cc': result.get('cc', None),
            })

        verdict = ('UNIVERSAL' if n_mandatory >= ARCH_UNIVERSAL_THRESHOLD
                    else 'NOT_UNIVERSAL')

        summary[target_name] = {
            'n_mandatory': n_mandatory,
            'n_tested': n_tested,
            'n_total_archs': n_archs,
            'verdict': verdict,
            'per_arch': per_arch,
        }

        logger.info("  %s: %d/%d architectures mandatory -> %s",
                     target_name, n_mandatory, n_tested, verdict)

    return summary
```

3. Implement `scripts/15_human_cross_patient.py`:

```python
#!/usr/bin/env python
"""
Script 15: Run cross-patient universality test.

Usage:
    cd "Working memory"
    python scripts/15_human_cross_patient.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from human_wm.analysis.cross_patient import cross_patient_test
from human_wm.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    schema = load_nwb_schema()
    if schema is None:
        logger.error("NWB schema not found.")
        sys.exit(1)

    inventory_path = PROCESSED_DIR / 'patient_inventory.json'
    with open(inventory_path) as f:
        inventory = json.load(f)

    summary = cross_patient_test(inventory, schema)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / 'cross_patient_results.json'

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(_convert(summary), f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Cross-Patient Universality Results")
    print(f"{'='*65}")
    print(f"  {'Target':<25s}  {'Mandatory':>9s}  {'Tested':>6s}  {'Verdict':<15s}")
    print(f"  {'-'*60}")
    for name, info in sorted(summary.items()):
        print(f"  {name:<25s}  {info['n_mandatory']:>5d}/{info['n_tested']:<3d}  "
              f"        {info['verdict']:<15s}")
    print(f"{'='*65}\n")

    logger.info("Saved to %s", output_path)


if __name__ == '__main__':
    main()
```

4. Implement `scripts/16_human_cross_architecture.py`:

```python
#!/usr/bin/env python
"""
Script 16: Run cross-architecture universality test.

Usage:
    cd "Working memory"
    python scripts/16_human_cross_architecture.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from human_wm.analysis.cross_architecture import cross_architecture_test
from human_wm.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    load_nwb_schema,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    schema = load_nwb_schema()
    if schema is None:
        logger.error("NWB schema not found.")
        sys.exit(1)

    # Load best patient
    inventory_path = PROCESSED_DIR / 'patient_inventory.json'
    with open(inventory_path) as f:
        inventory = json.load(f)

    best_patient = max(
        (p for p in inventory if p.get('usable', False)),
        key=lambda p: p.get('n_trials', 0) * p.get('n_frontal', 0),
        default=None,
    )
    if best_patient is None:
        logger.error("No usable patients.")
        sys.exit(1)

    patient_id = best_patient['patient_id']
    data = np.load(PROCESSED_DIR / f'{patient_id}_processed.npz',
                   allow_pickle=True)
    X = data['X']
    Y = data['Y']
    trial_info = pd.DataFrame(data['trial_info'].item())

    summary = cross_architecture_test(X, Y, trial_info, schema)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f'{patient_id}_cross_arch_results.json'

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(_convert(summary), f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Cross-Architecture Results: {patient_id}")
    print(f"{'='*65}")
    print(f"  {'Target':<25s}  {'Mandatory':>9s}  {'Tested':>6s}  {'Verdict':<15s}")
    print(f"  {'-'*60}")
    for name, info in sorted(summary.items()):
        archs_str = ', '.join(
            a['arch'] for a in info['per_arch']
            if a['status'] == 'OK' and
            a.get('category', '').startswith('MANDATORY')
        )
        print(f"  {name:<25s}  {info['n_mandatory']:>5d}/{info['n_tested']:<3d}  "
              f"        {info['verdict']:<15s}  [{archs_str}]")
    print(f"{'='*65}\n")

    logger.info("Saved to %s", output_path)


if __name__ == '__main__':
    main()
```

5. Commit:
```
git add human_wm/analysis/cross_patient.py human_wm/analysis/cross_architecture.py \
  scripts/15_human_cross_patient.py scripts/16_human_cross_architecture.py
git commit -m "human_wm: cross-patient and cross-architecture universality tests"
```

---

### Task 13: Universality Report

**Files:**
- `human_wm/analysis/universality_report.py`
- `scripts/17_human_report.py`

**Steps:**

1. Implement `human_wm/analysis/universality_report.py`:

```python
"""
DESCARTES Human Universality -- Final Report Generator

Combines results from:
  1. Cross-seed test (10 seeds, 1 patient, 1 architecture)
  2. Cross-patient test (all patients, 1 architecture)
  3. Cross-architecture test (4 architectures, 1 patient)

Final verdict per target:
  UNIVERSAL  -- all 3 tests pass
  ROBUST     -- 2/3 tests pass
  FRAGILE    -- 1/3 tests pass
  ZOMBIE     -- 0/3 tests pass
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _test_passed(test_result, target_name):
    """Check if a target passed a specific universality test."""
    if test_result is None or target_name not in test_result:
        return False
    verdict = test_result[target_name].get('verdict', '')
    return verdict in ('ROBUST', 'UNIVERSAL')


def generate_report(cross_seed_results, cross_patient_results,
                    cross_arch_results, output_dir=None):
    """Generate final universality report.

    Parameters
    ----------
    cross_seed_results : dict
        From cross_seed_test(). Per-target summaries.
    cross_patient_results : dict
        From cross_patient_test(). Per-target summaries.
    cross_arch_results : dict
        From cross_architecture_test(). Per-target summaries.
    output_dir : str or Path, optional
        Directory to save report files.

    Returns
    -------
    report : dict
        Final report with per-target verdicts and formatted text.
    """
    # Collect all target names
    all_targets = set()
    for results in [cross_seed_results, cross_patient_results,
                    cross_arch_results]:
        if results is not None:
            all_targets.update(results.keys())

    # Build per-target verdict
    target_verdicts = {}
    for target_name in sorted(all_targets):
        seed_pass = _test_passed(cross_seed_results, target_name)
        patient_pass = _test_passed(cross_patient_results, target_name)
        arch_pass = _test_passed(cross_arch_results, target_name)

        n_pass = sum([seed_pass, patient_pass, arch_pass])

        if n_pass == 3:
            verdict = 'UNIVERSAL'
        elif n_pass == 2:
            verdict = 'ROBUST'
        elif n_pass == 1:
            verdict = 'FRAGILE'
        else:
            verdict = 'ZOMBIE'

        # Collect details from each test
        seed_detail = cross_seed_results.get(target_name, {}) if cross_seed_results else {}
        patient_detail = cross_patient_results.get(target_name, {}) if cross_patient_results else {}
        arch_detail = cross_arch_results.get(target_name, {}) if cross_arch_results else {}

        target_verdicts[target_name] = {
            'final_verdict': verdict,
            'n_tests_passed': n_pass,
            'cross_seed': {
                'passed': seed_pass,
                'n_mandatory': seed_detail.get('n_mandatory', 0),
                'n_total': seed_detail.get('n_total', 0),
                'verdict': seed_detail.get('verdict', 'N/A'),
            },
            'cross_patient': {
                'passed': patient_pass,
                'n_mandatory': patient_detail.get('n_mandatory', 0),
                'n_tested': patient_detail.get('n_tested', 0),
                'verdict': patient_detail.get('verdict', 'N/A'),
            },
            'cross_architecture': {
                'passed': arch_pass,
                'n_mandatory': arch_detail.get('n_mandatory', 0),
                'n_tested': arch_detail.get('n_tested', 0),
                'verdict': arch_detail.get('verdict', 'N/A'),
            },
        }

    # Format text report
    lines = []
    lines.append("=" * 75)
    lines.append("  DESCARTES HUMAN UNIVERSALITY REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"  {'Target':<25s}  {'Seed':>6s}  {'Patient':>8s}  "
                 f"{'Arch':>6s}  {'VERDICT':<12s}")
    lines.append(f"  {'-'*70}")

    for target_name, tv in sorted(target_verdicts.items()):
        seed_str = ('PASS' if tv['cross_seed']['passed'] else 'FAIL')
        patient_str = ('PASS' if tv['cross_patient']['passed'] else 'FAIL')
        arch_str = ('PASS' if tv['cross_architecture']['passed'] else 'FAIL')

        lines.append(
            f"  {target_name:<25s}  {seed_str:>6s}  {patient_str:>8s}  "
            f"{arch_str:>6s}  {tv['final_verdict']:<12s}"
        )

    lines.append(f"  {'-'*70}")

    # Summary counts
    verdicts = [tv['final_verdict'] for tv in target_verdicts.values()]
    n_universal = verdicts.count('UNIVERSAL')
    n_robust = verdicts.count('ROBUST')
    n_fragile = verdicts.count('FRAGILE')
    n_zombie = verdicts.count('ZOMBIE')

    lines.append("")
    lines.append(f"  UNIVERSAL: {n_universal}  |  ROBUST: {n_robust}  |  "
                 f"FRAGILE: {n_fragile}  |  ZOMBIE: {n_zombie}")
    lines.append("")

    # Overall verdict
    if n_universal + n_robust > n_fragile + n_zombie:
        overall = "MAJORITY UNIVERSAL/ROBUST"
    elif n_zombie > n_universal + n_robust:
        overall = "MAJORITY ZOMBIE"
    else:
        overall = "MIXED"

    lines.append(f"  OVERALL: {overall}")
    lines.append("=" * 75)

    report_text = "\n".join(lines)

    # Build report dict
    report = {
        'generated': datetime.now().isoformat(),
        'targets': target_verdicts,
        'summary': {
            'n_universal': n_universal,
            'n_robust': n_robust,
            'n_fragile': n_fragile,
            'n_zombie': n_zombie,
            'overall': overall,
        },
        'report_text': report_text,
    }

    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_path = output_dir / 'universality_report.txt'
        with open(txt_path, 'w') as f:
            f.write(report_text)
        logger.info("Text report saved to %s", txt_path)

        json_path = output_dir / 'universality_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("JSON report saved to %s", json_path)

    # Print to console
    print(report_text)

    return report
```

2. Implement `scripts/17_human_report.py`:

```python
#!/usr/bin/env python
"""
Script 17: Generate final DESCARTES human universality report.

Usage:
    cd "Working memory"
    python scripts/17_human_report.py

Loads results from cross-seed, cross-patient, and cross-architecture
tests and generates a formatted report.
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from human_wm.analysis.universality_report import generate_report
from human_wm.config import PROCESSED_DIR, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)


def _load_json(path):
    """Load JSON file, return None if not found."""
    if not path.exists():
        logger.warning("Results file not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def main():
    # Find the best patient ID for cross-seed and cross-arch results
    inventory_path = PROCESSED_DIR / 'patient_inventory.json'
    if not inventory_path.exists():
        logger.error("Patient inventory not found.")
        sys.exit(1)

    with open(inventory_path) as f:
        inventory = json.load(f)

    best_patient = max(
        (p for p in inventory if p.get('usable', False)),
        key=lambda p: p.get('n_trials', 0) * p.get('n_frontal', 0),
        default=None,
    )
    if best_patient is None:
        logger.error("No usable patients.")
        sys.exit(1)

    patient_id = best_patient['patient_id']

    # Load all three test results
    cross_seed_results = _load_json(
        RESULTS_DIR / f'{patient_id}_cross_seed_results.json'
    )
    cross_patient_results = _load_json(
        RESULTS_DIR / 'cross_patient_results.json'
    )
    cross_arch_results = _load_json(
        RESULTS_DIR / f'{patient_id}_cross_arch_results.json'
    )

    # Count available results
    available = sum(1 for r in [cross_seed_results, cross_patient_results,
                                 cross_arch_results] if r is not None)
    if available == 0:
        logger.error("No test results found. Run scripts 14-16 first.")
        sys.exit(1)

    logger.info("Found %d/3 test result files", available)

    # Generate report
    report = generate_report(
        cross_seed_results=cross_seed_results,
        cross_patient_results=cross_patient_results,
        cross_arch_results=cross_arch_results,
        output_dir=RESULTS_DIR,
    )

    logger.info("Report generation complete.")


if __name__ == '__main__':
    main()
```

3. Commit:
```
git add human_wm/analysis/universality_report.py scripts/17_human_report.py
git commit -m "human_wm: universality report generator combining all 3 tests"
```

---

### Task 14: Update pyproject.toml

**Files:**
- `pyproject.toml` (modify)

**Steps:**

1. Modify `pyproject.toml` to include `human_wm` package and ensure `pynwb`/`dandi` dependencies are present:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "descartes-wm"
version = "0.1.0"
description = "DESCARTES Working Memory — Zombie test for thalamic working memory representations"
requires-python = ">=3.10"

dependencies = [
    # Data acquisition
    "dandi>=0.60",
    "pynwb>=2.5",
    "h5py>=3.9",

    # Numerical
    "numpy>=1.24",
    "scipy>=1.11",
    "numba>=0.58",

    # ML
    "torch>=2.0",
    "scikit-learn>=1.3",

    # Utilities
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "jupyter>=1.0",
]

[tool.setuptools.packages.find]
include = ["descartes_core*", "wm*", "human_wm*"]
```

The key change is adding `"human_wm*"` to the `include` list in `[tool.setuptools.packages.find]`. The `pynwb` and `dandi` dependencies are already present.

2. Verify:
```bash
cd "Working memory" && python -c "import human_wm; print('human_wm imported OK')"
```

3. Commit:
```
git add pyproject.toml
git commit -m "pyproject.toml: add human_wm to package discovery"
```

---

## Execution Handoff

Plan complete and saved. Two execution options:

**1. Subagent-Driven (this session)** -- I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** -- Open new session with executing-plans, batch execution with checkpoints

Which approach?
