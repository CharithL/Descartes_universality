"""
DESCARTES Human Universality -- Experiment-Specific Configuration

All hyperparameters, paths, and constants specific to the Rutishauser
Human Single-Neuron WM experiment (DANDI 000469). Methodology constants
(probing thresholds, ablation k-fractions) are inherited from
descartes_core.config.
"""

import json
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_NWB_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SURROGATE_DIR = DATA_DIR / 'surrogates'
RESULTS_DIR = DATA_DIR / 'results'
HIDDEN_DIR = SURROGATE_DIR / 'hidden'

# === DANDI Dataset ===
DANDISET_ID = '000469'

# === NWB Schema ===
NWB_SCHEMA_PATH = DATA_DIR / 'nwb_schema.json'


def load_nwb_schema(schema_path=None):
    """Load the NWB schema JSON file.

    Parameters
    ----------
    schema_path : Path or str, optional
        Path to the schema JSON file. Defaults to NWB_SCHEMA_PATH.

    Returns
    -------
    dict or None
        Parsed schema dictionary, or None if the file does not exist
        or cannot be read.
    """
    if schema_path is None:
        schema_path = NWB_SCHEMA_PATH
    schema_path = Path(schema_path)
    if not schema_path.exists():
        return None
    try:
        with open(schema_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# === Brain Region Identifiers (fallback patterns) ===
# MTL (medial temporal lobe) region patterns
# Supports both DANDI 000469 underscore style (e.g. "amygdala_right")
# and human-readable style (e.g. "Right Hippocampus")
MTL_REGION_PATTERNS = [
    'hippocampus', 'amygdala', 'entorhinal',
    'parahippocampal', 'perirhinal',
]
# Frontal region patterns
FRONTAL_REGION_PATTERNS = [
    'prefrontal', 'orbitofrontal', 'dorsolateral',
    'anterior_cingulate', 'anterior cingulate',
    'supplementary_motor', 'supplementary motor',
    'pre_supplementary', 'pre-SMA', 'pre-sma',
    'dACC', 'dacc', 'vmPFC', 'vmpfc',
    'ventromedial',
]

# Minimum neuron counts per region for a session to qualify
MIN_MTL_NEURONS = 3
MIN_FRONTAL_NEURONS = 3
# Minimum number of trials for a session to qualify
MIN_TRIALS = 50

# === Spike Binning ===
BIN_SIZE_MS = 50           # 50 ms bins
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

# === Reproducibility ===
N_SEEDS = 10
ARCHITECTURES = ['lstm', 'gru', 'transformer', 'linear']

# === Oscillation Bands (Hz) ===
THETA_BAND = (4, 8)
GAMMA_BAND = (30, 80)

# === Probe Target Levels ===
# Level B: Population-level cognitive variables (THE KEY LEVEL)
LEVEL_B_TARGETS = [
    'persistent_delay',       # Persistent activity during delay
    'memory_load',            # Memory load encoding
    'delay_stability',        # Stability of coding through delay
    'recognition_decision',   # Recognition decision signal
]

# Level C: Emergent dynamics
LEVEL_C_TARGETS = [
    'theta_modulation',       # 4-8 Hz oscillatory amplitude
    'gamma_modulation',       # 30-80 Hz oscillatory amplitude
    'population_synchrony',   # Pairwise correlation magnitude
]

# Level A: Individual neuron selectivity (expected mostly zombie)
LEVEL_A_TARGETS = [
    'concept_selectivity',    # Concept/category selectivity
    'mean_firing_rate',       # Mean firing rate
]

ALL_TARGETS = LEVEL_B_TARGETS + LEVEL_C_TARGETS + LEVEL_A_TARGETS

# === Quality Thresholds ===
MIN_CC_THRESHOLD = 0.3       # Minimum cross-condition correlation
GOOD_CC_THRESHOLD = 0.5      # "Meaningful" model performance
STRONG_CC_THRESHOLD = 0.7    # "Strong" model performance
