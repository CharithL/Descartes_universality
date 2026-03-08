"""
DESCARTES Working Memory — Experiment-Specific Configuration

All hyperparameters, paths, and constants specific to the Chen/Svoboda
ALM→Thalamus working memory experiment. Methodology constants (probing
thresholds, ablation k-fractions) are inherited from descartes_core.config.
"""

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
DANDISET_ID = '000363'
DANDISET_URL = 'https://dandiarchive.org/dandiset/000363'

# === Brain Region Identifiers ===
# CCF anno_name patterns for ALM (secondary motor area = MOs = ALM)
ALM_REGION_MARKERS = ['Secondary motor area']
# CCF anno_name patterns for thalamus
THAL_REGION_MARKERS = ['thalamus', 'Thalamus']
# electrode_group.location brain_regions patterns (coarse labels in small files)
ALM_ELECTRODE_MARKERS = ['ALM']
THAL_ELECTRODE_MARKERS = ['thalamus', 'Thalamus']

# Minimum neuron counts per region for a session to qualify
MIN_ALM_NEURONS = 30
MIN_THAL_NEURONS = 20

# === Task Timing (seconds, relative to trial start) ===
TONE_START_S = 0.0
TONE_END_S = 0.5
DELAY_START_S = 0.5
DELAY_END_S = 1.3
GO_CUE_S = 1.3
RESPONSE_WINDOW_END_S = 2.5

# Delay period duration
DELAY_DURATION_S = DELAY_END_S - DELAY_START_S  # 0.8s

# === Spike Binning ===
BIN_SIZE_MS = 10           # 10 ms bins
BIN_SIZE_S = BIN_SIZE_MS / 1000.0
N_DELAY_BINS = int(DELAY_DURATION_S / BIN_SIZE_S)  # 80 bins

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

# === Quality Thresholds ===
MIN_CC_THRESHOLD = 0.3    # Minimum cross-condition correlation for probing
GOOD_CC_THRESHOLD = 0.5   # "Meaningful" model performance
STRONG_CC_THRESHOLD = 0.7  # "Strong" model performance

# === Probe Target Levels ===
# Level A: Individual neuron selectivity (expected mostly zombie)
# Level B: Population-level cognitive variables (THE KEY LEVEL)
# Level C: Emergent dynamics

LEVEL_B_TARGETS = [
    'choice_signal',       # Persistent choice axis projection
    'ramp_signal',         # Temporal ramp toward go cue
    'population_rate',     # Total thalamic firing rate
    'choice_magnitude',    # Strength of choice encoding (|choice_signal|)
]

LEVEL_C_TARGETS = [
    'delay_stability',     # How constant is choice coding through delay?
    'theta_modulation',    # 4-8 Hz oscillatory amplitude
    'population_synchrony',  # Pairwise correlation magnitude
]

# === Oscillation Bands (Hz) ===
THETA_BAND = (4, 8)
GAMMA_BAND = (30, 80)
