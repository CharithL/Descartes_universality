"""
DESCARTES Core — Shared Configuration Constants

These are methodology constants (probing, ablation thresholds) that apply
across ALL DESCARTES experiments. Experiment-specific constants (data paths,
model dimensions, etc.) live in each experiment's own config module.
"""

# === Probing ===
RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
CV_FOLDS = 5
PREPROCESSING_OPTIONS = [
    'Raw', 'StandardScaler', 'PCA_5', 'PCA_10', 'PCA_20', 'PCA_50',
]
SELECTIVITY_PERMS = 20
P_THRESHOLD = 0.05
DELTA_THRESHOLD_LEARNED = 0.1     # ΔR² above this = non-zombie candidate
DELTA_THRESHOLD_STRONG = 0.2      # ΔR² above this = strong non-zombie

# === Ablation ===
ABLATION_K_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
ABLATION_N_RANDOM = 10
CAUSAL_Z_THRESHOLD = -2.0         # z-score below this = causal
