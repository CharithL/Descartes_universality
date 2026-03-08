"""
DESCARTES Core — Architecture-Agnostic Probing and Ablation Engine

Shared analytical machinery for the DESCARTES zombie test methodology.
This module is experiment-independent: it operates on hidden states
(n_samples, hidden_dim) and target variables (n_samples,) regardless
of whether those come from a biophysical surrogate (L5PC), a cognitive
surrogate (ALM→thalamus), or any future experiment.

Core functions:
    ridge_cv_score          — Cross-validated Ridge regression
    probe_single_variable   — Full ΔR² probe with trained/untrained comparison
    resample_ablation       — OOD-robust causal ablation
    forward_with_resample   — Per-timestep hidden-state resampling hook
    classify_variable       — Three-stage classification cascade
"""

from descartes_core.ridge_probe import (
    ridge_cv_score,
    probe_single_variable,
    preprocess,
    selectivity_permutation_test,
    logistic_cv_auc,
    probe_binary_variable,
)
from descartes_core.ablation import (
    resample_ablation,
    forward_with_resample,
    causal_ablation,
    forward_with_clamp,
    classify_mandatory_type,
    ood_norm_diagnostic,
)
from descartes_core.classify import (
    classify_variable,
    print_classification_summary,
    CATEGORIES,
)
from descartes_core.metrics import (
    cross_condition_correlation,
)
