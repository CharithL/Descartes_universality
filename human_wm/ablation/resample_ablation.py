"""
DESCARTES Human Universality -- Resample Ablation (Architecture-Agnostic)

Adapts the core resample ablation protocol from descartes_core.ablation for the
multi-architecture human pipeline. The core module assumes models with .lstm,
.hidden_size, and .n_layers attributes; the human pipeline uses 4 architectures
(LSTM, GRU, Transformer, Linear), so this module works on pre-extracted hidden
states and output projections rather than performing architecture-specific
forward-pass unrolling.

The resample ablation protocol (Grant et al. 2025):
  1. For each probe target that passed the LEARNED threshold (delta-R-squared > 0.1):
     a. Identify the top-k hidden dimensions most correlated with the target.
     b. For each k-fraction in ABLATION_K_FRACTIONS:
        - Targeted ablation: replace top-k dims with resampled values from
          other trials (preserves marginal distribution, breaks joint structure).
        - Random ablation: replace k random dims (repeat ABLATION_N_RANDOM times).
        - Measure cross-condition correlation (CC) degradation for both.
     c. If targeted ablation degrades CC significantly more than random
        (z < CAUSAL_Z_THRESHOLD) at any k-fraction -> MANDATORY.
     d. If not -> LEARNED_ZOMBIE.

Classification hierarchy:
  - ZOMBIE: delta-R-squared < DELTA_THRESHOLD_LEARNED from probing (never reaches ablation).
  - LEARNED_ZOMBIE: delta-R-squared >= threshold but no causal evidence from ablation.
  - MANDATORY: delta-R-squared >= threshold AND targeted ablation > random at any k.
"""

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from scipy import stats

from descartes_core.config import (
    ABLATION_K_FRACTIONS,
    ABLATION_N_RANDOM,
    CAUSAL_Z_THRESHOLD,
    DELTA_THRESHOLD_LEARNED,
)
from descartes_core.metrics import cross_condition_correlation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Identify top-k dimensions
# ---------------------------------------------------------------------------

def identify_top_k_dims(
    hidden_states: np.ndarray,
    target: np.ndarray,
    k: int,
) -> np.ndarray:
    """Identify the top-k hidden dimensions most correlated with a target.

    For each hidden dimension, computes the absolute Pearson correlation
    with the target variable. Returns the indices of the k dimensions
    with the highest absolute correlation.

    Parameters
    ----------
    hidden_states : ndarray, shape (n_samples, hidden_size)
        Hidden-state matrix. Each row is one trial/sample, each column
        is one hidden dimension.
    target : ndarray, shape (n_samples,)
        Target variable (e.g., a probe target like persistent_delay
        or memory_load).
    k : int
        Number of top dimensions to return. Clamped to [1, hidden_size].

    Returns
    -------
    top_dims : ndarray of int, shape (k,)
        Indices of the k hidden dimensions with highest absolute
        correlation to the target, sorted from highest to lowest.

    Notes
    -----
    Dimensions with near-zero variance (std < 1e-10) receive a
    correlation of 0.0 to avoid numerical issues in pearsonr.
    """
    hidden_size = hidden_states.shape[1]
    k = max(1, min(k, hidden_size))

    correlations = np.zeros(hidden_size, dtype=np.float64)
    for d in range(hidden_size):
        col = hidden_states[:, d]
        if np.std(col) > 1e-10:
            r, _ = stats.pearsonr(col, target)
            correlations[d] = abs(float(r))
        # else: remains 0.0

    sorted_dims = np.argsort(correlations)[::-1]
    return sorted_dims[:k]


# ---------------------------------------------------------------------------
# Step 2: Resample hidden dimensions
# ---------------------------------------------------------------------------

def resample_hidden_dims(
    hidden_states: np.ndarray,
    dims_to_ablate: Union[List[int], np.ndarray],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Create ablated hidden states by resampling specified dimensions.

    For each dimension in dims_to_ablate, replaces that dimension's values
    with random draws from its empirical marginal distribution (i.e., values
    sampled from other trials). This preserves the marginal distribution of
    each individual dimension but breaks the joint correlation structure
    between the ablated dimensions and the rest of the representation.

    Parameters
    ----------
    hidden_states : ndarray, shape (n_samples, hidden_size)
        Original hidden-state matrix.
    dims_to_ablate : list or ndarray of int
        Indices of hidden dimensions to resample.
    rng : np.random.RandomState
        Random number generator for reproducible resampling.

    Returns
    -------
    ablated : ndarray, shape (n_samples, hidden_size)
        Copy of hidden_states with specified dimensions resampled.
        The original array is not modified.
    """
    ablated = hidden_states.copy()
    n_samples = hidden_states.shape[0]

    for d in dims_to_ablate:
        col = hidden_states[:, d]
        ablated[:, d] = rng.choice(col, size=n_samples, replace=True)

    return ablated


# ---------------------------------------------------------------------------
# Step 3: Compute ablated output (architecture-agnostic)
# ---------------------------------------------------------------------------

def compute_ablated_output(
    model: torch.nn.Module,
    X: torch.Tensor,
    hidden_states: np.ndarray,
    dims_to_ablate: Union[List[int], np.ndarray],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Run a model with specified hidden dimensions resampled.

    This is the architecture-agnostic approach: rather than performing
    architecture-specific forward-pass unrolling (which requires knowing
    whether the model is LSTM, GRU, Transformer, or Linear), this function
    works with the hidden states directly.

    Strategy:
      1. Run the model's forward pass to obtain hidden states h_seq.
      2. Resample the specified dimensions in h_seq using values drawn
         from the empirical distribution in ``hidden_states``.
      3. Apply the model's output projection (model.proj, model.output_proj,
         or model.readout) to the ablated hidden states.

    This avoids the need for timestep-by-timestep unrolling and works
    identically across all 4 architectures, since every architecture in
    the human pipeline exposes ``return_hidden=True`` and a projection
    layer.

    Parameters
    ----------
    model : nn.Module
        Any of the 4 HumanXxxSurrogate architectures. Must support
        ``model(X, return_hidden=True)`` returning ``(y_pred, h_seq)``
        and must have an output projection accessible as ``.proj``,
        ``.output_proj``, or ``.readout``.
    X : torch.Tensor, shape (n_trials, T, input_dim)
        Input data.
    hidden_states : ndarray, shape (n_samples, hidden_size)
        Empirical distribution source for resampling. Typically the
        trial-averaged hidden states from a reference dataset.
    dims_to_ablate : list or ndarray of int
        Hidden dimensions to resample.
    rng : np.random.RandomState
        Random number generator for reproducible resampling.

    Returns
    -------
    Y_pred : ndarray, shape (n_trials, T, output_dim)
        Model output with the specified hidden dimensions resampled.
    """
    dims_to_ablate = list(dims_to_ablate)
    device = next(model.parameters()).device
    X_dev = X.to(device)

    # Get the output projection layer
    proj = _get_output_proj(model)

    with torch.no_grad():
        if len(dims_to_ablate) == 0:
            result = model(X_dev)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

        # Run forward to get hidden states
        result = model(X_dev, return_hidden=True)
        y_pred, h_seq = result  # h_seq: (n_trials, T, hidden_size)

        h_np = h_seq.cpu().numpy()
        n_trials, T, hidden_size = h_np.shape

        # Resample specified dims for each trial
        # Draw replacement values from the empirical distribution
        resample_values = np.zeros(
            (n_trials, len(dims_to_ablate)), dtype=np.float32
        )
        for j, d in enumerate(dims_to_ablate):
            col = hidden_states[:, d]
            resample_values[:, j] = rng.choice(
                col, size=n_trials, replace=True
            )

        # Apply resampling across all timesteps (consistent per trial)
        for j, d in enumerate(dims_to_ablate):
            h_np[:, :, d] = resample_values[:, j:j+1]  # broadcast over T

        # Recompute output from ablated hidden states
        h_ablated = torch.tensor(h_np, dtype=torch.float32, device=device)
        y_ablated = proj(h_ablated)

    return y_ablated.cpu().numpy()


def _get_output_proj(model: torch.nn.Module) -> torch.nn.Module:
    """Get the output projection layer from a model.

    Supports all naming conventions across the DESCARTES codebase:
      - model.proj        (Human pipeline convention)
      - model.output_proj (WM convention)
      - model.readout     (L5PC convention)

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    nn.Module
        The output projection layer.

    Raises
    ------
    AttributeError
        If the model has none of the recognized projection attributes.
    """
    if hasattr(model, 'proj'):
        return model.proj
    if hasattr(model, 'output_proj'):
        return model.output_proj
    if hasattr(model, 'readout'):
        return model.readout
    raise AttributeError(
        "Model must have one of 'proj', 'output_proj', or 'readout' "
        "as an output projection layer. Found attributes: "
        f"{[a for a in dir(model) if not a.startswith('_')]}"
    )


# ---------------------------------------------------------------------------
# Step 4: Full resample ablation for one target
# ---------------------------------------------------------------------------

def run_resample_ablation(
    model: torch.nn.Module,
    X_test: torch.Tensor,
    Y_test: np.ndarray,
    trial_info: Optional[np.ndarray],
    hidden_states: np.ndarray,
    target: np.ndarray,
    target_name: str,
    k_fractions: Optional[List[float]] = None,
    n_random: Optional[int] = None,
    cc_function: Optional[Callable] = None,
) -> Dict:
    """Full resample ablation protocol for one probe target.

    Runs the progressive resample ablation protocol: for each k-fraction,
    performs targeted ablation (top-k correlated dims) and random ablation
    (k random dims, repeated n_random times), then compares CC degradation
    via z-score.

    Parameters
    ----------
    model : nn.Module
        Any of the 4 HumanXxxSurrogate architectures.
    X_test : torch.Tensor, shape (n_trials, T, input_dim)
        Test inputs.
    Y_test : ndarray, shape (n_trials, T) or (n_trials, T, output_dim)
        Ground-truth test outputs for CC computation.
    trial_info : ndarray, shape (n_trials,), or None
        Condition labels for grouped cross-condition correlation.
        If None, uses ungrouped CC (simple Pearson over trial means).
    hidden_states : ndarray, shape (n_trials, hidden_size)
        Trial-level hidden states (empirical distribution source for
        resampling and for correlation-based dimension ranking).
    target : ndarray, shape (n_trials,)
        The probe target variable (e.g., persistent_delay values).
    target_name : str
        Human-readable name of the target (for logging).
    k_fractions : list of float, optional
        Fractions of hidden dimensions to ablate. Defaults to
        ABLATION_K_FRACTIONS from descartes_core.config.
    n_random : int, optional
        Number of random ablation repetitions. Defaults to
        ABLATION_N_RANDOM from descartes_core.config.
    cc_function : callable, optional
        Custom cross-condition correlation function with signature
        ``cc_function(predicted, actual) -> float``. If None, uses
        cross_condition_correlation or cross_condition_correlation_grouped
        depending on whether trial_info is provided.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'target_name' : str
        - 'baseline_cc' : float, CC with intact model
        - 'per_k' : list of dict, one per k-fraction, each containing:
            - 'k_frac' : float
            - 'n_ablated' : int, number of dims ablated
            - 'cc_targeted' : float, CC after targeted ablation
            - 'cc_targeted_drop' : float, baseline_cc - cc_targeted
            - 'cc_random_mean' : float, mean CC across random ablations
            - 'cc_random_std' : float, std of random CC values
            - 'z_score' : float, (cc_targeted - cc_random_mean) / cc_random_std
            - 'is_causal' : bool, True if z_score < CAUSAL_Z_THRESHOLD
        - 'any_causal' : bool, True if any k-fraction shows causal evidence
        - 'min_z_score' : float, most negative z-score across all k-fractions
        - 'causal_k_frac' : float or None, smallest k-fraction showing causality
    """
    if k_fractions is None:
        k_fractions = ABLATION_K_FRACTIONS
    if n_random is None:
        n_random = ABLATION_N_RANDOM

    # Resolve CC function
    if cc_function is None:
        if trial_info is not None:
            from descartes_core.metrics import cross_condition_correlation_grouped

            def cc_function(pred, actual):
                return cross_condition_correlation_grouped(
                    pred, actual, trial_info
                )
        else:
            cc_function = cross_condition_correlation

    hidden_size = hidden_states.shape[1]

    logger.info("Resample ablation for target '%s' (hidden_size=%d)",
                target_name, hidden_size)

    # Baseline: intact model output
    rng = np.random.RandomState(99)
    intact_output = compute_ablated_output(
        model, X_test, hidden_states, [], rng
    )
    baseline_cc = float(cc_function(intact_output, Y_test))
    logger.info("  Baseline CC = %.4f", baseline_cc)

    # Progressive ablation
    per_k_results = []

    for k_frac in k_fractions:
        n_ablate = max(1, int(round(k_frac * hidden_size)))

        # Identify top-k most correlated dims
        top_dims = identify_top_k_dims(hidden_states, target, n_ablate)

        # --- Targeted ablation ---
        rng_targeted = np.random.RandomState(99)
        targeted_output = compute_ablated_output(
            model, X_test, hidden_states, top_dims, rng_targeted
        )
        cc_targeted = float(cc_function(targeted_output, Y_test))

        # --- Random ablation (repeated) ---
        random_ccs = []
        rng_random = np.random.RandomState(42)
        for rep in range(n_random):
            rand_dims = rng_random.choice(
                hidden_size, size=n_ablate, replace=False
            )
            rand_output = compute_ablated_output(
                model, X_test, hidden_states, rand_dims,
                np.random.RandomState(42 + rep),
            )
            rand_cc = float(cc_function(rand_output, Y_test))
            random_ccs.append(rand_cc)

        cc_random_mean = float(np.mean(random_ccs))
        cc_random_std = float(np.std(random_ccs))

        # z-score: how much worse is targeted vs random?
        # Negative z means targeted degrades MORE than random
        if cc_random_std > 1e-10:
            z_score = (cc_targeted - cc_random_mean) / cc_random_std
        else:
            z_score = -10.0 if cc_targeted < cc_random_mean else 0.0

        is_causal = z_score < CAUSAL_Z_THRESHOLD

        per_k_results.append({
            'k_frac': float(k_frac),
            'n_ablated': int(n_ablate),
            'cc_targeted': cc_targeted,
            'cc_targeted_drop': float(baseline_cc - cc_targeted),
            'cc_random_mean': cc_random_mean,
            'cc_random_std': cc_random_std,
            'z_score': float(z_score),
            'is_causal': is_causal,
        })

        verdict_str = 'CAUSAL' if is_causal else 'NON_CAUSAL'
        logger.info(
            "  k=%.0f%% (%d dims): CC_targeted=%.3f  CC_random=%.3f+/-%.3f  "
            "z=%.2f  [%s]",
            k_frac * 100, n_ablate, cc_targeted,
            cc_random_mean, cc_random_std, z_score, verdict_str,
        )

    # Summary statistics
    any_causal = any(r['is_causal'] for r in per_k_results)
    z_scores = [r['z_score'] for r in per_k_results]
    min_z = float(min(z_scores)) if z_scores else 0.0

    causal_entries = [r for r in per_k_results if r['is_causal']]
    causal_k_frac = (
        min(r['k_frac'] for r in causal_entries) if causal_entries else None
    )

    result = {
        'target_name': target_name,
        'baseline_cc': baseline_cc,
        'per_k': per_k_results,
        'any_causal': any_causal,
        'min_z_score': min_z,
        'causal_k_frac': causal_k_frac,
    }

    summary = 'CAUSAL' if any_causal else 'NON_CAUSAL'
    logger.info(
        "  Summary for '%s': %s (min_z=%.2f, causal_k=%.2f%%)",
        target_name, summary, min_z,
        (causal_k_frac or 0.0) * 100,
    )

    return result


# ---------------------------------------------------------------------------
# Step 5: Classify variable
# ---------------------------------------------------------------------------

def classify_variable(
    probe_result: Dict,
    ablation_result: Optional[Dict],
) -> str:
    """Classify a variable into the DESCARTES zombie taxonomy.

    The three-tier classification:
      - ZOMBIE: The variable is not decodable from the model's hidden
        states. delta-R-squared < DELTA_THRESHOLD_LEARNED from probing.
        The model does not represent this variable at all.
      - LEARNED_ZOMBIE: The variable is decodable (delta-R-squared >= threshold)
        but there is no causal evidence that the model uses the
        representation to compute its output. The representation exists
        but is epiphenomenal.
      - MANDATORY: The variable is decodable AND ablating the most
        correlated hidden dimensions degrades output more than ablating
        random dimensions (at any k-fraction). The model causally uses
        this representation.

    Parameters
    ----------
    probe_result : dict
        Probing result for this target. Must contain:
        - 'delta_r2' : float, the delta-R-squared value from probing.
    ablation_result : dict or None
        Ablation result from run_resample_ablation(). Must contain:
        - 'any_causal' : bool
        If None (ablation was not run because probing failed the
        threshold), the variable is classified as ZOMBIE.

    Returns
    -------
    classification : str
        One of 'MANDATORY', 'LEARNED_ZOMBIE', or 'ZOMBIE'.

    Examples
    --------
    >>> classify_variable({'delta_r2': 0.05}, None)
    'ZOMBIE'
    >>> classify_variable({'delta_r2': 0.25}, {'any_causal': False})
    'LEARNED_ZOMBIE'
    >>> classify_variable({'delta_r2': 0.25}, {'any_causal': True})
    'MANDATORY'
    """
    delta_r2 = probe_result.get('delta_r2', 0.0)

    # Gate 1: Does the model represent this variable at all?
    if delta_r2 < DELTA_THRESHOLD_LEARNED:
        return 'ZOMBIE'

    # Gate 2: Does the model causally use the representation?
    if ablation_result is None:
        # Ablation was not run (should not happen if delta_r2 >= threshold,
        # but handle gracefully)
        logger.warning(
            "classify_variable: delta_r2=%.3f >= threshold but no "
            "ablation result provided. Defaulting to LEARNED_ZOMBIE.",
            delta_r2,
        )
        return 'LEARNED_ZOMBIE'

    if ablation_result.get('any_causal', False):
        return 'MANDATORY'
    else:
        return 'LEARNED_ZOMBIE'


# ---------------------------------------------------------------------------
# Convenience: classify with redundancy subtype
# ---------------------------------------------------------------------------

def classify_with_redundancy(
    probe_result: Dict,
    ablation_result: Optional[Dict],
) -> Dict:
    """Classify a variable and determine redundancy subtype if MANDATORY.

    Extends classify_variable with a redundancy classification based
    on the breaking point (smallest k-fraction showing causality):
      - MANDATORY_CONCENTRATED: causal at k <= 10% (few dims carry the signal)
      - MANDATORY_DISTRIBUTED: causal at 10% < k <= 60%
      - MANDATORY_REDUNDANT: causal only at k > 60% (highly distributed)

    Parameters
    ----------
    probe_result : dict
        Must contain 'delta_r2'.
    ablation_result : dict or None
        Must contain 'any_causal' and 'causal_k_frac'.

    Returns
    -------
    result : dict
        - 'classification' : str ('ZOMBIE', 'LEARNED_ZOMBIE', or 'MANDATORY')
        - 'redundancy_type' : str or None (only for MANDATORY variables)
        - 'breaking_point' : float or None (smallest causal k-fraction)
        - 'delta_r2' : float
    """
    classification = classify_variable(probe_result, ablation_result)
    delta_r2 = probe_result.get('delta_r2', 0.0)

    redundancy_type = None
    breaking_point = None

    if classification == 'MANDATORY' and ablation_result is not None:
        breaking_point = ablation_result.get('causal_k_frac')
        if breaking_point is not None:
            if breaking_point <= 0.10:
                redundancy_type = 'MANDATORY_CONCENTRATED'
            elif breaking_point <= 0.60:
                redundancy_type = 'MANDATORY_DISTRIBUTED'
            else:
                redundancy_type = 'MANDATORY_REDUNDANT'

    return {
        'classification': classification,
        'redundancy_type': redundancy_type,
        'breaking_point': breaking_point,
        'delta_r2': delta_r2,
    }
