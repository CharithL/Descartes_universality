"""
DESCARTES Core — Causal Ablation with Progressive Clamping & Resampling

The DECISIVE test: ΔR² establishes that a variable is decodable, but
decoding does not imply causal use. Progressive clamping answers:
does the network actually USE this representation to compute its output?

Supports both mean-clamping (original) and resample ablation (OOD-robust,
Grant et al. 2025). Resample ablation is preferred — mean-clamping can
create off-manifold artifacts that mimic causality.

This module is architecture-agnostic. It requires:
  - A model with .lstm, .hidden_size, .n_layers attributes
  - A readout method (model.readout or model.output_proj)
  - A cross-condition correlation metric

Extracted from the L5PC DESCARTES pipeline.
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model readout helper
# ---------------------------------------------------------------------------

def _get_readout(model):
    """Get the readout/output projection layer from a model.

    Supports both naming conventions:
      - model.readout  (L5PC convention: Linear(hidden, 1))
      - model.output_proj (WM convention: Linear(hidden, n_output))
    """
    if hasattr(model, 'readout'):
        return model.readout
    if hasattr(model, 'output_proj'):
        return model.output_proj
    raise AttributeError(
        "Model must have either 'readout' or 'output_proj' attribute"
    )


# ---------------------------------------------------------------------------
# Forward pass with mean-clamping
# ---------------------------------------------------------------------------

def forward_with_clamp(model, test_inputs, clamp_dims, hidden_means):
    """Run LSTM forward pass with specified hidden dims clamped to their mean.

    Parameters
    ----------
    model : nn.Module
        Must have .lstm, .hidden_size, .n_layers attributes and
        a readout layer (readout or output_proj).
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    clamp_dims : list or ndarray of int
    hidden_means : ndarray, (hidden_size,)

    Returns
    -------
    output : ndarray, (n_trials, T) or (n_trials, T, output_dim)
    """
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)
    clamp_dims = list(clamp_dims)
    readout = _get_readout(model)

    if len(clamp_dims) == 0:
        with torch.no_grad():
            result = model(test_inputs)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

    batch_size, T, _ = test_inputs.shape
    hidden_size = model.hidden_size
    n_layers = model.n_layers

    h = torch.zeros(n_layers, batch_size, hidden_size, device=device)
    c = torch.zeros(n_layers, batch_size, hidden_size, device=device)

    mean_tensor = torch.tensor(
        hidden_means, dtype=torch.float32, device=device
    )

    outputs = []
    with torch.no_grad():
        for t in range(T):
            x_t = test_inputs[:, t:t+1, :]
            _, (h, c) = model.lstm(x_t, (h, c))

            h_last = h[-1]
            h_last[:, clamp_dims] = mean_tensor[clamp_dims].unsqueeze(0)
            h[-1] = h_last

            out_t = readout(h_last)
            outputs.append(out_t)

    output = torch.stack(outputs, dim=1)
    # Squeeze trailing dim if readout is (hidden -> 1)
    if output.shape[-1] == 1:
        output = output.squeeze(-1)
    return output.cpu().numpy()


# ---------------------------------------------------------------------------
# Forward pass with resample (OOD-robust)
# ---------------------------------------------------------------------------

def forward_with_resample(model, test_inputs, clamp_dims, hidden_states,
                          rng=None):
    """Forward pass replacing clamped dims with random empirical samples.

    Instead of clamping to mean (off-manifold), replace each clamped dim
    with a random draw from its empirical distribution across trials.
    Preserves marginal statistics while breaking specific correlations.

    Parameters
    ----------
    model : nn.Module
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    clamp_dims : list or ndarray of int
    hidden_states : ndarray, (n_samples, hidden_size)
        Trial-averaged hidden states (empirical distribution source).
    rng : np.random.RandomState, optional

    Returns
    -------
    output : ndarray, (n_trials, T) or (n_trials, T, output_dim)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)
    clamp_dims = list(clamp_dims)
    readout = _get_readout(model)

    if len(clamp_dims) == 0:
        with torch.no_grad():
            result = model(test_inputs)
            out = result[0] if isinstance(result, tuple) else result
            return out.cpu().numpy()

    batch_size, T, _ = test_inputs.shape
    hidden_size = model.hidden_size
    n_layers = model.n_layers

    h = torch.zeros(n_layers, batch_size, hidden_size, device=device)
    c = torch.zeros(n_layers, batch_size, hidden_size, device=device)

    # Resample once per forward pass (consistent across timesteps)
    resample_values = np.zeros((batch_size, len(clamp_dims)))
    for j, d in enumerate(clamp_dims):
        col = hidden_states[:, d]
        resample_values[:, j] = rng.choice(col, size=batch_size, replace=True)
    resample_tensor = torch.tensor(
        resample_values, dtype=torch.float32, device=device
    )

    outputs = []
    with torch.no_grad():
        for t in range(T):
            x_t = test_inputs[:, t:t+1, :]
            _, (h, c) = model.lstm(x_t, (h, c))
            h_last = h[-1]
            for j, d in enumerate(clamp_dims):
                h_last[:, d] = resample_tensor[:, j]
            h[-1] = h_last
            out_t = readout(h_last)
            outputs.append(out_t)

    output = torch.stack(outputs, dim=1)
    if output.shape[-1] == 1:
        output = output.squeeze(-1)
    return output.cpu().numpy()


# ---------------------------------------------------------------------------
# Progressive ablation (mean-clamp)
# ---------------------------------------------------------------------------

def causal_ablation(model, test_inputs, test_outputs, target_y,
                    hidden_states, trial_types=None,
                    cc_fn=None, k_fractions=None, n_random_repeats=None):
    """Progressive mean-clamp ablation protocol.

    Parameters
    ----------
    model : nn.Module
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    test_outputs : ndarray, (n_trials, T) or (n_trials, T, output_dim)
    target_y : ndarray, (n_trials,)
    hidden_states : ndarray, (n_trials, hidden_size)
    trial_types : ndarray, (n_trials,), optional
        Condition labels for grouped CC. If None, uses simple CC.
    cc_fn : callable, optional
        Custom cross-condition correlation function.
        Signature: cc_fn(predicted, actual) -> float.
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
    hidden_means = np.mean(hidden_states, axis=0)

    correlations = np.array([
        abs(float(stats.pearsonr(hidden_states[:, d], target_y)[0]))
        if np.std(hidden_states[:, d]) > 1e-10 else 0.0
        for d in range(hidden_size)
    ])
    sorted_dims = np.argsort(correlations)[::-1]

    intact_output = forward_with_clamp(model, test_inputs, [], hidden_means)
    baseline_cc = cc_fn(intact_output, test_outputs)

    rng = np.random.RandomState(42)
    results = []

    for k_frac in k_fractions:
        n_clamp = max(1, int(round(k_frac * hidden_size)))

        target_dims = sorted_dims[:n_clamp]
        target_output = forward_with_clamp(model, test_inputs, target_dims,
                                           hidden_means)
        target_cc = cc_fn(target_output, test_outputs)

        random_ccs = []
        for _ in range(n_random_repeats):
            rand_dims = rng.choice(hidden_size, size=n_clamp, replace=False)
            rand_output = forward_with_clamp(model, test_inputs, rand_dims,
                                             hidden_means)
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
            "  k=%.0f%% (%d dims): target_cc=%.3f  random_cc=%.3f±%.3f  "
            "z=%.2f  [%s]",
            k_frac * 100, n_clamp, target_cc, random_mean, random_std,
            z_score, verdict,
        )

    return results, float(baseline_cc)


# ---------------------------------------------------------------------------
# Resample ablation (OOD-robust)
# ---------------------------------------------------------------------------

def resample_ablation(model, test_inputs, test_outputs, target_y,
                      hidden_states, trial_types=None,
                      cc_fn=None, k_fractions=None, n_random_repeats=None):
    """Resample ablation: OOD-robust alternative to mean-clamping.

    Same protocol as causal_ablation() but replaces clamped dims with
    random samples from the empirical distribution instead of mean values.

    Parameters
    ----------
    model : nn.Module
    test_inputs : torch.Tensor, (n_trials, T, input_dim)
    test_outputs : ndarray
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

    correlations = np.array([
        abs(float(stats.pearsonr(hidden_states[:, d], target_y)[0]))
        if np.std(hidden_states[:, d]) > 1e-10 else 0.0
        for d in range(hidden_size)
    ])
    sorted_dims = np.argsort(correlations)[::-1]

    # Baseline: intact model
    hidden_means = np.mean(hidden_states, axis=0)
    intact_output = forward_with_clamp(model, test_inputs, [], hidden_means)
    baseline_cc = cc_fn(intact_output, test_outputs)

    rng = np.random.RandomState(99)
    results = []

    for k_frac in k_fractions:
        n_clamp = max(1, int(round(k_frac * hidden_size)))
        target_dims = sorted_dims[:n_clamp]

        target_output = forward_with_resample(
            model, test_inputs, target_dims, hidden_states, rng=rng
        )
        target_cc = cc_fn(target_output, test_outputs)

        random_ccs = []
        for _ in range(n_random_repeats):
            rand_dims = rng.choice(hidden_size, size=n_clamp, replace=False)
            rand_output = forward_with_resample(
                model, test_inputs, rand_dims, hidden_states, rng=rng
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
            "  [resample] k=%.0f%%: target_cc=%.3f  z=%.2f  [%s]",
            k_frac * 100, target_cc, z_score, verdict,
        )

    return results, float(baseline_cc)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_mandatory_type(ablation_results, baseline_cc):
    """Classify redundancy type based on breaking point.

    Returns
    -------
    classification : str
    breaking_point : float or None
    """
    causal_entries = [r for r in ablation_results if r['verdict'] == 'CAUSAL']

    if not causal_entries:
        return 'NON_CAUSAL', None

    breaking_point = min(r['k_frac'] for r in causal_entries)

    if breaking_point <= 0.10:
        return 'MANDATORY_CONCENTRATED', breaking_point
    elif breaking_point <= 0.60:
        return 'MANDATORY_DISTRIBUTED', breaking_point
    else:
        return 'MANDATORY_REDUNDANT', breaking_point


# ---------------------------------------------------------------------------
# OOD norm diagnostic
# ---------------------------------------------------------------------------

def ood_norm_diagnostic(model, test_inputs, target_y, hidden_states,
                        k_frac=0.20):
    """Compare L2 norms of hidden states under clamping vs intact.

    Detects off-manifold artifacts from mean-clamping.

    Returns
    -------
    diagnostic : dict
    """
    hidden_size = hidden_states.shape[1]
    hidden_means = np.mean(hidden_states, axis=0)
    n_clamp = max(1, int(round(k_frac * hidden_size)))

    correlations = np.array([
        abs(float(stats.pearsonr(hidden_states[:, d], target_y)[0]))
        if np.std(hidden_states[:, d]) > 1e-10 else 0.0
        for d in range(hidden_size)
    ])
    sorted_dims = np.argsort(correlations)[::-1]
    target_dims = sorted_dims[:n_clamp]

    device = next(model.parameters()).device
    test_inputs_dev = test_inputs.to(device)

    def _collect_norms(clamp_fn):
        batch_size, T, _ = test_inputs_dev.shape
        h = torch.zeros(model.n_layers, batch_size, model.hidden_size,
                        device=device)
        c = torch.zeros(model.n_layers, batch_size, model.hidden_size,
                        device=device)
        mean_tensor = torch.tensor(hidden_means, dtype=torch.float32,
                                   device=device)
        norms = []
        with torch.no_grad():
            for t_step in range(T):
                x_t = test_inputs_dev[:, t_step:t_step+1, :]
                _, (h, c) = model.lstm(x_t, (h, c))
                h_last = h[-1].clone()
                clamp_fn(h_last, mean_tensor)
                h[-1] = h_last
                norms.append(
                    torch.norm(h_last, dim=1).cpu().numpy()
                )
        return np.mean(np.stack(norms, axis=0), axis=0)

    norms_intact = _collect_norms(lambda h, m: None)

    def _mean_clamp(h, m):
        for d in target_dims:
            h[:, d] = m[d]
    norms_mean_clamped = _collect_norms(_mean_clamp)

    ratio = norms_mean_clamped / np.maximum(norms_intact, 1e-10)

    return {
        'k_frac': float(k_frac),
        'n_clamped': int(n_clamp),
        'norm_intact_mean': float(np.mean(norms_intact)),
        'norm_clamped_mean': float(np.mean(norms_mean_clamped)),
        'ratio_mean': float(np.mean(ratio)),
        'ratio_std': float(np.std(ratio)),
        'ood_flag': bool(abs(np.mean(ratio) - 1.0) > 0.15),
    }
