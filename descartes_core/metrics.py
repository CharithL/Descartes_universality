"""
DESCARTES Core — Shared Metrics

Architecture-agnostic evaluation metrics used across all DESCARTES experiments.
"""

import numpy as np
from scipy import stats


def cross_condition_correlation(predicted, actual):
    """Cross-condition correlation: correlate MEAN outputs across conditions.

    This is the primary output metric for ablation evaluation.
    Compares condition-averaged model output to condition-averaged ground truth.

    Parameters
    ----------
    predicted : ndarray
        Model output. If 3D (n_trials, T, D), flattens to 2D first.
        If 2D (n_trials, T), trial-averages over time.
        If 1D (n_trials,), used directly.
    actual : ndarray
        Ground truth, same shape convention as predicted.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    # Flatten 3D (n_trials, T, output_dim) to 2D (n_trials, T*output_dim)
    if predicted.ndim == 3:
        predicted = predicted.reshape(predicted.shape[0], -1)
    if actual.ndim == 3:
        actual = actual.reshape(actual.shape[0], -1)

    if predicted.ndim == 2:
        pred_mean = predicted.mean(axis=1)
        true_mean = actual.mean(axis=1)
    else:
        pred_mean = predicted
        true_mean = actual
    if len(pred_mean) < 3:
        return 0.0
    r, _ = stats.pearsonr(pred_mean, true_mean)
    return float(r)


def cross_condition_correlation_grouped(predicted, actual, group_labels):
    """Cross-condition correlation with explicit condition grouping.

    Averages predictions and actuals within each condition group,
    then correlates the condition means.

    Parameters
    ----------
    predicted : ndarray, (n_trials, ...) or (n_trials,)
    actual : ndarray, same shape as predicted
    group_labels : ndarray, (n_trials,)
        Condition labels (e.g., trial types: 0=left, 1=right).

    Returns
    -------
    float
        Pearson correlation of condition-averaged outputs.
    """
    conditions = np.unique(group_labels)
    if len(conditions) < 2:
        return 0.0

    pred_means = []
    true_means = []
    for c in conditions:
        mask = group_labels == c
        pred_means.append(predicted[mask].mean(axis=0).flatten())
        true_means.append(actual[mask].mean(axis=0).flatten())

    pred_concat = np.concatenate(pred_means)
    true_concat = np.concatenate(true_means)

    if len(pred_concat) < 3:
        return 0.0
    r, _ = stats.pearsonr(pred_concat, true_concat)
    return float(r)
