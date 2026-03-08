"""
DESCARTES Core — Ridge ΔR² Probing

Architecture-agnostic probing methodology: for each target variable,
fit RidgeCV from trained hidden states and from untrained (random-init)
hidden states, then compute ΔR² = R²_trained - R²_untrained.

Variables with ΔR² < DELTA_THRESHOLD_LEARNED are classified ZOMBIE:
the network carries no more information about them than random projections.

Extracted from the L5PC DESCARTES pipeline, with all experiment-specific
dependencies removed.
"""

import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from descartes_core.config import (
    CV_FOLDS,
    DELTA_THRESHOLD_LEARNED,
    P_THRESHOLD,
    PREPROCESSING_OPTIONS,
    RIDGE_ALPHAS,
    SELECTIVITY_PERMS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X, method):
    """Apply preprocessing to hidden-state matrix.

    Parameters
    ----------
    X : ndarray, shape (n_samples, hidden_size)
    method : str
        One of 'Raw', 'StandardScaler', 'PCA_5', 'PCA_10', 'PCA_20', 'PCA_50'.

    Returns
    -------
    X_proc : ndarray
    """
    if method == 'Raw':
        return X.copy()

    if method == 'StandardScaler':
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    if method.startswith('PCA_'):
        n_components = int(method.split('_')[1])
        n_components = min(n_components, X.shape[0], X.shape[1])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=0)
        return pca.fit_transform(X_scaled)

    raise ValueError(f"Unknown preprocessing method: {method}")


# ---------------------------------------------------------------------------
# Ridge cross-validation
# ---------------------------------------------------------------------------

def ridge_cv_score(X, y, cv_folds=CV_FOLDS, alphas=None, target_name=None):
    """Fit RidgeCV with trial-level cross-validation.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    cv_folds : int
    alphas : list of float, optional
    target_name : str, optional

    Returns
    -------
    mean_r2 : float
    fold_r2s : list of float
    best_alpha : float
    """
    if alphas is None:
        alphas = RIDGE_ALPHAS

    y_std = np.std(y)
    if y_std < 1e-10:
        label = target_name or "unknown"
        logger.warning("Target '%s' has zero variance (std=%.2e) -- returning R2=0",
                       label, y_std)
        return 0.0, [0.0] * cv_folds, alphas[0]

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_r2s = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # StandardScaler fitted per fold to avoid leakage
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_test = scaler_x.transform(X_test)

        y_mean, y_std_fold = y_train.mean(), y_train.std()
        if y_std_fold < 1e-10:
            fold_r2s.append(0.0)
            continue
        y_train_s = (y_train - y_mean) / y_std_fold
        y_test_s = (y_test - y_mean) / y_std_fold

        model = RidgeCV(alphas=alphas)
        model.fit(X_train, y_train_s)
        r2 = model.score(X_test, y_test_s)
        fold_r2s.append(float(r2))

    # Fit once on all data to report best alpha
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    y_full_s = (y - y.mean()) / max(y.std(), 1e-10)
    model_full = RidgeCV(alphas=alphas)
    model_full.fit(X_full, y_full_s)

    return float(np.mean(fold_r2s)), fold_r2s, float(model_full.alpha_)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def selectivity_permutation_test(X, y, n_perms=SELECTIVITY_PERMS,
                                 cv_folds=CV_FOLDS):
    """Permutation test: shuffle y across trials, build null R² distribution.

    Returns
    -------
    p_value : float
    observed_r2 : float
    null_r2s : list of float
    """
    observed_r2, _, _ = ridge_cv_score(X, y, cv_folds=cv_folds)

    rng = np.random.RandomState(0)
    null_r2s = []
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        perm_r2, _, _ = ridge_cv_score(X, y_perm, cv_folds=cv_folds)
        null_r2s.append(perm_r2)

    p_value = float((np.sum(np.array(null_r2s) >= observed_r2) + 1)
                    / (n_perms + 1))
    return p_value, observed_r2, null_r2s


# ---------------------------------------------------------------------------
# Single-variable probe
# ---------------------------------------------------------------------------

def probe_single_variable(trained_H, untrained_H, target_y, var_name,
                          preprocessing_options=None, alphas=None,
                          cv_folds=CV_FOLDS):
    """Probe one target variable across all preprocessing pipelines.

    Parameters
    ----------
    trained_H : ndarray, shape (n_samples, hidden_size)
        Hidden states from the trained surrogate.
    untrained_H : ndarray, shape (n_samples, hidden_size)
        Hidden states from the untrained (random-init) model.
    target_y : ndarray, shape (n_samples,)
        Scalar target per sample.
    var_name : str

    Returns
    -------
    result : dict
        Keys: var_name, R2_trained, R2_untrained, delta_R2,
              best_preprocessing, p_value, category,
              all_preprocessing_results.
    """
    if preprocessing_options is None:
        preprocessing_options = PREPROCESSING_OPTIONS
    if alphas is None:
        alphas = RIDGE_ALPHAS

    best_r2_trained = -np.inf
    best_r2_untrained = -np.inf
    best_prep = None
    all_prep_results = {}

    for prep in preprocessing_options:
        try:
            X_tr = preprocess(trained_H, prep)
            X_un = preprocess(untrained_H, prep)
        except Exception as e:
            logger.warning("Preprocessing '%s' failed for %s: %s",
                           prep, var_name, e)
            continue

        r2_tr, folds_tr, alpha_tr = ridge_cv_score(X_tr, target_y,
                                                    cv_folds=cv_folds,
                                                    alphas=alphas,
                                                    target_name=var_name)
        r2_un, folds_un, alpha_un = ridge_cv_score(X_un, target_y,
                                                    cv_folds=cv_folds,
                                                    alphas=alphas,
                                                    target_name=var_name)

        all_prep_results[prep] = {
            'R2_trained': r2_tr,
            'R2_untrained': r2_un,
            'delta_R2': r2_tr - r2_un,
            'alpha_trained': alpha_tr,
            'alpha_untrained': alpha_un,
            'fold_R2s_trained': folds_tr,
            'fold_R2s_untrained': folds_un,
        }

        if r2_tr > best_r2_trained:
            best_r2_trained = r2_tr
            best_r2_untrained = r2_un
            best_prep = prep

    delta_r2 = best_r2_trained - best_r2_untrained

    # Permutation test on best preprocessing
    p_value = 1.0
    if best_prep is not None and delta_r2 > 0:
        X_best = preprocess(trained_H, best_prep)
        p_value, _, _ = selectivity_permutation_test(
            X_best, target_y, n_perms=SELECTIVITY_PERMS, cv_folds=cv_folds
        )

    if delta_r2 < DELTA_THRESHOLD_LEARNED or p_value > P_THRESHOLD:
        category = 'ZOMBIE'
    else:
        category = 'LEARNED'

    return {
        'var_name': var_name,
        'R2_trained': float(best_r2_trained),
        'R2_untrained': float(best_r2_untrained),
        'delta_R2': float(delta_r2),
        'best_preprocessing': best_prep,
        'p_value': float(p_value),
        'category': category,
        'all_preprocessing_results': all_prep_results,
    }


# ---------------------------------------------------------------------------
# Logistic probing for binary targets
# ---------------------------------------------------------------------------

def logistic_cv_auc(X, y, cv_folds=CV_FOLDS):
    """Logistic regression with AUC-ROC for binary targets.

    Returns
    -------
    mean_auc : float
    fold_aucs : list of float
    n_positive : int
    """
    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))

    min_per_class = max(2, cv_folds)
    if n_positive < min_per_class or n_negative < min_per_class:
        logger.warning(
            "Too few samples for logistic CV: %d positive, %d negative "
            "(need >= %d each). Returning AUC=0.5 (chance)",
            n_positive, n_negative, min_per_class,
        )
        return 0.5, [0.5] * cv_folds, n_positive

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_aucs = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            Cs=10, cv=3, solver='lbfgs',
            max_iter=1000, random_state=42,
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fold_aucs.append(float(auc))

    if not fold_aucs:
        return 0.5, [], n_positive

    return float(np.mean(fold_aucs)), fold_aucs, n_positive


def probe_binary_variable(trained_H, untrained_H, target_y, var_name,
                          preprocessing_options=None, cv_folds=CV_FOLDS):
    """Probe a binary target variable using logistic regression + AUC-ROC.

    Returns
    -------
    result : dict
    """
    if preprocessing_options is None:
        preprocessing_options = PREPROCESSING_OPTIONS

    n_positive = int(np.sum(target_y == 1))
    n_negative = int(np.sum(target_y == 0))

    best_auc_trained = 0.5
    best_auc_untrained = 0.5
    best_prep = None
    all_prep_results = {}

    for prep in preprocessing_options:
        try:
            X_tr = preprocess(trained_H, prep)
            X_un = preprocess(untrained_H, prep)
        except Exception as e:
            logger.warning("Preprocessing '%s' failed for %s: %s",
                           prep, var_name, e)
            continue

        auc_tr, folds_tr, _ = logistic_cv_auc(X_tr, target_y, cv_folds)
        auc_un, folds_un, _ = logistic_cv_auc(X_un, target_y, cv_folds)

        all_prep_results[prep] = {
            'AUC_trained': auc_tr,
            'AUC_untrained': auc_un,
            'delta_AUC': auc_tr - auc_un,
        }

        if auc_tr > best_auc_trained:
            best_auc_trained = auc_tr
            best_auc_untrained = auc_un
            best_prep = prep

    delta_auc = best_auc_trained - best_auc_untrained

    return {
        'var_name': var_name,
        'metric': 'AUC-ROC',
        'AUC_trained': float(best_auc_trained),
        'AUC_untrained': float(best_auc_untrained),
        'delta_AUC': float(delta_auc),
        'n_positive': n_positive,
        'n_negative': n_negative,
        'best_preprocessing': best_prep,
        'all_preprocessing_results': all_prep_results,
    }
