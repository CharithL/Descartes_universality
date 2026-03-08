"""
DESCARTES WM -- Photoinhibition Validation

The Chen/Svoboda dataset includes trials where ALM was optogenetically
silenced during the delay period. This module validates that the LSTM
surrogate captures the causal structure: when ALM input is disrupted,
the model's mandatory variables (theta_mod, delay_stability) should
collapse, matching what happens in the biological thalamus.

Protocol (from DESCARTES_WORKING_MEMORY_GUIDE Section 9):
    1. Train LSTM on normal (no-photostim) trials         [already done]
    2. Probe hidden states for mandatory variables         [already done]
    3. Feed model ALM activity from photostim trials
    4. Extract hidden states from photostim-input trials
    5. Probe photostim hidden states for the same variables
    6. Compare: mandatory variables should degrade
"""

import logging
from pathlib import Path

import json
import numpy as np
import torch

from wm.config import HIDDEN_SIZES
from wm.surrogate.model import WMSurrogate
from wm.surrogate.extract_hidden import extract_hidden_states
from wm.targets.choice_signal import (
    compute_choice_axis,
    compute_delay_stability,
    trial_average_choice_signal,
)
from wm.targets.ramp_signal import compute_ramp_signal, trial_average_ramp_signal
from wm.targets.emergent import (
    compute_theta_modulation,
    compute_population_synchrony,
)

logger = logging.getLogger(__name__)


def identify_photostim_from_nwb(nwb_path, n_correct_trials):
    """Extract photostim labels from NWB file and match to test split.

    Parameters
    ----------
    nwb_path : Path
        Path to the NWB file.
    n_correct_trials : int
        Number of correct (hit) trials used in preprocessing.

    Returns
    -------
    is_photostim : ndarray of bool, shape (n_test_trials,)
    """
    import pynwb
    from wm.config import TRAIN_FRAC, VAL_FRAC

    with pynwb.NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        trials = nwb.trials
        n_all = len(trials)

        outcomes = [str(trials['outcome'][i]) for i in range(n_all)]
        photostim = []
        for i in range(n_all):
            p = str(trials['photostim_power'][i])
            photostim.append(p != 'N/A' and p != 'nan')

    is_stim_all = np.array(photostim)
    is_correct = np.array([o == 'hit' for o in outcomes])
    is_stim_correct = is_stim_all[is_correct]

    n_correct = int(is_correct.sum())
    logger.info("  NWB: %d total trials, %d correct", n_all, n_correct)

    rng = np.random.RandomState(42)
    indices = rng.permutation(n_correct)

    n_train = int(n_correct * TRAIN_FRAC)
    n_val = int(n_correct * VAL_FRAC)
    test_idx = indices[n_train + n_val:]

    n_test_computed = len(test_idx)
    if n_test_computed != n_correct_trials:
        logger.warning(
            "  Test split mismatch: NWB gives %d test trials but "
            "processed data has %d. Wrong NWB file?",
            n_test_computed, n_correct_trials,
        )
        return None

    return is_stim_correct[test_idx]


def identify_photostim_in_processed(session_dir, trial_metadata_path):
    """Match photostim labels to processed/split trial indices.

    The preprocessing pipeline filters to correct trials and splits.
    We need to trace which test trials correspond to photostim.

    Parameters
    ----------
    session_dir : Path
        Processed session directory containing splits.
    trial_metadata_path : Path
        Path to trial_metadata.json with photostim_power.

    Returns
    -------
    is_photostim : ndarray of bool, shape (n_test_trials,)
    """
    with open(trial_metadata_path) as f:
        meta = json.load(f)

    photostim_power = meta.get('photostim_power', [])
    outcomes = meta.get('outcomes', [])

    from wm.config import TRAIN_FRAC, VAL_FRAC

    # Build is_photostim for ALL trials
    is_stim_all = np.array([p != 'N/A' for p in photostim_power])

    # Filter to correct trials (same as preprocessing)
    is_correct = np.array([o == 'hit' for o in outcomes])
    correct_indices = np.where(is_correct)[0]
    is_stim_correct = is_stim_all[correct_indices]

    # Reproduce the same split (must match split_data in preprocessing.py)
    n_correct = len(correct_indices)
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_correct)

    n_train = int(n_correct * TRAIN_FRAC)
    n_val = int(n_correct * VAL_FRAC)
    test_idx = indices[n_train + n_val:]

    # Extract photostim labels for test trials
    is_stim_test = is_stim_correct[test_idx]

    return is_stim_test


def _train_and_evaluate_probe(H_train, y_train, H_eval, y_eval, alphas=None):
    """Train Ridge probe on training set, evaluate R² on eval set.

    This avoids cross-validation on small subsets. We train on control
    data and evaluate generalization on photostim data.

    Returns
    -------
    r2_eval : float
        R² on evaluation set.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from descartes_core.config import RIDGE_ALPHAS

    if alphas is None:
        alphas = RIDGE_ALPHAS

    if y_train.std() < 1e-10 or y_eval.std() < 1e-10:
        return 0.0

    scaler_x = StandardScaler()
    H_train_s = scaler_x.fit_transform(H_train)
    H_eval_s = scaler_x.transform(H_eval)

    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_s = (y_train - y_mean) / y_std
    y_eval_s = (y_eval - y_mean) / y_std  # same normalization!

    model = RidgeCV(alphas=alphas)
    model.fit(H_train_s, y_train_s)
    r2 = float(model.score(H_eval_s, y_eval_s))
    return r2


def run_photoinhibition_validation(
    splits, session_info, model_dir, hidden_dir,
    is_photostim_test, save_dir=None,
):
    """Run the full photoinhibition validation protocol.

    Protocol (correct approach):
        1. Extract hidden states for control and photostim subsets
        2. For each variable, train a Ridge probe on CONTROL hidden states
        3. Evaluate that probe on CONTROL data (via cross-val) -> ctrl_R2
        4. Evaluate the SAME probe on PHOTOSTIM data -> stim_R2
        5. If stim_R2 << ctrl_R2, the variable depends on intact ALM input

    This avoids the small-sample degenerate regression problem by never
    training a probe on the (small) photostim subset.

    Parameters
    ----------
    splits : dict
        Data splits from preprocessing.
    session_info : dict
    model_dir : Path
        Directory containing trained model checkpoints.
    hidden_dir : Path
        Directory containing extracted hidden states.
    is_photostim_test : ndarray of bool
        Which test trials are photostim.
    save_dir : Path, optional

    Returns
    -------
    results : dict
        Validation results per hidden size.
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from descartes_core.config import RIDGE_ALPHAS, CV_FOLDS

    model_dir = Path(model_dir)
    hidden_dir = Path(hidden_dir)

    X_test = splits['test']['X']
    Y_test = splits['test']['Y']
    trial_types = splits['test']['trial_types']

    ctrl_mask = ~is_photostim_test
    stim_mask = is_photostim_test

    n_ctrl = int(ctrl_mask.sum())
    n_stim = int(stim_mask.sum())
    logger.info(
        "Photoinhibition validation: %d control, %d photostim test trials",
        n_ctrl, n_stim,
    )

    if n_stim < 3:
        logger.warning("Too few photostim test trials (%d). Skipping.", n_stim)
        return {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_test.shape[2]
    output_dim = Y_test.shape[2]

    results = {}

    for hs in HIDDEN_SIZES:
        model_path = model_dir / f'wm_h{hs}_best.pt'
        if not model_path.exists():
            continue

        logger.info("=== Photoinhibition h=%d ===", hs)

        # Load trained model
        model = WMSurrogate(
            input_dim=input_dim, output_dim=output_dim,
            hidden_size=hs,
        ).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        # Extract hidden states for control and photostim subsets
        H_ctrl, _ = extract_hidden_states(model, X_test[ctrl_mask])
        H_stim, _ = extract_hidden_states(model, X_test[stim_mask])

        # Also load the untrained baseline hidden states (from full test)
        untrained_path = hidden_dir / f'wm_h{hs}_untrained.npz'
        untrained_H = np.load(untrained_path)['hidden_states']
        H_untrained_ctrl = untrained_H[ctrl_mask]
        H_untrained_stim = untrained_H[stim_mask]

        # Compute probe targets for each subset
        Y_ctrl = Y_test[ctrl_mask]
        Y_stim = Y_test[stim_mask]
        tt_ctrl = trial_types[ctrl_mask]
        tt_stim = trial_types[stim_mask]

        targets_ctrl = _compute_targets_safe(Y_ctrl, tt_ctrl)
        targets_stim = _compute_targets_safe(Y_stim, tt_stim)

        hs_results = {
            'n_ctrl': n_ctrl,
            'n_stim': n_stim,
            'variables': {},
        }

        for var_name in targets_ctrl:
            if targets_ctrl[var_name] is None or targets_stim[var_name] is None:
                continue

            y_ctrl = targets_ctrl[var_name]
            y_stim = targets_stim[var_name]

            # --- Trained model hidden states ---
            # Cross-val R² on control (the "normal" score)
            ctrl_cv_r2 = _cross_val_r2(H_ctrl, y_ctrl, CV_FOLDS)

            # Train probe on ALL ctrl, evaluate on stim
            stim_r2_trained = _train_and_evaluate_probe(
                H_ctrl, y_ctrl, H_stim, y_stim
            )

            # --- Untrained baseline ---
            ctrl_cv_r2_untrained = _cross_val_r2(
                H_untrained_ctrl, y_ctrl, CV_FOLDS
            )
            stim_r2_untrained = _train_and_evaluate_probe(
                H_untrained_ctrl, y_ctrl,
                H_untrained_stim, y_stim,
            )

            # Delta R² (trained - untrained)
            ctrl_delta_r2 = ctrl_cv_r2 - ctrl_cv_r2_untrained
            stim_delta_r2 = stim_r2_trained - stim_r2_untrained

            # Degradation: stim ΔR² drops to < 50% of ctrl ΔR²
            if ctrl_delta_r2 > 0.05:
                degraded = stim_delta_r2 < ctrl_delta_r2 * 0.5
            else:
                degraded = False  # Not learned enough to degrade

            hs_results['variables'][var_name] = {
                'control_R2_trained': ctrl_cv_r2,
                'control_R2_untrained': ctrl_cv_r2_untrained,
                'control_delta_R2': ctrl_delta_r2,
                'photostim_R2_trained': stim_r2_trained,
                'photostim_R2_untrained': stim_r2_untrained,
                'photostim_delta_R2': stim_delta_r2,
                'degraded': degraded,
            }

            status = "DEGRADED*" if degraded else "PERSISTS"
            logger.info(
                "  %s: ctrl_dR2=%.3f  stim_dR2=%.3f  [%s]",
                var_name, ctrl_delta_r2, stim_delta_r2, status,
            )

        # Output prediction quality comparison
        model.train(False)
        with torch.no_grad():
            Y_pred_ctrl = model(
                torch.tensor(X_test[ctrl_mask], dtype=torch.float32).to(device)
            )[0].cpu().numpy()
            Y_pred_stim = model(
                torch.tensor(X_test[stim_mask], dtype=torch.float32).to(device)
            )[0].cpu().numpy()

        mse_ctrl = float(np.mean((Y_pred_ctrl - Y_ctrl) ** 2))
        mse_stim = float(np.mean((Y_pred_stim - Y_stim) ** 2))

        hs_results['output_mse_control'] = mse_ctrl
        hs_results['output_mse_photostim'] = mse_stim
        hs_results['mse_ratio'] = mse_stim / max(mse_ctrl, 1e-10)

        logger.info(
            "  Output MSE: ctrl=%.4f  stim=%.4f  ratio=%.2f",
            mse_ctrl, mse_stim, hs_results['mse_ratio'],
        )

        results[hs] = hs_results

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'photoinhibition_validation.json', 'w') as f:
            json.dump(_make_serializable(results), f, indent=2)
        logger.info("Saved validation results to %s", save_dir)

    return results


def _cross_val_r2(H, y, cv_folds=5):
    """Quick cross-validated Ridge R² without the full probe machinery."""
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeCV
    from descartes_core.config import RIDGE_ALPHAS

    if y.std() < 1e-10:
        return 0.0

    kf = KFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)
    fold_r2s = []

    for train_idx, test_idx in kf.split(H):
        scaler = StandardScaler()
        H_tr = scaler.fit_transform(H[train_idx])
        H_te = scaler.transform(H[test_idx])

        y_mean, y_std = y[train_idx].mean(), y[train_idx].std()
        if y_std < 1e-10:
            fold_r2s.append(0.0)
            continue
        y_tr = (y[train_idx] - y_mean) / y_std
        y_te = (y[test_idx] - y_mean) / y_std

        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(H_tr, y_tr)
        fold_r2s.append(float(model.score(H_te, y_te)))

    return float(np.mean(fold_r2s)) if fold_r2s else 0.0


def _compute_targets_safe(Y, trial_types):
    """Compute probe targets, returning None for any that fail."""
    targets = {}

    try:
        choice_signal, _ = compute_choice_axis(Y, trial_types)
        targets['choice_signal'] = trial_average_choice_signal(choice_signal)
        targets['delay_stability'] = compute_delay_stability(choice_signal)
    except Exception as e:
        logger.warning("Could not compute choice targets: %s", e)
        targets['choice_signal'] = None
        targets['delay_stability'] = None

    try:
        ramp_signal, _ = compute_ramp_signal(Y)
        targets['ramp_signal'] = trial_average_ramp_signal(ramp_signal)
    except Exception:
        targets['ramp_signal'] = None

    try:
        targets['theta_modulation'] = compute_theta_modulation(Y)
    except Exception:
        targets['theta_modulation'] = None

    try:
        targets['population_synchrony'] = compute_population_synchrony(Y)
    except Exception:
        targets['population_synchrony'] = None

    return targets


def _make_serializable(obj):
    """Convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def print_validation_summary(results):
    """Print formatted photoinhibition validation summary."""
    print("\n" + "=" * 70)
    print("  DESCARTES WM -- Photoinhibition Validation")
    print("=" * 70)

    for hs, hs_results in sorted(results.items()):
        print(
            f"\n  Hidden size: {hs}  "
            f"(ctrl={hs_results['n_ctrl']}, stim={hs_results['n_stim']})"
        )
        print("  " + "-" * 60)
        print(f"    {'Variable':<22s}  {'ctrl_dR2':>8s}  {'stim_dR2':>8s}  "
              f"{'ctrl_R2':>7s}  {'stim_R2':>7s}  {'Status'}")
        print("  " + "-" * 60)

        for var_name, vr in hs_results.get('variables', {}).items():
            ctrl_dr2 = vr['control_delta_R2']
            stim_dr2 = vr['photostim_delta_R2']
            ctrl_r2 = vr['control_R2_trained']
            stim_r2 = vr['photostim_R2_trained']
            status = "DEGRADED*" if vr['degraded'] else "persists"
            print(
                f"    {var_name:<22s}  {ctrl_dr2:+8.3f}  {stim_dr2:+8.3f}  "
                f"{ctrl_r2:7.3f}  {stim_r2:7.3f}  [{status}]"
            )

        mse_ratio = hs_results.get('mse_ratio', 0)
        print(f"\n    Output MSE ratio (stim/ctrl): {mse_ratio:.2f}")

    # Aggregate across all hidden sizes
    all_learned_degraded = 0
    all_learned_total = 0
    degraded_vars = set()
    persisted_vars = set()

    for hs, hs_results in results.items():
        for var_name, vr in hs_results.get('variables', {}).items():
            if vr['control_delta_R2'] > 0.05:  # Only count learned variables
                all_learned_total += 1
                if vr['degraded']:
                    all_learned_degraded += 1
                    degraded_vars.add(var_name)
                else:
                    persisted_vars.add(var_name)

    print(f"\n  Summary: {all_learned_degraded}/{all_learned_total} "
          f"learned variables degraded under photoinhibition")

    if degraded_vars:
        print(f"    Degraded: {', '.join(sorted(degraded_vars))}")
    if persisted_vars:
        print(f"    Persisted: {', '.join(sorted(persisted_vars))}")

    if all_learned_total > 0:
        ratio = all_learned_degraded / all_learned_total
        if ratio > 0.5:
            print("\n  => Model captures causal ALM->thalamus dependency")
        elif ratio > 0.25:
            print("\n  => Partial causal structure captured")
        else:
            print("\n  => Model may have learned shortcuts not present in biology")

    print("=" * 70 + "\n")
