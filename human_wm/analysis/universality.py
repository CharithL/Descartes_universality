"""
DESCARTES Human Universality -- Cross-Seed / Cross-Patient / Cross-Architecture Tests

Orchestrates the three universality tests described in the DESCARTES Universality
Guide.  Each test trains surrogate models, probes hidden states for all target
variables, runs resample ablation on LEARNED variables, and classifies every
variable as MANDATORY, LEARNED_ZOMBIE, or ZOMBIE.

    Test 1  Cross-Seed:         10 random seeds × 1 patient
    Test 2  Cross-Patient:      1 seed × N patients
    Test 3  Cross-Architecture: 4 architectures × 1 patient

The final deliverable is the Universality Table -- a matrix showing, for each
probe target, how many seeds / patients / architectures classify it as MANDATORY.

Usage (programmatic)::

    from human_wm.analysis.universality import (
        run_single_patient_pipeline,
        cross_seed_test,
        cross_patient_test,
        cross_architecture_test,
        format_universality_table,
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from descartes_core.config import DELTA_THRESHOLD_LEARNED
from descartes_core.ridge_probe import probe_single_variable
from human_wm.ablation.resample_ablation import (
    classify_variable,
    run_resample_ablation,
)
from human_wm.config import (
    ALL_TARGETS,
    ARCHITECTURES,
    HIDDEN_SIZES,
    MIN_CC_THRESHOLD,
    N_SEEDS,
)
from human_wm.surrogate.models import create_surrogate
from human_wm.surrogate.train import (
    compute_cross_condition_cc,
    create_dataloader,
    extract_hidden_states,
    train_surrogate,
)
from human_wm.targets.probe_targets import compute_all_targets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-patient full pipeline
# ---------------------------------------------------------------------------

def run_single_patient_pipeline(
    splits: dict,
    task_timing: dict,
    arch_name: str = 'lstm',
    hidden_size: int = 128,
    seed: int = 0,
    output_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Run the full DESCARTES pipeline for one patient with one model.

    Steps:
      1. Train surrogate
      2. Check quality (CC > MIN_CC_THRESHOLD)
      3. Extract hidden states (trained + untrained)
      4. Compute all probe targets
      5. Probe each target (Ridge ΔR²)
      6. Resample ablation on LEARNED targets
      7. Classify as MANDATORY / LEARNED_ZOMBIE / ZOMBIE

    Parameters
    ----------
    splits : dict
        Output of ``nwb_loader.split_data`` with keys 'train', 'val', 'test',
        each containing 'X', 'Y', 'trial_info'.
    task_timing : dict
        Task timing specification with 'encoding_bins', 'delay_bins',
        'probe_bins' as slice objects.
    arch_name : str
        Architecture name: 'lstm', 'gru', 'transformer', 'linear'.
    hidden_size : int
        Hidden dimension for the surrogate.
    seed : int
        Random seed for reproducibility.
    output_dir : Path or str, optional
        Directory to save model checkpoints and results.

    Returns
    -------
    dict or None
        Results dictionary with per-variable classification, or None if model
        quality is too low.
    """
    import torch  # lazy import to avoid DLL issues during pure-Python usage

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Set seeds ---
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = splits['train']['X'].shape[2]
    output_dim = splits['train']['Y'].shape[2]

    # --- 1. Train surrogate ---
    model = create_surrogate(arch_name, input_dim, output_dim, hidden_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = create_dataloader(
        splits['train']['X'], splits['train']['Y'], shuffle=True,
    )
    val_loader = create_dataloader(
        splits['val']['X'], splits['val']['Y'], shuffle=False,
    )

    save_path = None
    if output_dir is not None:
        save_path = output_dir / f'{arch_name}_h{hidden_size}_s{seed}_best.pt'

    model, history = train_surrogate(
        model, train_loader, val_loader, save_path=save_path,
    )

    # --- 2. Quality check ---
    cc, condition_col = compute_cross_condition_cc(
        model,
        splits['test']['X'],
        splits['test']['Y'],
        splits['test']['trial_info'],
    )
    logger.info(
        'Arch=%s  seed=%d  CC=%.3f  cond_col=%s  (threshold=%.2f)',
        arch_name, seed, cc, condition_col, MIN_CC_THRESHOLD,
    )
    if np.isnan(cc) or cc < MIN_CC_THRESHOLD:
        logger.warning(
            'Model quality too low (CC=%.3f < %.2f) — skipping probing',
            cc if not np.isnan(cc) else 0.0, MIN_CC_THRESHOLD,
        )
        return None

    # --- 3. Extract hidden states ---
    h_trained = extract_hidden_states(model, splits['test']['X'])

    # Untrained baseline (fresh random init, same architecture)
    torch.manual_seed(seed + 99999)  # different seed for untrained
    model_untrained = create_surrogate(
        arch_name, input_dim, output_dim, hidden_size,
    ).to(device)
    h_untrained = extract_hidden_states(model_untrained, splits['test']['X'])

    # Trial-average hidden states for probing
    h_trained_avg = h_trained.mean(axis=1)      # (n_test, hidden_size)
    h_untrained_avg = h_untrained.mean(axis=1)

    # --- 4. Compute probe targets ---
    Y_test = splits['test']['Y']
    trial_info_test = splits['test']['trial_info']
    targets = compute_all_targets(Y_test, trial_info_test, task_timing)

    # --- 5 & 6. Probe + ablation for each target ---
    results: dict[str, Any] = {
        'cc': cc,
        'arch': arch_name,
        'seed': seed,
        'hidden_size': hidden_size,
        'variables': {},
    }

    for var_name, target_values in targets.items():
        if target_values is None:
            results['variables'][var_name] = {
                'classification': 'SKIPPED',
                'reason': 'target computation returned None',
            }
            continue

        # Check zero variance
        if np.std(target_values) < 1e-8:
            results['variables'][var_name] = {
                'classification': 'ZERO_VARIANCE',
                'reason': 'target has zero variance',
            }
            continue

        # Probe
        probe_result = probe_single_variable(
            h_trained_avg, h_untrained_avg,
            target_values, var_name,
        )

        delta_r2 = probe_result['delta_R2']

        # Ablation (only if LEARNED)
        ablation_result = None
        if delta_r2 >= DELTA_THRESHOLD_LEARNED:
            try:
                ablation_result = run_resample_ablation(
                    model=model,
                    X_test=splits['test']['X'],
                    Y_test=splits['test']['Y'],
                    trial_info=trial_info_test,
                    hidden_states=h_trained,
                    target=target_values,
                    target_name=var_name,
                )
            except Exception as exc:
                logger.warning(
                    'Ablation failed for %s: %s', var_name, exc,
                )

        classification = classify_variable(probe_result, ablation_result)

        results['variables'][var_name] = {
            'classification': classification,
            'delta_R2': delta_r2,
            'R2_trained': probe_result['R2_trained'],
            'R2_untrained': probe_result['R2_untrained'],
            'p_value': probe_result['p_value'],
        }

        logger.info(
            '  %s: ΔR²=%.3f  class=%s',
            var_name, delta_r2, classification,
        )

    # --- Save results ---
    if output_dir is not None:
        results_path = (
            output_dir / f'results_{arch_name}_h{hidden_size}_s{seed}.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Test 1: Cross-Seed Consistency
# ---------------------------------------------------------------------------

def cross_seed_test(
    splits: dict,
    task_timing: dict,
    arch_name: str = 'lstm',
    hidden_size: int = 128,
    n_seeds: int = N_SEEDS,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train N models with different seeds on the SAME patient.

    Reports how many seeds classify each variable as MANDATORY.

    Parameters
    ----------
    splits : dict
        Split data for one patient.
    task_timing : dict
        Task timing specification.
    arch_name : str
        Architecture to use (default: 'lstm').
    hidden_size : int
        Hidden dimension.
    n_seeds : int
        Number of random seeds.
    output_dir : Path or str, optional

    Returns
    -------
    dict
        Summary with per-variable seed consistency and verdicts.
    """
    if output_dir is not None:
        output_dir = Path(output_dir) / 'cross_seed'
        output_dir.mkdir(parents=True, exist_ok=True)

    seed_results: dict[str, list[str]] = {}
    successful_seeds = 0

    for seed in range(n_seeds):
        logger.info('=== Cross-Seed: seed %d/%d ===', seed + 1, n_seeds)

        result = run_single_patient_pipeline(
            splits, task_timing,
            arch_name=arch_name,
            hidden_size=hidden_size,
            seed=seed,
            output_dir=output_dir,
        )
        if result is None:
            continue

        successful_seeds += 1
        for var_name, var_result in result['variables'].items():
            if var_name not in seed_results:
                seed_results[var_name] = []
            seed_results[var_name].append(var_result['classification'])

    # --- Summary ---
    summary: dict[str, Any] = {
        'test': 'cross_seed',
        'n_seeds': n_seeds,
        'successful_seeds': successful_seeds,
        'variables': {},
    }

    for var_name, classifications in seed_results.items():
        n_mandatory = sum(1 for c in classifications if c == 'MANDATORY')
        n_total = len(classifications)
        pct = (n_mandatory / n_total * 100) if n_total > 0 else 0

        if n_mandatory >= 8:
            verdict = 'ROBUST'
        elif n_mandatory >= 5:
            verdict = 'MODERATE'
        else:
            verdict = 'FRAGILE'

        summary['variables'][var_name] = {
            'n_mandatory': n_mandatory,
            'n_total': n_total,
            'pct': pct,
            'verdict': verdict,
            'classifications': classifications,
        }

    if output_dir is not None:
        with open(output_dir / 'cross_seed_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Test 2: Cross-Patient Universality
# ---------------------------------------------------------------------------

def cross_patient_test(
    patient_data: list[dict[str, Any]],
    task_timing: dict,
    arch_name: str = 'lstm',
    hidden_size: int = 128,
    seed: int = 0,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train a separate model per patient.

    Reports which variables are MANDATORY across patients — THE universality
    test for human cognition.

    Parameters
    ----------
    patient_data : list of dict
        Each dict has keys: 'patient_id', 'splits', containing patient ID
        and split data (from ``split_data``).
    task_timing : dict
        Task timing specification.
    arch_name : str
    hidden_size : int
    seed : int
    output_dir : Path or str, optional

    Returns
    -------
    dict
        Summary with per-variable patient universality.
    """
    if output_dir is not None:
        output_dir = Path(output_dir) / 'cross_patient'
        output_dir.mkdir(parents=True, exist_ok=True)

    patient_results: dict[str, dict[str, str]] = {}

    for pdata in patient_data:
        patient_id = pdata['patient_id']
        splits = pdata['splits']
        logger.info('=== Cross-Patient: %s ===', patient_id)

        pat_dir = None
        if output_dir is not None:
            pat_dir = output_dir / patient_id

        result = run_single_patient_pipeline(
            splits, task_timing,
            arch_name=arch_name,
            hidden_size=hidden_size,
            seed=seed,
            output_dir=pat_dir,
        )
        if result is None:
            continue

        patient_results[patient_id] = {
            var_name: var_result['classification']
            for var_name, var_result in result['variables'].items()
        }

    # --- Summary ---
    n_patients = len(patient_results)
    all_variables: set[str] = set()
    for results in patient_results.values():
        all_variables.update(results.keys())

    summary: dict[str, Any] = {
        'test': 'cross_patient',
        'n_patients': n_patients,
        'variables': {},
    }

    for var_name in sorted(all_variables):
        n_mandatory = sum(
            1 for p_results in patient_results.values()
            if p_results.get(var_name) == 'MANDATORY'
        )
        pct = (n_mandatory / n_patients * 100) if n_patients > 0 else 0

        if pct >= 80:
            verdict = 'UNIVERSAL'
        elif pct >= 50:
            verdict = 'PARTIAL'
        elif pct >= 20:
            verdict = 'MINORITY'
        else:
            verdict = 'NO'

        summary['variables'][var_name] = {
            'n_mandatory': n_mandatory,
            'n_patients': n_patients,
            'pct': pct,
            'verdict': verdict,
        }

    if output_dir is not None:
        with open(output_dir / 'cross_patient_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Test 3: Cross-Architecture
# ---------------------------------------------------------------------------

def cross_architecture_test(
    splits: dict,
    task_timing: dict,
    hidden_size: int = 128,
    seed: int = 0,
    architectures: list[str] | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Train LSTM, GRU, Transformer, Linear on the same patient.

    Reports which variables are MANDATORY across architectures.

    If a variable is mandatory in all 4 → mathematical necessity.
    If mandatory only in recurrent models → recurrence-specific.
    If mandatory only in LSTM → LSTM-specific artefact.

    Parameters
    ----------
    splits : dict
        Split data for one patient (the best patient).
    task_timing : dict
    hidden_size : int
    seed : int
    architectures : list of str, optional
        Defaults to ARCHITECTURES from config.
    output_dir : Path or str, optional

    Returns
    -------
    dict
        Summary with per-variable architecture universality.
    """
    if architectures is None:
        architectures = list(ARCHITECTURES)

    if output_dir is not None:
        output_dir = Path(output_dir) / 'cross_architecture'
        output_dir.mkdir(parents=True, exist_ok=True)

    arch_results: dict[str, dict[str, Any]] = {}

    for arch_name in architectures:
        logger.info('=== Cross-Architecture: %s ===', arch_name)

        arch_dir = None
        if output_dir is not None:
            arch_dir = output_dir / arch_name

        result = run_single_patient_pipeline(
            splits, task_timing,
            arch_name=arch_name,
            hidden_size=hidden_size,
            seed=seed,
            output_dir=arch_dir,
        )

        if result is None:
            arch_results[arch_name] = {
                'status': 'FAILED',
                'cc': 0.0,
                'variables': {},
            }
        else:
            arch_results[arch_name] = {
                'status': 'OK',
                'cc': result['cc'],
                'variables': {
                    var_name: var_result['classification']
                    for var_name, var_result in result['variables'].items()
                },
            }

    # --- Summary ---
    all_variables: set[str] = set()
    for arch_res in arch_results.values():
        all_variables.update(arch_res.get('variables', {}).keys())

    summary: dict[str, Any] = {
        'test': 'cross_architecture',
        'architectures': architectures,
        'variables': {},
    }

    for var_name in sorted(all_variables):
        mandatory_in: list[str] = []
        tested_in: list[str] = []

        for arch_name in architectures:
            arch_res = arch_results.get(arch_name, {})
            if arch_res.get('status') == 'OK':
                tested_in.append(arch_name)
                if arch_res['variables'].get(var_name) == 'MANDATORY':
                    mandatory_in.append(arch_name)

        n_mandatory = len(mandatory_in)
        n_tested = len(tested_in)

        if n_mandatory == n_tested and n_tested >= 3:
            verdict = 'UNIVERSAL'
        elif n_mandatory >= 2:
            verdict = 'PARTIAL'
        else:
            verdict = 'ARCH_SPECIFIC'

        summary['variables'][var_name] = {
            'n_mandatory': n_mandatory,
            'n_tested': n_tested,
            'mandatory_in': mandatory_in,
            'verdict': verdict,
        }

    if output_dir is not None:
        with open(output_dir / 'cross_arch_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


# ---------------------------------------------------------------------------
# Final Universality Table
# ---------------------------------------------------------------------------

def format_universality_table(
    cross_seed_summary: dict | None = None,
    cross_patient_summary: dict | None = None,
    cross_arch_summary: dict | None = None,
) -> str:
    """Format the combined Universality Table as a printable string.

    This is the final deliverable of the Human Universality pipeline.

    Parameters
    ----------
    cross_seed_summary : dict, optional
    cross_patient_summary : dict, optional
    cross_arch_summary : dict, optional

    Returns
    -------
    str
        Formatted table.
    """
    all_vars: set[str] = set()
    if cross_seed_summary:
        all_vars.update(cross_seed_summary.get('variables', {}).keys())
    if cross_patient_summary:
        all_vars.update(cross_patient_summary.get('variables', {}).keys())
    if cross_arch_summary:
        all_vars.update(cross_arch_summary.get('variables', {}).keys())

    lines = []
    lines.append('=' * 78)
    lines.append('HUMAN WORKING MEMORY — UNIVERSALITY REPORT')
    lines.append('=' * 78)
    lines.append('')

    header = (
        f'{"Variable":<25} | {"Cross-Seed":<14} | {"Cross-Patient":<15} '
        f'| {"Cross-Arch":<14} | VERDICT'
    )
    lines.append(header)

    n_seed = 10
    n_patient = '?'
    n_arch = 4
    if cross_seed_summary:
        n_seed = cross_seed_summary.get('successful_seeds', 10)
    if cross_patient_summary:
        n_patient = cross_patient_summary.get('n_patients', '?')
    if cross_arch_summary:
        n_arch = len(cross_arch_summary.get('architectures', ARCHITECTURES))

    subheader = (
        f'{"":25} | {"(N/" + str(n_seed) + ")":<14} '
        f'| {"(N/" + str(n_patient) + ")":<15} '
        f'| {"(N/" + str(n_arch) + ")":<14} |'
    )
    lines.append(subheader)
    lines.append('-' * 78)

    for var_name in sorted(all_vars):
        # Cross-seed
        if cross_seed_summary and var_name in cross_seed_summary['variables']:
            cs = cross_seed_summary['variables'][var_name]
            seed_str = f'{cs["n_mandatory"]}/{cs["n_total"]}'
        else:
            seed_str = '?/?'

        # Cross-patient
        if cross_patient_summary and var_name in cross_patient_summary['variables']:
            cp = cross_patient_summary['variables'][var_name]
            patient_str = (
                f'{cp["n_mandatory"]}/{cp["n_patients"]} '
                f'({cp["pct"]:.0f}%)'
            )
        else:
            patient_str = '?/?'

        # Cross-architecture
        if cross_arch_summary and var_name in cross_arch_summary['variables']:
            ca = cross_arch_summary['variables'][var_name]
            arch_str = f'{ca["n_mandatory"]}/{ca["n_tested"]}'
        else:
            arch_str = '?/?'

        # Overall verdict
        verdicts = []
        if cross_seed_summary and var_name in cross_seed_summary['variables']:
            verdicts.append(cross_seed_summary['variables'][var_name]['verdict'])
        if cross_patient_summary and var_name in cross_patient_summary['variables']:
            verdicts.append(cross_patient_summary['variables'][var_name]['verdict'])
        if cross_arch_summary and var_name in cross_arch_summary['variables']:
            verdicts.append(cross_arch_summary['variables'][var_name]['verdict'])

        if all(v in ('ROBUST', 'UNIVERSAL') for v in verdicts) and len(verdicts) >= 2:
            overall = 'UNIVERSAL'
        elif any(v in ('ROBUST', 'UNIVERSAL') for v in verdicts):
            overall = 'ROBUST'
        elif any(v in ('MODERATE', 'PARTIAL') for v in verdicts):
            overall = 'PARTIAL'
        else:
            overall = 'ZOMBIE'

        line = (
            f'{var_name:<25} | {seed_str:<14} | {patient_str:<15} '
            f'| {arch_str:<14} | {overall}'
        )
        lines.append(line)

    lines.append('')
    lines.append('VERDICT KEY:')
    lines.append('  UNIVERSAL:  ≥8/10 seeds, ≥80% patients, ≥3/4 architectures')
    lines.append('  ROBUST:     ≥8/10 seeds, ≥50% patients, ≥2/4 architectures')
    lines.append('  PARTIAL:    Some evidence but not consistent')
    lines.append('  ZOMBIE:     Not mandatory in any test')

    return '\n'.join(lines)
