"""
DESCARTES WM — Probing Orchestration

Run Ridge ΔR² probing across all probe targets and hidden sizes.
Uses descartes_core for the actual probing logic.
"""

import json
import logging
from pathlib import Path

import numpy as np

from descartes_core import probe_single_variable
from wm.config import HIDDEN_SIZES, LEVEL_B_TARGETS, LEVEL_C_TARGETS

logger = logging.getLogger(__name__)


def compute_all_targets(Y_test, trial_types_test):
    """Compute all probe target variables from test data.

    Parameters
    ----------
    Y_test : ndarray, (n_trials, T, n_thal)
    trial_types_test : ndarray, (n_trials,)

    Returns
    -------
    targets : dict mapping level -> dict mapping name -> ndarray (n_trials,)
    """
    from wm.targets.choice_signal import (
        compute_choice_axis,
        compute_choice_magnitude,
        compute_delay_stability,
        trial_average_choice_signal,
    )
    from wm.targets.ramp_signal import compute_ramp_signal, trial_average_ramp_signal
    from wm.targets.emergent import (
        compute_population_rate,
        compute_theta_modulation,
        compute_population_synchrony,
    )

    # Choice signal
    choice_signal, choice_axis = compute_choice_axis(Y_test, trial_types_test)

    # Level B targets
    ramp_signal, ramp_axis = compute_ramp_signal(Y_test)
    pop_rate = compute_population_rate(Y_test)

    level_b = {
        'choice_signal': trial_average_choice_signal(choice_signal),
        'ramp_signal': trial_average_ramp_signal(ramp_signal),
        'population_rate': pop_rate.mean(axis=1),
        'choice_magnitude': compute_choice_magnitude(choice_signal).mean(axis=1),
    }

    # Level C targets
    level_c = {
        'delay_stability': compute_delay_stability(choice_signal),
        'theta_modulation': compute_theta_modulation(Y_test),
        'population_synchrony': compute_population_synchrony(Y_test),
    }

    return {'B': level_b, 'C': level_c}


def run_probing_all(hidden_states_dict, targets, save_dir=None):
    """Run probing across all hidden sizes and target levels.

    Parameters
    ----------
    hidden_states_dict : dict
        hidden_size -> (trained_H, untrained_H)
    targets : dict
        level -> {name -> ndarray}
    save_dir : str or Path, optional

    Returns
    -------
    all_results : dict
        Nested: level -> hidden_size -> list of probe results
    """
    all_results = {}

    for level, level_targets in targets.items():
        all_results[level] = {}
        for hs, (trained_H, untrained_H) in hidden_states_dict.items():
            results = []

            for var_name, target_y in level_targets.items():
                n = min(trained_H.shape[0], len(target_y))
                if n < 5:
                    logger.warning("Skipping %s: only %d samples", var_name, n)
                    continue

                result = probe_single_variable(
                    trained_H[:n], untrained_H[:n],
                    target_y[:n], var_name,
                )
                result['level'] = level
                result['hidden_size'] = hs
                results.append(result)

                logger.info(
                    "  %s (h=%d): R2_tr=%.3f  R2_un=%.3f  ΔR²=%.3f  [%s]",
                    var_name, hs, result['R2_trained'],
                    result['R2_untrained'], result['delta_R2'],
                    result['category'],
                )

            all_results[level][hs] = results

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for level in all_results:
            for hs in all_results[level]:
                path = save_dir / f'probe_level{level}_h{hs}.json'
                # Convert numpy types for JSON serialization
                serializable = _make_serializable(all_results[level][hs])
                with open(path, 'w') as f:
                    json.dump(serializable, f, indent=2)

    return all_results


def _make_serializable(obj):
    """Convert numpy types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
