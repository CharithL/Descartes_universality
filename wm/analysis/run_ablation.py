"""
DESCARTES WM — Ablation Orchestration

Run resample ablation on all variables that passed the probing stage
(ΔR² > 0.1). Uses descartes_core for the actual ablation logic.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from descartes_core import resample_ablation, classify_mandatory_type
from descartes_core.metrics import cross_condition_correlation_grouped
from wm.config import HIDDEN_SIZES
from wm.surrogate.model import WMSurrogate

logger = logging.getLogger(__name__)


def run_ablation_on_learned(probe_results, splits, hidden_states_dict,
                            model_dir, targets, save_dir=None,
                            delta_threshold=0.1):
    """Run resample ablation on all LEARNED variables.

    Parameters
    ----------
    probe_results : dict
        level -> hidden_size -> list of probe result dicts
    splits : dict
        Data splits from preprocessing.
    hidden_states_dict : dict
        hidden_size -> (trained_H, untrained_H)
    model_dir : str or Path
    targets : dict
        level -> {name -> ndarray}
    save_dir : str or Path, optional
    delta_threshold : float

    Returns
    -------
    ablation_results : dict
    """
    model_dir = Path(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = splits['test']['X'].shape[2]
    output_dim = splits['test']['Y'].shape[2]
    trial_types = splits['test']['trial_types']

    X_test = torch.tensor(splits['test']['X'], dtype=torch.float32)
    Y_test = splits['test']['Y']

    ablation_results = {}

    for level, level_probes in probe_results.items():
        for hs, results_list in level_probes.items():
            # Load model
            model_path = model_dir / f'wm_h{hs}_best.pt'
            if not model_path.exists():
                logger.warning("Model not found: %s", model_path)
                continue

            model = WMSurrogate(
                input_dim=input_dim, output_dim=output_dim,
                hidden_size=hs,
            ).to(device)
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            model.train(False)

            trained_H = hidden_states_dict[hs][0]

            for probe_result in results_list:
                if probe_result.get('delta_R2', 0) < delta_threshold:
                    continue

                var_name = probe_result['var_name']
                target_y = targets[level][var_name]

                n = min(trained_H.shape[0], len(target_y),
                        X_test.shape[0], Y_test.shape[0])

                logger.info("Ablation: %s  level=%s  h=%d", var_name, level, hs)

                # Custom CC function using trial types
                def cc_fn(pred, actual):
                    return cross_condition_correlation_grouped(
                        pred, actual, trial_types[:n]
                    )

                abl_results, baseline_cc = resample_ablation(
                    model, X_test[:n], Y_test[:n],
                    target_y[:n], trained_H[:n],
                    trial_types=trial_types[:n],
                )

                classification, breaking_point = classify_mandatory_type(
                    abl_results, baseline_cc
                )

                key = f'{level}_{var_name}_h{hs}'
                ablation_results[key] = {
                    'var_name': var_name,
                    'level': level,
                    'hidden_size': hs,
                    'baseline_cc': baseline_cc,
                    'classification': classification,
                    'breaking_point': breaking_point,
                    'ablation_steps': abl_results,
                }

                logger.info("  -> %s (breaking at %.0f%%)",
                            classification,
                            (breaking_point or 0) * 100)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / 'ablation_results.json'
        with open(path, 'w') as f:
            json.dump(_make_serializable(ablation_results), f, indent=2)
        logger.info("Saved ablation results to %s", path)

    return ablation_results


def _make_serializable(obj):
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
