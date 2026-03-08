"""
DESCARTES WM — Hidden State Extraction

Extract hidden representations from trained AND untrained (random-init)
surrogates on the test set. The untrained baseline is critical:
    ΔR² = R²_trained - R²_untrained
isolates learned representations from random-projection artifacts.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from wm.config import BATCH_SIZE, HIDDEN_SIZES, N_LSTM_LAYERS
from wm.surrogate.model import WMSurrogate

logger = logging.getLogger(__name__)


def extract_hidden_states(model, test_X, batch_size=BATCH_SIZE):
    """Run model on test data and collect hidden states.

    Parameters
    ----------
    model : WMSurrogate
    test_X : ndarray, (n_trials, T, input_dim)
    batch_size : int

    Returns
    -------
    hidden : ndarray, (n_trials, hidden_dim)
        Trial-averaged hidden states (averaged over timesteps).
    hidden_full : ndarray, (n_trials, T, hidden_dim)
        Full timestep-level hidden states.
    """
    device = next(model.parameters()).device
    model.eval()

    all_hidden = []
    n_trials = test_X.shape[0]

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            end = min(start + batch_size, n_trials)
            x_batch = torch.tensor(
                test_X[start:end], dtype=torch.float32
            ).to(device)

            _, h_seq = model(x_batch, return_hidden=True)
            # h_seq: (batch, T, hidden_dim)
            all_hidden.append(h_seq.cpu().numpy())

    hidden_full = np.concatenate(all_hidden, axis=0)  # (n_trials, T, hidden)
    hidden_avg = hidden_full.mean(axis=1)  # (n_trials, hidden)

    logger.info(
        "Extracted hidden states: full=%s  avg=%s",
        hidden_full.shape, hidden_avg.shape,
    )
    return hidden_avg, hidden_full


def extract_trained_and_untrained(model_path, test_X, input_dim, output_dim,
                                  hidden_size, save_dir=None):
    """Extract hidden states from trained model and untrained baseline.

    Returns
    -------
    trained_H : ndarray, (n_trials, hidden_dim)
    untrained_H : ndarray, (n_trials, hidden_dim)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Trained model
    trained_model = WMSurrogate(
        input_dim=input_dim, output_dim=output_dim,
        hidden_size=hidden_size,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    trained_model.load_state_dict(state_dict)
    trained_H, _ = extract_hidden_states(trained_model, test_X)

    # Untrained baseline (random init, no weight loading)
    untrained_model = WMSurrogate(
        input_dim=input_dim, output_dim=output_dim,
        hidden_size=hidden_size,
    ).to(device)
    untrained_H, _ = extract_hidden_states(untrained_model, test_X)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_dir / f'wm_h{hidden_size}_trained.npz',
            hidden_states=trained_H,
        )
        np.savez_compressed(
            save_dir / f'wm_h{hidden_size}_untrained.npz',
            hidden_states=untrained_H,
        )
        logger.info("Saved hidden states to %s", save_dir)

    return trained_H, untrained_H


def extract_all_sizes(splits, model_dir, save_dir):
    """Extract hidden states for all hidden sizes.

    Returns
    -------
    all_hidden : dict mapping hidden_size -> (trained_H, untrained_H)
    """
    model_dir = Path(model_dir)
    input_dim = splits['test']['X'].shape[2]
    output_dim = splits['test']['Y'].shape[2]

    all_hidden = {}
    for hs in HIDDEN_SIZES:
        model_path = model_dir / f'wm_h{hs}_best.pt'
        if not model_path.exists():
            logger.warning("Model not found: %s", model_path)
            continue

        logger.info("=== Extracting h=%d ===", hs)
        trained_H, untrained_H = extract_trained_and_untrained(
            model_path, splits['test']['X'],
            input_dim, output_dim, hs,
            save_dir=save_dir,
        )
        all_hidden[hs] = (trained_H, untrained_H)

    return all_hidden
