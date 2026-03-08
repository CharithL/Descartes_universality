"""
DESCARTES Human Universality -- Surrogate Training Loop

Training loop for the Human WM surrogate pipeline. Adapted from the mouse
WM pipeline (wm.surrogate.train) with extensions required for the human
universality tests:

    - Multi-architecture training (LSTM, GRU, Transformer, Linear)
    - Multi-seed training for cross-seed reproducibility tests
    - Hidden-state extraction for downstream probing
    - Cross-condition correlation using trial_info dicts (not scalar arrays)

The core training mechanics (early stopping, cosine annealing, gradient
clipping, MSE loss) are identical to the mouse pipeline to keep the two
branches directly comparable.

Usage
-----
    from human_wm.surrogate.train import (
        train_multi_architecture,
        train_multi_seed,
    )

    # Train all 4 architectures at a fixed hidden size
    results = train_multi_architecture(splits, hidden_size=128,
                                       output_dir='surrogates/h128')

    # Train N seeds of one architecture for cross-seed test
    results = train_multi_seed(splits, arch_name='lstm', hidden_size=128,
                                n_seeds=10, output_dir='surrogates/seed_sweep')
"""

import logging
from pathlib import Path

import numpy as np

# NOTE: torch imports are lazy -- loaded inside functions that need them.
# This allows pure-Python utilities (e.g. _detect_condition_column) to be
# imported and tested without a working torch installation / DLL.

from human_wm.config import (
    ARCHITECTURES,
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    HIDDEN_SIZES,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_CC_THRESHOLD,
    N_SEEDS,
    WEIGHT_DECAY,
)
logger = logging.getLogger(__name__)

# Columns that are likely to encode meaningful experimental conditions.
# Ordered by priority -- the first column found in trial_info is used.
_CONDITION_COLUMN_PRIORITY = [
    'in_set',       # Sternberg: stimulus in memorised set?
    'match',        # match / non-match condition
    'correct',      # correct / incorrect response
    'set_size',     # memory load (number of items)
    'category',     # stimulus category
    'condition',    # generic condition label
]


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------

def create_dataloader(X, Y, batch_size=BATCH_SIZE, shuffle=True):
    """Create a PyTorch DataLoader from numpy arrays.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_bins, n_input)
        Input neural population activity (e.g. MTL spike counts).
    Y : np.ndarray, shape (n_trials, n_bins, n_output)
        Target neural population activity (e.g. Frontal spike counts).
    batch_size : int, optional
        Mini-batch size. Default from config (32).
    shuffle : bool, optional
        Whether to shuffle at each epoch. Default True.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader wrapping a TensorDataset of (X, Y).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_surrogate(model, train_loader, val_loader, n_epochs=MAX_EPOCHS,
                    lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE,
                    save_path=None):
    """Train a surrogate model with early stopping and cosine annealing.

    The training procedure is intentionally identical to the mouse WM
    pipeline to ensure that any differences in learned representations
    are attributable to the data, not the training procedure.

    Training details
    ~~~~~~~~~~~~~~~~
    - Optimiser: Adam with weight decay (AdamW-style via ``weight_decay``).
    - Scheduler: CosineAnnealingLR over ``n_epochs``.
    - Loss: MSE between predicted and actual target population activity.
    - Gradient clipping: max-norm clipping at ``GRAD_CLIP_NORM``.
    - Early stopping: stops when validation loss does not improve for
      ``patience`` consecutive epochs.

    Parameters
    ----------
    model : nn.Module
        A surrogate model instance (any architecture returned by
        ``create_surrogate``).  Must return ``(y_pred,)`` or
        ``(y_pred, hidden_states)`` from its ``forward`` method.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader (used for early stopping).
    n_epochs : int, optional
        Maximum number of training epochs. Default 200.
    lr : float, optional
        Initial learning rate. Default 1e-3.
    patience : int, optional
        Early-stopping patience (epochs without improvement). Default 20.
    save_path : str or Path or None
        Where to save the best checkpoint. If None, a default name is
        generated from the model's ``hidden_size`` attribute.

    Returns
    -------
    model : nn.Module
        The model with best-validation-loss weights loaded.
    history : dict
        Dictionary with keys ``'train_loss'`` and ``'val_loss'``, each
        a list of per-epoch loss values.
    """
    import torch
    import torch.nn as nn

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    # Determine save path
    if save_path is None:
        hs = getattr(model, 'hidden_size', 'unk')
        save_path = f'human_surrogate_h{hs}_best.pt'
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        n_train = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            Y_pred = model(X_batch)[0]
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            n_train += X_batch.size(0)

        train_loss /= max(n_train, 1)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                Y_pred = model(X_batch)[0]
                loss = criterion(Y_pred, Y_batch)
                val_loss += loss.item() * X_batch.size(0)
                n_val += X_batch.size(0)

        val_loss /= max(n_val, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter >= patience:
            logger.info(
                "Epoch %d/%d  train=%.6f  val=%.6f  best=%.6f  patience=%d/%d",
                epoch, n_epochs, train_loss, val_loss, best_val_loss,
                patience_counter, patience,
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    # Reload best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    logger.info("Loaded best model from %s (val_loss=%.6f)",
                save_path, best_val_loss)
    return model, history


# ---------------------------------------------------------------------------
# Cross-condition correlation
# ---------------------------------------------------------------------------

def _detect_condition_column(trial_info):
    """Auto-detect a suitable condition column from a trial_info dict.

    Searches ``trial_info`` keys against a priority-ordered list of
    candidate column names that typically encode experimental conditions
    in the Rutishauser dataset (e.g. 'in_set', 'match', 'correct',
    'set_size').

    Parameters
    ----------
    trial_info : dict[str, np.ndarray]
        Per-trial metadata dictionary.

    Returns
    -------
    str or None
        The name of the best matching condition column, or None if no
        suitable column is found.
    """
    for col in _CONDITION_COLUMN_PRIORITY:
        if col in trial_info:
            values = np.asarray(trial_info[col])
            n_unique = len(np.unique(values))
            # A column must have at least 2 but not too many unique values
            # to be a useful grouping variable. If every trial is unique
            # the mean-per-condition degenerates to single-trial means.
            if 2 <= n_unique <= 20:
                return col
    # Fallback: try any column that looks categorical
    for col, values in trial_info.items():
        values = np.asarray(values)
        n_unique = len(np.unique(values))
        if 2 <= n_unique <= 20:
            return col
    return None


def compute_cross_condition_cc(model, X_test, Y_test, trial_info):
    """Compute cross-condition correlation between predicted and true output.

    For each unique condition level the per-condition mean predicted and
    true time-series are computed, concatenated across conditions, and
    correlated. This measures how well the model captures condition-level
    differences rather than mere trial-level variance.

    The condition column is auto-detected from ``trial_info`` by searching
    for common column names (e.g. 'in_set', 'match', 'correct',
    'set_size') in priority order. If no suitable column is found, the
    function returns NaN with a warning.

    Parameters
    ----------
    model : nn.Module
        Trained surrogate model.
    X_test : np.ndarray, shape (n_trials, n_bins, n_input)
        Test-set input.
    Y_test : np.ndarray, shape (n_trials, n_bins, n_output)
        Test-set ground truth target.
    trial_info : dict[str, np.ndarray]
        Per-trial metadata dictionary from the test split.

    Returns
    -------
    cc : float
        Pearson correlation coefficient between condition-mean predicted
        and true activity. Returns ``float('nan')`` if no suitable
        condition column is detected or if correlation is undefined.
    condition_col : str or None
        The column name used for grouping, or None if none was found.
    """
    # Detect grouping variable
    condition_col = _detect_condition_column(trial_info)
    if condition_col is None:
        # Fallback: compute simple prediction correlation (no condition grouping)
        # This measures how well the model predicts overall temporal dynamics
        logger.info(
            "No condition column found in trial_info (keys: %s). "
            "Using trial-averaged prediction correlation as quality metric.",
            list(trial_info.keys()),
        )
        import torch as _torch

        device = next(model.parameters()).device
        model.eval()
        with _torch.no_grad():
            X_tensor = _torch.tensor(X_test, dtype=_torch.float32).to(device)
            Y_pred = model(X_tensor)[0].cpu().numpy()

        # Compare trial-averaged predicted vs true output across time × neurons
        pred_avg = Y_pred.mean(axis=0).flatten()
        true_avg = Y_test.mean(axis=0).flatten()

        if np.std(pred_avg) < 1e-12 or np.std(true_avg) < 1e-12:
            logger.warning("Zero variance in trial-avg predictions; CC undefined.")
            return float('nan'), None

        cc = float(np.corrcoef(pred_avg, true_avg)[0, 1])
        logger.info("Trial-averaged prediction CC = %.3f", cc)
        return cc, None

    conditions = np.asarray(trial_info[condition_col])
    logger.info("Using condition column '%s' with %d unique levels",
                condition_col, len(np.unique(conditions)))

    import torch

    # Forward pass
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_pred = model(X_tensor)[0].cpu().numpy()

    # Compute per-condition means
    unique_conditions = np.unique(conditions)
    pred_means = []
    true_means = []
    for c in unique_conditions:
        mask = conditions == c
        pred_means.append(Y_pred[mask].mean(axis=0).flatten())
        true_means.append(Y_test[mask].mean(axis=0).flatten())

    pred_concat = np.concatenate(pred_means)
    true_concat = np.concatenate(true_means)

    # Guard against constant vectors (zero variance)
    if np.std(pred_concat) < 1e-12 or np.std(true_concat) < 1e-12:
        logger.warning("Zero variance in condition means; CC undefined.")
        return float('nan'), condition_col

    cc = float(np.corrcoef(pred_concat, true_concat)[0, 1])
    return cc, condition_col


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(model, X):
    """Extract hidden-state trajectories from a trained surrogate.

    These hidden states are the substrate for downstream probing --
    linear classifiers/regressors are trained on top of frozen hidden
    states to test what cognitive variables are linearly decodable from
    the surrogate's internal representations.

    Parameters
    ----------
    model : nn.Module
        A trained surrogate model whose ``forward(x, return_hidden=True)``
        returns ``(y_pred, hidden_states)``.
    X : np.ndarray, shape (n_trials, n_bins, n_input)
        Input population activity.

    Returns
    -------
    hidden_states : np.ndarray, shape (n_trials, n_bins, hidden_size)
        The per-timestep hidden-state activations for every trial.

    Raises
    ------
    ValueError
        If the model does not support ``return_hidden=True`` or does not
        return the expected tuple.
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor, return_hidden=True)

    if not isinstance(outputs, tuple) or len(outputs) < 2:
        raise ValueError(
            f"Model {type(model).__name__} did not return hidden states. "
            "Ensure forward(x, return_hidden=True) returns (y_pred, h_seq)."
        )

    hidden_states = outputs[1].cpu().numpy()
    logger.info("Extracted hidden states: %s", hidden_states.shape)
    return hidden_states


# ---------------------------------------------------------------------------
# Multi-seed training
# ---------------------------------------------------------------------------

def train_multi_seed(splits, arch_name='lstm', hidden_size=128,
                     n_seeds=N_SEEDS, output_dir='surrogates/seed_sweep'):
    """Train the same architecture N times with different random seeds.

    This is the foundation of the *cross-seed universality test*: if
    representations are universal, different random initialisations of the
    same architecture should converge to linearly equivalent hidden-state
    geometries.

    Parameters
    ----------
    splits : dict
        Data splits dictionary with keys ``'train'``, ``'val'``, ``'test'``,
        each containing ``'X'``, ``'Y'``, and ``'trial_info'``.
    arch_name : str, optional
        Architecture name passed to ``create_surrogate``. One of
        ``'lstm'``, ``'gru'``, ``'transformer'``, ``'linear'``.
        Default ``'lstm'``.
    hidden_size : int, optional
        Hidden dimension. Default 128.
    n_seeds : int, optional
        Number of random seeds to train. Default from config (10).
    output_dir : str or Path, optional
        Directory for checkpoints. Each seed's model is saved as
        ``{arch_name}_h{hidden_size}_seed{seed}_best.pt``.

    Returns
    -------
    results : dict
        Mapping ``seed -> {model_path, cc, condition_col, n_params,
        history, hidden_path}``. Each entry also has ``hidden_path``
        pointing to the saved hidden states numpy file.
    """
    import torch
    from human_wm.surrogate.models import create_surrogate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = splits['train']['X'].shape[2]
    output_dim = splits['train']['Y'].shape[2]

    train_loader = create_dataloader(splits['train']['X'], splits['train']['Y'])
    val_loader = create_dataloader(splits['val']['X'], splits['val']['Y'],
                                   shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for seed in range(n_seeds):
        logger.info("=== Seed %d/%d  arch=%s  h=%d ===",
                     seed, n_seeds, arch_name, hidden_size)

        # Set global seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = create_surrogate(
            arch_name=arch_name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
        ).to(device)

        save_path = (
            output_dir / f'{arch_name}_h{hidden_size}_seed{seed}_best.pt'
        )
        model, history = train_surrogate(model, train_loader, val_loader,
                                         save_path=save_path)

        # Cross-condition correlation
        cc, condition_col = compute_cross_condition_cc(
            model,
            splits['test']['X'],
            splits['test']['Y'],
            splits['test']['trial_info'],
        )
        logger.info("Seed %d  CC=%.3f (col=%s)", seed, cc, condition_col)

        if not np.isnan(cc) and cc < MIN_CC_THRESHOLD:
            logger.warning(
                "Seed %d CC=%.3f is below MIN_CC_THRESHOLD=%.2f",
                seed, cc, MIN_CC_THRESHOLD,
            )

        # Extract and save hidden states for probing
        hidden = extract_hidden_states(model, splits['test']['X'])
        hidden_path = output_dir / f'{arch_name}_h{hidden_size}_seed{seed}_hidden.npy'
        np.save(hidden_path, hidden)

        results[seed] = {
            'model_path': str(save_path),
            'cc': cc,
            'condition_col': condition_col,
            'n_params': model.count_parameters(),
            'history': history,
            'hidden_path': str(hidden_path),
        }

    # Summary
    ccs = [r['cc'] for r in results.values() if not np.isnan(r['cc'])]
    if ccs:
        logger.info(
            "Multi-seed summary (%s h=%d): mean CC=%.3f +/- %.3f  "
            "(min=%.3f, max=%.3f, n=%d)",
            arch_name, hidden_size,
            np.mean(ccs), np.std(ccs),
            np.min(ccs), np.max(ccs), len(ccs),
        )

    return results


# ---------------------------------------------------------------------------
# Multi-architecture training
# ---------------------------------------------------------------------------

def train_multi_architecture(splits, hidden_size=128,
                             output_dir='surrogates/arch_sweep'):
    """Train all four architectures at a fixed hidden size.

    This is the foundation of the *cross-architecture universality test*:
    if representations are universal, different architectures (LSTM, GRU,
    Transformer, Linear) should converge to linearly equivalent
    hidden-state geometries despite having very different computational
    mechanisms.

    Each architecture is trained once (seed 0) here.  For the full
    cross-architecture x cross-seed matrix, call ``train_multi_seed``
    separately for each architecture.

    Parameters
    ----------
    splits : dict
        Data splits dictionary with keys ``'train'``, ``'val'``, ``'test'``,
        each containing ``'X'``, ``'Y'``, and ``'trial_info'``.
    hidden_size : int, optional
        Hidden dimension. Default 128.
    output_dir : str or Path, optional
        Directory for checkpoints. Each architecture's model is saved as
        ``{arch}_h{hidden_size}_best.pt``.

    Returns
    -------
    results : dict
        Mapping ``arch_name -> {model_path, cc, condition_col, n_params,
        history, hidden_path}``.
    """
    import torch
    from human_wm.surrogate.models import create_surrogate

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = splits['train']['X'].shape[2]
    output_dim = splits['train']['Y'].shape[2]

    train_loader = create_dataloader(splits['train']['X'], splits['train']['Y'])
    val_loader = create_dataloader(splits['val']['X'], splits['val']['Y'],
                                   shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for arch_name in ARCHITECTURES:
        logger.info("=== Architecture: %s  h=%d (in=%d, out=%d) ===",
                     arch_name, hidden_size, input_dim, output_dim)

        # Fixed seed for fair comparison across architectures
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        model = create_surrogate(
            arch_name=arch_name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
        ).to(device)

        save_path = output_dir / f'{arch_name}_h{hidden_size}_best.pt'
        model, history = train_surrogate(model, train_loader, val_loader,
                                         save_path=save_path)

        # Cross-condition correlation
        cc, condition_col = compute_cross_condition_cc(
            model,
            splits['test']['X'],
            splits['test']['Y'],
            splits['test']['trial_info'],
        )
        logger.info("arch=%s  CC=%.3f (col=%s)", arch_name, cc, condition_col)

        if not np.isnan(cc) and cc < MIN_CC_THRESHOLD:
            logger.warning(
                "arch=%s CC=%.3f is below MIN_CC_THRESHOLD=%.2f",
                arch_name, cc, MIN_CC_THRESHOLD,
            )

        # Extract and save hidden states for probing
        hidden = extract_hidden_states(model, splits['test']['X'])
        hidden_path = output_dir / f'{arch_name}_h{hidden_size}_hidden.npy'
        np.save(hidden_path, hidden)

        results[arch_name] = {
            'model_path': str(save_path),
            'cc': cc,
            'condition_col': condition_col,
            'n_params': model.count_parameters(),
            'history': history,
            'hidden_path': str(hidden_path),
        }

    # Summary table
    logger.info("=== Multi-architecture summary (h=%d) ===", hidden_size)
    for arch_name, r in results.items():
        logger.info(
            "  %-12s  CC=%.3f  params=%s",
            arch_name, r['cc'], f"{r['n_params']:,}",
        )

    return results
