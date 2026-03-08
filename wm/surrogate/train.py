"""
DESCARTES WM — Surrogate Training Loop

Standard training with early stopping, cosine annealing, and gradient
clipping. Trains the WMSurrogate to predict thalamic population activity
from ALM population activity during the delay period.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from wm.config import (
    BATCH_SIZE,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    HIDDEN_SIZES,
    LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
)
from wm.surrogate.model import WMSurrogate

logger = logging.getLogger(__name__)


def create_dataloader(X, Y, batch_size=BATCH_SIZE, shuffle=True):
    """Create a DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_surrogate(model, train_loader, val_loader, n_epochs=MAX_EPOCHS,
                    lr=LEARNING_RATE, patience=EARLY_STOP_PATIENCE,
                    save_path=None):
    """Train WMSurrogate with early stopping and cosine annealing.

    Returns
    -------
    model : WMSurrogate (best weights loaded)
    history : dict with 'train_loss', 'val_loss' lists
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    if save_path is None:
        save_path = f'wm_surrogate_h{model.hidden_size}_best.pt'
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
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

    model.load_state_dict(torch.load(save_path, weights_only=True))
    logger.info("Loaded best model from %s (val_loss=%.6f)",
                save_path, best_val_loss)
    return model, history


def compute_cross_condition_correlation(model, X_test, Y_test, trial_types):
    """Compute CC between predicted and actual thalamic activity.

    Returns
    -------
    cc : float
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        Y_pred = model(X_tensor)[0].cpu().numpy()

    conditions = np.unique(trial_types)
    pred_means = []
    true_means = []
    for c in conditions:
        mask = trial_types == c
        pred_means.append(Y_pred[mask].mean(axis=0).flatten())
        true_means.append(Y_test[mask].mean(axis=0).flatten())

    pred_concat = np.concatenate(pred_means)
    true_concat = np.concatenate(true_means)

    cc = np.corrcoef(pred_concat, true_concat)[0, 1]
    return float(cc)


def train_all_sizes(splits, session_info, output_dir):
    """Train surrogates at all hidden sizes for one session.

    Returns
    -------
    results : dict mapping hidden_size -> {model_path, cc, history}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dim = splits['train']['X'].shape[2]
    output_dim = splits['train']['Y'].shape[2]

    train_loader = create_dataloader(splits['train']['X'], splits['train']['Y'])
    val_loader = create_dataloader(splits['val']['X'], splits['val']['Y'],
                                   shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    for hs in HIDDEN_SIZES:
        logger.info("=== Training h=%d (in=%d, out=%d) ===",
                     hs, input_dim, output_dim)

        model = WMSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hs,
        ).to(device)

        save_path = output_dir / f'wm_h{hs}_best.pt'
        model, history = train_surrogate(model, train_loader, val_loader,
                                         save_path=save_path)

        cc = compute_cross_condition_correlation(
            model,
            splits['test']['X'],
            splits['test']['Y'],
            splits['test']['trial_types'],
        )
        logger.info("h=%d  CC=%.3f", hs, cc)

        results[hs] = {
            'model_path': str(save_path),
            'cc': cc,
            'n_params': model.count_parameters(),
            'history': history,
        }

    return results
