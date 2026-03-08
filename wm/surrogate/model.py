"""
DESCARTES WM — LSTM Surrogate for ALM → Thalamus Transformation

Input:  (batch, T, n_alm_neurons)  — ALM population activity per timestep
Output: (batch, T, n_thal_neurons) — Predicted thalamus activity per timestep

Architecture matches the L5PC surrogate to enable cross-circuit comparison
of learned representations. Key difference: multi-dimensional output
(population prediction) instead of scalar voltage.

Hidden states (batch, T, hidden_size) are the substrate for probing.
"""

import torch
import torch.nn as nn

from wm.config import N_LSTM_LAYERS


class WMSurrogate(nn.Module):
    """LSTM surrogate for the ALM → Thalamus transformation.

    Parameters
    ----------
    input_dim : int
        Number of ALM neurons (varies per session).
    output_dim : int
        Number of thalamus neurons (varies per session).
    hidden_size : int
        LSTM hidden dimension. Swept over [64, 128, 256].
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers.
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_layers=N_LSTM_LAYERS, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, return_hidden=False):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, (batch, T, input_dim)
        return_hidden : bool
            If True, return hidden states for probing.

        Returns
        -------
        y_pred : torch.Tensor, (batch, T, output_dim)
        hidden_states : torch.Tensor, optional
            (batch, T, hidden_size) — only if return_hidden=True.
        """
        h_seq, (h_n, c_n) = self.lstm(x)
        y_pred = self.output_proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"WMSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )
