"""
DESCARTES Human Universality -- Surrogate Architectures for Cross-Architecture Testing

Four surrogate architectures that learn the MTL -> Frontal (or Frontal -> MTL)
population transformation from human single-neuron recordings (Rutishauser
DANDI 000576). Each maps:

    Input:  (batch, T, n_source_neurons)  -- Source population activity per bin
    Output: (batch, T, n_target_neurons)  -- Predicted target population activity

All architectures expose an identical ``forward(x, return_hidden=True)``
interface so that downstream probing (Levels A / B / C) and ablation analyses
can treat them interchangeably. This is the key requirement for the
cross-architecture convergence test described in the universality guide:
if a representational motif (e.g. persistent delay coding) appears across
LSTM, GRU, Transformer, *and* Linear baselines, it is architecture-invariant.

Architectures
-------------
1. HumanLSTMSurrogate       -- LSTM  (matches mouse WMSurrogate pattern)
2. HumanGRUSurrogate        -- GRU   (gated recurrence, no cell state)
3. HumanTransformerSurrogate -- Transformer encoder with causal mask + sinusoidal PE
4. HumanLinearSurrogate      -- Temporal linear baseline (no recurrence)

Factory
-------
``create_surrogate(arch_name, input_dim, output_dim, hidden_size)``
    Instantiates any of the four by string key ('lstm', 'gru', 'transformer',
    'linear'), matching the ARCHITECTURES list in ``human_wm.config``.

Shared contract
---------------
Every model exposes:
    - ``hidden_size``  (int)  -- dimensionality of the hidden / embedding space
    - ``n_layers``     (int)  -- depth (stacked RNN layers, Transformer blocks, etc.)
    - ``output_proj``  (nn.Linear) -- final projection hidden_size -> output_dim
    - ``forward(x, return_hidden=False)``
          returns ``(y_pred,)`` or ``(y_pred, h_seq)``
          where h_seq is (batch, T, hidden_size)
    - ``count_parameters()``  -- total trainable scalar count
    - meaningful ``__repr__``
"""

import math

import torch
import torch.nn as nn

from human_wm.config import N_LSTM_LAYERS


# ---------------------------------------------------------------------------
# 1. LSTM
# ---------------------------------------------------------------------------

class HumanLSTMSurrogate(nn.Module):
    """LSTM surrogate -- direct analogue of the mouse ``WMSurrogate``.

    Two stacked LSTM layers with inter-layer dropout, followed by a linear
    output projection. Hidden states ``h_seq`` are the substrate for probing.

    Parameters
    ----------
    input_dim : int
        Number of source-region neurons.
    output_dim : int
        Number of target-region neurons.
    hidden_size : int
        LSTM hidden dimension (swept over HIDDEN_SIZES).
    n_layers : int
        Number of stacked LSTM layers (default from config).
    dropout : float
        Dropout probability between LSTM layers.
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
        x : torch.Tensor, shape (batch, T, input_dim)
            Source-region population activity.
        return_hidden : bool
            If True, also return the per-timestep hidden states for probing.

        Returns
        -------
        tuple
            ``(y_pred,)`` if ``return_hidden`` is False, else
            ``(y_pred, h_seq)`` where ``h_seq`` has shape
            (batch, T, hidden_size).
        """
        h_seq, (h_n, c_n) = self.lstm(x)
        y_pred = self.output_proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanLSTMSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# 2. GRU
# ---------------------------------------------------------------------------

class HumanGRUSurrogate(nn.Module):
    """GRU surrogate -- gated recurrence without a separate cell state.

    Structurally identical to the LSTM variant but uses ``nn.GRU``. The GRU
    has fewer parameters per layer (3 gates vs 4), providing a useful
    complexity contrast for the cross-architecture convergence test.

    Parameters
    ----------
    input_dim : int
        Number of source-region neurons.
    output_dim : int
        Number of target-region neurons.
    hidden_size : int
        GRU hidden dimension.
    n_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers.
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_layers=N_LSTM_LAYERS, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
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
        x : torch.Tensor, shape (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        tuple
            ``(y_pred,)`` or ``(y_pred, h_seq)``.
        """
        h_seq, h_n = self.gru(x)
        y_pred = self.output_proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanGRUSurrogate(in={self.input_dim}, out={self.output_dim}, "
            f"h={self.hidden_size}, layers={self.n_layers}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# 3. Transformer  (causal, encoder-only with sinusoidal positional encoding)
# ---------------------------------------------------------------------------

class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Registers a buffer of shape (1, max_len, d_model) so that the encodings
    move to the correct device automatically.

    Parameters
    ----------
    d_model : int
        Embedding / hidden dimension.
    max_len : int
        Maximum sequence length supported.
    dropout : float
        Dropout applied after adding positional encoding.
    """

    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)               # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                                       # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                               # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, d_model)

        Returns
        -------
        torch.Tensor, shape (batch, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HumanTransformerSurrogate(nn.Module):
    """Transformer-encoder surrogate with causal (autoregressive) masking.

    Uses a learnable linear projection from ``input_dim`` into ``hidden_size``
    (d_model), sinusoidal positional encoding, a stack of ``nn.TransformerEncoderLayer``
    blocks with causal attention, and a linear output projection.

    The causal mask ensures that predicted activity at time *t* depends only
    on source activity at times <= *t*, mirroring the strictly temporal
    nature of neural processing and matching the implicit causality of the
    LSTM / GRU variants.

    Parameters
    ----------
    input_dim : int
        Number of source-region neurons.
    output_dim : int
        Number of target-region neurons.
    hidden_size : int
        Transformer d_model dimension.
    n_layers : int
        Number of TransformerEncoderLayer blocks.
    n_heads : int
        Number of attention heads. Must divide ``hidden_size``.
    dropout : float
        Dropout for attention weights and feed-forward sublayers.
    dim_feedforward : int or None
        Inner dimension of the position-wise FFN. Defaults to 4 * hidden_size.
    max_len : int
        Maximum sequence length for positional encoding.
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_layers=N_LSTM_LAYERS, n_heads=4, dropout=0.1,
                 dim_feedforward=None, max_len=2048):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads

        if dim_feedforward is None:
            dim_feedforward = 4 * hidden_size

        # Project input neurons into d_model space
        self.input_proj = nn.Linear(input_dim, hidden_size)

        # Sinusoidal positional encoding
        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model=hidden_size, max_len=max_len, dropout=dropout,
        )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        # Output projection (same name as RNN variants for interface parity)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    # ---- causal mask helper ------------------------------------------------

    @staticmethod
    def _generate_causal_mask(seq_len, device):
        """Return an upper-triangular boolean causal mask.

        Parameters
        ----------
        seq_len : int
        device : torch.device

        Returns
        -------
        torch.Tensor, shape (seq_len, seq_len)
            ``True`` where attention is **blocked** (future positions).
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    # ---- forward -----------------------------------------------------------

    def forward(self, x, return_hidden=False):
        """Forward pass with causal masking.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        tuple
            ``(y_pred,)`` or ``(y_pred, h_seq)`` where ``h_seq`` is
            (batch, T, hidden_size) -- the Transformer encoder output
            *before* the output projection, analogous to LSTM hidden states.
        """
        T = x.size(1)
        mask = self._generate_causal_mask(T, x.device)

        h = self.input_proj(x)          # (batch, T, hidden_size)
        h = self.pos_enc(h)             # add positional encoding
        h_seq = self.transformer_encoder(h, mask=mask)  # (batch, T, hidden_size)
        y_pred = self.output_proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanTransformerSurrogate(in={self.input_dim}, "
            f"out={self.output_dim}, d_model={self.hidden_size}, "
            f"layers={self.n_layers}, heads={self.n_heads}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# 4. Linear baseline  (no recurrence, no non-linearity)
# ---------------------------------------------------------------------------

class HumanLinearSurrogate(nn.Module):
    """Per-timestep linear baseline with no recurrence.

    Maps each time-bin independently through ``input -> hidden -> output``
    using two linear layers with no activation in between. This provides
    a strict lower bound: any representational motif that appears in the
    linear model's hidden states is trivially decodable from the raw input
    and therefore not an *emergent* property of the architecture.

    Despite having no temporal modelling, the hidden layer still produces
    a ``(batch, T, hidden_size)`` tensor so the probing interface is
    identical to the recurrent and Transformer variants.

    Parameters
    ----------
    input_dim : int
        Number of source-region neurons.
    output_dim : int
        Number of target-region neurons.
    hidden_size : int
        Width of the intermediate linear layer.
    n_layers : int
        Stored for interface parity (always effectively 1).
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 n_layers=1, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers         # kept for interface parity

        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, x, return_hidden=False):
        """Forward pass -- independent per-timestep linear transform.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, input_dim)
        return_hidden : bool

        Returns
        -------
        tuple
            ``(y_pred,)`` or ``(y_pred, h_seq)``.
        """
        h_seq = self.input_proj(x)      # (batch, T, hidden_size)
        y_pred = self.output_proj(h_seq)

        if return_hidden:
            return y_pred, h_seq
        return (y_pred,)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"HumanLinearSurrogate(in={self.input_dim}, "
            f"out={self.output_dim}, h={self.hidden_size}, "
            f"params={self.count_parameters():,})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY = {
    'lstm': HumanLSTMSurrogate,
    'gru': HumanGRUSurrogate,
    'transformer': HumanTransformerSurrogate,
    'linear': HumanLinearSurrogate,
}


def create_surrogate(arch_name, input_dim, output_dim, hidden_size=128,
                     **kwargs):
    """Instantiate a surrogate model by architecture name.

    Parameters
    ----------
    arch_name : str
        One of ``'lstm'``, ``'gru'``, ``'transformer'``, ``'linear'``.
        Must match an entry in ``human_wm.config.ARCHITECTURES``.
    input_dim : int
        Number of source-region neurons.
    output_dim : int
        Number of target-region neurons.
    hidden_size : int
        Hidden / embedding dimension.
    **kwargs
        Additional keyword arguments forwarded to the constructor
        (e.g. ``n_heads`` for the Transformer, ``dropout``, etc.).

    Returns
    -------
    nn.Module
        An instance of the requested architecture.

    Raises
    ------
    ValueError
        If ``arch_name`` is not in the registry.

    Examples
    --------
    >>> model = create_surrogate('transformer', input_dim=40, output_dim=25,
    ...                          hidden_size=128, n_heads=8)
    >>> y, h = model(torch.randn(4, 60, 40), return_hidden=True)
    >>> y.shape
    torch.Size([4, 60, 25])
    >>> h.shape
    torch.Size([4, 60, 128])
    """
    arch_name = arch_name.lower().strip()
    if arch_name not in _REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Choose from {list(_REGISTRY.keys())}."
        )
    cls = _REGISTRY[arch_name]
    return cls(input_dim=input_dim, output_dim=output_dim,
               hidden_size=hidden_size, **kwargs)
