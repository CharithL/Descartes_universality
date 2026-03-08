"""Tests for human_wm.surrogate.models -- surrogate architecture definitions and factory."""

import pytest
import torch
import torch.nn as nn

from human_wm.surrogate.models import (
    HumanGRUSurrogate,
    HumanLinearSurrogate,
    HumanLSTMSurrogate,
    HumanTransformerSurrogate,
    create_surrogate,
)


# ---------------------------------------------------------------------------
# Shared test dimensions (small for speed)
# ---------------------------------------------------------------------------

INPUT_DIM = 10
OUTPUT_DIM = 5
HIDDEN_SIZE = 32
BATCH = 4
T = 20

ALL_ARCH_NAMES = ['lstm', 'gru', 'transformer', 'linear']

ALL_ARCH_CLASSES = [
    HumanLSTMSurrogate,
    HumanGRUSurrogate,
    HumanTransformerSurrogate,
    HumanLinearSurrogate,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    """Provide a random input tensor with standard test dimensions."""
    return torch.randn(BATCH, T, INPUT_DIM)


# ---------------------------------------------------------------------------
# Forward pass -- instantiation and output shapes
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Verify each architecture can be instantiated and produces correct output shapes."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_forward_returns_tuple(self, arch_name, sample_input):
        """forward(x) returns a tuple for every architecture."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        result = model(sample_input)
        assert isinstance(result, tuple)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_output_shape(self, arch_name, sample_input):
        """y_pred has shape (batch, T, output_dim) for every architecture."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        (y_pred,) = model(sample_input)
        assert y_pred.shape == (BATCH, T, OUTPUT_DIM)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_forward_without_hidden_returns_single_element(self, arch_name, sample_input):
        """forward(x) without return_hidden returns a 1-element tuple."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        result = model(sample_input)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Hidden state output
# ---------------------------------------------------------------------------

class TestHiddenState:
    """Verify return_hidden=True produces hidden states with correct shapes."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_return_hidden_gives_two_outputs(self, arch_name, sample_input):
        """forward(x, return_hidden=True) returns exactly two tensors."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        result = model(sample_input, return_hidden=True)
        assert len(result) == 2

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_hidden_state_shape(self, arch_name, sample_input):
        """h_seq has shape (batch, T, hidden_size) for every architecture."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        y_pred, h_seq = model(sample_input, return_hidden=True)
        assert h_seq.shape == (BATCH, T, HIDDEN_SIZE)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_y_pred_same_with_and_without_hidden(self, arch_name, sample_input):
        """y_pred is identical regardless of return_hidden flag."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        model.eval()
        with torch.no_grad():
            (y_no_hidden,) = model(sample_input, return_hidden=False)
            y_with_hidden, _ = model(sample_input, return_hidden=True)
        assert torch.allclose(y_no_hidden, y_with_hidden)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestFactory:
    """Tests for the create_surrogate factory function."""

    @pytest.mark.parametrize('arch_name, expected_class', [
        ('lstm', HumanLSTMSurrogate),
        ('gru', HumanGRUSurrogate),
        ('transformer', HumanTransformerSurrogate),
        ('linear', HumanLinearSurrogate),
    ])
    def test_creates_correct_class(self, arch_name, expected_class):
        """Factory returns an instance of the correct architecture class."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, expected_class)

    def test_invalid_name_raises_value_error(self):
        """Factory raises ValueError for an unrecognised architecture name."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_surrogate('resnet', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)

    def test_case_insensitive(self):
        """Factory handles uppercase and mixed-case names."""
        model = create_surrogate('LSTM', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, HumanLSTMSurrogate)

    def test_whitespace_stripped(self):
        """Factory strips leading/trailing whitespace from the name."""
        model = create_surrogate('  gru  ', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert isinstance(model, HumanGRUSurrogate)

    def test_factory_forwards_kwargs(self):
        """Factory passes extra kwargs through to the constructor."""
        model = create_surrogate(
            'transformer', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE, n_heads=4,
        )
        assert model.n_heads == 4


# ---------------------------------------------------------------------------
# Required attributes
# ---------------------------------------------------------------------------

class TestAttributes:
    """Verify all models expose the shared interface attributes."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_has_hidden_size(self, arch_name):
        """Every model stores its hidden_size attribute."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'hidden_size')
        assert model.hidden_size == HIDDEN_SIZE

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_has_n_layers(self, arch_name):
        """Every model stores its n_layers attribute."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'n_layers')
        assert isinstance(model.n_layers, int)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_has_output_proj(self, arch_name):
        """Every model has an output_proj nn.Linear layer."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert hasattr(model, 'output_proj')
        assert isinstance(model.output_proj, nn.Linear)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_output_proj_dimensions(self, arch_name):
        """output_proj maps hidden_size -> output_dim."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert model.output_proj.in_features == HIDDEN_SIZE
        assert model.output_proj.out_features == OUTPUT_DIM


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

class TestCountParameters:
    """Tests for the count_parameters method."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_returns_positive_integer(self, arch_name):
        """count_parameters() returns a positive int for every architecture."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        n_params = model.count_parameters()
        assert isinstance(n_params, int)
        assert n_params > 0

    def test_lstm_has_more_params_than_gru(self):
        """LSTM has 4 gates vs GRU's 3, so more parameters for same config."""
        lstm = create_surrogate('lstm', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        gru = create_surrogate('gru', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert lstm.count_parameters() > gru.count_parameters()

    def test_linear_has_fewest_params(self):
        """Linear baseline should have fewer parameters than recurrent models."""
        linear = create_surrogate('linear', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        lstm = create_surrogate('lstm', INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        assert linear.count_parameters() < lstm.count_parameters()


# ---------------------------------------------------------------------------
# Different hidden sizes (from config.HIDDEN_SIZES: 64, 128, 256)
# ---------------------------------------------------------------------------

class TestHiddenSizes:
    """Verify models work with the hidden sizes used in the project sweep."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    @pytest.mark.parametrize('hidden_size', [64, 128, 256])
    def test_output_shape_varies_with_hidden_size(self, arch_name, hidden_size,
                                                   sample_input):
        """Output shape is correct regardless of hidden_size setting."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, hidden_size)
        (y_pred,) = model(sample_input)
        assert y_pred.shape == (BATCH, T, OUTPUT_DIM)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    @pytest.mark.parametrize('hidden_size', [64, 128, 256])
    def test_hidden_shape_varies_with_hidden_size(self, arch_name, hidden_size,
                                                   sample_input):
        """h_seq dimension matches the configured hidden_size."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, hidden_size)
        _, h_seq = model(sample_input, return_hidden=True)
        assert h_seq.shape == (BATCH, T, hidden_size)

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    @pytest.mark.parametrize('hidden_size', [64, 128, 256])
    def test_param_count_increases_with_hidden_size(self, arch_name, hidden_size):
        """Larger hidden sizes should produce more parameters."""
        small = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, 32)
        large = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, hidden_size)
        assert large.count_parameters() >= small.count_parameters()


# ---------------------------------------------------------------------------
# Gradient flow (backward pass)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify that gradients flow through all parameters on backward pass."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_backward_populates_gradients(self, arch_name, sample_input):
        """After loss.backward(), every trainable parameter has a non-None gradient."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        model.train()

        (y_pred,) = model(sample_input)
        loss = y_pred.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Gradient is None for parameter '{name}' in {arch_name}"
                )

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_gradients_are_nonzero(self, arch_name, sample_input):
        """Gradients should be non-zero (not vanished) for at least some parameters."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        model.train()

        (y_pred,) = model(sample_input)
        loss = y_pred.sum()
        loss.backward()

        has_nonzero = any(
            param.grad.abs().sum().item() > 0
            for param in model.parameters()
            if param.requires_grad and param.grad is not None
        )
        assert has_nonzero, (
            f"All gradients are zero for {arch_name} -- gradient flow is broken"
        )

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_backward_through_hidden_states(self, arch_name, sample_input):
        """Gradients flow through hidden states to upstream parameters.

        Note: output_proj is not in the computational graph of h_seq
        (hidden states are produced *before* the output projection), so
        only parameters upstream of h_seq are expected to receive gradients.
        """
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        model.train()

        _, h_seq = model(sample_input, return_hidden=True)
        loss = h_seq.sum()
        loss.backward()

        # At least one parameter upstream of h_seq must receive a gradient
        grads_received = [
            name for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        ]
        assert len(grads_received) > 0, (
            f"No parameter received a gradient through hidden states in {arch_name}"
        )

        # output_proj should NOT have gradients (it is downstream of h_seq)
        assert model.output_proj.weight.grad is None, (
            "output_proj should not receive gradients from h_seq backward"
        )


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    """Verify that __repr__ produces a non-empty string for every architecture."""

    @pytest.mark.parametrize('arch_name', ALL_ARCH_NAMES)
    def test_repr_is_nonempty_string(self, arch_name):
        """__repr__ returns a meaningful, non-empty string."""
        model = create_surrogate(arch_name, INPUT_DIM, OUTPUT_DIM, HIDDEN_SIZE)
        r = repr(model)
        assert isinstance(r, str)
        assert len(r) > 0
