"""Tests for NILM model architectures."""

import pytest
import torch

from edge_pipeline.models.atcn import ATCNModel, ATCNModelLite
from edge_pipeline.models.lstm import AttentionLSTMModel, LSTMModel
from edge_pipeline.models.tcn import TCNModel, TCNModelSimple


class TestModelForwardPass:
    """Test forward passes for all model architectures."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample input batch."""
        batch_size = 4
        window_size = 60
        return torch.randn(batch_size, window_size)

    def test_lstm_forward(self, sample_batch):
        """Test LSTM forward pass."""
        model = LSTMModel(
            window_size=60,
            num_appliances=5,
            hidden_size=32,
            num_layers=2,
        )

        output = model(sample_batch)

        assert output.shape == (4, 5)

    def test_attention_lstm_forward(self, sample_batch):
        """Test Attention LSTM forward pass."""
        model = AttentionLSTMModel(
            window_size=60,
            num_appliances=5,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
        )

        output = model(sample_batch)

        assert output.shape == (4, 5)

    def test_tcn_forward(self, sample_batch):
        """Test TCN forward pass."""
        model = TCNModel(
            window_size=60,
            num_appliances=5,
            num_channels=32,
            num_layers=4,
        )

        output = model(sample_batch)

        assert output.shape == (4, 5)

    def test_tcn_simple_forward(self, sample_batch):
        """Test simple TCN forward pass."""
        model = TCNModelSimple(
            window_size=60,
            num_appliances=5,
            num_channels=32,
            num_layers=3,
        )

        output = model(sample_batch)

        assert output.shape == (4, 5)

    def test_atcn_forward(self, sample_batch):
        """Test ATCN forward pass."""
        model = ATCNModel(
            window_size=60,
            num_appliances=5,
            num_channels=32,
            num_tcn_layers=4,
            num_attention_heads=2,
            attention_layers=1,
        )

        output = model(sample_batch)

        assert output.shape == (4, 5)

    def test_atcn_lite_forward(self, sample_batch):
        """Test ATCN lite forward pass."""
        model = ATCNModelLite(
            window_size=60,
            num_appliances=5,
            num_channels=32,
            num_tcn_layers=3,
            num_attention_heads=2,
        )

        output = model(sample_batch)

        assert output.shape == (4, 5)


class TestModelGradients:
    """Test gradient flow through models."""

    def test_lstm_gradients(self):
        """Test LSTM has valid gradients."""
        model = LSTMModel(window_size=60, hidden_size=16, num_layers=1)

        x = torch.randn(2, 60, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_tcn_gradients(self):
        """Test TCN has valid gradients."""
        model = TCNModel(window_size=60, num_channels=16, num_layers=2)

        x = torch.randn(2, 60, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_atcn_gradients(self):
        """Test ATCN has valid gradients."""
        model = ATCNModel(
            window_size=60,
            num_channels=16,
            num_tcn_layers=2,
            num_attention_heads=2,
        )

        x = torch.randn(2, 60, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestExampleInput:
    """Test example input generation for ONNX export."""

    def test_lstm_example_input(self):
        """Test LSTM example input shape."""
        model = LSTMModel(window_size=60)
        example = model.get_example_input()

        assert example.shape == (1, 60)

    def test_tcn_example_input(self):
        """Test TCN example input shape."""
        model = TCNModel(window_size=60)
        example = model.get_example_input()

        assert example.shape == (1, 60)

    def test_atcn_example_input(self):
        """Test ATCN example input shape."""
        model = ATCNModel(window_size=60)
        example = model.get_example_input()

        assert example.shape == (1, 60)


class TestCustomAppliances:
    """Test models with custom appliance names."""

    @pytest.fixture
    def sample_batch(self):
        """Create sample input batch."""
        return torch.randn(4, 60)

    def test_tcn_custom_appliances(self, sample_batch):
        """Test TCN with 3 custom appliances."""
        model = TCNModel(
            window_size=60,
            num_channels=16,
            num_layers=2,
            appliance_names=["A", "B", "C"],
        )

        assert model.num_appliances == 3
        assert model.appliance_names == ["A", "B", "C"]

        output = model(sample_batch)
        assert output.shape == (4, 3)

    def test_lstm_custom_appliances(self, sample_batch):
        """Test LSTM with 3 custom appliances."""
        model = LSTMModel(
            window_size=60,
            hidden_size=16,
            num_layers=1,
            appliance_names=["X", "Y", "Z"],
        )

        assert model.num_appliances == 3
        output = model(sample_batch)
        assert output.shape == (4, 3)

    def test_atcn_custom_appliances(self, sample_batch):
        """Test ATCN with 2 custom appliances."""
        model = ATCNModel(
            window_size=60,
            num_channels=16,
            num_tcn_layers=2,
            num_attention_heads=2,
            appliance_names=["SOLAR", "BATTERY"],
        )

        assert model.num_appliances == 2
        output = model(sample_batch)
        assert output.shape == (4, 2)

    def test_default_appliance_names(self):
        """Test default SIDED appliance names when none provided."""
        model = TCNModel(window_size=60, num_channels=16, num_layers=2)

        assert model.num_appliances == 5
        assert model.appliance_names == [
            "BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"
        ]

    def test_num_appliances_without_names(self):
        """Test generic names when num_appliances != 5 and no names given."""
        model = TCNModel(
            window_size=60,
            num_appliances=3,
            num_channels=16,
            num_layers=2,
        )

        assert model.num_appliances == 3
        assert model.appliance_names == ["APPLIANCE_0", "APPLIANCE_1", "APPLIANCE_2"]


class TestModelParameters:
    """Test model parameter counts."""

    def test_lstm_params(self):
        """Test LSTM parameter count is reasonable."""
        model = LSTMModel(window_size=60, hidden_size=128, num_layers=3)
        params = sum(p.numel() for p in model.parameters())

        # Should be less than 2M params for this config
        assert params < 2_000_000

    def test_tcn_params(self):
        """Test TCN parameter count is reasonable."""
        model = TCNModel(window_size=60, num_channels=128, num_layers=8)
        params = sum(p.numel() for p in model.parameters())

        assert params < 2_000_000

    def test_atcn_params(self):
        """Test ATCN parameter count is reasonable."""
        model = ATCNModel(
            window_size=60,
            num_channels=128,
            num_tcn_layers=8,
            num_attention_heads=4,
        )
        params = sum(p.numel() for p in model.parameters())

        assert params < 5_000_000
