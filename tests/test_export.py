"""Tests for ONNX export functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from nilm_research.export.onnx_exporter import OnnxExporter
from nilm_research.models.lstm import LSTMModel
from nilm_research.models.tcn import TCNModel


class TestOnnxExporter:
    """Tests for ONNX export."""

    @pytest.fixture
    def sample_model(self):
        """Create a small model for testing."""
        return TCNModel(
            window_size=60,
            num_appliances=5,
            num_channels=16,
            num_layers=2,
        )

    def test_export_basic(self, sample_model):
        """Test basic ONNX export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OnnxExporter(
                model=sample_model,
                output_dir=tmpdir,
                model_name="test_model",
            )

            onnx_path = exporter.export(validate=True)

            assert onnx_path.exists()
            assert onnx_path.suffix == ".onnx"

    def test_export_with_metadata(self, sample_model):
        """Test export with normalisation metadata."""
        metadata = {
            "window_size": 60,
            "input": {"mean": 25000.0, "std": 20000.0, "min": -50000.0, "max": 150000.0},
            "outputs": {
                "EVSE": {"mean": 5000.0, "std": 5000.0, "min": 0.0, "max": 22000.0},
                "PV": {"mean": -15000.0, "std": 15000.0, "min": -50000.0, "max": 0.0},
                "CS": {"mean": 10000.0, "std": 8000.0, "min": 0.0, "max": 30000.0},
                "CHP": {"mean": -5000.0, "std": 8000.0, "min": -20000.0, "max": 5000.0},
                "BA": {"mean": 3000.0, "std": 2000.0, "min": 0.0, "max": 10000.0},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OnnxExporter(
                model=sample_model,
                output_dir=tmpdir,
                model_name="test_model",
            )

            onnx_path = exporter.export(normalisation_metadata=metadata, validate=True)

            # Check metadata file was created
            metadata_path = Path(tmpdir) / "test_model_metadata.json"
            assert metadata_path.exists()

            # Verify metadata contents
            with open(metadata_path) as f:
                loaded_metadata = json.load(f)

            assert loaded_metadata["window_size"] == 60
            assert "EVSE" in loaded_metadata["outputs"]

    def test_export_for_rust(self, sample_model):
        """Test export with Rust-compatible configuration."""
        metadata = {
            "window_size": 60,
            "input": {"mean": 25000.0, "std": 20000.0, "min": -50000.0, "max": 150000.0},
            "outputs": {
                "EVSE": {"mean": 5000.0, "std": 5000.0, "min": 0.0, "max": 22000.0},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OnnxExporter(
                model=sample_model,
                output_dir=tmpdir,
                model_name="nilm",
            )

            paths = exporter.export_for_rust(metadata)

            assert "onnx" in paths
            assert "config" in paths
            assert paths["onnx"].exists()
            assert paths["config"].exists()

            # Verify Rust config format
            with open(paths["config"]) as f:
                rust_config = json.load(f)

            assert "input_normalisation" in rust_config
            assert "output_normalisations" in rust_config
            assert rust_config["input_normalisation"]["std_dev"] == 20000.0

    def test_pytorch_onnx_equivalence(self, sample_model):
        """Test that PyTorch and ONNX outputs match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OnnxExporter(
                model=sample_model,
                output_dir=tmpdir,
                model_name="test_model",
            )

            onnx_path = exporter.export(validate=False)

            # Generate test input
            test_input = torch.randn(1, 60)

            # PyTorch inference
            sample_model.eval()
            with torch.no_grad():
                pytorch_output = sample_model(test_input).numpy()

            # ONNX inference
            session = ort.InferenceSession(str(onnx_path))
            onnx_output = session.run(
                ["output"],
                {"input": test_input.numpy()},
            )[0]

            # Compare outputs
            np.testing.assert_array_almost_equal(
                pytorch_output,
                onnx_output,
                decimal=5,
            )

    def test_onnx_input_output_shapes(self, sample_model):
        """Test ONNX model has correct input/output shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OnnxExporter(
                model=sample_model,
                output_dir=tmpdir,
                model_name="test_model",
            )

            onnx_path = exporter.export(validate=False)

            # Load and inspect model
            onnx_model = onnx.load(str(onnx_path))

            # Check input
            input_shape = [
                dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim
            ]
            assert input_shape == [1, 60]

            # Check output
            output_shape = [
                dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim
            ]
            assert output_shape == [1, 5]

    def test_lstm_export(self):
        """Test LSTM model exports correctly."""
        model = LSTMModel(
            window_size=60,
            num_appliances=5,
            hidden_size=32,
            num_layers=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OnnxExporter(
                model=model,
                output_dir=tmpdir,
                model_name="lstm_test",
            )

            onnx_path = exporter.export(validate=True)

            assert onnx_path.exists()

            # Verify inference works
            session = ort.InferenceSession(str(onnx_path))
            output = session.run(
                ["output"],
                {"input": np.random.randn(1, 60).astype(np.float32)},
            )[0]

            assert output.shape == (1, 5)
