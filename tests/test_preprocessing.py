"""Tests for preprocessing utilities."""

import numpy as np
import pytest

from edge_pipeline.data.preprocessing import (
    NormalisationParams,
    RobustScaler,
    WindowGenerator,
    ZScoreNormaliser,
)


class TestNormalisationParams:
    """Tests for NormalisationParams dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = NormalisationParams(mean=100.0, std=50.0, min_val=0.0, max_val=200.0)
        d = params.to_dict()

        assert d["mean"] == 100.0
        assert d["std"] == 50.0
        assert d["min"] == 0.0
        assert d["max"] == 200.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {"mean": 100.0, "std": 50.0, "min": 0.0, "max": 200.0}
        params = NormalisationParams.from_dict(d)

        assert params.mean == 100.0
        assert params.std == 50.0
        assert params.min_val == 0.0
        assert params.max_val == 200.0


class TestZScoreNormaliser:
    """Tests for Z-score normalisation."""

    def test_fit_transform(self):
        """Test fitting and transforming data."""
        data = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        normaliser = ZScoreNormaliser()

        transformed = normaliser.fit_transform(data)

        # Mean should be approximately 0
        assert abs(np.mean(transformed)) < 1e-10

        # Std should be approximately 1
        assert abs(np.std(transformed) - 1.0) < 1e-10

    def test_inverse_transform(self):
        """Test roundtrip normalisation."""
        data = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        normaliser = ZScoreNormaliser()

        normaliser.fit(data)
        transformed = normaliser.transform(data)
        recovered = normaliser.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, data)

    def test_get_params(self):
        """Test getting normalisation parameters."""
        data = np.array([0.0, 100.0, 200.0])
        normaliser = ZScoreNormaliser()
        normaliser.fit(data)

        params = normaliser.get_params()

        assert params.mean == 100.0
        assert params.min_val == 0.0
        assert params.max_val == 200.0


class TestRobustScaler:
    """Tests for robust scaling."""

    def test_robust_scaling(self):
        """Test robust scaling uses median and IQR."""
        # Data with outlier
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        scaler = RobustScaler(use_robust=True)
        scaler.fit(data)

        # Center should be median (3.5), not mean
        assert scaler.center_ == 3.5

    def test_standard_scaling_fallback(self):
        """Test non-robust mode uses mean/std."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        scaler = RobustScaler(use_robust=False)
        scaler.fit(data)

        assert scaler.center_ == np.mean(data)

    def test_roundtrip(self):
        """Test roundtrip transformation."""
        data = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        scaler = RobustScaler()

        scaler.fit(data)
        transformed = scaler.transform(data)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, data)


class TestWindowGenerator:
    """Tests for sliding window generation."""

    def test_window_generation(self):
        """Test basic window generation."""
        generator = WindowGenerator(window_size=5, stride=1)

        # Create simple test data
        aggregate = np.arange(10, dtype=np.float32)
        appliances = np.column_stack([
            np.arange(10),
            np.arange(10) * 2,
            np.arange(10) * 3,
            np.arange(10) * 4,
            np.arange(10) * 5,
        ]).astype(np.float32)

        inputs, targets = generator.generate_windows(aggregate, appliances, normalise=False)

        # Should have 6 windows (10 - 5 + 1)
        assert inputs.shape == (6, 5)
        assert targets.shape == (6, 5)

        # First window should be [0, 1, 2, 3, 4]
        np.testing.assert_array_equal(inputs[0], [0, 1, 2, 3, 4])

        # Target should be at midpoint (index 2)
        assert targets[0, 0] == 2

    def test_window_stride(self):
        """Test window generation with stride."""
        generator = WindowGenerator(window_size=5, stride=2)

        aggregate = np.arange(10, dtype=np.float32)
        appliances = np.zeros((10, 5), dtype=np.float32)

        inputs, targets = generator.generate_windows(aggregate, appliances, normalise=False)

        # With stride 2: windows at 0, 2, 4 -> 3 windows
        assert inputs.shape[0] == 3

    def test_fit_normalisers(self):
        """Test fitting normalisers on training data."""
        generator = WindowGenerator(window_size=5)

        aggregate = np.random.randn(100).astype(np.float32) * 10000 + 5000
        appliances = np.random.randn(100, 5).astype(np.float32) * 2000 + 1000

        generator.fit_normalisers(aggregate, appliances, use_robust=True)

        assert generator.input_normaliser is not None
        assert generator.output_normalisers is not None
        assert len(generator.output_normalisers) == 5

    def test_normalisation_metadata(self):
        """Test getting normalisation metadata for export."""
        generator = WindowGenerator(window_size=60)

        aggregate = np.random.randn(1000).astype(np.float32) * 10000
        appliances = np.random.randn(1000, 5).astype(np.float32) * 2000

        generator.fit_normalisers(aggregate, appliances)

        metadata = generator.get_normalisation_metadata()

        assert metadata["window_size"] == 60
        assert metadata["input"] is not None
        assert "mean" in metadata["input"]
        assert "std" in metadata["input"]
        assert len(metadata["outputs"]) == 5
