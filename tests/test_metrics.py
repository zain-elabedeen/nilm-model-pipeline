"""Tests for evaluation metrics."""

import numpy as np
import pytest
import torch

from nilm_research.evaluation.metrics import NilmMetrics, compute_metrics_numpy


class TestNilmMetrics:
    """Tests for NilmMetrics class."""

    @pytest.fixture
    def metrics(self):
        return NilmMetrics()

    def test_mae(self, metrics):
        """Test Mean Absolute Error."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 2.0, 4.0])

        mae = metrics.mae(y_pred, y_true)

        # MAE = (0 + 0 + 1) / 3 = 0.333...
        assert abs(mae.item() - 1 / 3) < 1e-6

    def test_mse(self, metrics):
        """Test Mean Squared Error."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 2.0, 4.0])

        mse = metrics.mse(y_pred, y_true)

        # MSE = (0 + 0 + 1) / 3 = 0.333...
        assert abs(mse.item() - 1 / 3) < 1e-6

    def test_rmse(self, metrics):
        """Test Root Mean Squared Error."""
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        y_true = torch.tensor([1.0, 2.0, 5.0])

        rmse = metrics.rmse(y_pred, y_true)

        # MSE = (0 + 0 + 4) / 3, RMSE = sqrt(4/3)
        expected = np.sqrt(4 / 3)
        assert abs(rmse.item() - expected) < 1e-6

    def test_r2_perfect(self, metrics):
        """Test R² for perfect predictions."""
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        r2 = metrics.r2(y_pred, y_true)

        assert abs(r2.item() - 1.0) < 1e-6

    def test_r2_baseline(self, metrics):
        """Test R² for mean prediction (should be 0)."""
        y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = torch.full_like(y_true, y_true.mean())

        r2 = metrics.r2(y_pred, y_true)

        assert abs(r2.item()) < 1e-6

    def test_nde(self, metrics):
        """Test Normalised Disaggregation Error."""
        y_pred = torch.tensor([100.0, 200.0])
        y_true = torch.tensor([100.0, 200.0])

        nde = metrics.nde(y_pred, y_true)

        # Perfect predictions should have NDE = 0
        assert abs(nde.item()) < 1e-6

    def test_mae_per_appliance(self, metrics):
        """Test per-appliance MAE."""
        y_pred = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        y_true = torch.tensor([
            [2.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 2.0, 3.0, 4.0, 5.0],
        ])

        mae_per_app = metrics.mae_per_appliance(y_pred, y_true)

        # First appliance: MAE = (1 + 1) / 2 = 1
        # Others: MAE = 0
        assert mae_per_app.shape == (5,)
        assert abs(mae_per_app[0].item() - 1.0) < 1e-6
        assert abs(mae_per_app[1].item()) < 1e-6

    def test_compute_all(self, metrics):
        """Test computing all metrics at once."""
        y_pred = torch.randn(10, 5)
        y_true = torch.randn(10, 5)

        results = metrics.compute_all(y_pred, y_true)

        assert "mae" in results
        assert "mse" in results
        assert "rmse" in results
        assert "r2" in results
        assert "nde" in results
        assert "sae" in results
        assert "mae_per_appliance" in results


class TestNumpyMetrics:
    """Tests for NumPy-based metrics."""

    def test_compute_metrics_numpy(self):
        """Test NumPy metrics function."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 3.0])

        results = compute_metrics_numpy(y_pred, y_true)

        assert results["mae"] < 1e-6
        assert results["mse"] < 1e-6
        assert results["r2"] > 0.999

    def test_metrics_type(self):
        """Test that metrics are Python floats."""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.5, 2.5, 3.5])

        results = compute_metrics_numpy(y_pred, y_true)

        assert isinstance(results["mae"], float)
        assert isinstance(results["mse"], float)
        assert isinstance(results["r2"], float)
