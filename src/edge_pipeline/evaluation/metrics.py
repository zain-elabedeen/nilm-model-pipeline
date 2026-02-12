"""Evaluation metrics for NILM models."""

import torch


class NilmMetrics:
    """
    Collection of evaluation metrics for NILM disaggregation.

    Metrics:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - R²: Coefficient of Determination
    - NDE: Normalised Disaggregation Error (NILM-specific)
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def mae(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Mean Absolute Error."""
        return torch.mean(torch.abs(y_pred - y_true))

    def mse(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error."""
        return torch.mean((y_pred - y_true) ** 2)

    def rmse(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Root Mean Squared Error."""
        return torch.sqrt(self.mse(y_pred, y_true))

    def r2(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Coefficient of Determination (R² score)."""
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1.0 - ss_res / (ss_tot + self.epsilon)

    def nde(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Normalised Disaggregation Error.

        NDE = sqrt(sum((y_pred - y_true)²)) / sum(y_true)

        Lower is better. Commonly used in NILM literature.
        """
        numerator = torch.sqrt(torch.sum((y_pred - y_true) ** 2))
        denominator = torch.sum(torch.abs(y_true)) + self.epsilon
        return numerator / denominator

    def signal_aggregate_error(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Signal Aggregate Error (SAE).

        SAE = |sum(y_pred) - sum(y_true)| / sum(y_true)

        Measures how well total energy is preserved.
        """
        pred_total = torch.sum(y_pred)
        true_total = torch.sum(torch.abs(y_true)) + self.epsilon
        return torch.abs(pred_total - true_total) / true_total

    def mae_per_appliance(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        """MAE computed per appliance (returns tensor of shape (5,))."""
        return torch.mean(torch.abs(y_pred - y_true), dim=0)

    def compute_all(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute all metrics."""
        return {
            "mae": self.mae(y_pred, y_true),
            "mse": self.mse(y_pred, y_true),
            "rmse": self.rmse(y_pred, y_true),
            "r2": self.r2(y_pred, y_true),
            "nde": self.nde(y_pred, y_true),
            "sae": self.signal_aggregate_error(y_pred, y_true),
            "mae_per_appliance": self.mae_per_appliance(y_pred, y_true),
        }


def compute_metrics_numpy(y_pred, y_true, epsilon: float = 1e-8) -> dict:
    """Compute metrics using NumPy arrays."""
    import numpy as np

    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + epsilon)

    nde = np.sqrt(np.sum((y_pred - y_true) ** 2)) / (np.sum(np.abs(y_true)) + epsilon)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "nde": float(nde),
    }
