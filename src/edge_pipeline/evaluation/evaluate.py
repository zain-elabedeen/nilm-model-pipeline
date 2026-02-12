"""Model evaluation utilities for NILM models."""

from typing import Literal

import numpy as np
import torch

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.evaluation.metrics import compute_metrics_numpy


APPLIANCE_NAMES = ["BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"]


def load_model(checkpoint_path: str, model_type: Literal["lstm", "tcn", "atcn"]):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint
        model_type: Model architecture type

    Returns:
        Loaded model in eval mode
    """
    from edge_pipeline.models.atcn import ATCNModel
    from edge_pipeline.models.lstm import LSTMModel
    from edge_pipeline.models.tcn import TCNModel

    model_classes = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "atcn": ATCNModel,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_classes[model_type].load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(
    model,
    datamodule: NilmDataModule,
    device: str = "cpu",
) -> dict:
    """Evaluate model on test set.

    Args:
        model: Trained NILM model
        datamodule: DataModule with test data (must have setup() called)
        device: Device to run inference on

    Returns:
        Dict of metric names to values, including per-appliance breakdowns
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    test_loader = datamodule.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y_pred = model(x).cpu().numpy()
            all_preds.append(y_pred)
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Overall metrics
    results = compute_metrics_numpy(preds, targets)

    # Per-appliance metrics
    for i, name in enumerate(APPLIANCE_NAMES):
        appliance_metrics = compute_metrics_numpy(preds[:, i], targets[:, i])
        results[f"{name}_mae"] = appliance_metrics["mae"]
        results[f"{name}_mse"] = appliance_metrics["mse"]
        results[f"{name}_r2"] = appliance_metrics["r2"]

    return results
