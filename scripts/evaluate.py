#!/usr/bin/env python
"""Evaluation script for trained NILM models."""

import argparse
from pathlib import Path

import numpy as np
import torch

from nilm_research.data.datamodule import NilmDataModule
from nilm_research.evaluation.metrics import NilmMetrics, compute_metrics_numpy
from nilm_research.models.atcn import ATCNModel
from nilm_research.models.lstm import LSTMModel
from nilm_research.models.tcn import TCNModel


APPLIANCE_NAMES = ["EVSE", "PV", "CS", "CHP", "BA"]


def load_model(checkpoint_path: str, model_type: str):
    """Load model from checkpoint."""
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
    """Evaluate model on test set."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate NILM model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "tcn", "atcn"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/.cache/nilm-research/sided",
        help="Data directory",
    )
    parser.add_argument("--window-size", type=int, default=60, help="Window size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Load data
    datamodule = NilmDataModule(
        data_dir=args.data_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        use_amda=False,  # No augmentation for evaluation
    )
    datamodule.setup()

    # Load model
    model = load_model(args.checkpoint, args.model_type)

    # Evaluate
    results = evaluate_model(model, datamodule, args.device)

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {results['mae']:.4f}")
    print(f"  MSE:  {results['mse']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  R²:   {results['r2']:.4f}")
    print(f"  NDE:  {results['nde']:.4f}")

    print(f"\nPer-Appliance MAE:")
    for name in APPLIANCE_NAMES:
        print(f"  {name}: {results[f'{name}_mae']:.4f}")

    # Save results
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
