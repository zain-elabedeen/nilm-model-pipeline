#!/usr/bin/env python
"""CLI wrapper for evaluating trained NILM models."""

import argparse

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.evaluation.evaluate import APPLIANCE_NAMES, evaluate_model, load_model


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
        default="~/.cache/edge-pipeline/sided",
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
