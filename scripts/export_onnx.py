#!/usr/bin/env python
"""Export trained NILM model to ONNX format for Rust deployment."""

import argparse
from pathlib import Path

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.export.onnx_exporter import OnnxExporter
from edge_pipeline.models.lstm import LSTMModel
from edge_pipeline.models.tcn import TCNModel


def load_model(checkpoint_path: str, model_type: str):
    """Load model from checkpoint."""
    model_classes = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_classes[model_type].load_from_checkpoint(checkpoint_path)


def get_loader(args):
    """Create a dataset loader from CLI args."""
    dataset = getattr(args, "dataset", "sided")

    if dataset == "csv":
        from edge_pipeline.data.csv_loader import CsvLoader

        return CsvLoader(
            data_dir=args.data_dir,
            aggregate_column=getattr(args, "aggregate_column", "aggregate"),
            timestamp_column=getattr(args, "timestamp_column", "timestamp"),
        )

    from edge_pipeline.data.sided_loader import SidedLoader
    return SidedLoader(cache_dir=args.data_dir)


def main():
    parser = argparse.ArgumentParser(description="Export NILM model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "tcn"],
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nilm",
        help="Output model name",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="~/.cache/edge-pipeline/sided",
        help="Data directory (for normalisation metadata)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["sided", "csv"],
        default="sided",
        help="Dataset type",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Window size",
    )
    parser.add_argument(
        "--copy-to-rust",
        type=str,
        help="Copy ONNX to Rust project models directory",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.model_type)

    # Get normalisation metadata from datamodule
    print("Loading normalisation metadata...")
    loader = get_loader(args)
    datamodule = NilmDataModule(
        loader=loader,
        data_dir=args.data_dir,
        window_size=args.window_size,
        use_amda=False,
    )
    datamodule.setup()
    normalisation_metadata = datamodule.get_normalisation_metadata()

    # Export
    print(f"Exporting to ONNX...")
    exporter = OnnxExporter(
        model=model,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    paths = exporter.export_for_rust(normalisation_metadata)

    print(f"\nExported files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Copy to Rust project if requested
    if args.copy_to_rust:
        import shutil

        rust_models_dir = Path(args.copy_to_rust)
        rust_models_dir.mkdir(parents=True, exist_ok=True)

        for path in paths.values():
            dest = rust_models_dir / path.name
            shutil.copy(path, dest)
            print(f"\nCopied to: {dest}")


if __name__ == "__main__":
    main()
