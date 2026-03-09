#!/usr/bin/env python
"""Export trained NILM model to ONNX format for Rust deployment."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.export.onnx_exporter import OnnxExporter
from edge_pipeline.models.atcn import ATCNModel
from edge_pipeline.models.lstm import LSTMModel
from edge_pipeline.models.tcn import TCNModel


def load_model(checkpoint_path: str, model_type: str):
    """Load model from checkpoint."""
    model_classes = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "atcn": ATCNModel,
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


def create_export_datamodule(
    *,
    dataset: str,
    data_dir: str,
    window_size: int,
    batch_size: int = 256,
    aggregate_column: str = "aggregate",
    timestamp_column: str = "timestamp",
) -> NilmDataModule:
    """Build the datamodule used to recover normalisation metadata."""
    args = SimpleNamespace(
        dataset=dataset,
        data_dir=data_dir,
        aggregate_column=aggregate_column,
        timestamp_column=timestamp_column,
    )
    loader = get_loader(args)
    datamodule = NilmDataModule(
        loader=loader,
        data_dir=data_dir,
        window_size=window_size,
        batch_size=batch_size,
        use_amda=False,
    )
    datamodule.setup()
    return datamodule


def export_checkpoint_artifacts(
    *,
    checkpoint_path: str | Path,
    model_type: str,
    output_dir: str | Path,
    model_name: str,
    data_dir: str,
    dataset: str,
    window_size: int,
    batch_size: int = 256,
    aggregate_column: str = "aggregate",
    timestamp_column: str = "timestamp",
) -> dict[str, Path]:
    """Export ONNX artifacts for a trained checkpoint."""
    model = load_model(str(checkpoint_path), model_type)
    datamodule = create_export_datamodule(
        dataset=dataset,
        data_dir=data_dir,
        window_size=window_size,
        batch_size=batch_size,
        aggregate_column=aggregate_column,
        timestamp_column=timestamp_column,
    )
    exporter = OnnxExporter(
        model=model,
        output_dir=output_dir,
        model_name=model_name,
    )
    return exporter.export_for_rust(datamodule.get_normalisation_metadata())


def _copy_if_exists(src: Path | None, dest: Path) -> bool:
    """Copy a file if it exists."""
    if src is None or not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def persist_training_artifacts(
    *,
    artifact_dir: str | Path,
    model_name: str,
    experiment_name: str | None,
    output_dir: str | Path,
    aip_model_dir: str | None,
    resolved_config_yaml: str,
    resolved_config: dict,
    results: dict,
    best_checkpoint_path: str | Path | None,
    last_checkpoint_path: str | Path,
    data_dir: str,
    dataset: str,
    window_size: int,
    batch_size: int = 256,
    aggregate_column: str = "aggregate",
    timestamp_column: str = "timestamp",
) -> tuple[dict, Path]:
    """Write exported training artifacts and return the manifest plus its path."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "config_resolved.yaml", "w") as f:
        f.write(resolved_config_yaml)

    with open(artifact_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    best_path = Path(best_checkpoint_path) if best_checkpoint_path else None
    last_path = Path(last_checkpoint_path)
    copied_best = _copy_if_exists(best_path, artifact_dir / "best.ckpt")
    copied_last = _copy_if_exists(last_path, artifact_dir / "last.ckpt")

    export_dir = artifact_dir / "export"
    export_checkpoint_path = best_path if copied_best else last_path
    export_paths: dict[str, Path] = {}
    if export_checkpoint_path.exists():
        export_paths = export_checkpoint_artifacts(
            checkpoint_path=export_checkpoint_path,
            model_type=model_name,
            output_dir=export_dir,
            model_name=model_name,
            data_dir=data_dir,
            dataset=dataset,
            window_size=window_size,
            batch_size=batch_size,
            aggregate_column=aggregate_column,
            timestamp_column=timestamp_column,
        )

    manifest = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "artifact_dir": str(artifact_dir),
        "best_checkpoint": str(artifact_dir / "best.ckpt") if copied_best else None,
        "last_checkpoint": str(artifact_dir / "last.ckpt") if copied_last else None,
        "onnx": str(export_paths.get("onnx")) if "onnx" in export_paths else None,
        "rust_config": str(export_paths.get("config")) if "config" in export_paths else None,
        "results_file": str(artifact_dir / "test_results.json"),
        "resolved_config_file": str(artifact_dir / "config_resolved.yaml"),
        "output_dir": str(output_dir),
        "aip_model_dir": aip_model_dir,
        "resolved_config": resolved_config,
    }
    manifest_path = artifact_dir / "artifact_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Artifacts] Exported artifacts to {artifact_dir}")
    if copied_best:
        print(f"[Artifacts] best checkpoint: {artifact_dir / 'best.ckpt'}")
    if copied_last:
        print(f"[Artifacts] last checkpoint: {artifact_dir / 'last.ckpt'}")
    for name, path in export_paths.items():
        print(f"[Artifacts] {name}: {path}")

    return manifest, manifest_path


def main():
    parser = argparse.ArgumentParser(description="Export NILM model to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "tcn", "atcn"],
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
    print("Loading normalisation metadata...")
    print("Exporting to ONNX...")
    paths = export_checkpoint_artifacts(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        output_dir=args.output_dir,
        model_name=args.model_name,
        data_dir=args.data_dir,
        dataset=args.dataset,
        window_size=args.window_size,
        aggregate_column=getattr(args, "aggregate_column", "aggregate"),
        timestamp_column=getattr(args, "timestamp_column", "timestamp"),
    )

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
