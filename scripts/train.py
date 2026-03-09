#!/usr/bin/env python
"""Training script for NILM models using Hydra configuration."""

import json
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.models.atcn import ATCNModel
from edge_pipeline.models.lstm import LSTMModel
from edge_pipeline.models.tcn import TCNModel
from edge_pipeline.training.trainer import NilmTrainer
from scripts.export_onnx import persist_training_artifacts
from scripts.vertex.register_model import register_uploaded_model


def get_loader(cfg: DictConfig):
    """Instantiate a dataset loader from config."""
    dataset_name = cfg.data.get("name", "sided")

    if dataset_name == "csv":
        from edge_pipeline.data.csv_loader import CsvLoader

        return CsvLoader(
            data_dir=cfg.data.data_dir,
            aggregate_column=cfg.data.get("aggregate_column", "aggregate"),
            timestamp_column=cfg.data.get("timestamp_column", "timestamp"),
            resolution_minutes=cfg.data.get("resolution_minutes", 1),
            file_pattern=cfg.data.get("file_pattern", "*.csv"),
        )

    if dataset_name == "sided":
        from edge_pipeline.data.sided_loader import SidedLoader

        return SidedLoader(cache_dir=cfg.data.data_dir)

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_model(cfg: DictConfig, appliance_names: list[str]):
    """Instantiate model from config."""
    model_classes = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "atcn": ATCNModel,
    }

    model_name = cfg.model.name
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = model_classes[model_name]

    # Common params
    common_params = {
        "window_size": cfg.data.window_size,
        "learning_rate": cfg.model.learning_rate,
        "weight_decay": cfg.model.weight_decay,
        "max_epochs": cfg.training.max_epochs,
        "appliance_names": appliance_names,
    }

    # Model-specific params
    if model_name == "lstm":
        return model_class(
            **common_params,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            bidirectional=cfg.model.bidirectional,
            dropout=cfg.model.dropout,
        )
    elif model_name == "tcn":
        return model_class(
            **common_params,
            num_channels=cfg.model.num_channels,
            num_layers=cfg.model.num_layers,
            kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        )
    elif model_name == "atcn":
        return model_class(
            **common_params,
            num_channels=cfg.model.num_channels,
            num_tcn_layers=cfg.model.num_tcn_layers,
            kernel_size=cfg.model.kernel_size,
            num_attention_heads=cfg.model.num_attention_heads,
            attention_layers=cfg.model.attention_layers,
            dropout=cfg.model.dropout,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _resolve_vertex_path(path_str: str | None) -> Path | None:
    """Map Vertex URIs like gs://bucket/path to the /gcs mount when possible."""
    if not path_str:
        return None
    if path_str.startswith("gs://"):
        return Path("/gcs") / path_str.removeprefix("gs://")
    return Path(path_str).expanduser()


def _artifact_dirs(output_dir: str | Path, artifact_uri: str | None = None) -> list[Path]:
    """Return all artifact destinations for this run."""
    dirs: list[Path] = [Path(output_dir).expanduser() / "artifacts"]

    aip_model_dir = _resolve_vertex_path(artifact_uri)
    if aip_model_dir is not None:
        dirs.append(aip_model_dir)

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        key = str(directory)
        if key not in seen:
            seen.add(key)
            unique_dirs.append(directory)
    return unique_dirs


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _register_vertex_model_if_enabled(
    *,
    cfg: DictConfig,
    artifact_uri: str | None,
    manifest: dict,
) -> None:
    """Register uploaded model artifacts in Vertex when explicitly enabled."""
    if not _env_flag("VERTEX_SETUP_MODEL", default=False):
        return

    if not artifact_uri or not artifact_uri.startswith("gs://"):
        print("[Artifacts] Skipping Vertex model registration: no gs:// artifact URI found")
        return

    serving_container_image_uri = os.getenv("VERTEX_SERVING_CONTAINER_IMAGE_URI")
    if not serving_container_image_uri:
        print(
            "[Artifacts] Skipping Vertex model registration: "
            "VERTEX_SERVING_CONTAINER_IMAGE_URI is not set"
        )
        return

    model = register_uploaded_model(
        project_id=(
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("CLOUD_ML_PROJECT_ID")
            or os.getenv("PROJECT_ID")
        ),
        region=(
            os.getenv("VERTEX_REGION")
            or os.getenv("CLOUD_ML_REGION")
            or os.getenv("GOOGLE_CLOUD_REGION")
            or "us-central1"
        ),
        artifact_uri=artifact_uri,
        display_name=os.getenv(
            "VERTEX_MODEL_DISPLAY_NAME",
            OmegaConf.select(cfg, "vertex.model_display_name", default=cfg.model.name),
        ),
        description=os.getenv("VERTEX_MODEL_DESCRIPTION", ""),
        serving_container_image_uri=serving_container_image_uri,
        predict_route=os.getenv("VERTEX_PREDICT_ROUTE", "/predict"),
        health_route=os.getenv("VERTEX_HEALTH_ROUTE", "/health"),
        serving_port=int(os.getenv("VERTEX_SERVING_PORT", "8080")),
    )
    manifest["vertex_model_resource_name"] = model.resource_name
    print(f"[Artifacts] Registered Vertex model: {model.resource_name}")


def export_training_artifacts(
    cfg: DictConfig,
    trainer: NilmTrainer,
    results: dict,
) -> None:
    """Persist checkpoints, config, metrics, and ONNX exports for Vertex/local runs."""
    output_dir = Path(cfg.output_dir).expanduser()
    best_checkpoint = trainer.get_best_checkpoint()
    best_checkpoint_path = Path(best_checkpoint) if best_checkpoint else None
    last_checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    resolved_config_yaml = OmegaConf.to_yaml(cfg, resolve=True)

    artifact_uri = OmegaConf.select(
        cfg,
        "vertex.artifact_uri",
        default=os.getenv("AIP_MODEL_DIR"),
    )

    for artifact_dir in _artifact_dirs(output_dir, artifact_uri=artifact_uri):
        manifest, manifest_path = persist_training_artifacts(
            artifact_dir=artifact_dir,
            model_name=cfg.model.name,
            experiment_name=cfg.get("name"),
            output_dir=output_dir,
            aip_model_dir=artifact_uri,
            resolved_config_yaml=resolved_config_yaml,
            resolved_config=resolved_config,
            results=results,
            best_checkpoint_path=best_checkpoint_path,
            last_checkpoint_path=last_checkpoint_path,
            data_dir=cfg.data.data_dir,
            dataset=cfg.data.get("name", "sided"),
            window_size=cfg.data.window_size,
            batch_size=cfg.data.batch_size,
            aggregate_column=cfg.data.get("aggregate_column", "aggregate"),
            timestamp_column=cfg.data.get("timestamp_column", "timestamp"),
        )

        if artifact_uri and str(artifact_dir) == str(_resolve_vertex_path(artifact_uri)):
            _register_vertex_model_if_enabled(
                cfg=cfg,
                artifact_uri=artifact_uri,
                manifest=manifest,
            )
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train a NILM model."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    pl.seed_everything(cfg.seed)

    # Create loader
    loader = get_loader(cfg)

    # Resolve site splits
    train_sites = list(cfg.data.train_facilities) if "train_facilities" in cfg.data else None
    val_sites = list(cfg.data.val_facilities) if "val_facilities" in cfg.data else None
    test_sites = list(cfg.data.test_facilities) if "test_facilities" in cfg.data else None

    # Create datamodule
    datamodule = NilmDataModule(
        loader=loader,
        data_dir=cfg.data.data_dir,
        window_size=cfg.data.window_size,
        stride=cfg.data.stride,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_sites=train_sites,
        val_sites=val_sites,
        test_sites=test_sites,
        use_amda=cfg.data.use_amda,
        amda_scale=cfg.data.amda_scale,
        use_robust_scaling=cfg.data.use_robust_scaling,
        strict_site_ids=cfg.data.get("strict_site_ids", False),
    )

    # Create model
    model = get_model(cfg, datamodule.appliance_names)

    # Create trainer
    experiment_name = f"{cfg.model.name}"
    if cfg.data.use_amda:
        experiment_name += f"_amda{cfg.data.amda_scale:.1f}"

    trainer = NilmTrainer(
        model=model,
        datamodule=datamodule,
        project_name=cfg.logging.project_name,
        experiment_name=experiment_name,
        output_dir=cfg.output_dir,
        max_epochs=cfg.training.max_epochs,
        patience=cfg.training.patience,
        check_val_every_n_epoch=OmegaConf.select(cfg, "training.check_val_every_n_epoch", default=1),
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        use_wandb=cfg.logging.use_wandb,
        wandb_offline=cfg.logging.wandb_offline,
        use_mlflow=OmegaConf.select(cfg, "logging.use_mlflow", default=False),
        mlflow_tracking_uri=OmegaConf.select(cfg, "logging.mlflow_tracking_uri", default=None),
        mlflow_experiment_name=OmegaConf.select(
            cfg, "logging.mlflow_experiment_name", default=cfg.logging.project_name
        ),
        mlflow_run_name=OmegaConf.select(cfg, "logging.mlflow_run_name", default=None),
        log_every_n_steps=OmegaConf.select(cfg, "logging.log_every_n_steps", default=50),
    )

    # Train
    trainer.train()

    # Test
    results = trainer.test()
    print(f"\nTest results: {results}")

    # Persist exported artifacts for Vertex and local runs
    export_training_artifacts(cfg, trainer, results)

    # Return validation loss for hyperparameter optimisation
    return results.get("test/loss", float("inf"))


if __name__ == "__main__":
    main()
