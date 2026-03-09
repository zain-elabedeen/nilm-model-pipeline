#!/usr/bin/env python
"""Training script for NILM models using Hydra configuration."""

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.models.atcn import ATCNModel
from edge_pipeline.models.lstm import LSTMModel
from edge_pipeline.models.tcn import TCNModel
from edge_pipeline.training.trainer import NilmTrainer


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
    )

    # Train
    trainer.train()

    # Test
    results = trainer.test()
    print(f"\nTest results: {results}")

    # Return validation loss for hyperparameter optimisation
    return results.get("test/loss", float("inf"))


if __name__ == "__main__":
    main()
