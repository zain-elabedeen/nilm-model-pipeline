#!/usr/bin/env python
"""Training script for NILM models using Hydra configuration."""

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nilm_research.data.datamodule import NilmDataModule
from nilm_research.models.atcn import ATCNModel
from nilm_research.models.lstm import LSTMModel
from nilm_research.models.tcn import TCNModel
from nilm_research.training.trainer import NilmTrainer


def get_model(cfg: DictConfig):
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

    # Create datamodule
    datamodule = NilmDataModule(
        data_dir=cfg.data.data_dir,
        window_size=cfg.data.window_size,
        stride=cfg.data.stride,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        train_facilities=list(cfg.data.train_facilities),
        val_facilities=list(cfg.data.val_facilities),
        test_facilities=list(cfg.data.test_facilities),
        use_amda=cfg.data.use_amda,
        amda_scale=cfg.data.amda_scale,
        use_robust_scaling=cfg.data.use_robust_scaling,
    )

    # Create model
    model = get_model(cfg)

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
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        use_wandb=cfg.logging.use_wandb,
        wandb_offline=cfg.logging.wandb_offline,
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
