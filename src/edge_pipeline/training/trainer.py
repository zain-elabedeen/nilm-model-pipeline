"""Training utilities for NILM models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from edge_pipeline.data.base import NilmDatasetLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from edge_pipeline.data.datamodule import NilmDataModule
from edge_pipeline.models.base import NilmModel


class NilmTrainer:
    """High-level trainer for NILM models."""

    def __init__(
        self,
        model: NilmModel,
        datamodule: NilmDataModule,
        project_name: str = "underscore-edge-model-pipeline",
        experiment_name: str | None = None,
        output_dir: str | Path = "outputs",
        max_epochs: int = 100,
        patience: int = 10,
        accelerator: Literal["auto", "cpu", "gpu", "mps"] = "auto",
        devices: int | str = "auto",
        precision: Literal["32", "16-mixed", "bf16-mixed"] = "32",
        gradient_clip_val: float = 1.0,
        use_wandb: bool = True,
        wandb_offline: bool = False,
        check_val_every_n_epoch:int = 1,
    ):
        """
        Initialise trainer.

        Args:
            model: NILM model to train
            datamodule: Data module with train/val/test splits
            project_name: W&B project name
            experiment_name: W&B run name (auto-generated if None)
            output_dir: Directory for checkpoints and outputs
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            accelerator: Device accelerator
            devices: Number of devices
            precision: Training precision
            gradient_clip_val: Gradient clipping value
            use_wandb: Whether to use Weights & Biases logging
            wandb_offline: Run W&B in offline mode
        """
        self.model = model
        self.datamodule = datamodule
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        logger = None
        if use_wandb:
            logger = WandbLogger(
                project=project_name,
                name=experiment_name,
                save_dir=str(self.output_dir),
                offline=wandb_offline,
                log_model=True,
            )
            logger.watch(model, log="gradients", log_freq=100)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val/loss",
                patience=patience,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints",
                filename="best-{epoch:02d}-{val_loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            gradient_clip_val=gradient_clip_val,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            log_every_n_steps=20,
            check_val_every_n_epoch = check_val_every_n_epoch,
        )

    def train(self) -> None:
        """Train the model."""
        self.trainer.fit(self.model, datamodule=self.datamodule)

    def test(self) -> dict:
        """Test the model."""
        results = self.trainer.test(self.model, datamodule=self.datamodule)
        return results[0] if results else {}

    def get_best_checkpoint(self) -> str | None:
        """Get path to best checkpoint."""
        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback is None:
            return None

        return checkpoint_callback.best_model_path


def train_model(
    model_type: Literal["lstm", "tcn"],
    window_size: int = 60,
    batch_size: int = 256,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    stride: int = 5,
    use_amda: bool = True,
    amda_scale: float = 2.5,
    num_workers: int = 4,
    output_dir: str = "outputs",
    use_wandb: bool = True,
    check_val_every_n_epoch:int = 1,
    loader: NilmDatasetLoader | None = None,
) -> tuple[NilmModel, NilmTrainer]:
    """
    Convenience function to train a NILM model.

    Args:
        model_type: Type of model to train
        window_size: Input window size
        batch_size: Training batch size
        max_epochs: Maximum epochs
        learning_rate: Learning rate
        use_amda: Whether to use AMDA augmentation
        amda_scale: AMDA scaling factor
        num_workers: DataLoader workers (reduce to 2 on Colab)
        output_dir: Output directory
        use_wandb: Whether to use W&B logging
        check_val_every_n_epoch: Run Validation every n epochs
        loader: Optional dataset loader. If None, uses SIDED.

    Returns:
        Tuple of (trained model, trainer)
    """
    from edge_pipeline.models.lstm import LSTMModel
    from edge_pipeline.models.tcn import TCNModel

    # Create datamodule
    datamodule = NilmDataModule(
        loader=loader,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amda=use_amda,
        amda_scale=amda_scale,
        stride=stride,
    )

    # Create model
    model_classes = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
    }

    model = model_classes[model_type](
        window_size=window_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        appliance_names=datamodule.appliance_names,
    )

    # Create trainer
    trainer = NilmTrainer(
        model=model,
        datamodule=datamodule,
        experiment_name=f"{model_type}_amda{amda_scale:.1f}" if use_amda else model_type,
        output_dir=output_dir,
        max_epochs=max_epochs,
        use_wandb=use_wandb,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    # Train
    trainer.train()

    return model, trainer
