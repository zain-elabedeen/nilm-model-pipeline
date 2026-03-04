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
        wandb_watch_gradients: bool = False,
        wandb_watch_log_freq: int = 500,
        use_mlflow: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_experiment_name: str | None = None,
        mlflow_run_name: str | None = None,
        check_val_every_n_epoch: int = 1,
        log_every_n_steps: int = 50,
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
            wandb_watch_gradients: Log W&B gradient histograms (can slow training)
            wandb_watch_log_freq: Gradient watch logging frequency
            use_mlflow: Whether to use MLflow tracking
            mlflow_tracking_uri: MLflow tracking URI (e.g. file:./mlruns)
            mlflow_experiment_name: MLflow experiment name
            mlflow_run_name: MLflow run name
            log_every_n_steps: Lightning metric logging frequency
        """
        self.model = model
        self.datamodule = datamodule
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        loggers = []
        if use_wandb:
            wandb_logger = WandbLogger(
                project=project_name,
                name=experiment_name,
                save_dir=str(self.output_dir),
                offline=wandb_offline,
                log_model=True,
            )
            if wandb_watch_gradients:
                wandb_logger.watch(model, log="gradients", log_freq=wandb_watch_log_freq)
            loggers.append(wandb_logger)

        if use_mlflow:
            try:
                import mlflow  # noqa: F401
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "MLflow logging requested but mlflow is not installed. "
                    "Install dependencies with `uv sync` (or `pip install mlflow`)."
                ) from exc

            from pytorch_lightning.loggers import MLFlowLogger

            mlflow_logger = MLFlowLogger(
                experiment_name=mlflow_experiment_name or project_name,
                run_name=mlflow_run_name or experiment_name,
                tracking_uri=mlflow_tracking_uri,
                save_dir=str(self.output_dir / "mlruns"),
                log_model=False,
            )
            loggers.append(mlflow_logger)

        if len(loggers) == 1:
            logger = loggers[0]
        elif len(loggers) > 1:
            logger = loggers
        else:
            logger = False

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
        ]

        # LearningRateMonitor requires an active logger.
        if logger is not False:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

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
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
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
    model_type: Literal["lstm", "tcn", "atcn"],
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
    wandb_watch_gradients: bool = False,
    wandb_watch_log_freq: int = 500,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_run_name: str | None = None,
    check_val_every_n_epoch: int = 1,
    log_every_n_steps: int = 50,
    accelerator: Literal["auto", "cpu", "gpu", "mps"] = "auto",
    devices: int | str = "auto",
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32",
    train_sites: list[str] | None = None,
    val_sites: list[str] | None = None,
    test_sites: list[str] | None = None,
    auto_split_shuffle: bool = True,
    auto_split_seed: int = 42,
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
        wandb_watch_gradients: Log W&B gradient histograms (can slow training)
        wandb_watch_log_freq: Gradient watch logging frequency
        use_mlflow: Whether to use MLflow logging
        mlflow_tracking_uri: MLflow tracking URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        check_val_every_n_epoch: Run Validation every n epochs
        log_every_n_steps: Lightning metric logging frequency
        accelerator: Device accelerator
        devices: Number of devices
        precision: Training precision
        train_sites: Optional explicit list of train site IDs
        val_sites: Optional explicit list of validation site IDs
        test_sites: Optional explicit list of test site IDs
        auto_split_shuffle: Shuffle sites before auto split (when splits not provided)
        auto_split_seed: Random seed for deterministic auto split
        loader: Optional dataset loader. If None, uses SIDED.

    Returns:
        Tuple of (trained model, trainer)
    """
    from edge_pipeline.models.atcn import ATCNModel
    from edge_pipeline.models.lstm import LSTMModel
    from edge_pipeline.models.tcn import TCNModel

    # Create datamodule
    datamodule = NilmDataModule(
        loader=loader,
        window_size=window_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_sites=train_sites,
        val_sites=val_sites,
        test_sites=test_sites,
        use_amda=use_amda,
        amda_scale=amda_scale,
        stride=stride,
        auto_split_shuffle=auto_split_shuffle,
        auto_split_seed=auto_split_seed,
    )

    # Create model
    model_classes = {
        "lstm": LSTMModel,
        "tcn": TCNModel,
        "atcn": ATCNModel,
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
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        use_wandb=use_wandb,
        wandb_watch_gradients=wandb_watch_gradients,
        wandb_watch_log_freq=wandb_watch_log_freq,
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=log_every_n_steps,
    )

    # Train
    trainer.train()

    return model, trainer
