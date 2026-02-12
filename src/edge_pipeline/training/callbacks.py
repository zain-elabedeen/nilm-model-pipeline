"""Custom callbacks for NILM training."""

from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class EarlyStoppingWithLogging(EarlyStopping):
    """Early stopping with additional logging."""

    def __init__(
        self,
        monitor: str = "val/loss",
        min_delta: float = 0.0001,
        patience: int = 10,
        mode: Literal["min", "max"] = "min",
        verbose: bool = True,
        check_on_train_epoch_end: bool = False,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            verbose=verbose,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )
        self.best_epoch = 0

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        super().on_validation_end(trainer, pl_module)

        # Track best epoch
        if self.best_score is not None:
            current = self._get_current_score(trainer)
            if self._is_improvement(current):
                self.best_epoch = trainer.current_epoch

    def _get_current_score(self, trainer: pl.Trainer) -> float:
        """Get current monitored value."""
        logs = trainer.callback_metrics
        return logs.get(self.monitor, float("inf" if self.mode == "min" else "-inf"))

    def _is_improvement(self, current: float) -> bool:
        """Check if current score is an improvement."""
        if self.best_score is None:
            return True

        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta


class GradientLoggingCallback(pl.Callback):
    """Log gradient statistics during training."""

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = []
        for p in pl_module.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.data.norm(2).item())

        if grad_norms:
            import torch

            grad_norm = torch.tensor(grad_norms).mean().item()
            max_grad = max(grad_norms)

            pl_module.log("train/grad_norm", grad_norm)
            pl_module.log("train/grad_max", max_grad)


class ModelSummaryCallback(pl.Callback):
    """Log model summary at start of training."""

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        print(f"\nModel: {pl_module.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print()

        if hasattr(pl_module, "logger") and pl_module.logger is not None:
            pl_module.logger.experiment.config.update({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
            })
