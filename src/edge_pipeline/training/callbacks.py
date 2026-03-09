"""Custom callbacks for NILM training."""

from time import perf_counter
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.rank_zero import rank_zero_info


def _metric_to_float(value: object) -> float | None:
    """Convert scalar tensors/numbers to Python floats for logging."""
    if value is None:
        return None

    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu().item())
    except ModuleNotFoundError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


class PlainTextProgressCallback(pl.Callback):
    """Emit plain stdout progress lines that Vertex/Cloud Logging can capture."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = max(1, log_every_n_steps)
        self._epoch_start_time: float | None = None

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._epoch_start_time = perf_counter()
        rank_zero_info(
            f"[Training] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} started"
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking:
            return

        total_batches = trainer.num_training_batches
        should_log = (
            (batch_idx + 1) % self.log_every_n_steps == 0
            or batch_idx + 1 == total_batches
        )
        if not should_log:
            return

        train_loss = None
        if hasattr(outputs, "get"):
            train_loss = _metric_to_float(outputs.get("loss"))
        elif outputs is not None:
            train_loss = _metric_to_float(outputs)

        if train_loss is None:
            train_loss = _metric_to_float(trainer.callback_metrics.get("train/loss"))

        message = (
            f"[Training] epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"step {batch_idx + 1}/{total_batches}"
        )
        if train_loss is not None:
            message += f" train/loss={train_loss:.4f}"
        rank_zero_info(message)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        elapsed = None
        if self._epoch_start_time is not None:
            elapsed = perf_counter() - self._epoch_start_time

        train_loss = _metric_to_float(trainer.callback_metrics.get("train/loss"))
        message = f"[Training] epoch {trainer.current_epoch + 1} finished"
        if train_loss is not None:
            message += f" train/loss={train_loss:.4f}"
        if elapsed is not None:
            message += f" duration={elapsed:.1f}s"
        rank_zero_info(message)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        val_loss = _metric_to_float(trainer.callback_metrics.get("val/loss"))
        val_mae = _metric_to_float(trainer.callback_metrics.get("val/mae"))
        val_r2 = _metric_to_float(trainer.callback_metrics.get("val/r2"))

        message = f"[Validation] epoch {trainer.current_epoch + 1}"
        if val_loss is not None:
            message += f" val/loss={val_loss:.4f}"
        if val_mae is not None:
            message += f" val/mae={val_mae:.4f}"
        if val_r2 is not None:
            message += f" val/r2={val_r2:.4f}"
        rank_zero_info(message)

    def on_test_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        test_loss = _metric_to_float(trainer.callback_metrics.get("test/loss"))
        test_mae = _metric_to_float(trainer.callback_metrics.get("test/mae"))
        test_r2 = _metric_to_float(trainer.callback_metrics.get("test/r2"))

        message = "[Test]"
        if test_loss is not None:
            message += f" test/loss={test_loss:.4f}"
        if test_mae is not None:
            message += f" test/mae={test_mae:.4f}"
        if test_r2 is not None:
            message += f" test/r2={test_r2:.4f}"
        rank_zero_info(message)
