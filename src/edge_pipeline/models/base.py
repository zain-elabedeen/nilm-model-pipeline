"""Base NILM model interface."""

from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from edge_pipeline.evaluation.metrics import NilmMetrics


class NilmModel(pl.LightningModule, ABC):
    """
    Abstract base class for NILM sequence-to-point models.

    All models take input shape (batch, window_size) and output
    (batch, num_appliances) for the configured appliance types.
    """

    # Class-level default for backwards compatibility with old checkpoints
    APPLIANCE_NAMES = ["BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"]

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        appliance_names: list[str] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.window_size = window_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        if appliance_names is not None:
            self.appliance_names = appliance_names
            self.num_appliances = len(appliance_names)
        else:
            self.num_appliances = num_appliances
            # Fall back to class-level SIDED defaults when num_appliances matches
            if num_appliances == len(self.APPLIANCE_NAMES):
                self.appliance_names = list(self.APPLIANCE_NAMES)
            else:
                self.appliance_names = [f"APPLIANCE_{i}" for i in range(num_appliances)]

        self.metrics = NilmMetrics()
        self.loss_fn = nn.MSELoss()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size)

        Returns:
            Output tensor of shape (batch, num_appliances)
        """
        pass

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Compute metrics
        metrics = self.metrics.compute_all(y_hat, y)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mae", metrics["mae"], prog_bar=True)
        self.log("val/mse", metrics["mse"])
        self.log("val/r2", metrics["r2"])
        self.log("val/nde", metrics["nde"])

        # Per-appliance metrics
        for i, name in enumerate(self.appliance_names):
            self.log(f"val/mae_{name}", metrics["mae_per_appliance"][i])

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        metrics = self.metrics.compute_all(y_hat, y)

        self.log("test/loss", loss)
        self.log("test/mae", metrics["mae"])
        self.log("test/mse", metrics["mse"])
        self.log("test/r2", metrics["r2"])
        self.log("test/nde", metrics["nde"])

        for i, name in enumerate(self.appliance_names):
            self.log(f"test/mae_{name}", metrics["mae_per_appliance"][i])

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_example_input(self) -> torch.Tensor:
        """Get example input for ONNX export."""
        return torch.randn(1, self.window_size)
