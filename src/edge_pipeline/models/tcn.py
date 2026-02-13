"""Temporal Convolutional Network for NILM."""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from edge_pipeline.models.base import NilmModel


class CausalConv1d(nn.Module):
    """Causal convolution with appropriate padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Remove future samples (causal)
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class TemporalBlock(nn.Module):
    """Single TCN block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        return self.relu_out(x + residual)


class TCNModel(NilmModel):
    """
    Temporal Convolutional Network for sequence-to-point NILM.

    Architecture from SIDED paper:
    - 8 dilated convolutional layers
    - 128 channels per layer
    - Kernel size 3, dilations doubling each layer
    - Dropout 0.33
    """

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        num_channels: int = 128,
        num_layers: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.33,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        appliance_names: list[str] | None = None,
    ):
        super().__init__(
            window_size=window_size,
            num_appliances=num_appliances,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            appliance_names=appliance_names,
        )
        self.save_hyperparameters()

        self.num_channels = num_channels

        # Input projection
        self.input_proj = nn.Conv1d(1, num_channels, 1)

        # TCN blocks with exponentially increasing dilations
        layers = []
        for i in range(num_layers):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels, self.num_appliances),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, window_size)

        Returns:
            Output of shape (batch, num_appliances)
        """
        # Add channel dimension: (batch, 1, seq)
        x = x.unsqueeze(1)

        # Project and process through TCN
        x = self.input_proj(x)
        x = self.tcn(x)

        # Output projection
        return self.fc(x)


class TCNModelSimple(NilmModel):
    """Simpler TCN variant for faster training."""

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        num_channels: int = 64,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
        appliance_names: list[str] | None = None,
    ):
        super().__init__(
            window_size=window_size,
            num_appliances=num_appliances,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            appliance_names=appliance_names,
        )
        self.save_hyperparameters()

        layers = [nn.Conv1d(1, num_channels, kernel_size, padding=kernel_size // 2)]

        for i in range(num_layers - 1):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    num_channels,
                    num_channels,
                    kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
            ])

        self.encoder = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(num_channels, self.num_appliances),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return self.head(x)
