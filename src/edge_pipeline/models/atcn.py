"""Attention-augmented Temporal Convolutional Network for NILM."""

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from edge_pipeline.models.base import NilmModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class CausalConv1d(nn.Module):
    """Causal 1D convolution."""

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
        if self.padding > 0:
            x = x[:, :, : -self.padding]
        return x


class TemporalBlock(nn.Module):
    """TCN block with residual."""

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

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))

        return self.relu(x + residual)


class ATCNModel(NilmModel):
    """
    Attention-augmented Temporal Convolutional Network.

    Combines TCN for local temporal patterns with multi-head self-attention
    for capturing long-range dependencies.

    Architecture:
    - Input projection
    - TCN encoder (8 layers, dilated convolutions)
    - Positional encoding
    - Multi-head self-attention (4 heads)
    - Output projection
    """

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        num_channels: int = 128,
        num_tcn_layers: int = 8,
        kernel_size: int = 3,
        num_attention_heads: int = 4,
        attention_layers: int = 2,
        dropout: float = 0.33,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
    ):
        super().__init__(
            window_size=window_size,
            num_appliances=num_appliances,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
        )
        self.save_hyperparameters()

        self.num_channels = num_channels

        # Input projection
        self.input_proj = nn.Conv1d(1, num_channels, 1)

        # TCN encoder
        tcn_layers = []
        for i in range(num_tcn_layers):
            dilation = 2**i
            tcn_layers.append(
                TemporalBlock(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*tcn_layers)

        # Positional encoding for attention
        self.pos_encoding = PositionalEncoding(num_channels, max_len=window_size, dropout=dropout)

        # Multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_channels,
            nhead=num_attention_heads,
            dim_feedforward=num_channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)

        # Layer norm before output
        self.norm = nn.LayerNorm(num_channels)

        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_appliances),
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

        # TCN encoding: (batch, channels, seq)
        x = self.input_proj(x)
        x = self.tcn(x)

        # Transpose for attention: (batch, seq, channels)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Self-attention
        x = self.transformer(x)

        # Layer norm
        x = self.norm(x)

        # Pool at midpoint for sequence-to-point
        midpoint = x.size(1) // 2
        x = x[:, midpoint, :]

        # Output projection
        return self.fc(x)


class ATCNModelLite(NilmModel):
    """Lighter ATCN variant with fewer parameters."""

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        num_channels: int = 64,
        num_tcn_layers: int = 4,
        kernel_size: int = 5,
        num_attention_heads: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 100,
    ):
        super().__init__(
            window_size=window_size,
            num_appliances=num_appliances,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
        )
        self.save_hyperparameters()

        # Simple TCN
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(1, num_channels, kernel_size, padding=kernel_size // 2))

        for i in range(num_tcn_layers - 1):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            self.conv_layers.append(
                nn.Conv1d(
                    num_channels, num_channels, kernel_size, padding=padding, dilation=dilation
                )
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Single attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(num_channels)

        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels, num_appliances),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = self.dropout(self.relu(conv(x)))

        # (batch, seq, channels)
        x = x.transpose(1, 2)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # Midpoint pooling
        midpoint = x.size(1) // 2
        x = x[:, midpoint, :]

        return self.fc(x)
