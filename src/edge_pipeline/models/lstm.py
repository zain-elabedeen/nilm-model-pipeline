"""LSTM model for NILM."""

import torch
import torch.nn as nn

from edge_pipeline.models.base import NilmModel


class LSTMModel(NilmModel):
    """
    3-layer bidirectional LSTM for sequence-to-point NILM.

    Architecture from SIDED paper:
    - 3 LSTM layers with 128 hidden units
    - Bidirectional processing
    - Dropout between layers
    - Dense output layer
    """

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
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

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection
        self.input_proj = nn.Linear(1, hidden_size)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output layers
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_appliances),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, window_size)

        Returns:
            Output of shape (batch, num_appliances)
        """
        # Add feature dimension: (batch, seq, 1)
        x = x.unsqueeze(-1)

        # Project input
        x = self.input_proj(x)

        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Use output at sequence midpoint (sequence-to-point)
        midpoint = lstm_out.size(1) // 2
        x = lstm_out[:, midpoint, :]

        # Output projection
        return self.fc(x)


class AttentionLSTMModel(NilmModel):
    """LSTM with attention mechanism for improved sequence modelling."""

    def __init__(
        self,
        window_size: int = 60,
        num_appliances: int = 5,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
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

        self.hidden_size = hidden_size

        # Input projection
        self.input_proj = nn.Linear(1, hidden_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Attention
        lstm_output_size = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_appliances),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add feature dimension
        x = x.unsqueeze(-1)
        x = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Pool at midpoint
        midpoint = attn_out.size(1) // 2
        x = attn_out[:, midpoint, :]

        return self.fc(x)
