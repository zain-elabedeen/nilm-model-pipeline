"""Data preprocessing for NILM models.

Implements normalisation and windowing that aligns with the Rust production system.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class NormalisationParams:
    """Parameters for normalisation, matching Rust preprocessing.rs."""

    mean: float
    std: float
    min_val: float
    max_val: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for JSON serialisation."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
        }

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "NormalisationParams":
        """Create from dictionary."""
        return cls(
            mean=d["mean"],
            std=d["std"],
            min_val=d["min"],
            max_val=d["max"],
        )


# Default parameters matching Rust inference.rs output_normalisers
DEFAULT_INPUT_PARAMS = NormalisationParams(
    mean=25_000.0,
    std=20_000.0,
    min_val=-50_000.0,
    max_val=150_000.0,
)

SIDED_DEFAULT_OUTPUT_PARAMS = {
    "BATTERY": NormalisationParams(mean=0.0, std=15_000.0, min_val=-50_000.0, max_val=50_000.0),
    "SOLAR": NormalisationParams(mean=-20_000.0, std=20_000.0, min_val=-100_000.0, max_val=0.0),
    "COOLING": NormalisationParams(mean=20_000.0, std=20_000.0, min_val=0.0, max_val=100_000.0),
    "GENERATOR": NormalisationParams(mean=-50_000.0, std=80_000.0, min_val=-500_000.0, max_val=0.0),
    "BASE_LOAD": NormalisationParams(mean=40_000.0, std=30_000.0, min_val=0.0, max_val=200_000.0),
}

# Backwards compatibility alias
DEFAULT_OUTPUT_PARAMS = SIDED_DEFAULT_OUTPUT_PARAMS


class RobustScaler:
    """
    Robust scaler using median and IQR, with fallback to mean/std.

    More resistant to outliers than standard z-score normalisation.
    """

    def __init__(self, use_robust: bool = True):
        self.use_robust = use_robust
        self.center_: float | None = None
        self.scale_: float | None = None
        self.min_: float | None = None
        self.max_: float | None = None

    def fit(self, data: np.ndarray) -> "RobustScaler":
        """Fit the scaler to the data."""
        data = data.flatten()

        if self.use_robust:
            self.center_ = float(np.median(data))
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            self.scale_ = float(iqr) if iqr > 1e-10 else float(np.std(data))
        else:
            self.center_ = float(np.mean(data))
            self.scale_ = float(np.std(data))

        if self.scale_ < 1e-10:
            self.scale_ = 1.0

        self.min_ = float(np.min(data))
        self.max_ = float(np.max(data))

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters."""
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (data - self.center_) / self.scale_

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalised data."""
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return data * self.scale_ + self.center_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def get_params(self) -> NormalisationParams:
        """Get normalisation parameters for export."""
        if self.center_ is None:
            raise ValueError("Scaler not fitted.")
        return NormalisationParams(
            mean=self.center_,
            std=self.scale_,  # type: ignore
            min_val=self.min_,  # type: ignore
            max_val=self.max_,  # type: ignore
        )


class ZScoreNormaliser:
    """Z-score normalisation matching Rust preprocessing."""

    def __init__(self, params: NormalisationParams | None = None):
        self.params = params

    def fit(self, data: np.ndarray) -> "ZScoreNormaliser":
        """Fit to data."""
        data = data.flatten()
        self.params = NormalisationParams(
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalise data."""
        if self.params is None:
            raise ValueError("Normaliser not fitted.")
        if abs(self.params.std) < 1e-10:
            return np.zeros_like(data)
        return (data - self.params.mean) / self.params.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Denormalise data."""
        if self.params is None:
            raise ValueError("Normaliser not fitted.")
        return data * self.params.std + self.params.mean

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(data).transform(data)

    def get_params(self) -> NormalisationParams:
        """Get normalisation parameters for export."""
        if self.params is None:
            raise ValueError("Normaliser not fitted.")
        return self.params


class WindowGenerator:
    """
    Generates sliding windows for sequence-to-point NILM.

    Aligns with Rust WindowBuilder in preprocessing.rs.
    """

    def __init__(
        self,
        window_size: int = 60,
        stride: int = 1,
        input_normaliser: ZScoreNormaliser | RobustScaler | None = None,
        output_normalisers: dict[str, ZScoreNormaliser] | None = None,
        appliance_names: list[str] | None = None,
    ):
        """
        Initialise window generator.

        Args:
            window_size: Number of timesteps per window (60 for 1 hour at 1-min)
            stride: Step size between consecutive windows
            input_normaliser: Normaliser for aggregate power input
            output_normalisers: Dict mapping appliance name to normaliser
            appliance_names: Ordered appliance names. If None, defaults to SIDED names.
        """
        self.window_size = window_size
        self.stride = stride
        self.input_normaliser = input_normaliser
        self.output_normalisers = output_normalisers
        self.appliance_names = appliance_names or [
            "BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"
        ]

    def generate_windows(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
        normalise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate windows from continuous time series.

        Args:
            aggregate: Shape (timesteps,) aggregate power
            appliances: Shape (timesteps, num_appliances) appliance powers

        Returns:
            Tuple of (inputs, targets):
            - inputs: Shape (n_windows, window_size)
            - targets: Shape (n_windows, num_appliances)
        """
        num_appliances = appliances.shape[1]
        n_samples = len(aggregate)
        n_windows = (n_samples - self.window_size) // self.stride + 1

        inputs = np.zeros((n_windows, self.window_size), dtype=np.float32)
        targets = np.zeros((n_windows, num_appliances), dtype=np.float32)

        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            midpoint = start + self.window_size // 2

            inputs[i] = aggregate[start:end]
            targets[i] = appliances[midpoint]

        if normalise:
            if self.input_normaliser is not None:
                inputs = self.input_normaliser.transform(inputs)

            if self.output_normalisers is not None:
                for j, name in enumerate(self.appliance_names):
                    if name in self.output_normalisers:
                        targets[:, j] = self.output_normalisers[name].transform(targets[:, j])

        return inputs, targets

    def fit_normalisers(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
        use_robust: bool = True,
    ) -> None:
        """Fit normalisers to training data."""
        if use_robust:
            self.input_normaliser = RobustScaler(use_robust=True)
        else:
            self.input_normaliser = ZScoreNormaliser()

        self.input_normaliser.fit(aggregate)

        num_appliances = appliances.shape[1]
        if len(self.appliance_names) != num_appliances:
            raise ValueError(
                f"appliance_names has {len(self.appliance_names)} entries "
                f"but data has {num_appliances} appliance columns"
            )

        self.output_normalisers = {}

        for j, name in enumerate(self.appliance_names):
            if use_robust:
                normaliser = RobustScaler(use_robust=True)
            else:
                normaliser = ZScoreNormaliser()
            normaliser.fit(appliances[:, j])
            self.output_normalisers[name] = normaliser

    def get_normalisation_metadata(self) -> dict:
        """Get normalisation parameters for ONNX export."""
        metadata = {
            "window_size": self.window_size,
            "input": None,
            "outputs": {},
        }

        if self.input_normaliser is not None:
            metadata["input"] = self.input_normaliser.get_params().to_dict()

        if self.output_normalisers is not None:
            for name, normaliser in self.output_normalisers.items():
                metadata["outputs"][name] = normaliser.get_params().to_dict()

        return metadata
