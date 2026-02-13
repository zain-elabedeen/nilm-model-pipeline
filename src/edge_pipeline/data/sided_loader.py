"""SIDED dataset loader.

Loads the Synthetic Industrial Dataset for Energy Disaggregation from
Hugging Face and maps SIDED categories to Underscore appliance types.
"""

from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from edge_pipeline.data.base import FacilityData, NilmDatasetLoader

# Hugging Face dataset identifier
DATASET_ID = "CInterno/Synthetic_Industrial_Dataset_For_Energy_Disaggregation_SIDED"

# Underscore appliance types in standard order (matches Rust inference.rs)
APPLIANCE_ORDER = ["BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"]

# Mapping from SIDED CSV column names to Underscore categories.
# This is an imperfect mapping — diesel generators behave nothing like CHP,
# batteries aren't EV chargers — but gets the pipeline working end-to-end.
SIDED_TO_UNDERSCORE = {
    "EVSE": "BATTERY",
    "PV": "SOLAR",
    "CS": "COOLING",
    "CHP": "GENERATOR",
    "BA": "BASE_LOAD",
}

# SIDED CSV column names (used when reading raw data files)
SIDED_COLUMN_ORDER = ["EVSE", "PV", "CS", "CHP", "BA"]


class ApplianceType(Enum):
    """Underscore appliance types for industrial NILM."""

    BATTERY = 0    # Battery storage (charge/discharge, bidirectional)
    SOLAR = 1      # Solar PV generation
    COOLING = 2    # Cooling systems (HVAC, refrigeration)
    GENERATOR = 3  # Diesel/gas generator
    BASE_LOAD = 4  # Factory production machinery + misc


class SidedLoader(NilmDatasetLoader):
    """Loads SIDED dataset from Hugging Face."""

    def __init__(
        self,
        cache_dir: Path | str = "~/.cache/edge-pipeline/sided",
        force_download: bool = False,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.force_download = force_download
        self._facilities: dict[str, FacilityData] | None = None

    @property
    def appliance_names(self) -> list[str]:
        return APPLIANCE_ORDER

    def available_sites(self) -> list[str]:
        if self._facilities is None:
            self._load_hf_dataset()
        return sorted(self._facilities.keys())

    def _load_hf_dataset(self) -> None:
        """Load the full dataset from Hugging Face and split into facilities."""
        from datasets import load_dataset

        download_mode = "force_redownload" if self.force_download else None
        ds = load_dataset(
            DATASET_ID,
            split="train",
            cache_dir=str(self.cache_dir),
            download_mode=download_mode,
        )
        df = ds.to_pandas()

        # Split into individual facilities by detecting timestamp resets
        # (each facility is one year of data; timestamps jump backwards at boundaries)
        times = df["Time"].values
        diffs = np.diff(times)
        boundary_indices = np.where(diffs < 0)[0] + 1
        boundaries = [0] + boundary_indices.tolist() + [len(df)]

        self._facilities = {}
        for i in range(len(boundaries) - 1):
            facility_id = f"F{i + 1:02d}"
            chunk = df.iloc[boundaries[i] : boundaries[i + 1]]

            timestamps = pd.to_datetime(chunk["Time"], unit="s")
            aggregate = chunk["Aggregate"].values.astype(np.float32)
            appliances = np.column_stack([
                chunk[col].values.astype(np.float32) for col in SIDED_COLUMN_ORDER
            ])

            self._facilities[facility_id] = FacilityData(
                facility_id=facility_id,
                aggregate=aggregate,
                appliances=appliances,
                timestamps=pd.DatetimeIndex(timestamps),
            )

    def load_site(self, site_id: str) -> FacilityData:
        """Load data for a single facility, mapping SIDED columns to Underscore types."""
        if self._facilities is None:
            self._load_hf_dataset()
        if site_id not in self._facilities:
            available = sorted(self._facilities.keys())
            raise ValueError(
                f"Facility {site_id} not found. Available: {available}"
            )
        return self._facilities[site_id]

    def load_facility(self, facility_id: str) -> FacilityData:
        """Alias for load_site (backwards compatibility)."""
        return self.load_site(facility_id)


class SidedDataset:
    """PyTorch-compatible dataset wrapper for SIDED data."""

    def __init__(
        self,
        facilities: list[FacilityData],
        window_size: int = 60,
        stride: int = 1,
        target_resolution_minutes: int = 1,
    ):
        self.window_size = window_size
        self.stride = stride
        self.target_resolution_minutes = target_resolution_minutes

        # Combine all facilities into continuous arrays
        self._build_windows(facilities)

    def _resample_to_target(self, data: FacilityData) -> tuple[np.ndarray, np.ndarray]:
        """Resample data to target resolution if needed."""
        if data.resolution_minutes == self.target_resolution_minutes:
            return data.aggregate, data.appliances

        # Resample using linear interpolation
        factor = data.resolution_minutes // self.target_resolution_minutes
        n_original = len(data.aggregate)
        n_target = n_original * factor

        # Create interpolated arrays
        x_original = np.arange(n_original)
        x_target = np.linspace(0, n_original - 1, n_target)

        aggregate = np.interp(x_target, x_original, data.aggregate)
        appliances = np.column_stack([
            np.interp(x_target, x_original, data.appliances[:, i])
            for i in range(data.appliances.shape[1])
        ])

        return aggregate.astype(np.float32), appliances.astype(np.float32)

    def _build_windows(self, facilities: list[FacilityData]) -> None:
        """Build sliding windows from all facilities."""
        all_aggregates = []
        all_appliances = []

        for facility in facilities:
            aggregate, appliances = self._resample_to_target(facility)
            all_aggregates.append(aggregate)
            all_appliances.append(appliances)

        # Concatenate all data
        self.aggregate = np.concatenate(all_aggregates)
        self.appliances = np.concatenate(all_appliances)

        # Calculate valid window indices
        n_samples = len(self.aggregate)
        self.window_starts = list(range(0, n_samples - self.window_size + 1, self.stride))

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get a single window.

        Returns:
            Tuple of (input_window, target):
            - input_window: Shape (window_size,) aggregate power
            - target: Shape (5,) appliance powers at window midpoint
        """
        start = self.window_starts[idx]
        end = start + self.window_size

        # Input: full window of aggregate power
        input_window = self.aggregate[start:end]

        # Target: appliance values at the midpoint (sequence-to-point)
        midpoint = start + self.window_size // 2
        target = self.appliances[midpoint]

        return input_window, target

    def iter_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches of windows."""
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            inputs = np.stack([self[idx][0] for idx in batch_indices])
            targets = np.stack([self[idx][1] for idx in batch_indices])
            yield inputs, targets
