"""SIDED dataset loader.

Loads the Synthetic Industrial Dataset for Energy Disaggregation from
Hugging Face and maps SIDED categories to Underscore appliance types.
"""

import os
from enum import Enum
from pathlib import Path
from time import perf_counter
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

# Stable site ordering derived from the dataset repository layout.
SIDED_SITE_FILES = [
    ("Dealer_LA", "SIDED/Dealer/Dealer_LA.csv"),
    ("Dealer_Offenbach", "SIDED/Dealer/Dealer_Offenbach.csv"),
    ("Dealer_Tokyo", "SIDED/Dealer/Dealer_Tokyo.csv"),
    ("Logistic_LA", "SIDED/Logistic/Logistic_LA.csv"),
    ("Logistic_Offenbach", "SIDED/Logistic/Logistic_Offenbach.csv"),
    ("Logistic_Tokyo", "SIDED/Logistic/Logistic_Tokyo.csv"),
    ("Office_LA", "SIDED/Office/Office_LA.csv"),
    ("Office_Offenbach", "SIDED/Office/Office_Offenbach.csv"),
    ("Office_Tokyo", "SIDED/Office/Office_Tokyo.csv"),
]

SIDED_SITE_IDS = [site_id for site_id, _ in SIDED_SITE_FILES]
LEGACY_SITE_ALIASES = {
    f"F{i:02d}": site_id for i, site_id in enumerate(SIDED_SITE_IDS, start=1)
}
EXPECTED_COLUMNS = {"Time", "Aggregate", *SIDED_COLUMN_ORDER}


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
        return SIDED_SITE_IDS.copy()

    def _resolve_site_id(self, site_id: str) -> str:
        """Resolve legacy aliases like F01 to canonical SIDED site IDs."""
        return LEGACY_SITE_ALIASES.get(site_id, site_id)

    def _load_site_file(self, site_id: str, filename: str) -> FacilityData:
        """Download and parse a single SIDED CSV file."""
        from huggingface_hub import hf_hub_download

        token = os.getenv("HF_TOKEN")
        try:
            csv_path = hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=filename,
                cache_dir=str(self.cache_dir),
                force_download=self.force_download,
                token=token,
            )
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SIDED facility '{site_id}' from '{filename}'"
            ) from exc

        missing_columns = sorted(EXPECTED_COLUMNS.difference(df.columns))
        if missing_columns:
            raise ValueError(
                f"SIDED file '{filename}' is missing expected columns: {missing_columns}"
            )

        df = df.sort_values("Time").reset_index(drop=True)
        timestamps = pd.to_datetime(df["Time"], unit="s")
        aggregate = df["Aggregate"].values.astype(np.float32)
        appliances = np.column_stack([
            df[col].values.astype(np.float32) for col in SIDED_COLUMN_ORDER
        ])

        valid = np.isfinite(aggregate) & np.all(np.isfinite(appliances), axis=1)
        if not np.all(valid):
            aggregate = aggregate[valid]
            appliances = appliances[valid]
            timestamps = timestamps[valid]

        return FacilityData(
            facility_id=site_id,
            aggregate=aggregate,
            appliances=appliances,
            timestamps=pd.DatetimeIndex(timestamps),
        )

    def _load_hf_dataset(self) -> None:
        """Load the SIDED dataset from Hugging Face, one file per facility."""
        start = perf_counter()
        print(f"[SidedLoader] Loading {len(SIDED_SITE_FILES)} SIDED facility files from Hugging Face...")
        facilities: dict[str, FacilityData] = {}
        for site_id, filename in SIDED_SITE_FILES:
            facilities[site_id] = self._load_site_file(site_id, filename)

        if len(facilities) != len(SIDED_SITE_FILES):
            raise RuntimeError(
                f"Expected {len(SIDED_SITE_FILES)} SIDED facilities, loaded {len(facilities)}"
            )

        self._facilities = facilities
        print(
            f"[SidedLoader] Prepared {len(self._facilities)} facilities in "
            f"{perf_counter() - start:.1f}s"
        )

    def load_site(self, site_id: str) -> FacilityData:
        """Load data for a single facility, mapping SIDED columns to Underscore types."""
        if self._facilities is None:
            self._load_hf_dataset()
        resolved_site_id = self._resolve_site_id(site_id)
        if resolved_site_id not in self._facilities:
            available = sorted(self._facilities.keys())
            raise ValueError(
                f"Facility {site_id} not found. Available: {available}"
            )
        return self._facilities[resolved_site_id]

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
