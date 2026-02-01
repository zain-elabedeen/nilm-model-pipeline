"""PyTorch Lightning DataModule for NILM training."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from nilm_research.data.augmentation import AMDATransform, ComposeTransforms
from nilm_research.data.preprocessing import WindowGenerator
from nilm_research.data.sided_loader import FacilityData, SidedLoader


class NilmTorchDataset(Dataset):
    """PyTorch Dataset wrapper for NILM data."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        transform: Callable | None = None,
    ):
        """
        Initialise dataset.

        Args:
            inputs: Shape (n_samples, window_size)
            targets: Shape (n_samples, 5)
            transform: Optional augmentation transform
        """
        self.inputs = torch.from_numpy(inputs).float()
        self.targets = torch.from_numpy(targets).float()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.inputs[idx]
        y = self.targets[idx]

        if self.transform is not None:
            x_np = x.numpy()
            y_np = y.numpy()
            x_np, y_np = self.transform(x_np, y_np)
            x = torch.from_numpy(x_np).float()
            y = torch.from_numpy(y_np).float()

        return x, y


class NilmDataModule(pl.LightningDataModule):
    """Lightning DataModule for NILM training."""

    def __init__(
        self,
        data_dir: str | Path = "~/.cache/nilm-research/sided",
        window_size: int = 60,
        stride: int = 1,
        batch_size: int = 256,
        num_workers: int = 4,
        train_facilities: list[str] | None = None,
        val_facilities: list[str] | None = None,
        test_facilities: list[str] | None = None,
        use_amda: bool = True,
        amda_scale: float = 2.5,
        use_robust_scaling: bool = True,
    ):
        """
        Initialise DataModule.

        Args:
            data_dir: Directory to cache SIDED data
            window_size: Samples per window (60 = 1 hour at 1-min resolution)
            stride: Step between windows
            batch_size: Training batch size
            num_workers: DataLoader workers
            train_facilities: Facility IDs for training (default: F01-F07)
            val_facilities: Facility IDs for validation (default: F08)
            test_facilities: Facility IDs for testing (default: F09-F10)
            use_amda: Whether to apply AMDA augmentation
            amda_scale: AMDA base scaling factor
            use_robust_scaling: Use robust (median/IQR) vs standard scaling
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir).expanduser()
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amda = use_amda
        self.amda_scale = amda_scale
        self.use_robust_scaling = use_robust_scaling

        # Default facility splits
        self.train_facilities = train_facilities or [f"F{i:02d}" for i in range(1, 8)]
        self.val_facilities = val_facilities or ["F08"]
        self.test_facilities = test_facilities or ["F09", "F10"]

        self.window_generator: WindowGenerator | None = None
        self.train_dataset: NilmTorchDataset | None = None
        self.val_dataset: NilmTorchDataset | None = None
        self.test_dataset: NilmTorchDataset | None = None

    def prepare_data(self) -> None:
        """Download data if needed."""
        loader = SidedLoader(cache_dir=self.data_dir)
        all_facilities = self.train_facilities + self.val_facilities + self.test_facilities
        for fid in all_facilities:
            try:
                loader.load_facility(fid)
            except Exception:
                pass  # Will handle in setup

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training/validation/testing."""
        loader = SidedLoader(cache_dir=self.data_dir)

        # Load facility data
        train_data = self._load_facilities(loader, self.train_facilities)
        val_data = self._load_facilities(loader, self.val_facilities)
        test_data = self._load_facilities(loader, self.test_facilities)

        # Combine training data for fitting normalisers
        train_agg = np.concatenate([f.aggregate for f in train_data])
        train_app = np.concatenate([f.appliances for f in train_data])

        # Fit normalisers on training data
        self.window_generator = WindowGenerator(
            window_size=self.window_size,
            stride=self.stride,
        )
        self.window_generator.fit_normalisers(
            train_agg, train_app, use_robust=self.use_robust_scaling
        )

        # Generate windows
        train_inputs, train_targets = self._generate_all_windows(train_data)
        val_inputs, val_targets = self._generate_all_windows(val_data)
        test_inputs, test_targets = self._generate_all_windows(test_data)

        # Create augmentation transform for training
        train_transform = None
        if self.use_amda:
            train_transform = AMDATransform(
                base_scale=self.amda_scale,
                probability=0.5,
            )

        # Create datasets
        self.train_dataset = NilmTorchDataset(
            train_inputs, train_targets, transform=train_transform
        )
        self.val_dataset = NilmTorchDataset(val_inputs, val_targets)
        self.test_dataset = NilmTorchDataset(test_inputs, test_targets)

    def _load_facilities(
        self, loader: SidedLoader, facility_ids: list[str]
    ) -> list[FacilityData]:
        """Load facilities, generating synthetic data if download fails."""
        facilities = []
        for fid in facility_ids:
            try:
                facilities.append(loader.load_facility(fid))
            except Exception:
                # Generate synthetic data for testing/development
                facilities.append(self._generate_synthetic_facility(fid))
        return facilities

    def _generate_synthetic_facility(self, facility_id: str) -> FacilityData:
        """Generate synthetic facility data for development."""
        import pandas as pd

        # 7 days at 1-minute resolution
        n_samples = 7 * 24 * 60
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1min")

        # Generate realistic-ish patterns
        t = np.arange(n_samples)
        hour_of_day = (t % (24 * 60)) / 60

        # Base load with daily pattern
        ba = 3000 + 1000 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 200, n_samples)

        # Cooling with daytime peak
        cs = np.maximum(
            0,
            8000 * np.exp(-((hour_of_day - 14) ** 2) / 20) + np.random.normal(0, 500, n_samples),
        )

        # PV with solar pattern (negative = generation)
        pv = -np.maximum(
            0,
            20000 * np.exp(-((hour_of_day - 12) ** 2) / 8) + np.random.normal(0, 1000, n_samples),
        )

        # EVSE with random charging events
        evse = np.zeros(n_samples)
        for _ in range(20):  # 20 charging events per week
            start = np.random.randint(0, n_samples - 120)
            duration = np.random.randint(30, 120)
            power = np.random.uniform(3000, 22000)
            evse[start : start + duration] = power

        # CHP (mostly off, occasional generation)
        chp = np.zeros(n_samples)
        if np.random.random() > 0.5:
            chp = -np.random.uniform(2000, 8000) * np.ones(n_samples)

        appliances = np.column_stack([evse, pv, cs, chp, ba]).astype(np.float32)
        aggregate = appliances.sum(axis=1).astype(np.float32)

        return FacilityData(
            facility_id=facility_id,
            aggregate=aggregate,
            appliances=appliances,
            timestamps=timestamps,
            resolution_minutes=1,
        )

    def _generate_all_windows(
        self, facilities: list[FacilityData]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate windows from multiple facilities."""
        all_inputs = []
        all_targets = []

        for facility in facilities:
            inputs, targets = self.window_generator.generate_windows(
                facility.aggregate,
                facility.appliances,
                normalise=True,
            )
            all_inputs.append(inputs)
            all_targets.append(targets)

        return np.concatenate(all_inputs), np.concatenate(all_targets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_normalisation_metadata(self) -> dict:
        """Get normalisation parameters for ONNX export."""
        if self.window_generator is None:
            raise ValueError("DataModule not set up. Call setup() first.")
        return self.window_generator.get_normalisation_metadata()
