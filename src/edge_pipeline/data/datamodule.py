"""PyTorch Lightning DataModule for NILM training."""

import warnings
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset

from edge_pipeline.data.augmentation import AMDATransform
from edge_pipeline.data.base import FacilityData, NilmDatasetLoader
from edge_pipeline.data.preprocessing import WindowGenerator


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
            targets: Shape (n_samples, num_appliances)
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
        loader: NilmDatasetLoader | None = None,
        data_dir: str | Path = "~/.cache/edge-pipeline/sided",
        window_size: int = 60,
        stride: int = 1,
        batch_size: int = 256,
        num_workers: int = 4,
        train_sites: list[str] | None = None,
        val_sites: list[str] | None = None,
        test_sites: list[str] | None = None,
        train_facilities: list[str] | None = None,
        val_facilities: list[str] | None = None,
        test_facilities: list[str] | None = None,
        use_amda: bool = True,
        amda_scale: float = 2.5,
        use_robust_scaling: bool = True,
        auto_split_shuffle: bool = True,
        auto_split_seed: int = 42,
    ):
        """
        Initialise DataModule.

        Args:
            loader: Dataset loader. If None, creates a SidedLoader using data_dir.
            data_dir: Directory to cache SIDED data (used when loader is None)
            window_size: Samples per window (60 = 1 hour at 1-min resolution)
            stride: Step between windows
            batch_size: Training batch size
            num_workers: DataLoader workers
            train_sites: Site IDs for training (alias: train_facilities)
            val_sites: Site IDs for validation (alias: val_facilities)
            test_sites: Site IDs for testing (alias: test_facilities)
            train_facilities: Legacy alias for train_sites
            val_facilities: Legacy alias for val_sites
            test_facilities: Legacy alias for test_sites
            use_amda: Whether to apply AMDA augmentation
            amda_scale: AMDA base scaling factor
            use_robust_scaling: Use robust (median/IQR) vs standard scaling
            auto_split_shuffle: Shuffle sites before auto-splitting
            auto_split_seed: Random seed for deterministic auto-splitting
        """
        super().__init__()
        self.save_hyperparameters(ignore=["loader"])

        # Create loader if not provided
        if loader is not None:
            self._loader = loader
        else:
            from edge_pipeline.data.sided_loader import SidedLoader
            self._loader = SidedLoader(cache_dir=data_dir)

        self.data_dir = Path(data_dir).expanduser()
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_amda = use_amda
        self.amda_scale = amda_scale
        self.use_robust_scaling = use_robust_scaling
        self.auto_split_shuffle = auto_split_shuffle
        self.auto_split_seed = auto_split_seed
        self._used_auto_split = False
        self._did_log_setup_summary = False

        # Resolve site splits (new names take precedence over legacy aliases)
        self.train_sites = train_sites or train_facilities
        self.val_sites = val_sites or val_facilities
        self.test_sites = test_sites or test_facilities

        # Auto-split if no explicit splits given
        if self.train_sites is None and self.val_sites is None and self.test_sites is None:
            self._used_auto_split = True
            self._auto_split()

        self.window_generator: WindowGenerator | None = None
        self.train_dataset: NilmTorchDataset | None = None
        self.val_dataset: NilmTorchDataset | None = None
        self.test_dataset: NilmTorchDataset | None = None

    def _auto_split(self) -> None:
        """Split available sites 70/15/15 when no explicit splits are given."""
        sites = sorted(self._loader.available_sites())
        if not sites:
            raise ValueError("No sites available from dataset loader.")

        if self.auto_split_shuffle:
            rng = np.random.default_rng(self.auto_split_seed)
            sites = list(rng.permutation(sites))

        n = len(sites)
        if n < 3:
            self.train_sites = [sites[0]]
            self.val_sites = [sites[-1]]
            self.test_sites = [sites[-1]]
            return

        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))

        # Guarantee at least one test site.
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        if n_train + n_val >= n:
            n_train = max(1, n - n_val - 1)

        self.train_sites = sorted(sites[:n_train])
        self.val_sites = sorted(sites[n_train:n_train + n_val])
        self.test_sites = sorted(sites[n_train + n_val:])

        # Keep every split non-empty for tiny datasets.
        if not self.val_sites:
            self.val_sites = [self.train_sites[-1]]
        # Ensure test has at least one site
        if not self.test_sites:
            self.test_sites = [self.val_sites[-1] if self.val_sites else self.train_sites[-1]]

    def _resolve_split_sites(self) -> None:
        """Filter missing configured sites and ensure split fallbacks are valid."""
        available = set(self._loader.available_sites())
        missing: list[str] = []

        def _filter(sites: list[str] | None) -> list[str]:
            if not sites:
                return []
            kept = [sid for sid in sites if sid in available]
            missing.extend([sid for sid in sites if sid not in available])
            return kept

        train_sites = _filter(self.train_sites)
        val_sites = _filter(self.val_sites)
        test_sites = _filter(self.test_sites)

        if missing:
            missing_unique = sorted(set(missing))
            warnings.warn(
                f"Configured site IDs not found and will be skipped: {missing_unique}",
                UserWarning,
                stacklevel=2,
            )

        available_sorted = sorted(available)
        if not train_sites:
            if available_sorted:
                train_sites = [available_sorted[0]]
            else:
                raise ValueError("No valid sites available for training.")
        if not val_sites:
            val_sites = [train_sites[-1]]
        if not test_sites:
            test_sites = [val_sites[-1]]

        self.train_sites = train_sites
        self.val_sites = val_sites
        self.test_sites = test_sites

    @property
    def appliance_names(self) -> list[str]:
        return self._loader.appliance_names

    @property
    def num_appliances(self) -> int:
        return len(self.appliance_names)

    def prepare_data(self) -> None:
        """Download data if needed."""
        all_sites = (self.train_sites or []) + (self.val_sites or []) + (self.test_sites or [])
        unique_sites = sorted(set(all_sites))
        if not unique_sites:
            return

        start = perf_counter()
        rank_zero_info(
            f"[NilmDataModule] prepare_data: preloading {len(unique_sites)} site(s): {unique_sites}"
        )
        for sid in unique_sites:
            self._loader.load_site(sid)
        elapsed = perf_counter() - start
        rank_zero_info(f"[NilmDataModule] prepare_data: done in {elapsed:.1f}s")

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training/validation/testing."""
        setup_start = perf_counter()
        rank_zero_info(f"[NilmDataModule] setup(stage={stage}) starting")
        self._resolve_split_sites()

        # Load site data
        load_start = perf_counter()
        train_data = [self._loader.load_site(sid) for sid in self.train_sites]
        val_data = [self._loader.load_site(sid) for sid in self.val_sites]
        test_data = [self._loader.load_site(sid) for sid in self.test_sites]
        rank_zero_info(
            f"[NilmDataModule] setup: loaded split site data in {perf_counter() - load_start:.1f}s"
        )

        # Combine training data for fitting normalisers
        train_agg = np.concatenate([f.aggregate for f in train_data])
        train_app = np.concatenate([f.appliances for f in train_data])

        # Fit normalisers on training data
        norm_start = perf_counter()
        self.window_generator = WindowGenerator(
            window_size=self.window_size,
            stride=self.stride,
            appliance_names=self.appliance_names,
        )
        self.window_generator.fit_normalisers(
            train_agg, train_app, use_robust=self.use_robust_scaling
        )
        rank_zero_info(
            f"[NilmDataModule] setup: fit normalisers in {perf_counter() - norm_start:.1f}s"
        )

        # Generate windows
        window_start = perf_counter()
        train_inputs, train_targets = self._generate_all_windows(train_data)
        val_inputs, val_targets = self._generate_all_windows(val_data)
        test_inputs, test_targets = self._generate_all_windows(test_data)
        rank_zero_info(
            f"[NilmDataModule] setup: generated windows in {perf_counter() - window_start:.1f}s"
        )

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

        if not self._did_log_setup_summary:
            self._log_setup_summary(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                train_inputs=train_inputs,
                val_inputs=val_inputs,
                test_inputs=test_inputs,
            )
            self._did_log_setup_summary = True

        rank_zero_info(
            f"[NilmDataModule] setup(stage={stage}) completed in {perf_counter() - setup_start:.1f}s"
        )

    def _log_setup_summary(
        self,
        train_data: list[FacilityData],
        val_data: list[FacilityData],
        test_data: list[FacilityData],
        train_inputs: np.ndarray,
        val_inputs: np.ndarray,
        test_inputs: np.ndarray,
    ) -> None:
        """Log split and window summary once per run."""
        split_mode = "explicit"
        if self._used_auto_split:
            split_mode = "auto-shuffled" if self.auto_split_shuffle else "auto-contiguous"

        rank_zero_info(
            "[NilmDataModule] split="
            f"{split_mode}, window_size={self.window_size}, stride={self.stride}, "
            f"robust_scaling={self.use_robust_scaling}, use_amda={self.use_amda}, "
            f"amda_scale={self.amda_scale}, auto_split_seed={self.auto_split_seed}"
        )

        train_sites = self.train_sites or []
        val_sites = self.val_sites or []
        test_sites = self.test_sites or []

        rank_zero_info(f"[NilmDataModule] train_sites ({len(train_sites)}): {train_sites}")
        rank_zero_info(f"[NilmDataModule] val_sites ({len(val_sites)}): {val_sites}")
        rank_zero_info(f"[NilmDataModule] test_sites ({len(test_sites)}): {test_sites}")

        train_steps = sum(len(f.aggregate) for f in train_data)
        val_steps = sum(len(f.aggregate) for f in val_data)
        test_steps = sum(len(f.aggregate) for f in test_data)
        rank_zero_info(
            "[NilmDataModule] timesteps "
            f"train={train_steps:,}, val={val_steps:,}, test={test_steps:,}"
        )
        rank_zero_info(
            "[NilmDataModule] windows "
            f"train={len(train_inputs):,}, val={len(val_inputs):,}, test={len(test_inputs):,}"
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
