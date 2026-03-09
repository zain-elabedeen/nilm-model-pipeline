"""Tests for NILM DataModule split handling."""

import numpy as np
import pandas as pd
import pytest

from edge_pipeline.data.base import FacilityData, NilmDatasetLoader
from edge_pipeline.data.datamodule import NilmDataModule


class DummyLoader(NilmDatasetLoader):
    """Minimal in-memory loader for datamodule tests."""

    def __init__(self, site_ids: list[str]):
        self._appliance_names = ["BATTERY", "SOLAR", "COOLING", "GENERATOR", "BASE_LOAD"]
        self._sites = {sid: self._make_facility(sid) for sid in site_ids}

    @property
    def appliance_names(self) -> list[str]:
        return self._appliance_names

    def load_site(self, site_id: str) -> FacilityData:
        if site_id not in self._sites:
            raise ValueError(f"missing site {site_id}")
        return self._sites[site_id]

    def available_sites(self) -> list[str]:
        return sorted(self._sites.keys())

    def _make_facility(self, site_id: str) -> FacilityData:
        n_samples = 300
        timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
        aggregate = np.linspace(1000, 2000, n_samples, dtype=np.float32)
        appliances = np.column_stack([
            aggregate * 0.2,
            aggregate * 0.2,
            aggregate * 0.2,
            aggregate * 0.2,
            aggregate * 0.2,
        ]).astype(np.float32)
        return FacilityData(
            facility_id=site_id,
            aggregate=aggregate,
            appliances=appliances,
            timestamps=timestamps,
        )


def test_setup_recovers_from_missing_configured_sites() -> None:
    loader = DummyLoader(["F01", "F02"])
    dm = NilmDataModule(
        loader=loader,
        window_size=60,
        stride=10,
        batch_size=8,
        num_workers=0,
        train_sites=["F03"],
        val_sites=["F02"],
        test_sites=["F99"],
        use_amda=False,
    )

    with pytest.warns(UserWarning, match="Configured site IDs not found"):
        dm.setup()

    assert dm.train_sites == ["F01"]
    assert dm.val_sites == ["F02"]
    assert dm.test_sites == ["F02"]
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert len(dm.test_dataset) > 0


def test_setup_raises_when_strict_site_ids_enabled() -> None:
    loader = DummyLoader(["F01", "F02"])
    dm = NilmDataModule(
        loader=loader,
        window_size=60,
        stride=10,
        batch_size=8,
        num_workers=0,
        train_sites=["F03"],
        val_sites=["F02"],
        test_sites=["F99"],
        use_amda=False,
        strict_site_ids=True,
    )

    with pytest.raises(ValueError, match="Configured site IDs not found"):
        dm.setup()


def test_auto_split_with_single_site() -> None:
    loader = DummyLoader(["F01"])
    dm = NilmDataModule(
        loader=loader,
        window_size=60,
        stride=10,
        batch_size=8,
        num_workers=0,
        use_amda=False,
    )
    dm.setup()

    assert dm.train_sites == ["F01"]
    assert dm.val_sites == ["F01"]
    assert dm.test_sites == ["F01"]
    assert len(dm.train_dataset) > 0


def test_auto_split_shuffles_deterministically() -> None:
    loader = DummyLoader([f"F{i:02d}" for i in range(1, 10)])

    dm_a = NilmDataModule(
        loader=loader,
        window_size=60,
        stride=10,
        batch_size=8,
        num_workers=0,
        use_amda=False,
        auto_split_shuffle=True,
        auto_split_seed=123,
    )
    dm_b = NilmDataModule(
        loader=loader,
        window_size=60,
        stride=10,
        batch_size=8,
        num_workers=0,
        use_amda=False,
        auto_split_shuffle=True,
        auto_split_seed=123,
    )

    assert dm_a.train_sites == dm_b.train_sites
    assert dm_a.val_sites == dm_b.val_sites
    assert dm_a.test_sites == dm_b.test_sites
    assert dm_a.train_sites != ["F01", "F02", "F03", "F04", "F05", "F06"]


def test_auto_split_can_disable_shuffle() -> None:
    loader = DummyLoader([f"F{i:02d}" for i in range(1, 10)])

    dm = NilmDataModule(
        loader=loader,
        window_size=60,
        stride=10,
        batch_size=8,
        num_workers=0,
        use_amda=False,
        auto_split_shuffle=False,
    )

    assert dm.train_sites == ["F01", "F02", "F03", "F04", "F05", "F06"]
    assert dm.val_sites == ["F07"]
    assert dm.test_sites == ["F08", "F09"]
