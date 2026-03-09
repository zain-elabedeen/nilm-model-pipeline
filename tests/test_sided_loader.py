"""Tests for SIDED Hugging Face loading logic."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_pipeline.data.sided_loader import SIDED_SITE_FILES, SidedLoader


def _write_sided_csv(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "Time": np.array([offset, offset + 60, offset + 120], dtype=np.int64),
        "Aggregate": np.array([10.0, 11.0, 12.0], dtype=np.float32),
        "EVSE": np.array([1.0, 1.1, 1.2], dtype=np.float32),
        "PV": np.array([2.0, 2.1, 2.2], dtype=np.float32),
        "CS": np.array([3.0, 3.1, 3.2], dtype=np.float32),
        "CHP": np.array([4.0, 4.1, 4.2], dtype=np.float32),
        "BA": np.array([5.0, 5.1, 5.2], dtype=np.float32),
    })
    df.to_csv(path, index=False)


def test_sided_loader_loads_all_expected_facilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for idx, (_, filename) in enumerate(SIDED_SITE_FILES):
        _write_sided_csv(tmp_path / filename, offset=idx * 1000)

    def fake_hf_hub_download(
        repo_id: str,
        repo_type: str,
        filename: str,
        cache_dir: str,
        force_download: bool,
        token: str | None,
    ) -> str:
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    loader = SidedLoader(cache_dir=tmp_path)

    assert loader.available_sites() == [site_id for site_id, _ in SIDED_SITE_FILES]

    alias_site = loader.load_site("F09")
    assert alias_site.facility_id == "Office_Tokyo"
    assert alias_site.aggregate.shape == (3,)
    assert alias_site.appliances.shape == (3, 5)


def test_sided_loader_raises_for_missing_expected_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for idx, (_, filename) in enumerate(SIDED_SITE_FILES[:-1]):
        _write_sided_csv(tmp_path / filename, offset=idx * 1000)

    def fake_hf_hub_download(
        repo_id: str,
        repo_type: str,
        filename: str,
        cache_dir: str,
        force_download: bool,
        token: str | None,
    ) -> str:
        path = tmp_path / filename
        if not path.exists():
            raise FileNotFoundError(filename)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    loader = SidedLoader(cache_dir=tmp_path)

    with pytest.raises(RuntimeError, match="Failed to load SIDED facility 'Office_Tokyo'"):
        loader.available_sites()
