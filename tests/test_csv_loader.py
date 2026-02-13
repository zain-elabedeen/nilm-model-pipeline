"""Tests for CSV dataset loader."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edge_pipeline.data.csv_loader import CsvLoader


def _write_csv(
    path: Path,
    n_rows: int = 200,
    appliance_cols: list[str] | None = None,
    aggregate_col: str = "aggregate",
    timestamp_col: str = "timestamp",
) -> None:
    """Write a sample CSV file for testing."""
    if appliance_cols is None:
        appliance_cols = ["heater", "fridge", "lighting"]

    rng = np.random.default_rng(42)
    data = {aggregate_col: rng.normal(5000, 1000, n_rows).astype(np.float32)}
    for col in appliance_cols:
        data[col] = rng.normal(1000, 500, n_rows).astype(np.float32)
    data[timestamp_col] = pd.date_range("2024-01-01", periods=n_rows, freq="1min")

    pd.DataFrame(data).to_csv(path, index=False)


class TestCsvLoader:
    """Tests for CsvLoader."""

    def test_available_sites(self):
        """Test site listing from CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir) / "site_a.csv")
            _write_csv(Path(tmpdir) / "site_b.csv")

            loader = CsvLoader(data_dir=tmpdir)
            sites = loader.available_sites()

            assert sites == ["site_a", "site_b"]

    def test_load_site(self):
        """Test loading a single site."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir) / "factory1.csv")

            loader = CsvLoader(data_dir=tmpdir)
            data = loader.load_site("factory1")

            assert data.facility_id == "factory1"
            assert data.aggregate.shape == (200,)
            assert data.appliances.shape == (200, 3)

    def test_appliance_names_auto_detected(self):
        """Test auto-detection of appliance columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(
                Path(tmpdir) / "site.csv",
                appliance_cols=["solar", "battery"],
            )

            loader = CsvLoader(data_dir=tmpdir)

            assert loader.appliance_names == ["solar", "battery"]

    def test_appliance_names_explicit(self):
        """Test explicit appliance columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(
                Path(tmpdir) / "site.csv",
                appliance_cols=["heater", "fridge", "lighting"],
            )

            loader = CsvLoader(
                data_dir=tmpdir,
                appliance_columns=["heater", "fridge"],
            )

            assert loader.appliance_names == ["heater", "fridge"]
            data = loader.load_site("site")
            assert data.appliances.shape == (200, 2)

    def test_missing_site_raises(self):
        """Test error for non-existent site."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir) / "site.csv")

            loader = CsvLoader(data_dir=tmpdir)

            with pytest.raises(ValueError, match="not found"):
                loader.load_site("nonexistent")

    def test_missing_column_raises(self):
        """Test error for missing appliance column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir) / "site.csv", appliance_cols=["heater"])

            loader = CsvLoader(
                data_dir=tmpdir,
                appliance_columns=["heater", "nonexistent"],
            )

            with pytest.raises(ValueError, match="not found"):
                loader.load_site("site")

    def test_load_all(self):
        """Test loading all sites."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir) / "a.csv")
            _write_csv(Path(tmpdir) / "b.csv")

            loader = CsvLoader(data_dir=tmpdir)
            all_data = loader.load_all()

            assert len(all_data) == 2

    def test_no_timestamp_column(self):
        """Test loading CSV without timestamp column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "site.csv"
            rng = np.random.default_rng(42)
            data = {
                "aggregate": rng.normal(5000, 1000, 100),
                "heater": rng.normal(1000, 500, 100),
            }
            pd.DataFrame(data).to_csv(path, index=False)

            loader = CsvLoader(data_dir=tmpdir, timestamp_column=None)
            site = loader.load_site("site")

            assert len(site.timestamps) == 100

    def test_empty_directory(self):
        """Test error when no CSV files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = CsvLoader(data_dir=tmpdir)

            assert loader.available_sites() == []

            with pytest.raises(ValueError, match="No CSV files"):
                _ = loader.appliance_names
