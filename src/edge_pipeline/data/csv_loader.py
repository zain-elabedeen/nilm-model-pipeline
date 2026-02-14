"""CSV-based dataset loader for custom NILM datasets.

Each CSV file represents one site. The CSV must contain an aggregate power
column and one or more appliance columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from edge_pipeline.data.base import FacilityData, NilmDatasetLoader


class CsvLoader(NilmDatasetLoader):
    """Loads NILM data from CSV files.

    Each CSV file in the data directory is treated as one site. The file
    stem (filename without extension) is used as the site ID.

    Args:
        data_dir: Directory containing CSV files.
        aggregate_column: Name of the aggregate power column.
        appliance_columns: Ordered list of appliance column names. If None,
            auto-detected from CSV headers (all columns except aggregate and
            timestamp).
        timestamp_column: Name of the timestamp column, or None if no
            timestamp column exists.
        resolution_minutes: Time resolution in minutes.
        file_pattern: Glob pattern for CSV files.
    """

    def __init__(
        self,
        data_dir: str | Path,
        aggregate_column: str = "aggregate",
        appliance_columns: list[str] | None = None,
        timestamp_column: str | None = "timestamp",
        resolution_minutes: int = 1,
        file_pattern: str = "*.csv",
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.aggregate_column = aggregate_column
        self._appliance_columns = appliance_columns
        self.timestamp_column = timestamp_column
        self.resolution_minutes = resolution_minutes
        self.file_pattern = file_pattern
        self._detected_appliance_columns: list[str] | None = None

    @property
    def appliance_names(self) -> list[str]:
        if self._appliance_columns is not None:
            return self._appliance_columns

        # Auto-detect from first available CSV
        if self._detected_appliance_columns is None:
            files = self._csv_files()
            if not files:
                raise ValueError(
                    f"No CSV files matching '{self.file_pattern}' "
                    f"found in {self.data_dir}"
                )
            self._detected_appliance_columns = self._detect_columns(files[0])

        return self._detected_appliance_columns

    def _csv_files(self) -> list[Path]:
        """Return sorted list of CSV files in data_dir."""
        return sorted(self.data_dir.glob(self.file_pattern))

    def _detect_columns(self, csv_path: Path) -> list[str]:
        """Auto-detect appliance columns from a CSV file."""
        df = pd.read_csv(csv_path, nrows=0)
        exclude = {self.aggregate_column}
        if self.timestamp_column is not None:
            exclude.add(self.timestamp_column)
        return [col for col in df.columns if col not in exclude]

    def available_sites(self) -> list[str]:
        return [f.stem for f in self._csv_files()]

    def load_site(self, site_id: str) -> FacilityData:
        csv_path = self.data_dir / f"{site_id}.csv"
        if not csv_path.exists():
            available = self.available_sites()
            raise ValueError(
                f"Site {site_id} not found (no file {csv_path}). "
                f"Available: {available}"
            )

        df = pd.read_csv(csv_path)

        if self.aggregate_column not in df.columns:
            raise ValueError(
                f"Aggregate column '{self.aggregate_column}' "
                f"not found in {csv_path}. Columns: {list(df.columns)}"
            )

        aggregate = df[self.aggregate_column].values.astype(np.float32)

        columns = self.appliance_names
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Appliance columns {missing} not found in {csv_path}. "
                f"Columns: {list(df.columns)}"
            )

        appliances = np.column_stack(
            [df[col].values.astype(np.float32) for col in columns]
        )

        # Drop rows containing NaN or inf in either aggregate or appliances
        valid = np.isfinite(aggregate) & np.all(np.isfinite(appliances), axis=1)
        if not np.all(valid):
            aggregate = aggregate[valid]
            appliances = appliances[valid]
            df = df[valid]

        if self.timestamp_column is not None and self.timestamp_column in df.columns:
            timestamps = pd.DatetimeIndex(pd.to_datetime(df[self.timestamp_column]))
        else:
            timestamps = pd.date_range(
                "2024-01-01",
                periods=len(aggregate),
                freq=f"{self.resolution_minutes}min",
            )

        return FacilityData(
            facility_id=site_id,
            aggregate=aggregate,
            appliances=appliances,
            timestamps=timestamps,
            resolution_minutes=self.resolution_minutes,
        )
