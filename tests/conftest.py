"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from edge_pipeline.data.base import FacilityData


def generate_synthetic_facility(
    facility_id: str = "TEST01",
    n_days: int = 7,
    resolution_minutes: int = 1,
    num_appliances: int = 5,
) -> FacilityData:
    """Generate synthetic facility data for testing.

    Args:
        facility_id: ID for the facility.
        n_days: Number of days of data.
        resolution_minutes: Time resolution.
        num_appliances: Number of appliance columns.

    Returns:
        FacilityData with random but plausible patterns.
    """
    n_samples = n_days * 24 * 60 // resolution_minutes
    timestamps = pd.date_range(
        "2024-01-01", periods=n_samples, freq=f"{resolution_minutes}min"
    )
    t = np.arange(n_samples)
    hour_of_day = (t % (24 * 60 // resolution_minutes)) / (60 // resolution_minutes)

    rng = np.random.default_rng(42)

    columns = []
    for i in range(num_appliances):
        phase = rng.uniform(0, 24)
        amplitude = rng.uniform(5000, 50000)
        base = rng.uniform(-10000, 20000)
        signal = amplitude * np.exp(-((hour_of_day - phase) ** 2) / 10) + base
        signal += rng.normal(0, 1000, n_samples)
        columns.append(signal.astype(np.float32))

    appliances = np.column_stack(columns)
    aggregate = appliances.sum(axis=1).astype(np.float32)

    return FacilityData(
        facility_id=facility_id,
        aggregate=aggregate,
        appliances=appliances,
        timestamps=timestamps,
        resolution_minutes=resolution_minutes,
    )


@pytest.fixture
def synthetic_facility() -> FacilityData:
    """A single synthetic facility with 5 appliances."""
    return generate_synthetic_facility()


@pytest.fixture
def synthetic_facility_3app() -> FacilityData:
    """A single synthetic facility with 3 appliances."""
    return generate_synthetic_facility(num_appliances=3)
