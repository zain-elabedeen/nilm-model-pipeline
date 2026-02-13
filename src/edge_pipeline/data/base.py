"""Abstract base classes for NILM dataset loaders."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FacilityData:
    """Data for a single facility/site.

    Attributes:
        facility_id: Unique identifier for this site.
        aggregate: Total power consumption. Shape: (timesteps,)
        appliances: Per-appliance power consumption. Shape: (timesteps, num_appliances)
        timestamps: Datetime index for each timestep.
        resolution_minutes: Time resolution in minutes.
    """

    facility_id: str
    aggregate: np.ndarray
    appliances: np.ndarray
    timestamps: pd.DatetimeIndex
    resolution_minutes: int = 1

    def __len__(self) -> int:
        return len(self.aggregate)

    @property
    def num_days(self) -> int:
        return len(self) // (24 * 60 // self.resolution_minutes)


class NilmDatasetLoader(ABC):
    """Abstract base class for NILM dataset loaders.

    Implementors must provide appliance names, site loading, and site listing.
    The loader is the single source of truth for appliance names — everything
    downstream derives from it.
    """

    @property
    @abstractmethod
    def appliance_names(self) -> list[str]:
        """Ordered list of appliance/category names."""
        ...

    @abstractmethod
    def load_site(self, site_id: str) -> FacilityData:
        """Load data for a single site.

        Args:
            site_id: Identifier for the site to load.

        Returns:
            FacilityData for the requested site.

        Raises:
            ValueError: If site_id is not found.
        """
        ...

    @abstractmethod
    def available_sites(self) -> list[str]:
        """Return sorted list of available site IDs."""
        ...

    def load_all(self, site_ids: list[str] | None = None) -> list[FacilityData]:
        """Load data for multiple sites.

        Args:
            site_ids: Sites to load. If None, loads all available sites.

        Returns:
            List of FacilityData, one per site.
        """
        if site_ids is None:
            site_ids = self.available_sites()

        return [self.load_site(sid) for sid in site_ids]
