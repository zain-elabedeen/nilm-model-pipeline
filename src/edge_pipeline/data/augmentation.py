"""AMDA (Appliance Magnitude-aware Data Augmentation) implementation.

Based on the SIDED paper methodology for improving NILM model generalisation.
"""

import numpy as np


class AMDATransform:
    """
    Appliance Magnitude-aware Data Augmentation.

    Scales appliance signals inversely proportional to their contribution,
    amplifying underrepresented appliances while reducing dominant ones.

    From the SIDED paper:
        p_i = P_total_i / P_total  (relative contribution)
        S_i = s * (1 - p_i)        (scaling factor)
        x̃_i,t = S_i * x_i,t       (scaled signal)

    Where s is the base scaling factor (optimal s ≈ 2.5).
    """

    def __init__(
        self,
        base_scale: float = 2.5,
        min_scale: float = 0.5,
        max_scale: float = 4.0,
        probability: float = 0.5,
    ):
        """
        Initialise AMDA transform.

        Args:
            base_scale: Base scaling factor s (paper optimal: 2.5)
            min_scale: Minimum allowed scaling factor
            max_scale: Maximum allowed scaling factor
            probability: Probability of applying augmentation per sample
        """
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.probability = probability

    def __call__(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply AMDA augmentation.

        Args:
            aggregate: Shape (batch, window_size) or (window_size,)
            appliances: Shape (batch, 5) or (5,)

        Returns:
            Tuple of (augmented_aggregate, augmented_appliances)
        """
        if np.random.random() > self.probability:
            return aggregate, appliances

        # Handle both batched and single samples
        single_sample = aggregate.ndim == 1
        if single_sample:
            aggregate = aggregate[np.newaxis, :]
            appliances = appliances[np.newaxis, :]

        batch_size = aggregate.shape[0]
        aug_aggregate = np.zeros_like(aggregate)
        aug_appliances = np.zeros_like(appliances)

        for b in range(batch_size):
            aug_agg, aug_app = self._augment_single(aggregate[b], appliances[b])
            aug_aggregate[b] = aug_agg
            aug_appliances[b] = aug_app

        if single_sample:
            return aug_aggregate[0], aug_appliances[0]

        return aug_aggregate, aug_appliances

    def _augment_single(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply AMDA to a single sample."""
        # Calculate total power contributions (absolute values for generators)
        total_powers = np.abs(appliances)
        p_total = np.sum(total_powers)

        if p_total < 1e-6:
            # No power, return unchanged
            return aggregate, appliances

        # Relative contributions
        p_i = total_powers / p_total

        # Scaling factors: S_i = s * (1 - p_i)
        scale_factors = self.base_scale * (1.0 - p_i)

        # Clamp to valid range
        scale_factors = np.clip(scale_factors, self.min_scale, self.max_scale)

        # Apply scaling to appliances
        aug_appliances = appliances * scale_factors

        # Reconstruct aggregate from scaled appliances
        # Aggregate = sum of appliances (accounting for sign)
        aug_aggregate = np.full_like(aggregate, np.sum(aug_appliances))

        return aug_aggregate, aug_appliances

    def augment_batch(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply AMDA to a batch with individual probability checks.

        Args:
            aggregate: Shape (batch, window_size)
            appliances: Shape (batch, 5)

        Returns:
            Augmented batch
        """
        mask = np.random.random(aggregate.shape[0]) < self.probability
        aug_aggregate = aggregate.copy()
        aug_appliances = appliances.copy()

        for i in np.where(mask)[0]:
            aug_aggregate[i], aug_appliances[i] = self._augment_single(
                aggregate[i], appliances[i]
            )

        return aug_aggregate, aug_appliances


class MixupTransform:
    """
    Mixup augmentation for NILM.

    Linearly interpolates between pairs of samples.
    """

    def __init__(self, alpha: float = 0.2, probability: float = 0.3):
        """
        Initialise Mixup.

        Args:
            alpha: Beta distribution parameter for mixing coefficient
            probability: Probability of applying mixup
        """
        self.alpha = alpha
        self.probability = probability

    def __call__(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply mixup to a batch."""
        if np.random.random() > self.probability:
            return aggregate, appliances

        batch_size = aggregate.shape[0]
        if batch_size < 2:
            return aggregate, appliances

        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Shuffle indices for pairing
        indices = np.random.permutation(batch_size)

        # Mix samples
        mixed_aggregate = lam * aggregate + (1 - lam) * aggregate[indices]
        mixed_appliances = lam * appliances + (1 - lam) * appliances[indices]

        return mixed_aggregate, mixed_appliances


class GaussianNoiseTransform:
    """Adds Gaussian noise to simulate sensor measurement noise."""

    def __init__(
        self,
        std_ratio: float = 0.02,
        probability: float = 0.5,
    ):
        """
        Initialise noise transform.

        Args:
            std_ratio: Noise standard deviation as ratio of signal std
            probability: Probability of applying noise
        """
        self.std_ratio = std_ratio
        self.probability = probability

    def __call__(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian noise to aggregate (not targets)."""
        if np.random.random() > self.probability:
            return aggregate, appliances

        signal_std = np.std(aggregate)
        noise_std = signal_std * self.std_ratio
        noise = np.random.normal(0, noise_std, aggregate.shape)

        return aggregate + noise, appliances


class ComposeTransforms:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(
        self,
        aggregate: np.ndarray,
        appliances: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            aggregate, appliances = transform(aggregate, appliances)
        return aggregate, appliances
