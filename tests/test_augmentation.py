"""Tests for AMDA and other data augmentation transforms."""

import numpy as np
import pytest

from edge_pipeline.data.augmentation import (
    AMDATransform,
    ComposeTransforms,
    GaussianNoiseTransform,
    MixupTransform,
)


class TestAMDATransform:
    """Tests for AMDA augmentation."""

    def test_amda_basic(self):
        """Test basic AMDA transformation."""
        amda = AMDATransform(base_scale=2.5, probability=1.0)

        # Create sample data
        aggregate = np.array([10000.0] * 60, dtype=np.float32)
        appliances = np.array([5000.0, -3000.0, 2000.0, 0.0, 1000.0], dtype=np.float32)

        aug_agg, aug_app = amda(aggregate, appliances)

        # Output shapes should match
        assert aug_agg.shape == aggregate.shape
        assert aug_app.shape == appliances.shape

        # Appliances should be scaled
        assert not np.allclose(aug_app, appliances)

    def test_amda_scaling_logic(self):
        """Test AMDA scaling factors are computed correctly."""
        amda = AMDATransform(base_scale=2.5, min_scale=0.5, max_scale=4.0, probability=1.0)

        # Dominant appliance should be scaled down
        aggregate = np.array([10000.0] * 60, dtype=np.float32)
        appliances = np.array([8000.0, -1000.0, 500.0, 0.0, 500.0], dtype=np.float32)

        _, aug_app = amda(aggregate, appliances)

        # The dominant appliance (index 0) should have smaller scale
        # Relative contribution: 8000/10000 = 0.8
        # Scale factor: 2.5 * (1 - 0.8) = 0.5
        # So it should be scaled down
        assert abs(aug_app[0]) < abs(appliances[0])

    def test_amda_probability(self):
        """Test AMDA respects probability parameter."""
        np.random.seed(42)
        amda = AMDATransform(base_scale=2.5, probability=0.0)

        aggregate = np.array([10000.0] * 60, dtype=np.float32)
        appliances = np.array([5000.0, -3000.0, 2000.0, 0.0, 1000.0], dtype=np.float32)

        aug_agg, aug_app = amda(aggregate, appliances)

        # With probability 0, should return unchanged
        np.testing.assert_array_equal(aug_agg, aggregate)
        np.testing.assert_array_equal(aug_app, appliances)

    def test_amda_batch(self):
        """Test AMDA on batched data."""
        amda = AMDATransform(base_scale=2.5, probability=0.5)

        batch_size = 32
        aggregate = np.random.randn(batch_size, 60).astype(np.float32) * 10000 + 5000
        appliances = np.random.randn(batch_size, 5).astype(np.float32) * 2000 + 1000

        aug_agg, aug_app = amda.augment_batch(aggregate, appliances)

        assert aug_agg.shape == aggregate.shape
        assert aug_app.shape == appliances.shape


class TestMixupTransform:
    """Tests for Mixup augmentation."""

    def test_mixup_basic(self):
        """Test basic mixup."""
        np.random.seed(42)
        mixup = MixupTransform(alpha=0.2, probability=1.0)

        batch_size = 8
        aggregate = np.random.randn(batch_size, 60).astype(np.float32)
        appliances = np.random.randn(batch_size, 5).astype(np.float32)

        mixed_agg, mixed_app = mixup(aggregate, appliances)

        assert mixed_agg.shape == aggregate.shape
        assert mixed_app.shape == appliances.shape

    def test_mixup_probability_zero(self):
        """Test mixup with zero probability."""
        mixup = MixupTransform(alpha=0.2, probability=0.0)

        aggregate = np.random.randn(8, 60).astype(np.float32)
        appliances = np.random.randn(8, 5).astype(np.float32)

        mixed_agg, mixed_app = mixup(aggregate, appliances)

        np.testing.assert_array_equal(mixed_agg, aggregate)
        np.testing.assert_array_equal(mixed_app, appliances)


class TestGaussianNoiseTransform:
    """Tests for Gaussian noise augmentation."""

    def test_noise_basic(self):
        """Test noise is added to aggregate."""
        np.random.seed(42)
        noise = GaussianNoiseTransform(std_ratio=0.02, probability=1.0)

        aggregate = (np.arange(60, dtype=np.float32) * 100 + 5000)
        appliances = np.ones((5,), dtype=np.float32) * 2000

        aug_agg, aug_app = noise(aggregate, appliances)

        # Aggregate should have noise added
        assert not np.allclose(aug_agg, aggregate)

        # Appliances (targets) should be unchanged
        np.testing.assert_array_equal(aug_app, appliances)


class TestComposeTransforms:
    """Tests for composed transforms."""

    def test_compose(self):
        """Test composing multiple transforms."""
        transforms = ComposeTransforms([
            GaussianNoiseTransform(std_ratio=0.01, probability=1.0),
            AMDATransform(base_scale=2.5, probability=1.0),
        ])

        aggregate = np.random.randn(60).astype(np.float32) * 10000 + 5000
        appliances = np.random.randn(5).astype(np.float32) * 2000 + 1000

        aug_agg, aug_app = transforms(aggregate, appliances)

        assert aug_agg.shape == aggregate.shape
        assert aug_app.shape == appliances.shape
