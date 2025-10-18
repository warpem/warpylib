"""
Tests for bandpass filtering operations
"""

import torch
import pytest
import numpy as np

from warpylib.ops.bandpass import bandpass
from warpylib.ops.bandpass_rft import bandpass_rft


class TestBandpass2D:
    """Test 2D real-space bandpass filtering"""

    def test_lowpass_basic(self):
        """Test basic low-pass filtering"""
        x = torch.randn(64, 64)

        filtered = bandpass(x, dimensionality=2, high_freq=0.5)

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_highpass_basic(self):
        """Test basic high-pass filtering"""
        x = torch.randn(64, 64)

        filtered = bandpass(x, dimensionality=2, low_freq=0.1)

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_bandpass_basic(self):
        """Test basic band-pass filtering"""
        x = torch.randn(64, 64)

        filtered = bandpass(x, dimensionality=2, low_freq=0.1, high_freq=0.5)

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_soft_edges(self):
        """Test bandpass with soft edges"""
        x = torch.randn(64, 64)

        filtered = bandpass(
            x,
            dimensionality=2,
            low_freq=0.1,
            high_freq=0.5,
            soft_edge_low=0.05,
            soft_edge_high=0.05
        )

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_rectangular_image(self):
        """Test with non-square image"""
        x = torch.randn(128, 64)

        filtered = bandpass(x, dimensionality=2, high_freq=0.5)

        assert filtered.shape == x.shape

    def test_batch_dimensions(self):
        """Test with batch dimensions"""
        x = torch.randn(4, 64, 64)

        filtered = bandpass(x, dimensionality=2, high_freq=0.5)

        assert filtered.shape == x.shape

    def test_multiple_batch_dimensions(self):
        """Test with multiple batch dimensions"""
        x = torch.randn(2, 3, 64, 64)

        filtered = bandpass(x, dimensionality=2, low_freq=0.1)

        assert filtered.shape == x.shape

    def test_preserves_dtype(self):
        """Test that dtype is preserved"""
        x32 = torch.randn(64, 64, dtype=torch.float32)
        filtered32 = bandpass(x32, dimensionality=2, high_freq=0.5)
        assert filtered32.dtype == torch.float32

        x64 = torch.randn(64, 64, dtype=torch.float64)
        filtered64 = bandpass(x64, dimensionality=2, high_freq=0.5)
        assert filtered64.dtype == torch.float64

    def test_no_filtering(self):
        """Test with no filters applied (both None)"""
        x = torch.randn(64, 64)

        # No filters - should return similar to input
        filtered = bandpass(x, dimensionality=2)

        assert filtered.shape == x.shape
        # Without any filtering, output should be very close to input
        assert torch.allclose(filtered, x, atol=1e-5)


class TestBandpass3D:
    """Test 3D real-space bandpass filtering"""

    def test_lowpass_3d(self):
        """Test 3D low-pass filtering"""
        x = torch.randn(32, 32, 32)

        filtered = bandpass(x, dimensionality=3, high_freq=0.5)

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_highpass_3d(self):
        """Test 3D high-pass filtering"""
        x = torch.randn(32, 32, 32)

        filtered = bandpass(x, dimensionality=3, low_freq=0.1)

        assert filtered.shape == x.shape

    def test_bandpass_3d(self):
        """Test 3D band-pass filtering"""
        x = torch.randn(32, 32, 32)

        filtered = bandpass(
            x,
            dimensionality=3,
            low_freq=0.1,
            high_freq=0.5,
            soft_edge_low=0.05,
            soft_edge_high=0.05
        )

        assert filtered.shape == x.shape

    def test_non_cubic_volume(self):
        """Test with non-cubic volume"""
        x = torch.randn(64, 32, 48)

        filtered = bandpass(x, dimensionality=3, high_freq=0.5)

        assert filtered.shape == x.shape

    def test_batch_3d(self):
        """Test 3D filtering with batch"""
        x = torch.randn(2, 32, 32, 32)

        filtered = bandpass(x, dimensionality=3, low_freq=0.1, high_freq=0.5)

        assert filtered.shape == x.shape


class TestBandpassRFT2D:
    """Test 2D RFFT bandpass filtering"""

    def test_rft_lowpass(self):
        """Test low-pass filtering on RFFT data"""
        x = torch.randn(64, 64)
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))

        filtered_fft = bandpass_rft(x_fft, dimensionality=2, high_freq=0.5)

        assert filtered_fft.shape == x_fft.shape
        assert torch.is_complex(filtered_fft)
        assert torch.isfinite(filtered_fft).all()

    def test_rft_highpass(self):
        """Test high-pass filtering on RFFT data"""
        x = torch.randn(64, 64)
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))

        filtered_fft = bandpass_rft(x_fft, dimensionality=2, low_freq=0.1)

        assert filtered_fft.shape == x_fft.shape
        assert torch.is_complex(filtered_fft)

    def test_rft_bandpass(self):
        """Test band-pass filtering on RFFT data"""
        x = torch.randn(64, 64)
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))

        filtered_fft = bandpass_rft(
            x_fft,
            dimensionality=2,
            low_freq=0.1,
            high_freq=0.5,
            soft_edge_low=0.05,
            soft_edge_high=0.05
        )

        assert filtered_fft.shape == x_fft.shape

    def test_rft_batch(self):
        """Test RFFT filtering with batch"""
        x = torch.randn(4, 64, 64)
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))

        filtered_fft = bandpass_rft(x_fft, dimensionality=2, high_freq=0.5)

        assert filtered_fft.shape == x_fft.shape


class TestBandpassRFT3D:
    """Test 3D RFFT bandpass filtering"""

    def test_rft3d_lowpass(self):
        """Test 3D RFFT low-pass filtering"""
        x = torch.randn(32, 32, 32)
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        filtered_fft = bandpass_rft(x_fft, dimensionality=3, high_freq=0.5)

        assert filtered_fft.shape == x_fft.shape
        assert torch.is_complex(filtered_fft)

    def test_rft3d_bandpass(self):
        """Test 3D RFFT band-pass filtering"""
        x = torch.randn(32, 32, 32)
        x_fft = torch.fft.rfftn(x, dim=(-3, -2, -1))

        filtered_fft = bandpass_rft(
            x_fft,
            dimensionality=3,
            low_freq=0.1,
            high_freq=0.5
        )

        assert filtered_fft.shape == x_fft.shape


class TestConsistency:
    """Test consistency between bandpass and bandpass_rft"""

    def test_bandpass_rft_equivalence(self):
        """Test that bandpass and manual rft filtering are equivalent"""
        x = torch.randn(64, 64)

        # Via bandpass wrapper
        result1 = bandpass(x, dimensionality=2, low_freq=0.1, high_freq=0.5)

        # Via manual rft
        x_fft = torch.fft.rfftn(x, dim=(-2, -1))
        filtered_fft = bandpass_rft(x_fft, dimensionality=2, low_freq=0.1, high_freq=0.5)
        result2 = torch.fft.irfftn(filtered_fft, s=x.shape, dim=(-2, -1))

        # Should be identical (within numerical precision)
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_deterministic(self):
        """Test that filtering is deterministic"""
        x = torch.randn(64, 64)

        filtered1 = bandpass(x, dimensionality=2, low_freq=0.1, high_freq=0.5)
        filtered2 = bandpass(x, dimensionality=2, low_freq=0.1, high_freq=0.5)

        assert torch.allclose(filtered1, filtered2)

    def test_batch_independence(self):
        """Test that batch elements are filtered with same filter"""
        # Create batch with identical elements
        x = torch.randn(64, 64)
        batch = x.unsqueeze(0).expand(4, -1, -1).clone()

        filtered_batch = bandpass(batch, dimensionality=2, high_freq=0.5)

        # All batch elements should be identical
        for i in range(1, 4):
            assert torch.allclose(filtered_batch[0], filtered_batch[i], atol=1e-6)


class TestFilterProperties:
    """Test that filters have expected properties"""

    def test_lowpass_reduces_high_freq(self):
        """Test that low-pass filter reduces high frequencies"""
        # Create image with high-frequency content
        x = torch.zeros(64, 64)
        # Add checkerboard pattern (high frequency)
        for i in range(64):
            for j in range(64):
                x[i, j] = 1.0 if (i + j) % 2 == 0 else -1.0

        # Apply low-pass filter
        filtered = bandpass(x, dimensionality=2, high_freq=0.3)

        # Filtered should be smoother (lower variance)
        assert filtered.var() < x.var()

    def test_highpass_reduces_dc(self):
        """Test that high-pass filter reduces DC component"""
        # Create image with DC offset
        x = torch.randn(64, 64) + 10.0

        # Apply high-pass filter (removes low frequencies including DC)
        filtered = bandpass(x, dimensionality=2, low_freq=0.1)

        # Mean should be closer to zero
        assert torch.abs(filtered.mean()) < torch.abs(x.mean())

    def test_soft_edge_smoothness(self):
        """Test that soft edges create smooth transitions"""
        x = torch.randn(64, 64)

        # Filter with hard edge
        hard = bandpass(x, dimensionality=2, high_freq=0.5, soft_edge_high=0.0)

        # Filter with soft edge
        soft = bandpass(x, dimensionality=2, high_freq=0.5, soft_edge_high=0.1)

        # Both should work
        assert torch.isfinite(hard).all()
        assert torch.isfinite(soft).all()


class TestValidation:
    """Test input validation and error handling"""

    def test_invalid_dimensionality(self):
        """Test that invalid dimensionality raises error"""
        x = torch.randn(64, 64)

        with pytest.raises(ValueError, match="Dimensionality must be 2 or 3"):
            bandpass(x, dimensionality=1, high_freq=0.5)

    def test_dimensionality_exceeds_tensor_dims(self):
        """Test error when dimensionality > tensor dims"""
        x = torch.randn(64, 64)

        with pytest.raises(ValueError, match="cannot exceed tensor dimensions"):
            bandpass(x, dimensionality=3, high_freq=0.5)


class TestEdgeCases:
    """Test edge cases"""

    def test_very_low_cutoff(self):
        """Test with very low frequency cutoff"""
        x = torch.randn(64, 64)

        filtered = bandpass(x, dimensionality=2, high_freq=0.01)

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_very_high_cutoff(self):
        """Test with cutoff near Nyquist"""
        x = torch.randn(64, 64)

        filtered = bandpass(x, dimensionality=2, low_freq=0.99)

        assert filtered.shape == x.shape
        assert torch.isfinite(filtered).all()

    def test_narrow_bandpass(self):
        """Test with very narrow bandpass"""
        x = torch.randn(64, 64)

        filtered = bandpass(x, dimensionality=2, low_freq=0.4, high_freq=0.45)

        assert filtered.shape == x.shape

    def test_large_soft_edge(self):
        """Test with soft edge larger than passband"""
        x = torch.randn(64, 64)

        filtered = bandpass(
            x,
            dimensionality=2,
            low_freq=0.2,
            high_freq=0.3,
            soft_edge_low=0.15,
            soft_edge_high=0.15
        )

        assert filtered.shape == x.shape


class TestNumericalStability:
    """Test numerical stability"""

    def test_gradient_flow(self):
        """Test that gradients flow correctly through bandpass"""
        x = torch.randn(64, 64, requires_grad=True)

        filtered = bandpass(x, dimensionality=2, high_freq=0.5)
        loss = (filtered ** 2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_extreme_values(self):
        """Test with extreme input values"""
        x = torch.randn(64, 64) * 1000.0 + 5000.0

        filtered = bandpass(x, dimensionality=2, high_freq=0.5)

        assert torch.isfinite(filtered).all()

    def test_very_small_values(self):
        """Test with very small input values"""
        x = torch.randn(64, 64) * 1e-6

        filtered = bandpass(x, dimensionality=2, high_freq=0.5)

        assert torch.isfinite(filtered).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])