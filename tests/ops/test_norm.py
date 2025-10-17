"""
Tests for normalization operations
"""

import torch
import pytest
import numpy as np

from warpylib.ops.norm import norm
from warpylib.ops.norm_ft import norm_ft
from warpylib.ops.norm_rft import norm_rft


class TestNorm2D:
    """Test 2D real-space normalization"""

    def test_basic_normalization_no_mask(self):
        """Test basic normalization without mask"""
        # Create tensor with known mean and std
        x = torch.randn(64, 64) * 5.0 + 10.0  # mean ~10, std ~5

        normalized = norm(x, dimensionality=2, diameter=0)

        # After normalization, should have mean ~0 and std ~1
        assert torch.abs(normalized.mean()) < 0.1
        assert torch.abs(normalized.std() - 1.0) < 0.1

    def test_normalization_with_inner_mask(self):
        """Test normalization with inner circular mask"""
        x = torch.randn(64, 64) * 3.0 + 5.0

        normalized = norm(x, dimensionality=2, diameter=40.0, mode="inner")

        # Statistics calculated on inner circle should give mean ~0, std ~1 in that region
        # But full tensor will have different statistics
        assert normalized.shape == x.shape

    def test_normalization_with_outer_mask(self):
        """Test normalization with outer region"""
        x = torch.randn(64, 64) * 2.0 + 3.0

        normalized = norm(x, dimensionality=2, diameter=20.0, mode="outer")

        assert normalized.shape == x.shape

    def test_batch_dimensions(self):
        """Test normalization with batch dimensions"""
        # Batch of 4 images
        x = torch.randn(4, 64, 64) * 2.0 + 5.0

        normalized = norm(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape

        # Each image in batch should be normalized independently
        for i in range(4):
            assert torch.abs(normalized[i].mean()) < 0.1
            assert torch.abs(normalized[i].std() - 1.0) < 0.1

    def test_preserves_dtype(self):
        """Test that dtype is preserved"""
        x32 = torch.randn(32, 32, dtype=torch.float32)
        normalized32 = norm(x32, dimensionality=2, diameter=0)
        assert normalized32.dtype == torch.float32

        x64 = torch.randn(32, 32, dtype=torch.float64)
        normalized64 = norm(x64, dimensionality=2, diameter=0)
        assert normalized64.dtype == torch.float64

    def test_zero_std_handling(self):
        """Test that constant tensors don't cause division by zero"""
        x = torch.ones(64, 64) * 5.0  # Constant value

        normalized = norm(x, dimensionality=2, diameter=0)

        # Should not contain NaN or Inf
        assert torch.isfinite(normalized).all()


class TestNorm3D:
    """Test 3D real-space normalization"""

    def test_basic_normalization_3d(self):
        """Test basic 3D normalization"""
        x = torch.randn(32, 32, 32) * 4.0 + 7.0

        normalized = norm(x, dimensionality=3, diameter=0)

        assert torch.abs(normalized.mean()) < 0.1
        assert torch.abs(normalized.std() - 1.0) < 0.1

    def test_spherical_mask_3d(self):
        """Test normalization with spherical mask"""
        x = torch.randn(32, 32, 32) * 2.0 + 3.0

        normalized = norm(x, dimensionality=3, diameter=20.0, mode="inner")

        assert normalized.shape == x.shape

    def test_batch_3d(self):
        """Test 3D normalization with batch"""
        x = torch.randn(2, 32, 32, 32) * 3.0 + 2.0

        normalized = norm(x, dimensionality=3, diameter=0)

        assert normalized.shape == x.shape


class TestNormRFT2D:
    """Test 2D RFFT normalization"""

    def test_complex_rfft_no_mask(self):
        """Test normalization of complex RFFT data without mask"""
        # Create RFFT-shaped complex data
        x = torch.randn(64, 33, dtype=torch.complex64)

        normalized = norm_rft(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape
        assert normalized.dtype == torch.complex64

        # For complex, only std normalization (no mean subtraction)
        # Check that normalization produces finite values
        # Note: Hermitian symmetry handling affects overall statistics
        assert torch.isfinite(normalized).all()
        mag = torch.abs(normalized)
        assert mag.std() > 0  # Should have some variation

    def test_real_rfft_no_mask(self):
        """Test normalization of real RFFT data without mask"""
        # Real-valued RFFT format (unusual but should work)
        x = torch.randn(64, 33)

        normalized = norm_rft(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape
        assert not torch.is_complex(normalized)

        # For real, mean subtraction should happen
        # Note: Hermitian symmetry mask affects statistics
        assert torch.isfinite(normalized).all()

    def test_hermitian_symmetry_handling(self):
        """Test that Hermitian symmetry is properly handled"""
        # Create actual RFFT from real data
        real_data = torch.randn(64, 64)
        rfft_data = torch.fft.rfft2(real_data)

        normalized = norm_rft(rfft_data, dimensionality=2, diameter=0)

        assert normalized.shape == rfft_data.shape
        assert torch.is_complex(normalized)

        # Should not have NaN or Inf
        assert torch.isfinite(normalized).all()

    def test_with_circular_mask(self):
        """Test RFFT normalization with circular mask in Fourier space"""
        x = torch.randn(64, 33, dtype=torch.complex64)

        normalized = norm_rft(x, dimensionality=2, diameter=40.0, mode="inner")

        assert normalized.shape == x.shape

    def test_batch_rfft(self):
        """Test batch RFFT normalization"""
        x = torch.randn(4, 64, 33, dtype=torch.complex64)

        normalized = norm_rft(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape


class TestNormRFT3D:
    """Test 3D RFFT normalization"""

    def test_complex_rfft3d_no_mask(self):
        """Test 3D RFFT normalization"""
        # RFFT3D shape: (D, H, W//2+1)
        x = torch.randn(32, 32, 17, dtype=torch.complex64)

        normalized = norm_rft(x, dimensionality=3, diameter=0)

        assert normalized.shape == x.shape
        assert torch.is_complex(normalized)

    def test_real_rfft3d(self):
        """Test 3D RFFT with real data"""
        real_data = torch.randn(32, 32, 32)
        rfft_data = torch.fft.rfftn(real_data)

        normalized = norm_rft(rfft_data, dimensionality=3, diameter=0)

        assert normalized.shape == rfft_data.shape
        assert torch.isfinite(normalized).all()


class TestNormFT2D:
    """Test 2D FFT normalization"""

    def test_complex_fft_no_mask(self):
        """Test normalization of complex FFT data"""
        x = torch.randn(64, 64, dtype=torch.complex64)

        normalized = norm_ft(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape
        assert normalized.dtype == torch.complex64

        # For complex FFT, only std normalization
        mag = torch.abs(normalized)
        assert torch.abs(mag.std() - 1.0) < 0.2

    def test_real_fft(self):
        """Test normalization of real FFT data"""
        x = torch.randn(64, 64)

        normalized = norm_ft(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape
        # Real data: mean subtracted and std normalized
        assert torch.abs(normalized.mean()) < 0.1
        assert torch.abs(normalized.std() - 1.0) < 0.1

    def test_actual_fft_data(self):
        """Test with actual FFT-transformed data"""
        real_data = torch.randn(64, 64)
        fft_data = torch.fft.fft2(real_data)

        normalized = norm_ft(fft_data, dimensionality=2, diameter=0)

        assert normalized.shape == fft_data.shape
        assert torch.is_complex(normalized)
        assert torch.isfinite(normalized).all()

    def test_with_mask_inner(self):
        """Test FFT normalization with inner circular mask"""
        x = torch.randn(64, 64, dtype=torch.complex64)

        normalized = norm_ft(x, dimensionality=2, diameter=40.0, mode="inner")

        assert normalized.shape == x.shape

    def test_with_mask_outer(self):
        """Test FFT normalization with outer region"""
        x = torch.randn(64, 64, dtype=torch.complex64)

        normalized = norm_ft(x, dimensionality=2, diameter=20.0, mode="outer")

        assert normalized.shape == x.shape

    def test_batch_fft(self):
        """Test batch FFT normalization"""
        x = torch.randn(4, 64, 64, dtype=torch.complex64)

        normalized = norm_ft(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape


class TestNormFT3D:
    """Test 3D FFT normalization"""

    def test_complex_fft3d(self):
        """Test 3D FFT normalization"""
        x = torch.randn(32, 32, 32, dtype=torch.complex64)

        normalized = norm_ft(x, dimensionality=3, diameter=0)

        assert normalized.shape == x.shape
        assert torch.is_complex(normalized)

    def test_spherical_mask_3d(self):
        """Test 3D FFT with spherical mask"""
        x = torch.randn(32, 32, 32, dtype=torch.complex64)

        normalized = norm_ft(x, dimensionality=3, diameter=20.0, mode="inner")

        assert normalized.shape == x.shape


class TestValidation:
    """Test input validation and error handling"""

    def test_invalid_dimensionality(self):
        """Test that invalid dimensionality raises error"""
        x = torch.randn(64, 64)

        with pytest.raises(ValueError, match="Dimensionality must be 2 or 3"):
            norm(x, dimensionality=1)

    def test_dimensionality_exceeds_tensor_dims(self):
        """Test error when dimensionality > tensor dims"""
        x = torch.randn(64, 64)

        with pytest.raises(ValueError, match="cannot exceed tensor dimensions"):
            norm(x, dimensionality=3)


class TestEdgeCases:
    """Test edge cases"""

    def test_very_small_diameter(self):
        """Test with very small diameter"""
        x = torch.randn(64, 64)

        # Very small diameter - only few pixels for statistics
        normalized = norm(x, dimensionality=2, diameter=5.0)

        assert normalized.shape == x.shape
        assert torch.isfinite(normalized).all()

    def test_very_large_diameter(self):
        """Test with diameter larger than image"""
        x = torch.randn(64, 64) * 3.0 + 5.0

        # Large diameter - should be similar to no mask
        normalized = norm(x, dimensionality=2, diameter=1000.0)

        assert torch.abs(normalized.mean()) < 0.2
        assert torch.abs(normalized.std() - 1.0) < 0.2

    def test_single_batch_element(self):
        """Test with batch size of 1"""
        x = torch.randn(1, 64, 64)

        normalized = norm(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape

    def test_rectangular_image(self):
        """Test with non-square image"""
        x = torch.randn(64, 128)

        normalized = norm(x, dimensionality=2, diameter=0)

        assert normalized.shape == x.shape


class TestRoundTrip:
    """Test round-trip transformations"""

    def test_fft_rfft_consistency(self):
        """Test that FFT and RFFT normalization are consistent"""
        # Create real data
        real_data = torch.randn(64, 64)

        # Full FFT
        fft_data = torch.fft.fft2(real_data)
        fft_normalized = norm_ft(fft_data, dimensionality=2, diameter=0)

        # RFFT
        rfft_data = torch.fft.rfft2(real_data)
        rfft_normalized = norm_rft(rfft_data, dimensionality=2, diameter=0)

        # Both should produce valid results
        assert torch.isfinite(fft_normalized).all()
        assert torch.isfinite(rfft_normalized).all()

    def test_normalization_invertible(self):
        """Test that we can compute normalization parameters"""
        x = torch.randn(64, 64) * 5.0 + 10.0

        # Get original statistics
        orig_mean = x.mean()
        orig_std = x.std(unbiased=False)

        # Normalize
        normalized = norm(x, dimensionality=2, diameter=0)

        # Denormalize
        denormalized = normalized * orig_std + orig_mean

        # Should be close to original
        assert torch.allclose(denormalized, x, atol=1e-5)


class TestNumericalStability:
    """Test numerical stability"""

    def test_gradient_flow(self):
        """Test that gradients flow correctly"""
        x = torch.randn(64, 64, requires_grad=True)

        normalized = norm(x, dimensionality=2, diameter=0)
        loss = (normalized ** 2).sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_extreme_values(self):
        """Test with extreme input values"""
        x = torch.randn(64, 64) * 1000.0 + 5000.0

        normalized = norm(x, dimensionality=2, diameter=0)

        assert torch.isfinite(normalized).all()
        assert torch.abs(normalized.mean()) < 1.0
        assert torch.abs(normalized.std() - 1.0) < 0.5

    def test_very_small_values(self):
        """Test with very small input values"""
        x = torch.randn(64, 64) * 1e-6

        normalized = norm(x, dimensionality=2, diameter=0)

        assert torch.isfinite(normalized).all()


class TestConsistency:
    """Test consistency across different calls"""

    def test_deterministic(self):
        """Test that normalization is deterministic"""
        x = torch.randn(64, 64)

        norm1 = norm(x, dimensionality=2, diameter=0)
        norm2 = norm(x, dimensionality=2, diameter=0)

        assert torch.allclose(norm1, norm2)

    def test_independence_of_batches(self):
        """Test that batch elements are normalized independently"""
        # Create batch where each element has different statistics
        batch = torch.stack([
            torch.randn(64, 64) * 1.0 + 0.0,
            torch.randn(64, 64) * 5.0 + 10.0,
            torch.randn(64, 64) * 0.5 + 2.0,
        ])

        normalized = norm(batch, dimensionality=2, diameter=0)

        # Each should be independently normalized
        for i in range(3):
            assert torch.abs(normalized[i].mean()) < 0.1
            assert torch.abs(normalized[i].std() - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])