"""
Tests for masking operations
"""

import torch
import pytest
import numpy as np

from warpylib.ops.masking import mask_sphere, mask_sphere_ft


class TestMaskSphere2D:
    """Test 2D circular masking"""

    def test_basic_2d_mask(self):
        """Test basic 2D circular mask application"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=32.0, soft_edge_width=0.0)

        assert masked.shape == (64, 64)
        # Center should be 1.0 (within radius)
        assert masked[32, 32] == 1.0
        # Corners should be 0.0 (outside radius)
        assert masked[0, 0] == 0.0

    def test_hard_edge_2d(self):
        """Test hard edge mask (no soft falloff)"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=0.0)

        # Calculate distances from center
        center = 31.5
        y_coords = torch.arange(64) - center
        x_coords = torch.arange(64) - center
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(xx**2 + yy**2)

        # Points inside radius should be 1.0
        inside_mask = distances < 20.0
        assert torch.allclose(masked[inside_mask], torch.ones_like(masked[inside_mask]))

        # Points outside radius should be 0.0
        outside_mask = distances >= 20.0
        assert torch.allclose(masked[outside_mask], torch.zeros_like(masked[outside_mask]))

    def test_soft_edge_2d(self):
        """Test soft edge with raised cosine falloff"""
        x = torch.ones(128, 128)
        diameter = 80.0
        soft_edge_width = 10.0
        masked = mask_sphere(x, diameter=diameter, soft_edge_width=soft_edge_width)

        # Calculate distances from center
        center = 63.5
        y_coords = torch.arange(128) - center
        x_coords = torch.arange(128) - center
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(xx**2 + yy**2)

        radius = diameter / 2.0

        # Points well inside should be 1.0
        inside_mask = distances < radius - 1.0
        assert torch.allclose(masked[inside_mask], torch.ones_like(masked[inside_mask]), atol=1e-5)

        # Points in transition region should be between 0 and 1
        transition_mask = (distances >= radius) & (distances < radius + soft_edge_width)
        transition_values = masked[transition_mask]
        assert (transition_values > 0).all() and (transition_values < 1).all()

        # Points outside should be 0.0
        outside_mask = distances >= radius + soft_edge_width
        assert torch.allclose(masked[outside_mask], torch.zeros_like(masked[outside_mask]), atol=1e-5)

    def test_soft_edge_smoothness(self):
        """Test that soft edge follows raised cosine profile"""
        x = torch.ones(100, 100)
        diameter = 50.0
        soft_edge_width = 10.0
        masked = mask_sphere(x, diameter=diameter, soft_edge_width=soft_edge_width)

        # Sample along a horizontal line through center
        center_line = masked[50, :]

        # Find the transition region
        # Should smoothly decrease from 1 to 0
        diff = torch.diff(center_line)
        # In transition region, differences should be negative (decreasing)
        # and relatively smooth (no sudden jumps)
        assert (diff <= 0).sum() > 0  # Some decreasing values exist

    def test_preserves_values_inside_mask(self):
        """Test that values inside mask are preserved"""
        x = torch.randn(64, 64)
        masked = mask_sphere(x, diameter=60.0, soft_edge_width=0.0)

        # Center region should equal original values
        assert torch.allclose(masked[32, 32], x[32, 32])
        assert torch.allclose(masked[30:35, 30:35], x[30:35, 30:35], atol=1e-6)

    def test_zeros_outside_mask(self):
        """Test that values outside mask are zeroed"""
        x = torch.randn(64, 64) + 5.0  # Non-zero values
        masked = mask_sphere(x, diameter=20.0, soft_edge_width=0.0)

        # Corners should be zero
        assert masked[0, 0] == 0.0
        assert masked[0, -1] == 0.0
        assert masked[-1, 0] == 0.0
        assert masked[-1, -1] == 0.0


class TestMaskSphere3D:
    """Test 3D spherical masking"""

    def test_basic_3d_mask(self):
        """Test basic 3D spherical mask application"""
        x = torch.ones(32, 32, 32)
        masked = mask_sphere(x, diameter=20.0, soft_edge_width=0.0)

        assert masked.shape == (32, 32, 32)
        # Center should be 1.0
        assert masked[16, 16, 16] == 1.0
        # Corners should be 0.0
        assert masked[0, 0, 0] == 0.0

    def test_hard_edge_3d(self):
        """Test hard edge spherical mask"""
        x = torch.ones(40, 40, 40)
        masked = mask_sphere(x, diameter=30.0, soft_edge_width=0.0)

        # Calculate distances from center
        center = 19.5
        z_coords = torch.arange(40) - center
        y_coords = torch.arange(40) - center
        x_coords = torch.arange(40) - center
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(xx**2 + yy**2 + zz**2)

        radius = 15.0

        # Points inside should be 1.0
        inside_mask = distances < radius
        assert torch.allclose(masked[inside_mask], torch.ones_like(masked[inside_mask]))

        # Points outside should be 0.0
        outside_mask = distances >= radius
        assert torch.allclose(masked[outside_mask], torch.zeros_like(masked[outside_mask]))

    def test_soft_edge_3d(self):
        """Test soft edge spherical mask"""
        x = torch.ones(64, 64, 64)
        diameter = 40.0
        soft_edge_width = 8.0
        masked = mask_sphere(x, diameter=diameter, soft_edge_width=soft_edge_width)

        # Calculate distances from center
        center = 31.5
        z_coords = torch.arange(64) - center
        y_coords = torch.arange(64) - center
        x_coords = torch.arange(64) - center
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(xx**2 + yy**2 + zz**2)

        radius = diameter / 2.0

        # Points in transition should be between 0 and 1
        transition_mask = (distances >= radius) & (distances < radius + soft_edge_width)
        transition_values = masked[transition_mask]
        assert (transition_values > 0).all() and (transition_values < 1).all()


class TestMaskSphereFT:
    """Test FFT-format masking"""

    def test_mask_sphere_ft_2d(self):
        """Test 2D FFT mask is ifftshifted"""
        x = torch.ones(64, 64, dtype=torch.complex64)
        masked = mask_sphere_ft(x, diameter=40.0, soft_edge_width=0.0)

        assert masked.shape == (64, 64)
        assert masked.dtype == torch.complex64

        # The mask should be different from centered version
        masked_centered = mask_sphere(x, diameter=40.0, soft_edge_width=0.0)
        assert not torch.allclose(masked, masked_centered)

        # DC component (at index 0,0 in FFT format) should have mask value
        # from the center of the centered mask
        centered_mask = mask_sphere(torch.ones(64, 64), diameter=40.0, soft_edge_width=0.0)
        expected_dc_value = centered_mask[32, 32]
        assert torch.isclose(masked[0, 0].real, expected_dc_value, atol=1e-6)

    def test_mask_sphere_ft_3d(self):
        """Test 3D FFT mask is ifftshifted"""
        x = torch.ones(32, 32, 32, dtype=torch.complex64)
        masked = mask_sphere_ft(x, diameter=20.0, soft_edge_width=0.0)

        assert masked.shape == (32, 32, 32)
        assert masked.dtype == torch.complex64

        # DC component should correspond to center of centered mask
        centered_mask = mask_sphere(torch.ones(32, 32, 32), diameter=20.0, soft_edge_width=0.0)
        expected_dc_value = centered_mask[16, 16, 16]
        assert torch.isclose(masked[0, 0, 0].real, expected_dc_value, atol=1e-6)

    def test_ifftshift_correctness(self):
        """Test that mask_sphere_ft correctly applies ifftshift"""
        x = torch.ones(64, 64)

        # Manual approach
        centered_masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)
        manual_fft_masked = torch.fft.ifftshift(centered_masked)

        # Using mask_sphere_ft
        auto_fft_masked = mask_sphere_ft(x, diameter=40.0, soft_edge_width=5.0)

        assert torch.allclose(manual_fft_masked, auto_fft_masked)

    def test_mask_sphere_ft_with_actual_fft(self):
        """Test masking actual FFT data"""
        # Create a test image
        x = torch.randn(64, 64)

        # Transform to Fourier space
        x_fft = torch.fft.fft2(x)

        # Apply mask in FFT format
        masked_fft = mask_sphere_ft(x_fft, diameter=50.0, soft_edge_width=5.0)

        # Transform back
        masked_real = torch.fft.ifft2(masked_fft).real

        assert masked_real.shape == x.shape
        # Result should be different from original
        assert not torch.allclose(masked_real, x)


class TestValidation:
    """Test input validation and error handling"""

    def test_1d_tensor_raises(self):
        """Test that 1D tensor raises ValueError"""
        x = torch.ones(64)

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            mask_sphere(x, diameter=30.0, soft_edge_width=0.0)

    def test_4d_tensor_raises(self):
        """Test that 4D tensor raises ValueError"""
        x = torch.ones(2, 32, 32, 32)

        with pytest.raises(ValueError, match="must be 2D or 3D"):
            mask_sphere(x, diameter=30.0, soft_edge_width=0.0)

    def test_negative_diameter_produces_all_zeros(self):
        """Test that negative diameter produces all zeros"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=-10.0, soft_edge_width=0.0)

        # Everything should be outside the mask
        assert torch.allclose(masked, torch.zeros_like(masked))

    def test_zero_diameter_produces_all_zeros(self):
        """Test that zero diameter produces all zeros"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=0.0, soft_edge_width=0.0)

        # Everything should be outside the mask (or at the boundary)
        assert torch.allclose(masked, torch.zeros_like(masked))

    def test_very_large_diameter(self):
        """Test that very large diameter masks nothing"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=1000.0, soft_edge_width=0.0)

        # Everything should be inside the mask
        assert torch.allclose(masked, x)


class TestDtypeAndDevice:
    """Test dtype and device preservation"""

    def test_preserves_float32(self):
        """Test that float32 dtype is preserved"""
        x = torch.randn(64, 64, dtype=torch.float32)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert masked.dtype == torch.float32

    def test_preserves_float64(self):
        """Test that float64 dtype is preserved"""
        x = torch.randn(64, 64, dtype=torch.float64)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert masked.dtype == torch.float64

    def test_preserves_complex64(self):
        """Test that complex64 dtype is preserved"""
        x = torch.randn(64, 64, dtype=torch.complex64)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert masked.dtype == torch.complex64

    def test_preserves_complex128(self):
        """Test that complex128 dtype is preserved"""
        x = torch.randn(64, 64, dtype=torch.complex128)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert masked.dtype == torch.complex128

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_preserves_device_cuda(self):
        """Test that CUDA device is preserved"""
        x = torch.randn(64, 64, device='cuda')
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert masked.device.type == 'cuda'

    def test_preserves_device_cpu(self):
        """Test that CPU device is preserved"""
        x = torch.randn(64, 64, device='cpu')
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert masked.device.type == 'cpu'


class TestGradients:
    """Test gradient flow for autograd"""

    def test_gradient_flow_2d(self):
        """Test that gradients flow through 2D mask"""
        x = torch.randn(64, 64, requires_grad=True)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)
        loss = masked.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_3d(self):
        """Test that gradients flow through 3D mask"""
        x = torch.randn(32, 32, 32, requires_grad=True)
        masked = mask_sphere(x, diameter=20.0, soft_edge_width=3.0)
        loss = masked.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_finite(self):
        """Test that gradients are finite"""
        x = torch.randn(64, 64, requires_grad=True)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=10.0)
        loss = (masked ** 2).sum()
        loss.backward()

        assert torch.isfinite(x.grad).all()


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_diameter_equals_dimension(self):
        """Test when diameter equals image dimension"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=64.0, soft_edge_width=0.0)

        # Center should definitely be 1
        assert masked[32, 32] == 1.0

    def test_soft_edge_larger_than_radius(self):
        """Test when soft edge width is larger than radius"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=10.0, soft_edge_width=20.0)

        # Should still produce valid mask
        assert masked.shape == (64, 64)
        assert (masked >= 0).all() and (masked <= 1).all()

    def test_small_image_large_mask(self):
        """Test small image with large mask diameter"""
        x = torch.ones(16, 16)
        masked = mask_sphere(x, diameter=100.0, soft_edge_width=0.0)

        # Everything should be inside
        assert torch.allclose(masked, x)

    def test_rectangular_image(self):
        """Test with non-square image"""
        x = torch.ones(64, 128)
        masked = mask_sphere(x, diameter=50.0, soft_edge_width=0.0)

        assert masked.shape == (64, 128)
        # Center should be masked
        center_y = 31.5
        center_x = 63.5
        # Point near center should be 1
        assert masked[32, 64] == 1.0

    def test_rectangular_volume(self):
        """Test with non-cubic volume"""
        x = torch.ones(32, 48, 64)
        masked = mask_sphere(x, diameter=30.0, soft_edge_width=0.0)

        assert masked.shape == (32, 48, 64)


class TestNumericalPrecision:
    """Test numerical precision and stability"""

    def test_symmetry_2d(self):
        """Test that mask is symmetric about center"""
        x = torch.ones(64, 64)
        masked = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        # Check horizontal symmetry
        left_half = masked[:, :32]
        right_half = masked[:, 32:].flip(-1)
        assert torch.allclose(left_half, right_half, atol=1e-5)

        # Check vertical symmetry
        top_half = masked[:32, :]
        bottom_half = masked[32:, :].flip(-2)
        assert torch.allclose(top_half, bottom_half, atol=1e-5)

    def test_symmetry_3d(self):
        """Test that spherical mask is symmetric"""
        x = torch.ones(32, 32, 32)
        masked = mask_sphere(x, diameter=24.0, soft_edge_width=3.0)

        # Check symmetry along one axis
        left_half = masked[:, :, :16]
        right_half = masked[:, :, 16:].flip(-1)
        assert torch.allclose(left_half, right_half, atol=1e-5)

    def test_consistency_across_calls(self):
        """Test that repeated calls give same output"""
        x = torch.randn(64, 64)

        masked1 = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)
        masked2 = mask_sphere(x, diameter=40.0, soft_edge_width=5.0)

        assert torch.allclose(masked1, masked2)

    def test_raised_cosine_values(self):
        """Test that transition region has correct raised cosine values"""
        x = torch.ones(100, 100)
        diameter = 50.0
        soft_edge_width = 10.0
        radius = diameter / 2.0

        masked = mask_sphere(x, diameter=diameter, soft_edge_width=soft_edge_width)

        # Check a specific point in transition region
        # At radius + soft_edge_width/2, should be approximately 0.5
        center = 49.5
        # Point at distance = radius + soft_edge_width/2
        test_dist = radius + soft_edge_width / 2.0
        test_x = int(center + test_dist)
        test_y = int(center)

        if test_x < 100:
            # Should be approximately 0.5 (middle of cosine transition)
            assert 0.4 < masked[test_y, test_x] < 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])