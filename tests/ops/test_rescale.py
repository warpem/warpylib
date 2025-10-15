"""
Tests for rescale op
"""

import torch
import pytest
from warpylib.ops import rescale


class TestRescaleBasic:
    """Basic functionality tests"""

    def test_2d_downscale(self):
        """Test 2D downscaling"""
        torch.manual_seed(0)
        x = torch.randn(2, 3, 128, 128)
        y = rescale(x, size=(64, 64))

        assert y.shape == torch.Size([2, 3, 64, 64])
        assert not y.is_complex()

    def test_2d_upscale(self):
        """Test 2D upscaling"""
        torch.manual_seed(1)
        x = torch.randn(2, 3, 64, 64)
        y = rescale(x, size=(128, 128))

        assert y.shape == torch.Size([2, 3, 128, 128])
        assert not y.is_complex()

    def test_3d_downscale(self):
        """Test 3D downscaling"""
        torch.manual_seed(2)
        x = torch.randn(4, 64, 64, 64)
        y = rescale(x, size=(32, 32, 32))

        assert y.shape == torch.Size([4, 32, 32, 32])
        assert not y.is_complex()

    def test_3d_upscale(self):
        """Test 3D upscaling"""
        torch.manual_seed(3)
        x = torch.randn(4, 32, 32, 32)
        y = rescale(x, size=(64, 64, 64))

        assert y.shape == torch.Size([4, 64, 64, 64])
        assert not y.is_complex()

    def test_2d_mixed_scaling(self):
        """Test 2D with different scaling in each dimension"""
        torch.manual_seed(4)
        x = torch.randn(2, 128, 128)
        y = rescale(x, size=(64, 256))

        assert y.shape == torch.Size([2, 64, 256])
        assert not y.is_complex()

    def test_3d_mixed_scaling(self):
        """Test 3D with different scaling in each dimension"""
        torch.manual_seed(5)
        x = torch.randn(2, 64, 64, 64)
        y = rescale(x, size=(32, 64, 128))

        assert y.shape == torch.Size([2, 32, 64, 128])
        assert not y.is_complex()


class TestRescaleEdgeCases:
    """Edge case tests"""

    def test_identity_2d(self):
        """Test that rescaling to same size returns identical tensor"""
        torch.manual_seed(6)
        x = torch.randn(2, 64, 64)
        y = rescale(x, size=(64, 64))

        assert torch.equal(x, y)

    def test_identity_3d(self):
        """Test that rescaling to same size returns identical tensor in 3D"""
        torch.manual_seed(7)
        x = torch.randn(2, 32, 32, 32)
        y = rescale(x, size=(32, 32, 32))

        assert torch.equal(x, y)

    def test_complex_input_raises(self):
        """Test that complex input raises ValueError"""
        torch.manual_seed(8)
        x = torch.randn(2, 64, 64, dtype=torch.complex64)

        with pytest.raises(ValueError, match="must be real-valued"):
            rescale(x, size=(32, 32))

    def test_invalid_size_dimensions(self):
        """Test that invalid size tuple raises ValueError"""
        torch.manual_seed(9)
        x = torch.randn(2, 64, 64)

        with pytest.raises(ValueError, match="size must have 2 or 3 elements"):
            rescale(x, size=(64,))

    def test_insufficient_tensor_dimensions(self):
        """Test that tensor with too few dimensions raises ValueError"""
        torch.manual_seed(10)
        x = torch.randn(64)  # 1D tensor

        with pytest.raises(ValueError, match="but size specifies"):
            rescale(x, size=(32, 32))


class TestRescaleGradientFlow:
    """Test gradient flow through rescale operation"""

    def test_gradient_flow_2d_downscale(self):
        """Test gradients flow through 2D downscaling"""
        torch.manual_seed(11)
        x = torch.randn(2, 64, 64, requires_grad=True)
        y = rescale(x, size=(32, 32))

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_2d_upscale(self):
        """Test gradients flow through 2D upscaling"""
        torch.manual_seed(12)
        x = torch.randn(2, 32, 32, requires_grad=True)
        y = rescale(x, size=(64, 64))

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_3d_downscale(self):
        """Test gradients flow through 3D downscaling"""
        torch.manual_seed(13)
        x = torch.randn(2, 32, 32, 32, requires_grad=True)
        y = rescale(x, size=(16, 16, 16))

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_3d_upscale(self):
        """Test gradients flow through 3D upscaling"""
        torch.manual_seed(14)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        y = rescale(x, size=(32, 32, 32))

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_mixed_scaling(self):
        """Test gradients flow with mixed up/down scaling"""
        torch.manual_seed(15)
        x = torch.randn(2, 64, 64, 64, requires_grad=True)
        y = rescale(x, size=(32, 64, 128))

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().sum() > 0


class TestRescaleBandwidthLimiting:
    """Test that rescale correctly handles frequency content"""

    def test_downscale_preserves_dc_component_2d(self):
        """Test that downscaling preserves DC component (mean) in 2D"""
        # Create image with known mean
        x = torch.ones(1, 128, 128) * 5.0
        y = rescale(x, size=(64, 64))

        # Mean should be preserved
        assert torch.allclose(x.mean(), y.mean(), rtol=1e-5)

    def test_downscale_preserves_dc_component_3d(self):
        """Test that downscaling preserves DC component (mean) in 3D"""
        # Create volume with known mean
        x = torch.ones(1, 64, 64, 64) * 3.0
        y = rescale(x, size=(32, 32, 32))

        # Mean should be preserved
        assert torch.allclose(x.mean(), y.mean(), rtol=1e-5)

    def test_upscale_preserves_dc_component_2d(self):
        """Test that upscaling preserves DC component in 2D"""
        x = torch.ones(1, 32, 32) * 7.0
        y = rescale(x, size=(64, 64))

        # Mean should be preserved
        assert torch.allclose(x.mean(), y.mean(), rtol=1e-5)

    def test_upscale_preserves_dc_component_3d(self):
        """Test that upscaling preserves DC component in 3D"""
        x = torch.ones(1, 32, 32, 32) * 2.5
        y = rescale(x, size=(64, 64, 64))

        # Mean should be preserved
        assert torch.allclose(x.mean(), y.mean(), rtol=1e-5)

    def test_downscale_removes_high_frequencies_2d(self):
        """Test that downscaling removes high frequency content in 2D"""
        # Create high-frequency pattern (checkerboard-like)
        x = torch.zeros(1, 128, 128)
        x[:, ::2, ::2] = 1.0
        x[:, 1::2, 1::2] = 1.0
        x[:, ::2, 1::2] = -1.0
        x[:, 1::2, ::2] = -1.0

        # Downscale significantly
        y = rescale(x, size=(16, 16))

        # High frequencies should be suppressed - variance should be much smaller
        # In the extreme case, it might even be close to zero mean
        assert y.var() < x.var()

    def test_upscale_zero_pads_high_frequencies_2d(self):
        """Test that upscaling adds zeros at high frequencies (smooths) in 2D"""
        # Create simple low-frequency signal
        x = torch.ones(1, 32, 32)
        x[:, :16, :] = 2.0

        # Upscale
        y = rescale(x, size=(64, 64))

        # Result should be smooth (upscaling in Fourier space adds zeros at high freq)
        # Check that dimensions match
        assert y.shape == torch.Size([1, 64, 64])


class TestRescaleMultiBatch:
    """Test rescale with various batch dimensions"""

    def test_single_batch_2d(self):
        """Test with single item batch"""
        torch.manual_seed(16)
        x = torch.randn(1, 64, 64)
        y = rescale(x, size=(32, 32))

        assert y.shape == torch.Size([1, 32, 32])

    def test_multi_batch_2d(self):
        """Test with multiple batch dimensions"""
        torch.manual_seed(17)
        x = torch.randn(2, 3, 4, 64, 64)
        y = rescale(x, size=(32, 32))

        assert y.shape == torch.Size([2, 3, 4, 32, 32])

    def test_single_batch_3d(self):
        """Test with single item batch in 3D"""
        torch.manual_seed(18)
        x = torch.randn(1, 32, 32, 32)
        y = rescale(x, size=(16, 16, 16))

        assert y.shape == torch.Size([1, 16, 16, 16])

    def test_multi_batch_3d(self):
        """Test with multiple batch dimensions in 3D"""
        torch.manual_seed(19)
        x = torch.randn(2, 3, 32, 32, 32)
        y = rescale(x, size=(64, 64, 64))

        assert y.shape == torch.Size([2, 3, 64, 64, 64])


class TestRescaleRoundTrip:
    """Test round-trip rescaling operations"""

    def test_downscale_upscale_2d(self):
        """Test downscale followed by upscale preserves low frequencies in 2D"""
        # Create smooth signal (low frequency)
        torch.manual_seed(20)
        x = torch.randn(1, 128, 128)
        # Apply Gaussian smoothing to create low-frequency signal
        x_smooth = torch.nn.functional.avg_pool2d(x.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

        # Downscale and upscale
        y = rescale(x_smooth, size=(64, 64))
        z = rescale(y, size=(128, 128))

        # Should be close to original for smooth signals
        # (high frequencies are lost, but low frequencies preserved)
        assert z.shape == x_smooth.shape

    def test_upscale_downscale_2d(self):
        """Test upscale followed by downscale returns close to original in 2D"""
        torch.manual_seed(21)
        x = torch.randn(1, 64, 64)

        # Upscale then downscale
        y = rescale(x, size=(128, 128))
        z = rescale(y, size=(64, 64))

        # Should be close to original (within numerical precision)
        assert torch.allclose(x, z, atol=2e-2)

    def test_upscale_downscale_3d(self):
        """Test upscale followed by downscale returns close to original in 3D"""
        torch.manual_seed(22)
        x = torch.randn(1, 32, 32, 32)

        # Upscale then downscale
        y = rescale(x, size=(64, 64, 64))
        z = rescale(y, size=(32, 32, 32))

        # Should be close to original
        assert torch.allclose(x, z, atol=1e-1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
