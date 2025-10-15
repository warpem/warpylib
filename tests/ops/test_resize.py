"""
Tests for resize operation
"""

import torch
import pytest
import numpy as np

from warpylib.ops import resize


class TestResize2D:
    """Test 2D resize operations"""

    def test_crop_both_dimensions(self):
        """Test cropping in both dimensions"""
        # Create a 4x4 tensor with known values
        x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

        # Crop to 2x2 (should keep center)
        y = resize(x, size=(2, 2))

        assert y.shape == (1, 2, 2)
        # Center 2x2 should be elements [5, 6, 9, 10]
        expected = torch.tensor([[[5, 6], [9, 10]]], dtype=torch.float32)
        assert torch.allclose(y, expected)

    def test_pad_both_dimensions(self):
        """Test padding in both dimensions"""
        x = torch.ones(1, 2, 2, dtype=torch.float32)

        # Pad to 4x4
        y = resize(x, size=(4, 4), padding_mode='constant', padding_value=0.0)

        assert y.shape == (1, 4, 4)
        # Center 2x2 should be ones, rest zeros
        assert torch.allclose(y[0, 1:3, 1:3], torch.ones(2, 2))
        # Check corners are zero
        assert y[0, 0, 0] == 0.0
        assert y[0, 0, 3] == 0.0
        assert y[0, 3, 0] == 0.0
        assert y[0, 3, 3] == 0.0

    def test_mixed_crop_and_pad(self):
        """Test cropping one dimension while padding another"""
        x = torch.ones(1, 8, 4, dtype=torch.float32)

        # Crop height to 4, pad width to 8
        y = resize(x, size=(4, 8), padding_mode='constant', padding_value=0.0)

        assert y.shape == (1, 4, 8)
        # Center 4x4 should be ones
        assert torch.allclose(y[0, :, 2:6], torch.ones(4, 4))
        # Left and right padding should be zeros
        assert torch.allclose(y[0, :, :2], torch.zeros(4, 2))
        assert torch.allclose(y[0, :, 6:], torch.zeros(4, 2))

    def test_same_size_returns_unchanged(self):
        """Test that same size returns input unchanged"""
        x = torch.randn(2, 3, 8, 8)
        y = resize(x, size=(8, 8))

        assert y is x  # Should return same object

    def test_padding_mode_replicate(self):
        """Test replicate padding mode"""
        x = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2)
        # [[0, 1],
        #  [2, 3]]

        y = resize(x, size=(4, 4), padding_mode='replicate')

        assert y.shape == (1, 4, 4)
        # Corners should be replicated
        assert y[0, 0, 0] == 0.0  # Top-left corner
        assert y[0, 0, 3] == 1.0  # Top-right corner
        assert y[0, 3, 0] == 2.0  # Bottom-left corner
        assert y[0, 3, 3] == 3.0  # Bottom-right corner

    def test_padding_mode_reflect(self):
        """Test reflect padding mode"""
        x = torch.ones(1, 2, 2, dtype=torch.float32) * 5.0

        y = resize(x, size=(4, 4), padding_mode='reflect')

        assert y.shape == (1, 4, 4)
        # With reflect, all values should still be 5.0 (mirror of uniform values)
        assert torch.allclose(y, torch.ones(1, 4, 4) * 5.0)

    def test_custom_padding_value(self):
        """Test custom padding value"""
        x = torch.ones(1, 2, 2)

        y = resize(x, size=(4, 4), padding_mode='constant', padding_value=-1.0)

        assert y.shape == (1, 4, 4)
        # Center should be ones
        assert torch.allclose(y[0, 1:3, 1:3], torch.ones(2, 2))
        # Edges should be -1
        assert y[0, 0, 0] == -1.0

    def test_multiple_batch_dimensions(self):
        """Test with multiple batch dimensions"""
        x = torch.randn(2, 3, 4, 8, 8)

        y = resize(x, size=(4, 4))

        assert y.shape == (2, 3, 4, 4, 4)

    def test_centering_crop(self):
        """Verify that cropping keeps data centered"""
        # Create tensor with distinct corners
        x = torch.zeros(1, 8, 8)
        x[0, 0, 0] = 1.0  # Top-left
        x[0, 0, 7] = 2.0  # Top-right
        x[0, 7, 0] = 3.0  # Bottom-left
        x[0, 7, 7] = 4.0  # Bottom-right
        x[0, 3:5, 3:5] = 10.0  # Center marker

        y = resize(x, size=(4, 4))

        # Center marker should be preserved
        assert torch.allclose(y[0, 1:3, 1:3], torch.ones(2, 2) * 10.0)
        # Corners should be cropped away (zeros in result)
        assert y[0, 0, 0] == 0.0

    def test_centering_pad(self):
        """Verify that padding keeps data centered"""
        # Create 2x2 with known values
        x = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.float32)

        y = resize(x, size=(6, 6), padding_mode='constant', padding_value=0.0)

        # Original data should be at center (indices 2:4, 2:4)
        expected_center = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        assert torch.allclose(y[0, 2:4, 2:4], expected_center)


class TestResize3D:
    """Test 3D resize operations"""

    def test_crop_all_dimensions(self):
        """Test cropping in all three dimensions"""
        x = torch.randn(1, 8, 8, 8)

        y = resize(x, size=(4, 4, 4))

        assert y.shape == (1, 4, 4, 4)
        # Verify we got the center region
        expected = x[0, 2:6, 2:6, 2:6]
        assert torch.allclose(y[0], expected)

    def test_pad_all_dimensions(self):
        """Test padding in all three dimensions"""
        x = torch.ones(1, 2, 2, 2)

        y = resize(x, size=(4, 4, 4), padding_mode='constant', padding_value=0.0)

        assert y.shape == (1, 4, 4, 4)
        # Center 2x2x2 should be ones
        assert torch.allclose(y[0, 1:3, 1:3, 1:3], torch.ones(2, 2, 2))

    def test_mixed_crop_pad_3d(self):
        """Test mixed operations in 3D"""
        x = torch.ones(1, 8, 4, 6)

        # Crop depth, keep height, pad width
        y = resize(x, size=(4, 4, 8), padding_mode='constant', padding_value=0.0)

        assert y.shape == (1, 4, 4, 8)
        # Center should have original data
        assert torch.allclose(y[0, :, :, 1:7], torch.ones(4, 4, 6))

    def test_3d_centering(self):
        """Verify 3D centering behavior"""
        x = torch.zeros(1, 6, 6, 6)
        # Put marker in center
        x[0, 2:4, 2:4, 2:4] = 5.0

        y = resize(x, size=(4, 4, 4))

        # Center marker should be preserved
        assert torch.allclose(y[0, 1:3, 1:3, 1:3], torch.ones(2, 2, 2) * 5.0)

    def test_3d_multiple_batches(self):
        """Test 3D with multiple batch dimensions"""
        x = torch.randn(2, 3, 4, 4, 4)

        y = resize(x, size=(8, 8, 8), padding_mode='constant')

        assert y.shape == (2, 3, 8, 8, 8)


class TestValidation:
    """Test input validation and error handling"""

    def test_odd_current_dimension_raises(self):
        """Test that odd current dimensions raise ValueError"""
        x = torch.randn(1, 3, 4)  # Height is odd

        with pytest.raises(ValueError, match="odd size"):
            resize(x, size=(4, 4))

    def test_odd_target_dimension_raises(self):
        """Test that odd target dimensions raise ValueError"""
        x = torch.randn(1, 4, 4)

        with pytest.raises(ValueError, match="odd size"):
            resize(x, size=(3, 4))

    def test_invalid_size_length_raises(self):
        """Test that invalid size tuple length raises ValueError"""
        x = torch.randn(1, 4, 4)

        with pytest.raises(ValueError, match="size must have 2 or 3 elements"):
            resize(x, size=(4, 4, 4, 4))

    def test_insufficient_tensor_dimensions_raises(self):
        """Test that tensor with too few dimensions raises ValueError"""
        x = torch.randn(4)  # 1D tensor

        with pytest.raises(ValueError, match="Tensor has .* dimensions"):
            resize(x, size=(4, 4))

    def test_2d_size_on_3d_tensor_raises(self):
        """Test that 2D size on properly shaped 3D batched tensor works"""
        # This should actually work - a (batch, H, W) tensor with 2D size
        x = torch.randn(2, 4, 4)
        y = resize(x, size=(6, 6))
        assert y.shape == (2, 6, 6)

    def test_3d_size_on_2d_tensor_raises(self):
        """Test that 3D size on insufficient tensor raises"""
        x = torch.randn(4, 4)  # Only 2D total, need at least 3D for 3D spatial

        with pytest.raises(ValueError, match="Tensor has .* dimensions"):
            resize(x, size=(4, 4, 4))


class TestDtypeAndDevice:
    """Test dtype and device preservation"""

    def test_preserves_float32(self):
        """Test that float32 dtype is preserved"""
        x = torch.randn(1, 4, 4, dtype=torch.float32)
        y = resize(x, size=(8, 8))

        assert y.dtype == torch.float32

    def test_preserves_float64(self):
        """Test that float64 dtype is preserved"""
        x = torch.randn(1, 4, 4, dtype=torch.float64)
        y = resize(x, size=(8, 8))

        assert y.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_preserves_device_cuda(self):
        """Test that CUDA device is preserved"""
        x = torch.randn(1, 4, 4, device='cuda')
        y = resize(x, size=(8, 8))

        assert y.device.type == 'cuda'

    def test_preserves_device_cpu(self):
        """Test that CPU device is preserved"""
        x = torch.randn(1, 4, 4, device='cpu')
        y = resize(x, size=(8, 8))

        assert y.device.type == 'cpu'


class TestGradients:
    """Test gradient flow for autograd"""

    def test_gradient_flow_crop(self):
        """Test that gradients flow through crop operation"""
        x = torch.randn(1, 8, 8, requires_grad=True)
        y = resize(x, size=(4, 4))

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # Only center region should have gradients
        assert x.grad[0, 2:6, 2:6].abs().sum() > 0

    def test_gradient_flow_pad(self):
        """Test that gradients flow through pad operation"""
        x = torch.randn(1, 2, 2, requires_grad=True)
        y = resize(x, size=(4, 4), padding_mode='constant')

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_flow_replicate_pad(self):
        """Test gradients with replicate padding"""
        x = torch.randn(1, 2, 2, requires_grad=True)
        y = resize(x, size=(4, 4), padding_mode='replicate')

        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # Edges should have higher gradients due to replication
        assert x.grad.abs().sum() > 0


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_large_batch_dimensions(self):
        """Test with many batch dimensions"""
        x = torch.randn(2, 3, 4, 5, 6, 4, 4)
        y = resize(x, size=(8, 8))

        assert y.shape == (2, 3, 4, 5, 6, 8, 8)

    def test_single_element_batch(self):
        """Test with single element batches"""
        x = torch.randn(1, 1, 1, 4, 4)
        y = resize(x, size=(8, 8))

        assert y.shape == (1, 1, 1, 8, 8)

    def test_asymmetric_resize(self):
        """Test highly asymmetric resize operations"""
        x = torch.ones(1, 2, 16)
        y = resize(x, size=(16, 2))

        assert y.shape == (1, 16, 2)

    def test_very_large_padding(self):
        """Test padding to much larger size"""
        x = torch.ones(1, 2, 2)
        y = resize(x, size=(32, 32), padding_mode='constant', padding_value=0.0)

        assert y.shape == (1, 32, 32)
        # Center 2x2 should be ones
        assert torch.allclose(y[0, 15:17, 15:17], torch.ones(2, 2))

    def test_very_large_crop(self):
        """Test cropping from much larger size"""
        x = torch.randn(1, 64, 64)
        y = resize(x, size=(2, 2))

        assert y.shape == (1, 2, 2)
        # Should get center region
        expected = x[0, 31:33, 31:33]
        assert torch.allclose(y[0], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
