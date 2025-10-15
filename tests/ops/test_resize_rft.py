"""
Tests for resize_rft operation
"""

import torch
import pytest

from warpylib.ops import resize_rft


class TestResizeFT2D:
    """Test 2D Fourier space resize operations"""

    def test_fft_roundtrip_pad_crop(self):
        """Test that pad then crop in FT space is reversible"""
        # Start with FT of a real image
        x_real = torch.randn(2, 64, 64)
        x_ft_original = torch.fft.rfft2(x_real)

        # Pad in FT space
        x_ft_padded = resize_rft(x_ft_original, size=(128, 128))

        # Transform back and forth
        x_real_padded = torch.fft.irfft2(x_ft_padded, s=(128, 128))
        x_ft_padded_again = torch.fft.rfft2(x_real_padded)

        # Crop back to original size
        x_ft_recovered = resize_rft(x_ft_padded_again, size=(64, 64))
        x_recovered = torch.fft.irfft2(x_ft_recovered, s=(64, 64))

        biggest_error = (x_real - x_recovered).abs().max().item()
        print(f"Biggest error after pad/crop: {biggest_error}")

        # Should match original (within numerical precision)
        assert torch.allclose(x_real, x_recovered, atol=2e-2)

    def test_output_shape_2d(self):
        """Test output shapes are correct for 2D"""
        x_ft = torch.randn(2, 64, 33, dtype=torch.complex64)  # 64x64 in real space

        # Crop
        y = resize_rft(x_ft, size=(32, 32))
        assert y.shape == (2, 32, 17)  # 32//2+1 = 17

        # Pad
        y = resize_rft(x_ft, size=(128, 128))
        assert y.shape == (2, 128, 65)  # 128//2+1 = 65

        # Mixed
        y = resize_rft(x_ft, size=(32, 128))
        assert y.shape == (2, 32, 65)

    def test_dc_preservation_crop(self):
        """Test that DC component is preserved when cropping"""
        x_ft = torch.randn(1, 64, 33, dtype=torch.complex64)
        dc_original = x_ft[0, 0, 0].clone()

        y_ft = resize_rft(x_ft, size=(32, 32))

        # DC should be at [0, 0] and unchanged
        assert torch.allclose(y_ft[0, 0, 0], dc_original)

    def test_dc_preservation_pad(self):
        """Test that DC component is preserved when padding"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.complex64)
        dc_original = x_ft[0, 0, 0].clone()

        y_ft = resize_rft(x_ft, size=(64, 64))

        # DC should still be at [0, 0] and unchanged
        assert torch.allclose(y_ft[0, 0, 0], dc_original)

    def test_real_valued_tensor(self):
        """Test with real-valued tensor (e.g., CTF)"""
        # Simulate a real-valued CTF in Fourier space
        ctf_ft = torch.randn(2, 64, 33)  # Real, not complex

        y = resize_rft(ctf_ft, size=(32, 32))

        assert y.shape == (2, 32, 17)
        assert y.dtype == torch.float32  # Should preserve real dtype

    def test_complex_valued_tensor(self):
        """Test with complex-valued tensor (actual RFFT output)"""
        x_ft = torch.randn(2, 64, 33, dtype=torch.complex64)

        y = resize_rft(x_ft, size=(32, 32))

        assert y.shape == (2, 32, 17)
        assert y.dtype == torch.complex64  # Should preserve complex dtype

    def test_same_size_returns_unchanged(self):
        """Test that same size returns input unchanged"""
        x_ft = torch.randn(2, 64, 33, dtype=torch.complex64)

        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft is x_ft  # Should return same object

    def test_custom_padding_value_real(self):
        """Test custom real padding value"""
        x_ft = torch.ones(1, 32, 17)

        y_ft = resize_rft(x_ft, size=(64, 64), padding_value=5.0)

        # Check that padded regions have the custom value
        # In the height dimension, padding should be in the middle
        # For unshifted FFT, this is at indices [16:48]
        assert torch.allclose(y_ft[0, 16:48, :], torch.ones(32, 33) * 5.0)

    def test_custom_padding_value_complex(self):
        """Test custom complex padding value"""
        x_ft = torch.ones(1, 32, 17, dtype=torch.complex64)

        y_ft = resize_rft(x_ft, size=(64, 64), padding_value=1.0 + 2.0j)

        # Padded regions should have the custom complex value
        assert torch.allclose(y_ft[0, 16:48, :], torch.ones(32, 33) * (1.0 + 2.0j))


class TestResizeFT3D:
    """Test 3D Fourier space resize operations"""

    def test_fft_roundtrip_pad_crop_3d(self):
        """Test 3D pad then crop is reversible"""
        # Start with FT of a real volume
        x_real = torch.randn(2, 32, 32, 32)
        x_ft_original = torch.fft.rfftn(x_real, dim=(-3, -2, -1))

        # Pad in FT space
        x_ft_padded = resize_rft(x_ft_original, size=(64, 64, 64))

        # Transform back and forth
        x_real_padded = torch.fft.irfftn(x_ft_padded, s=(64, 64, 64), dim=(-3, -2, -1))
        x_ft_padded_again = torch.fft.rfftn(x_real_padded, dim=(-3, -2, -1))

        # Crop back to original size
        x_ft_recovered = resize_rft(x_ft_padded_again, size=(32, 32, 32))
        x_recovered = torch.fft.irfftn(x_ft_recovered, s=(32, 32, 32), dim=(-3, -2, -1))

        biggest_error = (x_real - x_recovered).abs().max().item()
        print(f"Biggest error after pad/crop: {biggest_error}")

        # Should match original
        assert torch.allclose(x_real, x_recovered, atol=1e-1)

    def test_output_shape_3d(self):
        """Test output shapes are correct for 3D"""
        x_ft = torch.randn(2, 64, 64, 33, dtype=torch.complex64)

        # Crop
        y = resize_rft(x_ft, size=(32, 32, 32))
        assert y.shape == (2, 32, 32, 17)

        # Pad
        y = resize_rft(x_ft, size=(128, 128, 128))
        assert y.shape == (2, 128, 128, 65)

        # Mixed
        y = resize_rft(x_ft, size=(32, 128, 64))
        assert y.shape == (2, 32, 128, 33)

    def test_dc_preservation_3d(self):
        """Test DC preservation in 3D"""
        x_ft = torch.randn(1, 64, 64, 33, dtype=torch.complex64)
        dc_original = x_ft[0, 0, 0, 0].clone()

        # Crop
        y_ft = resize_rft(x_ft, size=(32, 32, 32))
        assert torch.allclose(y_ft[0, 0, 0, 0], dc_original)

        # Pad
        y_ft = resize_rft(x_ft, size=(128, 128, 128))
        assert torch.allclose(y_ft[0, 0, 0, 0], dc_original)

    def test_3d_multiple_batches(self):
        """Test 3D with multiple batch dimensions"""
        x_ft = torch.randn(2, 3, 4, 32, 32, 17, dtype=torch.complex64)

        y_ft = resize_rft(x_ft, size=(64, 64, 64))

        assert y_ft.shape == (2, 3, 4, 64, 64, 33)


class TestValidation:
    """Test input validation and error handling"""

    def test_odd_height_current_raises(self):
        """Test that odd non-RFFT current dimension raises"""
        x_ft = torch.randn(1, 33, 17)  # Height is odd

        with pytest.raises(ValueError, match="odd size"):
            resize_rft(x_ft, size=(32, 32))

    def test_odd_height_target_raises(self):
        """Test that odd non-RFFT target dimension raises"""
        x_ft = torch.randn(1, 32, 17)

        with pytest.raises(ValueError, match="odd size"):
            resize_rft(x_ft, size=(33, 32))

    def test_rfft_dimension_not_validated(self):
        """Test that current RFFT dimension is not validated for evenness"""
        # Create tensor with "wrong" RFFT dimension (e.g., from odd original size)
        # This should still work since we don't validate the current RFFT dim
        x_ft = torch.randn(1, 32, 18)  # Could be from 34x34 image (odd)

        # Should work without error
        y_ft = resize_rft(x_ft, size=(32, 32))
        assert y_ft.shape == (1, 32, 17)

    def test_invalid_size_length_raises(self):
        """Test that invalid size tuple length raises"""
        x_ft = torch.randn(1, 32, 17)

        with pytest.raises(ValueError, match="size must have 2 or 3 elements"):
            resize_rft(x_ft, size=(32, 32, 32, 32))

    def test_insufficient_dimensions_raises(self):
        """Test that tensor with too few dimensions raises"""
        x_ft = torch.randn(17)  # 1D tensor

        with pytest.raises(ValueError, match="Tensor has .* dimensions"):
            resize_rft(x_ft, size=(32, 32))


class TestFrequencyPreservation:
    """Test that low frequencies are preserved correctly"""

    def test_high_frequency_removal_crop(self):
        """Test that high frequencies are removed when cropping"""
        # Create signal with high frequency noise
        x_real = torch.randn(1, 128, 128)

        x_ft = torch.fft.rfft2(x_real)

        # Crop heavily to remove high frequencies
        x_ft_cropped = resize_rft(x_ft, size=(16, 16))

        # Pad back to original size
        x_ft_padded = resize_rft(x_ft_cropped, size=(128, 128))

        # High frequencies should be zero
        # Check that high frequency region is zero
        # For unshifted FFT, high freqs are in middle and at Nyquist
        assert torch.allclose(
            x_ft_padded[0, 8:120, :],  # Middle frequencies in height
            torch.zeros(112, 65, dtype=torch.complex64),
            atol=1e-6
        )


class TestDtypeAndDevice:
    """Test dtype and device preservation"""

    def test_preserves_complex64(self):
        """Test that complex64 dtype is preserved"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.complex64)
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.dtype == torch.complex64

    def test_preserves_complex128(self):
        """Test that complex128 dtype is preserved"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.complex128)
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.dtype == torch.complex128

    def test_preserves_float32(self):
        """Test that float32 dtype is preserved for real tensors"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.float32)
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.dtype == torch.float32

    def test_preserves_float64(self):
        """Test that float64 dtype is preserved for real tensors"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.float64)
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_preserves_device_cuda(self):
        """Test that CUDA device is preserved"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.complex64, device='cuda')
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.device.type == 'cuda'

    def test_preserves_device_cpu(self):
        """Test that CPU device is preserved"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.complex64, device='cpu')
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.device.type == 'cpu'


class TestGradients:
    """Test gradient flow for autograd"""

    def test_gradient_flow_crop(self):
        """Test that gradients flow through crop operation"""
        x_ft = torch.randn(1, 64, 33, dtype=torch.complex64, requires_grad=True)
        y_ft = resize_rft(x_ft, size=(32, 32))

        loss = y_ft.abs().sum()
        loss.backward()

        assert x_ft.grad is not None
        assert x_ft.grad.abs().sum() > 0

    def test_gradient_flow_pad(self):
        """Test that gradients flow through pad operation"""
        x_ft = torch.randn(1, 32, 17, dtype=torch.complex64, requires_grad=True)
        y_ft = resize_rft(x_ft, size=(64, 64))

        loss = y_ft.abs().sum()
        loss.backward()

        assert x_ft.grad is not None
        assert x_ft.grad.abs().sum() > 0

    def test_gradient_flow_real_tensor(self):
        """Test gradients with real-valued tensor"""
        x_ft = torch.randn(1, 32, 17, requires_grad=True)
        y_ft = resize_rft(x_ft, size=(64, 64))

        loss = y_ft.sum()
        loss.backward()

        assert x_ft.grad is not None
        assert x_ft.grad.abs().sum() > 0


class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_large_batch_dimensions(self):
        """Test with many batch dimensions"""
        x_ft = torch.randn(2, 3, 4, 5, 32, 17, dtype=torch.complex64)
        y_ft = resize_rft(x_ft, size=(64, 64))

        assert y_ft.shape == (2, 3, 4, 5, 64, 33)

    def test_asymmetric_resize(self):
        """Test highly asymmetric resize operations"""
        x_ft = torch.randn(1, 16, 65, dtype=torch.complex64)
        y_ft = resize_rft(x_ft, size=(128, 16))

        assert y_ft.shape == (1, 128, 9)

    def test_very_large_padding(self):
        """Test padding to much larger size"""
        x_ft = torch.randn(1, 16, 9, dtype=torch.complex64)
        y_ft = resize_rft(x_ft, size=(256, 256))

        assert y_ft.shape == (1, 256, 129)
        # DC should be preserved
        assert y_ft[0, 0, 0] == x_ft[0, 0, 0]

    def test_very_large_crop(self):
        """Test cropping from much larger size"""
        x_ft = torch.randn(1, 256, 129, dtype=torch.complex64)
        y_ft = resize_rft(x_ft, size=(16, 16))

        assert y_ft.shape == (1, 16, 9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
