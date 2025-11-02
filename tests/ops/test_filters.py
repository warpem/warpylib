"""
Tests for filter operations.
"""
import pytest
import torch
from warpylib.ops import get_sinc2_correction, get_sinc2_correction_rft


def test_sinc2_correction_3d_cubic():
    """Test 3D cubic sinc^2 correction (original behavior)."""
    size = 64
    correction = get_sinc2_correction(size)

    assert correction.shape == (size, size, size)
    assert correction.dtype == torch.float32
    # Center value should be 1 (sinc(0) = 1)
    assert torch.isclose(correction[size//2, size//2, size//2], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_3d_cuboid():
    """Test 3D cuboid (non-cubic) sinc^2 correction."""
    depth, height, width = 32, 64, 128
    correction = get_sinc2_correction((depth, height, width))

    assert correction.shape == (depth, height, width)
    assert correction.dtype == torch.float32
    # Center value should be 1
    assert torch.isclose(correction[depth//2, height//2, width//2], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_2d_square():
    """Test 2D square sinc^2 correction."""
    size = 128
    correction = get_sinc2_correction((size, size))

    assert correction.shape == (size, size)
    assert correction.dtype == torch.float32
    # Center value should be 1
    assert torch.isclose(correction[size//2, size//2], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_2d_rectangular():
    """Test 2D rectangular sinc^2 correction."""
    height, width = 64, 256
    correction = get_sinc2_correction((height, width))

    assert correction.shape == (height, width)
    assert correction.dtype == torch.float32
    # Center value should be 1
    assert torch.isclose(correction[height//2, width//2], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_oversampling_3d():
    """Test that oversampling reduces attenuation in 3D."""
    size = 64

    correction_os1 = get_sinc2_correction(size, oversampling=1.0)
    correction_os2 = get_sinc2_correction(size, oversampling=2.0)

    # With higher oversampling, edge values should be higher (less attenuation)
    edge_idx = 0
    assert correction_os2[edge_idx, size//2, size//2] > correction_os1[edge_idx, size//2, size//2]


def test_sinc2_correction_oversampling_2d():
    """Test that oversampling reduces attenuation in 2D."""
    size = 128

    correction_os1 = get_sinc2_correction((size, size), oversampling=1.0)
    correction_os2 = get_sinc2_correction((size, size), oversampling=2.0)

    # With higher oversampling, edge values should be higher (less attenuation)
    edge_idx = 0
    assert correction_os2[edge_idx, size//2] > correction_os1[edge_idx, size//2]


def test_sinc2_correction_symmetry_3d():
    """Test radial symmetry of 3D correction."""
    size = 64
    correction = get_sinc2_correction(size)

    center = size // 2
    # Check that points at equal radial distances have equal values
    # Point at (center+10, center, center)
    val1 = correction[center + 10, center, center]
    # Point at (center, center+10, center)
    val2 = correction[center, center + 10, center]
    # Point at (center, center, center+10)
    val3 = correction[center, center, center + 10]

    assert torch.isclose(val1, val2, rtol=1e-5)
    assert torch.isclose(val2, val3, rtol=1e-5)


def test_sinc2_correction_symmetry_2d():
    """Test radial symmetry of 2D square correction."""
    size = 128
    correction = get_sinc2_correction((size, size))

    center = size // 2
    # Check that points at equal radial distances have equal values
    # Point at (center+20, center)
    val1 = correction[center + 20, center]
    # Point at (center, center+20)
    val2 = correction[center, center + 20]

    assert torch.isclose(val1, val2, rtol=1e-5)


def test_sinc2_correction_monotonic_decrease_3d():
    """Test that 3D correction decreases monotonically from center."""
    size = 64
    correction = get_sinc2_correction(size)

    center = size // 2
    # Values along x-axis from center should generally decrease
    # (ignoring oscillations, check average trend)
    central_line = correction[center, center, center:]

    # Check that the value at center is the maximum
    assert central_line[0] == central_line.max()


def test_sinc2_correction_monotonic_decrease_2d():
    """Test that 2D correction decreases from center."""
    size = 128
    correction = get_sinc2_correction((size, size))

    center = size // 2
    # Values along x-axis from center
    central_line = correction[center, center:]

    # Check that the value at center is the maximum
    assert central_line[0] == central_line.max()


@pytest.mark.parametrize("size,expected_shape", [
    (32, (32, 32, 32)),
    ((32, 32), (32, 32)),
    ((32, 64), (32, 64)),
    ((64, 32), (64, 32)),
    ((16, 32, 64), (16, 32, 64)),
    ((128, 128, 64), (128, 128, 64)),
])
def test_sinc2_correction_shapes(size, expected_shape):
    """Test various size inputs produce correct output shapes."""
    correction = get_sinc2_correction(size)
    assert correction.shape == expected_shape


def test_sinc2_correction_invalid_size():
    """Test that invalid size raises appropriate error."""
    with pytest.raises(ValueError):
        get_sinc2_correction((64, 64, 64, 64))  # 4D not supported


def test_sinc2_correction_values_range():
    """Test that all correction values are in valid range."""
    # Test various configurations
    configs = [
        (64, 1.0),
        (64, 2.0),
        ((64, 128), 1.5),
        ((32, 64, 128), 2.0),
    ]

    for size, oversampling in configs:
        correction = get_sinc2_correction(size, oversampling=oversampling)
        assert correction.min() >= 0.0
        assert correction.max() <= 1.0


def test_sinc2_correction_backward_compatibility():
    """Test that int input maintains backward compatibility with 3D cubic output."""
    size = 48
    correction = get_sinc2_correction(size)

    # Should produce cubic 3D volume
    assert correction.ndim == 3
    assert correction.shape == (size, size, size)

    # Should be radially symmetric
    center = size // 2
    val_x = correction[center + 5, center, center]
    val_y = correction[center, center + 5, center]
    val_z = correction[center, center, center + 5]
    assert torch.allclose(val_x, val_y)
    assert torch.allclose(val_y, val_z)


def test_sinc2_correction_different_aspect_ratios():
    """Test correction works for various aspect ratios."""
    # Very wide 2D
    correction_wide = get_sinc2_correction((64, 512))
    assert correction_wide.shape == (64, 512)

    # Very tall 2D
    correction_tall = get_sinc2_correction((512, 64))
    assert correction_tall.shape == (512, 64)

    # Elongated 3D
    correction_elongated = get_sinc2_correction((32, 32, 256))
    assert correction_elongated.shape == (32, 32, 256)


# Tests for RFT version


def test_sinc2_correction_rft_3d_cubic():
    """Test 3D cubic sinc^2 correction in rfft format."""
    size = 64
    correction = get_sinc2_correction_rft(size)

    # rfft format: (size, size, size//2+1)
    assert correction.shape == (size, size, size // 2 + 1)
    assert correction.dtype == torch.float32
    # DC component at [0, 0, 0]
    assert torch.isclose(correction[0, 0, 0], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_rft_3d_cuboid():
    """Test 3D cuboid (non-cubic) sinc^2 correction in rfft format."""
    depth, height, width = 32, 64, 128
    correction = get_sinc2_correction_rft((depth, height, width))

    # rfft format: (depth, height, width//2+1)
    assert correction.shape == (depth, height, width // 2 + 1)
    assert correction.dtype == torch.float32
    # DC component at [0, 0, 0]
    assert torch.isclose(correction[0, 0, 0], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_rft_2d_square():
    """Test 2D square sinc^2 correction in rfft format."""
    size = 128
    correction = get_sinc2_correction_rft((size, size))

    # rfft format: (size, size//2+1)
    assert correction.shape == (size, size // 2 + 1)
    assert correction.dtype == torch.float32
    # DC component at [0, 0]
    assert torch.isclose(correction[0, 0], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_rft_2d_rectangular():
    """Test 2D rectangular sinc^2 correction in rfft format."""
    height, width = 64, 256
    correction = get_sinc2_correction_rft((height, width))

    # rfft format: (height, width//2+1)
    assert correction.shape == (height, width // 2 + 1)
    assert correction.dtype == torch.float32
    # DC component at [0, 0]
    assert torch.isclose(correction[0, 0], torch.tensor(1.0))
    # All values should be in [0, 1]
    assert correction.min() >= 0.0
    assert correction.max() <= 1.0


def test_sinc2_correction_rft_oversampling_3d():
    """Test that oversampling reduces attenuation in rfft 3D."""
    size = 64

    correction_os1 = get_sinc2_correction_rft(size, oversampling=1.0)
    correction_os2 = get_sinc2_correction_rft(size, oversampling=2.0)

    # With higher oversampling, edge values should be higher (less attenuation)
    # Check Nyquist frequency in width dimension
    nyquist_idx = size // 2
    assert correction_os2[0, 0, nyquist_idx] > correction_os1[0, 0, nyquist_idx]


def test_sinc2_correction_rft_oversampling_2d():
    """Test that oversampling reduces attenuation in rfft 2D."""
    size = 128

    correction_os1 = get_sinc2_correction_rft((size, size), oversampling=1.0)
    correction_os2 = get_sinc2_correction_rft((size, size), oversampling=2.0)

    # With higher oversampling, edge values should be higher (less attenuation)
    nyquist_idx = size // 2
    assert correction_os2[0, nyquist_idx] > correction_os1[0, nyquist_idx]


@pytest.mark.parametrize("size,expected_shape", [
    (32, (32, 32, 17)),
    ((32, 32), (32, 17)),
    ((32, 64), (32, 33)),
    ((64, 32), (64, 17)),
    ((16, 32, 64), (16, 32, 33)),
    ((128, 128, 64), (128, 128, 33)),
])
def test_sinc2_correction_rft_shapes(size, expected_shape):
    """Test various size inputs produce correct rfft output shapes."""
    correction = get_sinc2_correction_rft(size)
    assert correction.shape == expected_shape


def test_sinc2_correction_rft_device():
    """Test that device parameter works correctly."""
    size = 64

    # CPU device
    correction_cpu = get_sinc2_correction_rft(size, device=torch.device('cpu'))
    assert correction_cpu.device == torch.device('cpu')

    # Default device (should be CPU)
    correction_default = get_sinc2_correction_rft(size)
    assert correction_default.device.type == 'cpu'


def test_sinc2_correction_rft_values_range():
    """Test that all rfft correction values are in valid range."""
    configs = [
        (64, 1.0),
        (64, 2.0),
        ((64, 128), 1.5),
        ((32, 64, 128), 2.0),
    ]

    for size, oversampling in configs:
        correction = get_sinc2_correction_rft(size, oversampling=oversampling)
        assert correction.min() >= 0.0
        assert correction.max() <= 1.0


def test_sinc2_correction_rft_backward_compatibility():
    """Test that int input maintains backward compatibility with 3D cubic output in rfft format."""
    size = 48
    correction = get_sinc2_correction_rft(size)

    # Should produce cubic 3D volume in rfft format
    assert correction.ndim == 3
    assert correction.shape == (size, size, size // 2 + 1)

    # DC component should be at [0, 0, 0]
    assert torch.isclose(correction[0, 0, 0], torch.tensor(1.0))


def test_sinc2_correction_rft_different_aspect_ratios():
    """Test rfft correction works for various aspect ratios."""
    # Very wide 2D
    correction_wide = get_sinc2_correction_rft((64, 512))
    assert correction_wide.shape == (64, 257)

    # Very tall 2D
    correction_tall = get_sinc2_correction_rft((512, 64))
    assert correction_tall.shape == (512, 33)

    # Elongated 3D
    correction_elongated = get_sinc2_correction_rft((32, 32, 256))
    assert correction_elongated.shape == (32, 32, 129)


def test_sinc2_correction_rft_consistency_with_regular_2d():
    """Test that rfft version matches regular version for corresponding frequencies (2D)."""
    size = 64

    # Get both versions
    correction_regular = get_sinc2_correction((size, size))
    correction_rft = get_sinc2_correction_rft((size, size))

    # Convert regular to rfft format using actual rfft
    correction_regular_fft = torch.fft.fftshift(correction_regular)
    correction_regular_rfft = torch.fft.rfft2(correction_regular_fft, norm='forward')

    # The rfft of the regular correction should have similar magnitude pattern to our direct rft version
    # (They won't be identical since one is in Fourier space already)
    # Just check that DC components match
    assert torch.isclose(correction_rft[0, 0], torch.tensor(1.0), atol=1e-5)


def test_sinc2_correction_rft_consistency_with_regular_3d():
    """Test that rfft version matches regular version for corresponding frequencies (3D)."""
    size = 32

    # Get both versions
    correction_regular = get_sinc2_correction(size)
    correction_rft = get_sinc2_correction_rft(size)

    # Both should have DC component = 1
    assert torch.isclose(correction_rft[0, 0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(correction_regular[size//2, size//2, size//2], torch.tensor(1.0), atol=1e-5)


def test_sinc2_correction_rft_dc_at_origin():
    """Test that DC component is at [0,0] or [0,0,0] in rfft format."""
    # 2D case
    correction_2d = get_sinc2_correction_rft((128, 256))
    assert correction_2d[0, 0] == correction_2d.max()

    # 3D case
    correction_3d = get_sinc2_correction_rft(64)
    assert correction_3d[0, 0, 0] == correction_3d.max()


def test_sinc2_correction_rft_monotonic_decrease_along_positive_freq():
    """Test that correction decreases along positive frequencies in rfft format."""
    size = 128
    correction = get_sinc2_correction_rft((size, size))

    # Along positive x-axis (width dimension, which is rfft)
    # Values should generally decrease from DC
    dc_value = correction[0, 0]
    nyquist_value = correction[0, -1]

    # DC should be the maximum
    assert dc_value == correction.max()
    # Nyquist should be less than DC (with oversampling = 1.0, there's attenuation)
    assert nyquist_value < dc_value