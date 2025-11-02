"""
Tests for rectangular CTF generation.
"""
import pytest
import torch
from warpylib.ctf import CTF


def test_square_ctf():
    """Test that CTF.get_2d works with square dimensions (original functionality)."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    ctf_square = ctf.get_2d(256)

    assert ctf_square.shape == (256, 129)
    assert -1.5 <= ctf_square.min() <= 0
    assert 0 <= ctf_square.max() <= 1.5


def test_rectangular_ctf_tall():
    """Test rectangular CTF where height > width."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    ctf_rect = ctf.get_2d((512, 256))

    assert ctf_rect.shape == (512, 129)  # height=512, width//2+1 = 129
    assert -1.5 <= ctf_rect.min() <= 0
    assert 0 <= ctf_rect.max() <= 1.5


def test_rectangular_ctf_wide():
    """Test rectangular CTF where width > height."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    ctf_rect = ctf.get_2d((256, 512))

    assert ctf_rect.shape == (256, 257)  # height=256, width//2+1 = 257
    assert -1.5 <= ctf_rect.min() <= 0
    assert 0 <= ctf_rect.max() <= 1.5


def test_rectangular_ctf_with_original_size():
    """Test rectangular CTF with original_size parameter."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    # Size (128, 256) but frequencies calculated as if from (512, 1024)
    ctf_rect = ctf.get_2d((128, 256), original_size=(512, 1024))

    assert ctf_rect.shape == (128, 129)


def test_rectangular_ctf_with_original_size_tuple():
    """Test rectangular CTF with tuple original_size."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    ctf_rect = ctf.get_2d((200, 300), original_size=(400, 600))

    assert ctf_rect.shape == (200, 151)


def test_get_ctf_coords_rectangular():
    """Test get_ctf_coords with rectangular dimensions."""
    coords_r, coords_angle = CTF.get_ctf_coords((384, 256))

    assert coords_r.shape == (384, 129)
    assert coords_angle.shape == (384, 129)
    assert coords_r.min() >= 0
    assert coords_r.max() > 0


def test_get_ctf_coords_square():
    """Test get_ctf_coords with square dimensions."""
    coords_r, coords_angle = CTF.get_ctf_coords(256)

    assert coords_r.shape == (256, 129)
    assert coords_angle.shape == (256, 129)


def test_rectangular_ctf_with_astigmatism():
    """Test rectangular CTF with astigmatism."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.defocus_delta = 0.5
    ctf.defocus_angle = 45.0
    ctf.pixel_size = 1.5

    ctf_rect = ctf.get_2d((300, 400))

    assert ctf_rect.shape == (300, 201)
    assert -1.5 <= ctf_rect.min() <= 0
    assert 0 <= ctf_rect.max() <= 1.5


def test_rectangular_ctf_amp_squared():
    """Test rectangular CTF with amp_squared option."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    ctf_rect = ctf.get_2d((256, 384), amp_squared=True)

    assert ctf_rect.shape == (256, 193)
    assert ctf_rect.min() >= 0  # amp_squared should be non-negative


def test_rectangular_ctf_batched():
    """Test rectangular CTF with batched parameters."""
    ctf = CTF()
    ctf.defocus = torch.tensor([1.0, 2.0, 3.0])  # Batch of 3
    ctf.pixel_size = 1.5

    ctf_rect = ctf.get_2d((200, 300))

    assert ctf_rect.shape == (3, 200, 151)  # (batch, height, width//2+1)


@pytest.mark.parametrize("size,expected_shape", [
    (128, (128, 65)),
    ((128, 128), (128, 65)),
    ((100, 200), (100, 101)),
    ((200, 100), (200, 51)),
    ((64, 128), (64, 65)),
])
def test_rectangular_ctf_shapes(size, expected_shape):
    """Test various rectangular CTF dimensions."""
    ctf = CTF()
    ctf.defocus = 2.0
    ctf.pixel_size = 1.5

    ctf_result = ctf.get_2d(size)

    assert ctf_result.shape == expected_shape