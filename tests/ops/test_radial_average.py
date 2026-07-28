"""
Tests for radial_average_rft operation
"""

import torch
import pytest

from warpylib.ops import radial_average_rft


class TestRadialAverageRft:
    def test_shapes_2d(self):
        power = torch.rand(3, 64, 33)  # rfft layout of a 64x64 image, batch 3
        profile, counts = radial_average_rft(power, image_shape=(64, 64))
        assert profile.shape == (3, 33)
        assert counts.shape == (33,)
        assert counts.dtype == torch.int64

    def test_shapes_3d(self):
        power = torch.rand(16, 16, 9)  # rfft layout of 16^3
        profile, counts = radial_average_rft(power, image_shape=(16, 16, 16))
        assert profile.shape == (9,)
        assert counts.shape == (9,)

    def test_dc_shell_isolated(self):
        power = torch.zeros(8, 5)
        power[0, 0] = 7.0  # DC pixel only
        profile, counts = radial_average_rft(power, image_shape=(8, 8))
        assert counts[0] == 1
        assert profile[0] == pytest.approx(7.0)

    def test_constant_field_averages_to_constant(self):
        power = torch.ones(64, 33)
        profile, counts = radial_average_rft(power, image_shape=(64, 64))
        # Every shell that has pixels must average to exactly 1.0.
        assert torch.allclose(profile[counts > 0], torch.ones_like(profile[counts > 0]))

    def test_radius_function_recovered(self):
        # Build a field whose value equals its integer shell radius, then check
        # the radial average returns that radius for populated shells.
        h, w = 32, 32
        ky = torch.fft.fftfreq(h) * h
        kx = torch.arange(w // 2 + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ky, kx, indexing="ij")
        shell = torch.round(torch.sqrt(yy**2 + xx**2))
        profile, counts = radial_average_rft(shell, image_shape=(h, w))
        populated = counts > 0
        expected = torch.arange(w // 2 + 1, dtype=profile.dtype)
        assert torch.allclose(profile[populated], expected[populated], atol=1e-5)

    def test_rejects_odd_image_shape(self):
        with pytest.raises(ValueError, match="even"):
            radial_average_rft(torch.zeros(7, 5), image_shape=(7, 8))

    def test_rejects_mismatched_shape(self):
        with pytest.raises(ValueError, match="does not match"):
            radial_average_rft(torch.zeros(64, 64), image_shape=(64, 64))
