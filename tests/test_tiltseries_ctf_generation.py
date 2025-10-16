"""
Tests for TiltSeries CTF generation methods
"""

import pytest
import torch

from warpylib.tilt_series import TiltSeries


class TestGetCTFsForParticles:
    """Test get_ctfs_for_particles and get_ctfs_for_particles_single methods"""

    def test_single_particle_single_position(self):
        """Test CTF generation for single particle with single position"""
        from warpylib.cubic_grid import CubicGrid

        # Create simple tilt series with 3 tilts
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set up CTF
        ts.ctf.voltage = 300.0
        ts.ctf.cs = 2.7
        ts.ctf.amplitude = 0.07

        # Set up defocus grid with reasonable values
        ts.grid_ctf_defocus = CubicGrid(
            dimensions=(1, 1, 3),
            values=torch.tensor([[[2.0]], [[2.5]], [[3.0]]])  # Defocus in µm
        )

        # Single coordinate: (3,)
        coord = torch.tensor([50.0, 50.0, 25.0])
        pixel_size = 1.5

        # Get CTFs
        ctf = ts.get_ctfs_for_particles_single(coord, pixel_size, weighted=False)

        # Check that defocus is batched per tilt
        assert ctf.defocus.shape == (3,)

        # All defocus values should be positive (underfocus)
        assert torch.all(ctf.defocus > 0)

        # Pixel size should be set
        assert ctf.pixel_size == pixel_size

    def test_batched_particles_single_position(self):
        """Test CTF generation for multiple particles with single position each"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Batched coordinates: (2, 3) - 2 particles
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [75.0, 75.0, 25.0],
        ])
        pixel_size = 1.5

        # Get CTFs
        ctf = ts.get_ctfs_for_particles_single(coords, pixel_size)

        # Check that defocus is batched: (2, 3) - 2 particles, 3 tilts
        assert ctf.defocus.shape == (2, 3)

        # All defocus values should be finite
        assert torch.all(torch.isfinite(ctf.defocus))

    def test_per_tilt_positions(self):
        """Test CTF generation with per-tilt positions"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Per-tilt coordinates: (2, 3, 3) - 2 particles, 3 tilts, 3 coords
        coords = torch.tensor([
            [[25.0, 25.0, 20.0], [25.0, 25.0, 25.0], [25.0, 25.0, 30.0]],  # Particle 0
            [[75.0, 75.0, 20.0], [75.0, 75.0, 25.0], [75.0, 75.0, 30.0]],  # Particle 1
        ])
        pixel_size = 1.5

        # Get CTFs
        ctf = ts.get_ctfs_for_particles(coords, pixel_size)

        # Check that defocus is batched: (2, 3) - 2 particles, 3 tilts
        assert ctf.defocus.shape == (2, 3)

        # Defocus should vary with Z position
        assert not torch.allclose(ctf.defocus[:, 0], ctf.defocus[:, 1])

    def test_weighted_ctf(self):
        """Test CTF generation with weighting enabled"""
        from warpylib.cubic_grid import CubicGrid

        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set up location weights grid (non-zero)
        ts.grid_location_weights = CubicGrid(
            dimensions=(1, 1, 1),
            values=torch.tensor([[[1.0]]])
        )

        coord = torch.tensor([50.0, 50.0, 25.0])
        pixel_size = 1.5

        # Get CTFs with weighting
        ctf = ts.get_ctfs_for_particles_single(coord, pixel_size, weighted=True)

        # Check that scale and bfactor are set
        assert ctf.scale.shape == (3,)
        assert ctf.bfactor.shape == (3,)

        # Scale should vary with dose (cosine weighting by default)
        assert torch.all(ctf.scale > 0)

        # B-factor should increase with dose (more negative)
        assert ctf.bfactor[2] < ctf.bfactor[0]  # Higher dose = more negative B-factor

    def test_weights_only(self):
        """Test CTF generation with weights_only flag"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coord = torch.tensor([50.0, 50.0, 25.0])
        pixel_size = 1.5

        # Get CTFs with weights_only=True
        ctf = ts.get_ctfs_for_particles_single(coord, pixel_size,
                                               weighted=True, weights_only=True)

        # Defocus should be zero
        assert torch.all(ctf.defocus == 0)

        # Cs should be zero
        assert ctf.cs == 0

        # Amplitude should be 1
        assert ctf.amplitude == 1.0

        # But scale and bfactor should still be set
        assert ctf.scale.shape == (3,)
        assert ctf.bfactor.shape == (3,)

    def test_use_tilt_mask(self):
        """Test that UseTilt mask is applied"""
        from warpylib.cubic_grid import CubicGrid

        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set up location weights grid (non-zero)
        ts.grid_location_weights = CubicGrid(
            dimensions=(1, 1, 1),
            values=torch.tensor([[[1.0]]])
        )

        # Disable middle tilt
        ts.use_tilt = torch.tensor([True, False, True])

        coord = torch.tensor([50.0, 50.0, 25.0])
        pixel_size = 1.5

        # Get CTFs with weighting
        ctf = ts.get_ctfs_for_particles_single(coord, pixel_size, weighted=True)

        # Middle tilt should have very small scale
        assert ctf.scale[1] < 0.001
        assert ctf.scale[0] > 0.1  # Other tilts should have normal scale
        assert ctf.scale[2] > 0.1


class TestGetCTFsForOneTilt:
    """Test get_ctfs_for_one_tilt method"""

    def test_single_tilt_single_particle(self):
        """Test CTF generation for single particle at one tilt"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_id = 1
        defocus = torch.tensor([2.5])
        coord = torch.tensor([[50.0, 50.0, 25.0]])
        pixel_size = 1.5

        # Get CTF
        ctf = ts.get_ctfs_for_one_tilt(tilt_id, defocus, coord, pixel_size)

        # Check that defocus is set
        assert ctf.defocus.shape == (1,)
        assert torch.allclose(ctf.defocus, defocus)

        # Pixel size should be set
        assert ctf.pixel_size == pixel_size

    def test_single_tilt_multiple_particles(self):
        """Test CTF generation for multiple particles at one tilt"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_id = 1
        defoci = torch.tensor([2.0, 2.5, 3.0])
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [50.0, 50.0, 25.0],
            [75.0, 75.0, 25.0],
        ])
        pixel_size = 1.5

        # Get CTF
        ctf = ts.get_ctfs_for_one_tilt(tilt_id, defoci, coords, pixel_size)

        # Check that defocus is batched per particle
        assert ctf.defocus.shape == (3,)
        assert torch.allclose(ctf.defocus, defoci)

    def test_single_tilt_with_weighting(self):
        """Test CTF generation with location-based weighting"""
        from warpylib.cubic_grid import CubicGrid

        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set up location weights grid (non-zero)
        ts.grid_location_weights = CubicGrid(
            dimensions=(1, 1, 1),
            values=torch.tensor([[[1.0]]])
        )

        tilt_id = 1
        defoci = torch.tensor([2.0, 2.5, 3.0])
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [50.0, 50.0, 25.0],
            [75.0, 75.0, 25.0],
        ])
        pixel_size = 1.5

        # Get CTF with weighting
        ctf = ts.get_ctfs_for_one_tilt(tilt_id, defoci, coords, pixel_size, weighted=True)

        # Check that scale and bfactor are set
        assert ctf.scale.shape == (3,)
        assert ctf.bfactor.shape == (3,)

        # All scales should be positive
        assert torch.all(ctf.scale > 0)

    def test_invalid_tilt_id(self):
        """Test error handling for invalid tilt_id"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        defocus = torch.tensor([2.5])
        coord = torch.tensor([[50.0, 50.0, 25.0]])
        pixel_size = 1.5

        # Test negative tilt_id
        with pytest.raises(ValueError, match="tilt_id must be between"):
            ts.get_ctfs_for_one_tilt(-1, defocus, coord, pixel_size)

        # Test tilt_id >= n_tilts
        with pytest.raises(ValueError, match="tilt_id must be between"):
            ts.get_ctfs_for_one_tilt(3, defocus, coord, pixel_size)

    def test_batched_coordinates(self):
        """Test with higher-dimensional batch shapes"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_id = 1
        # Shape: (2, 3) - 2 batches, 3 particles each
        defoci = torch.tensor([[2.0, 2.5, 3.0], [2.2, 2.7, 3.2]])
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [50.0, 50.0, 25.0], [75.0, 75.0, 25.0]],
            [[30.0, 30.0, 25.0], [55.0, 55.0, 25.0], [80.0, 80.0, 25.0]],
        ])
        pixel_size = 1.5

        # Get CTF
        ctf = ts.get_ctfs_for_one_tilt(tilt_id, defoci, coords, pixel_size)

        # Check that defocus has correct batch shape
        assert ctf.defocus.shape == (2, 3)
        assert torch.allclose(ctf.defocus, defoci)


class TestCTFParameterBroadcasting:
    """Test that tilt-specific CTF parameters broadcast correctly"""

    def test_defocus_delta_broadcasting(self):
        """Test that defocus_delta broadcasts correctly"""
        from warpylib.cubic_grid import CubicGrid

        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set different defocus_delta for each tilt via grid
        # Grid dimensions: (X=1, Y=1, Z=3) where Z is tilt dimension
        ts.grid_ctf_defocus_delta = CubicGrid(
            dimensions=(1, 1, 3),
            values=torch.tensor([[[0.1]], [[0.2]], [[0.3]]])  # Shape (Z,Y,X)
        )

        coord = torch.tensor([50.0, 50.0, 25.0])
        pixel_size = 1.5

        # Get CTFs
        ctf = ts.get_ctfs_for_particles_single(coord, pixel_size)

        # defocus_delta should be a tensor with per-tilt values
        assert isinstance(ctf.defocus_delta, torch.Tensor)
        assert ctf.defocus_delta.shape == (3,)
        assert torch.allclose(ctf.defocus_delta, torch.tensor([0.1, 0.2, 0.3]), atol=1e-5)

    def test_phase_shift_broadcasting(self):
        """Test that phase_shift broadcasts correctly"""
        from warpylib.cubic_grid import CubicGrid

        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set different phase_shift for each tilt via grid
        ts.grid_ctf_phase = CubicGrid(
            dimensions=(1, 1, 3),
            values=torch.tensor([[[0.0]], [[0.5]], [[1.0]]])  # Shape (Z,Y,X)
        )

        coord = torch.tensor([50.0, 50.0, 25.0])
        pixel_size = 1.5

        # Get CTFs
        ctf = ts.get_ctfs_for_particles_single(coord, pixel_size)

        # phase_shift should be a tensor with per-tilt values
        assert isinstance(ctf.phase_shift, torch.Tensor)
        assert ctf.phase_shift.shape == (3,)
        assert torch.allclose(ctf.phase_shift, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
