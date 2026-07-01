"""
Tests for TiltSeries.reconstruct_subvolume_ctfs method
"""

import pytest
import torch

from warpylib.tilt_series import TiltSeries


class TestReconstructSubvolumeCTFs:
    """Test reconstruct_subvolume_ctfs method"""

    def test_basic_ctf_volume_reconstruction(self):
        """Test basic CTF volume reconstruction with flat weighting"""
        # Create simple tilt series with 3 tilts
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Create test coordinates: shape (1, 3, 3) - 1 particle, 3 tilts, 3 coords
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],  # Volume center, tilt 0
            [50.0, 50.0, 25.0],  # Volume center, tilt 1
            [50.0, 50.0, 25.0],  # Volume center, tilt 2
        ]])

        # Reconstruct CTF volume with flat weighting (no CTF oscillations)
        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,  # Use flat weighting for basic test
            ctf_weighted=True
        )

        # Check shape: (1, 32, 32, 17) - rfft format (size//2+1 in last dimension)
        assert result.shape == (1, 32, 32, 17)

        # Result should be real-valued (not complex)
        assert not result.is_complex()
        assert result.dtype == torch.float32

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_batched_particles(self):
        """Test CTF volume reconstruction with multiple particles"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Shape: (2, 3, 3) - 2 particles, 3 tilts, 3 coords
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],  # Particle 0
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],  # Particle 1
        ])

        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Check shape: (2, 32, 32, 17)
        assert result.shape == (2, 32, 32, 17)

        # Result should be real-valued (not complex)
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_different_oversamplings(self):
        """Test CTF volume reconstruction with different oversampling factors"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-10.0, 10.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Test with oversampling=1.0
        result_1x = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            oversampling=1.0,
            apply_ctf=True
        )
        assert result_1x.shape == (1, 32, 32, 17)

        # Test with oversampling=2.0
        result_2x = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            oversampling=2.0,
            apply_ctf=True
        )
        assert result_2x.shape == (1, 32, 32, 17)

        # Both should be finite and real-valued
        assert torch.all(torch.isfinite(result_1x))
        assert torch.all(torch.isfinite(result_2x))
        assert not result_1x.is_complex()
        assert not result_2x.is_complex()

    def test_single_coordinate_convenience_method(self):
        """Test convenience method for single coordinate"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate: (3,)
        coord = torch.tensor([50.0, 50.0, 25.0])

        result = ts.reconstruct_subvolume_ctfs_single(
            coords=coord,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )

        # Should return one CTF volume: (32, 32, 17)
        assert result.shape == (32, 32, 17)

        # Result should be real-valued
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with manual replication
        coords_manual = coord.unsqueeze(0).unsqueeze(0).expand(1, 3, 3)  # (1, 3, 3)
        result_manual = ts.reconstruct_subvolume_ctfs(
            coords=coords_manual,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )
        assert torch.allclose(result, result_manual.squeeze(0), atol=1e-5)

    def test_batched_convenience_method(self):
        """Test convenience method with batched coordinates"""
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

        result = ts.reconstruct_subvolume_ctfs_single(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )

        # Should return: (2, 32, 32, 17) - 2 particles
        assert result.shape == (2, 32, 32, 17)

        # Result should be real-valued
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_invalid_tilt_count(self):
        """Test error handling for coordinate/tilt mismatch"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Coords with wrong number of tilts
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])  # Only 2 tilts, but ts has 3

        with pytest.raises(ValueError, match="coords has 2 tilts but TiltSeries has 3"):
            ts.reconstruct_subvolume_ctfs(
                coords=coords,
                pixel_size=10.0,
                size=32
            )

    def test_apply_ctf_false(self):
        """Test CTF volume reconstruction with apply_ctf=False (flat weighting)"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct with flat weighting (no CTF oscillations)
        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            ctf_weighted=True  # Still applies dose weighting
        )

        # Check shape
        assert result.shape == (1, 32, 32, 17)

        # Result should be real-valued
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_tilt_ids_subset(self):
        """Test CTF volume reconstruction using a subset of tilts via tilt_ids"""
        ts = TiltSeries(n_tilts=5)
        ts.angles = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
        ts.dose = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Coordinates for all 5 tilts
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct using only tilts 0, 2, 4 (every other tilt)
        tilt_ids = torch.tensor([0, 2, 4])
        result_subset = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,  # Use flat weighting (no CTF parameters set)
            ctf_weighted=False,  # Disable dose weighting
            tilt_ids=tilt_ids
        )

        # Check shape is still correct
        assert result_subset.shape == (1, 32, 32, 17)
        assert torch.all(torch.isfinite(result_subset))
        assert not result_subset.is_complex()

        # Result should be different from using all tilts
        result_all = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            ctf_weighted=False  # Disable dose weighting
        )
        assert not torch.allclose(result_subset, result_all, atol=1e-5)

    def test_tilt_ids_all_tilts(self):
        """Test that using all tilt_ids gives same result as no tilt_ids"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct without tilt_ids
        result_no_ids = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )

        # Reconstruct with all tilt_ids
        tilt_ids = torch.tensor([0, 1, 2])
        result_with_ids = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        # Results should be identical
        assert torch.allclose(result_no_ids, result_with_ids, atol=1e-5)

    def test_tilt_ids_single_convenience_method(self):
        """Test tilt_ids with single coordinate convenience method"""
        ts = TiltSeries(n_tilts=4)
        ts.angles = torch.tensor([-30.0, -10.0, 10.0, 30.0])
        ts.dose = torch.tensor([0.0, 33.0, 66.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate
        coord = torch.tensor([50.0, 50.0, 25.0])

        # Use only first and last tilts
        tilt_ids = torch.tensor([0, 3])
        result = ts.reconstruct_subvolume_ctfs_single(
            coords=coord,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        assert result.shape == (32, 32, 17)
        assert torch.all(torch.isfinite(result))
        assert not result.is_complex()

    def test_tilt_ids_batched(self):
        """Test tilt_ids with multiple particles"""
        ts = TiltSeries(n_tilts=5)
        ts.angles = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
        ts.dose = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # 2 particles, 5 tilts each
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0], [25.0, 25.0, 25.0],
             [25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0], [75.0, 75.0, 25.0],
             [75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],
        ])

        # Use tilts 1, 2, 3
        tilt_ids = torch.tensor([1, 2, 3])
        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        assert result.shape == (2, 32, 32, 17)
        assert torch.all(torch.isfinite(result))
        assert not result.is_complex()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
