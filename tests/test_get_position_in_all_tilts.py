"""
Tests for TiltSeries.get_position_in_all_tilts method
"""

import pytest
import torch

from warpylib.tilt_series import TiltSeries


class TestGetPositionInAllTilts:
    """Test get_position_in_all_tilts method"""

    def test_basic_transformation(self):
        """Test basic coordinate transformation"""
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

        # Transform
        result = ts.get_position_in_all_tilts(coords)

        # Check shape: (1, 3, 3)
        assert result.shape == (1, 3, 3)

        # At volume center with no warping/movement, X/Y should be near image center
        # (small deviations due to rotation)
        assert torch.allclose(result[0, :, 0], torch.tensor([50.0, 50.0, 50.0]), atol=1.0)
        assert torch.allclose(result[0, :, 1], torch.tensor([50.0, 50.0, 50.0]), atol=1.0)

    def test_batched_particles(self):
        """Test transformation with multiple particles"""
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

        result = ts.get_position_in_all_tilts(coords)

        # Check shape: (2, 3, 3)
        assert result.shape == (2, 3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_with_tilt_axis_offsets(self):
        """Test transformation with tilt axis offsets"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([0.0, 0.0])
        ts.dose = torch.tensor([0.0, 50.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Add tilt axis offsets
        ts.tilt_axis_offset_x = torch.tensor([10.0, -10.0])
        ts.tilt_axis_offset_y = torch.tensor([5.0, -5.0])

        # Shape: (1, 2, 3) - 1 particle, 2 tilts, 3 coords
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],  # Tilt 0
            [50.0, 50.0, 25.0],  # Tilt 1
        ]])

        result = ts.get_position_in_all_tilts(coords)

        # X positions should differ by the offset amounts
        assert abs(result[0, 0, 0] - result[0, 1, 0]) > 15.0  # Should differ by ~20 (offset difference)
        assert abs(result[0, 0, 1] - result[0, 1, 1]) > 5.0   # Should differ by ~10

    def test_multiple_positions_per_tilt(self):
        """Test with multiple coordinate positions for each tilt"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-10.0, 10.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Shape: (3, 2, 3) - 3 particles, 2 tilts, 3 coords
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],  # Particle 0
            [[50.0, 50.0, 25.0], [50.0, 50.0, 25.0]],  # Particle 1
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],  # Particle 2
        ])

        result = ts.get_position_in_all_tilts(coords)

        # Check shape
        assert result.shape == (3, 2, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_single_coordinate_convenience_method(self):
        """Test convenience method for single coordinate"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate: (3,)
        coord = torch.tensor([50.0, 50.0, 25.0])

        result = ts.get_position_in_all_tilts_single(coord)

        # Should return one position per tilt: (3, 3)
        assert result.shape == (3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with manual replication
        coords_manual = coord.unsqueeze(0).unsqueeze(0).expand(1, 3, 3)  # (1, 3, 3)
        result_manual = ts.get_position_in_all_tilts(coords_manual)
        assert torch.allclose(result, result_manual.squeeze(0))

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

        result = ts.get_position_in_all_tilts_single(coords)

        # Should return: (2, 3, 3) - 2 particles, 3 tilts, 3 coords
        assert result.shape == (2, 3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_get_positions_in_one_tilt(self):
        """Test transformation for a single specific tilt"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Multiple coordinates to transform for tilt 1
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [50.0, 50.0, 25.0],
            [75.0, 75.0, 25.0],
        ])

        result = ts.get_positions_in_one_tilt(coords, tilt_id=1)

        # Check shape
        assert result.shape == (3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with get_position_in_all_tilts for the same tilt
        # Create coords for all tilts: (3, 3, 3) - 3 particles, 3 tilts, 3 coords
        coords_all_tilts = coords.unsqueeze(1).expand(3, 3, 3)

        result_all = ts.get_position_in_all_tilts(coords_all_tilts)

        # Extract results for tilt 1
        result_all_tilt1 = result_all[:, 1, :]  # All particles, tilt 1

        # Should match
        assert torch.allclose(result, result_all_tilt1, atol=1e-5)

    def test_get_positions_in_one_tilt_invalid_id(self):
        """Test error handling for invalid tilt_id"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[50.0, 50.0, 25.0]])

        # Test negative tilt_id
        with pytest.raises(ValueError, match="tilt_id must be between"):
            ts.get_positions_in_one_tilt(coords, tilt_id=-1)

        # Test tilt_id >= n_tilts
        with pytest.raises(ValueError, match="tilt_id must be between"):
            ts.get_positions_in_one_tilt(coords, tilt_id=3)

    def test_get_positions_in_one_tilt_with_offsets(self):
        """Test one-tilt transformation with tilt axis offsets"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([0.0, 0.0])
        ts.dose = torch.tensor([0.0, 50.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])
        ts.tilt_axis_offset_x = torch.tensor([10.0, -10.0])
        ts.tilt_axis_offset_y = torch.tensor([5.0, -5.0])

        coords = torch.tensor([
            [50.0, 50.0, 25.0],
            [60.0, 60.0, 25.0],
        ])

        # Get positions for tilt 0
        result_tilt0 = ts.get_positions_in_one_tilt(coords, tilt_id=0)

        # Get positions for tilt 1
        result_tilt1 = ts.get_positions_in_one_tilt(coords, tilt_id=1)

        # Results should differ due to different offsets
        assert not torch.allclose(result_tilt0[:, 0], result_tilt1[:, 0], atol=5.0)
        assert not torch.allclose(result_tilt0[:, 1], result_tilt1[:, 1], atol=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
