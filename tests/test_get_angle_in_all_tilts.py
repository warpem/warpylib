"""
Tests for TiltSeries.get_angle_in_all_tilts and related angle methods
"""

import pytest
import torch
import numpy as np

from warpylib.tilt_series import TiltSeries
from warpylib.euler import euler_to_matrix, matrix_to_euler


class TestGetAngleInAllTilts:
    """Test angle transformation methods"""

    def test_basic_angle_transformation(self):
        """Test basic angle transformation without grid corrections"""
        # Create simple tilt series with 3 tilts
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Create test coordinates (3 positions, one for each tilt)
        coords = torch.tensor([
            [50.0, 50.0, 25.0],  # Volume center, tilt 0
            [50.0, 50.0, 25.0],  # Volume center, tilt 1
            [50.0, 50.0, 25.0],  # Volume center, tilt 2
        ])

        # Transform
        result = ts.get_angle_in_all_tilts(coords)

        # Check shape
        assert result.shape == (3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Convert result angles to matrices to verify they're valid rotations
        result_matrices = torch.zeros((3, 3, 3))
        for t in range(3):
            result_matrices[t] = euler_to_matrix(result[t].unsqueeze(0)).squeeze(0)

        # Check that matrices are valid rotations
        for t in range(3):
            mat = result_matrices[t]
            # Determinant should be 1
            det = torch.det(mat)
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)
            # Matrix should be orthonormal
            identity = torch.matmul(mat.T, mat)
            assert torch.allclose(identity, torch.eye(3), atol=1e-5)

        # Tilt 1 (0 deg) should be close to identity since no grid corrections
        assert torch.allclose(result_matrices[1], torch.eye(3), atol=1e-4)

        # Tilt 0 and Tilt 2 should be different due to different tilt angles
        assert not torch.allclose(result_matrices[0], result_matrices[2], atol=0.01)

    def test_angle_single_coordinate(self):
        """Test convenience method for single coordinate"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate
        coord = torch.tensor([50.0, 50.0, 25.0])

        result = ts.get_angle_in_all_tilts_single(coord)

        # Should return one angle per tilt
        assert result.shape == (3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with manual replication
        coords_manual = coord.unsqueeze(0).repeat(3, 1)
        result_manual = ts.get_angle_in_all_tilts(coords_manual)
        assert torch.allclose(result, result_manual)

    def test_particle_angle_transformation(self):
        """Test particle angle transformation"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-10.0, 10.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Coordinates
        coords = torch.tensor([
            [50.0, 50.0, 25.0],  # Tilt 0
            [50.0, 50.0, 25.0],  # Tilt 1
        ])

        # Particle angles (in radians)
        particle_angles = torch.tensor([
            [0.1, 0.2, 0.3],  # Tilt 0
            [0.1, 0.2, 0.3],  # Tilt 1
        ])

        # Transform
        result = ts.get_particle_angle_in_all_tilts(coords, particle_angles)

        # Check shape
        assert result.shape == (2, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # The angles should be different from the input due to tilt transformation
        assert not torch.allclose(result, particle_angles, atol=0.01)

    def test_particle_angle_single(self):
        """Test particle angle transformation for single coordinate"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate and angle
        coord = torch.tensor([50.0, 50.0, 25.0])
        angle = torch.tensor([0.1, 0.2, 0.3])

        result = ts.get_particle_angle_in_all_tilts_single(coord, angle)

        # Should return one angle per tilt
        assert result.shape == (3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with manual replication
        coords_manual = coord.unsqueeze(0).repeat(3, 1)
        angles_manual = angle.unsqueeze(0).repeat(3, 1)
        result_manual = ts.get_particle_angle_in_all_tilts(coords_manual, angles_manual)
        assert torch.allclose(result, result_manual)

    def test_angles_in_one_tilt(self):
        """Test angle transformation for single tilt"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Multiple coordinates
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [50.0, 50.0, 25.0],
            [75.0, 75.0, 25.0],
        ])

        # Particle angles (in radians)
        particle_angles = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
        ])

        # Transform for tilt 1
        result = ts.get_angles_in_one_tilt(coords, particle_angles, tilt_id=1)

        # Check shape
        assert result.shape == (3, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with get_particle_angle_in_all_tilts for the same tilt
        # Create coords for all tilts (pattern: coord0_tilt0, coord0_tilt1, coord0_tilt2, ...)
        coords_all_tilts = torch.zeros((9, 3))
        angles_all_tilts = torch.zeros((9, 3))
        for i in range(3):
            for t in range(3):
                coords_all_tilts[i * 3 + t] = coords[i]
                angles_all_tilts[i * 3 + t] = particle_angles[i]

        result_all = ts.get_particle_angle_in_all_tilts(coords_all_tilts, angles_all_tilts)

        # Extract results for tilt 1
        result_all_tilt1 = result_all[[1, 4, 7]]  # Indices for tilt 1

        # Should match
        assert torch.allclose(result, result_all_tilt1, atol=1e-5)

    def test_angles_in_one_tilt_invalid_id(self):
        """Test error handling for invalid tilt_id"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[50.0, 50.0, 25.0]])
        angles = torch.tensor([[0.0, 0.0, 0.0]])

        # Test negative tilt_id
        with pytest.raises(ValueError, match="tilt_id must be between"):
            ts.get_angles_in_one_tilt(coords, angles, tilt_id=-1)

        # Test tilt_id >= n_tilts
        with pytest.raises(ValueError, match="tilt_id must be between"):
            ts.get_angles_in_one_tilt(coords, angles, tilt_id=3)

    def test_rotation_matrix_output(self):
        """Test that rotation matrix method produces valid rotation matrices"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-15.0, 15.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ])

        particle_angles = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

        # Get rotation matrices
        matrices = ts.get_particle_rotation_matrix_in_all_tilts(coords, particle_angles)

        # Check shape
        assert matrices.shape == (2, 3, 3)

        # Check that matrices are valid rotations (det = 1, orthonormal)
        for i in range(2):
            mat = matrices[i]

            # Determinant should be 1
            det = torch.det(mat)
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

            # Matrix should be orthonormal (M^T M = I)
            identity = torch.matmul(mat.T, mat)
            assert torch.allclose(identity, torch.eye(3), atol=1e-5)

    def test_with_grid_angle_corrections(self):
        """Test angle transformation with non-zero grid angle corrections"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([0.0, 0.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Set non-zero angle grids
        # Grid values are in degrees
        from warpylib.cubic_grid import CubicGrid
        ts.grid_angle_x = CubicGrid((1, 1, 2), values=torch.tensor([5.0, 10.0], dtype=torch.float32))
        ts.grid_angle_y = CubicGrid((1, 1, 2), values=torch.tensor([3.0, 6.0], dtype=torch.float32))
        ts.grid_angle_z = CubicGrid((1, 1, 2), values=torch.tensor([2.0, 4.0], dtype=torch.float32))

        coords = torch.tensor([
            [50.0, 50.0, 25.0],  # Tilt 0
            [50.0, 50.0, 25.0],  # Tilt 1
        ])

        # Transform
        result = ts.get_angle_in_all_tilts(coords)

        # Check shape
        assert result.shape == (2, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # The two tilts should have different results due to different grid values
        assert not torch.allclose(result[0], result[1], atol=0.01)

    def test_multiple_positions_per_tilt(self):
        """Test with multiple coordinate positions for each tilt"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-10.0, 10.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # 6 coordinates: 3 positions × 2 tilts
        coords = torch.tensor([
            [25.0, 25.0, 25.0],  # Position 0, tilt 0
            [25.0, 25.0, 25.0],  # Position 0, tilt 1
            [50.0, 50.0, 25.0],  # Position 1, tilt 0
            [50.0, 50.0, 25.0],  # Position 1, tilt 1
            [75.0, 75.0, 25.0],  # Position 2, tilt 0
            [75.0, 75.0, 25.0],  # Position 2, tilt 1
        ])

        result = ts.get_angle_in_all_tilts(coords)

        # Check shape
        assert result.shape == (6, 3)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_euler_roundtrip(self):
        """Test that angles can be converted to matrices and back"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-20.0, 20.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ])

        particle_angles = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ])

        # Get angles
        angles = ts.get_particle_angle_in_all_tilts(coords, particle_angles)

        # Get matrices
        matrices = ts.get_particle_rotation_matrix_in_all_tilts(coords, particle_angles)

        # Convert matrices back to angles
        for i in range(2):
            reconstructed = matrix_to_euler(matrices[i].unsqueeze(0))[0]

            # Should match the angles from get_particle_angle_in_all_tilts
            assert torch.allclose(reconstructed, angles[i], atol=1e-5)

    def test_invalid_coordinate_count(self):
        """Test error when coordinate count doesn't match n_tilts"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # 4 coordinates, not divisible by 3
        coords = torch.tensor([
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ])

        with pytest.raises(ValueError, match="must be divisible by n_tilts"):
            ts.get_angle_in_all_tilts(coords)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
