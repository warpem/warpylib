"""
Tests for Euler angle conversions
"""

import numpy as np
import torch
import pytest

from warpylib.euler import (
    euler_to_matrix,
    matrix_to_euler,
    euler_xyz_extrinsic_to_matrix,
    matrix_to_euler_xyz_extrinsic,
)


class TestEulerZYZ:
    """Test ZYZ convention Euler angles"""

    def test_identity(self):
        """Test identity rotation"""
        angles = torch.tensor([[0.0, 0.0, 0.0]])

        matrix = euler_to_matrix(angles)

        # Should be identity matrix
        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_roundtrip_identity(self):
        """Test roundtrip conversion for identity"""
        angles = torch.tensor([[0.0, 0.0, 0.0]])

        matrix = euler_to_matrix(angles)
        angles_out = matrix_to_euler(matrix)

        # Should recover original angles (or equivalent)
        matrix_out = euler_to_matrix(angles_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-6)

    def test_single_rotation_z(self):
        """Test rotation around Z axis only"""
        angles = torch.tensor([[0.5, 0.0, 0.3]])

        matrix = euler_to_matrix(angles)

        # When tilt=0, the rotation simplifies to rotation around Z
        # The ZYZ convention gives: R_z(psi) @ R_y(0) @ R_z(rot) = R_z(psi + rot)
        # But with the specific matrix form from the C# implementation
        total_angle = angles[0, 0] + angles[0, 2]

        # Verify it's a Z rotation by checking third row and column
        assert torch.allclose(matrix[..., 2, :2], torch.zeros(1, 2), atol=1e-6)
        assert torch.allclose(matrix[..., :2, 2], torch.zeros(1, 2), atol=1e-6)
        assert torch.allclose(matrix[..., 2, 2], torch.ones(1), atol=1e-6)

        # Verify the rotation angle matches via determinant of upper-left 2x2
        upper_left = matrix[..., :2, :2]
        # For a 2D rotation matrix, trace = 2*cos(angle)
        trace = upper_left[..., 0, 0] + upper_left[..., 1, 1]
        expected_trace = 2 * torch.cos(total_angle)
        assert torch.allclose(trace, expected_trace, atol=1e-6)

    def test_batch_conversion(self):
        """Test batch processing"""
        batch_size = 10
        angles = torch.stack([
            torch.rand(batch_size) * 2 * np.pi,  # rot
            torch.rand(batch_size) * np.pi,      # tilt
            torch.rand(batch_size) * 2 * np.pi   # psi
        ], dim=-1)

        matrices = euler_to_matrix(angles)

        assert matrices.shape == (batch_size, 3, 3)

        # Check all matrices are valid rotations (determinant ~= 1)
        dets = torch.det(matrices)
        assert torch.allclose(dets, torch.ones(batch_size), atol=1e-5)

    def test_roundtrip_random(self):
        """Test roundtrip conversion with random angles"""
        angles = torch.stack([
            torch.rand(5) * 2 * np.pi,  # rot
            torch.rand(5) * np.pi,      # tilt
            torch.rand(5) * 2 * np.pi   # psi
        ], dim=-1)

        matrices = euler_to_matrix(angles)
        angles_out = matrix_to_euler(matrices)

        # Reconstruct matrices
        matrices_out = euler_to_matrix(angles_out)

        # Matrices should match
        assert torch.allclose(matrices, matrices_out, atol=1e-5)

    def test_specific_angles(self):
        """Test with specific known angles"""
        # Test case from C# implementation
        angles = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.5, 0.7, 0.4],
            [1.0, 1.5, 0.8]
        ])

        matrices = euler_to_matrix(angles)

        # Verify orthogonality (R^T @ R = I)
        matrices_t = matrices.transpose(-2, -1)
        identity_check = torch.bmm(matrices_t, matrices)
        identity = torch.eye(3).unsqueeze(0).expand(3, -1, -1)

        assert torch.allclose(identity_check, identity, atol=1e-5)

    def test_gimbal_lock(self):
        """Test gimbal lock case (tilt = 0 or pi)"""
        # Case 1: tilt = 0
        angles = torch.tensor([[0.5, 0.0, 0.3]])

        matrix = euler_to_matrix(angles)
        angles_out = matrix_to_euler(matrix)

        # Should handle gimbal lock gracefully
        matrix_out = euler_to_matrix(angles_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-5)

        # Case 2: tilt = pi
        angles = torch.tensor([[0.5, np.pi, 0.3]])

        matrix = euler_to_matrix(angles)
        angles_out = matrix_to_euler(matrix)

        matrix_out = euler_to_matrix(angles_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-5)


class TestEulerXYZExtrinsic:
    """Test XYZ extrinsic convention Euler angles"""

    def test_identity(self):
        """Test identity rotation"""
        angles = torch.tensor([[0.0, 0.0, 0.0]])

        matrix = euler_xyz_extrinsic_to_matrix(angles)

        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_roundtrip_identity(self):
        """Test roundtrip for identity"""
        angles = torch.tensor([[0.0, 0.0, 0.0]])

        matrix = euler_xyz_extrinsic_to_matrix(angles)
        angles_out = matrix_to_euler_xyz_extrinsic(matrix)

        matrix_out = euler_xyz_extrinsic_to_matrix(angles_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-6)

    def test_rotation_x(self):
        """Test pure X rotation"""
        angles = torch.tensor([[0.5, 0.0, 0.0]])

        matrix = euler_xyz_extrinsic_to_matrix(angles)

        # Should match R_x(0.5)
        c, s = np.cos(0.5), np.sin(0.5)
        expected = torch.tensor([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=torch.float32).unsqueeze(0)

        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_rotation_y(self):
        """Test pure Y rotation"""
        angles = torch.tensor([[0.0, 0.5, 0.0]])

        matrix = euler_xyz_extrinsic_to_matrix(angles)

        # Should match R_y(0.5)
        c, s = np.cos(0.5), np.sin(0.5)
        expected = torch.tensor([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=torch.float32).unsqueeze(0)

        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_rotation_z(self):
        """Test pure Z rotation"""
        angles = torch.tensor([[0.0, 0.0, 0.5]])

        matrix = euler_xyz_extrinsic_to_matrix(angles)

        # Should match R_z(0.5)
        c, s = np.cos(0.5), np.sin(0.5)
        expected = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=torch.float32).unsqueeze(0)

        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_batch_conversion(self):
        """Test batch processing"""
        batch_size = 10
        angles = torch.stack([
            torch.rand(batch_size) * 2 * np.pi,  # k1
            torch.rand(batch_size) * 2 * np.pi,  # k2
            torch.rand(batch_size) * 2 * np.pi   # k3
        ], dim=-1)

        matrices = euler_xyz_extrinsic_to_matrix(angles)

        assert matrices.shape == (batch_size, 3, 3)

        # Check determinants
        dets = torch.det(matrices)
        assert torch.allclose(dets, torch.ones(batch_size), atol=1e-5)

    def test_roundtrip_random(self):
        """Test roundtrip with random angles"""
        angles = torch.stack([
            (torch.rand(5) - 0.5) * np.pi,        # k1: Avoid gimbal lock region
            (torch.rand(5) - 0.5) * np.pi * 0.8,  # k2: Stay away from ±pi/2
            (torch.rand(5) - 0.5) * np.pi         # k3
        ], dim=-1)

        matrices = euler_xyz_extrinsic_to_matrix(angles)
        angles_out = matrix_to_euler_xyz_extrinsic(matrices)

        matrices_out = euler_xyz_extrinsic_to_matrix(angles_out)

        assert torch.allclose(matrices, matrices_out, atol=1e-5)

    def test_gimbal_lock_xyz(self):
        """Test gimbal lock handling for XYZ"""
        # Gimbal lock at k2 = ±pi/2
        angles = torch.tensor([[0.3, np.pi / 2 - 0.001, 0.5]])  # Near gimbal lock

        matrix = euler_xyz_extrinsic_to_matrix(angles)
        angles_out = matrix_to_euler_xyz_extrinsic(matrix)

        matrix_out = euler_xyz_extrinsic_to_matrix(angles_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-4)


class TestGradients:
    """Test that conversions support autograd"""

    def test_euler_to_matrix_gradient(self):
        """Test gradients flow through euler_to_matrix"""
        angles = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)

        matrix = euler_to_matrix(angles)
        loss = matrix.sum()
        loss.backward()

        assert angles.grad is not None

    def test_matrix_to_euler_gradient(self):
        """Test gradients flow through matrix_to_euler"""
        angles_in = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)

        matrix = euler_to_matrix(angles_in)
        angles_out = matrix_to_euler(matrix)

        loss = angles_out.sum()
        loss.backward()

        assert angles_in.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
