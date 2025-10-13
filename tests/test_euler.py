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
        rot = torch.tensor([0.0])
        tilt = torch.tensor([0.0])
        psi = torch.tensor([0.0])

        matrix = euler_to_matrix(rot, tilt, psi)

        # Should be identity matrix
        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_roundtrip_identity(self):
        """Test roundtrip conversion for identity"""
        rot = torch.tensor([0.0])
        tilt = torch.tensor([0.0])
        psi = torch.tensor([0.0])

        matrix = euler_to_matrix(rot, tilt, psi)
        rot_out, tilt_out, psi_out = matrix_to_euler(matrix)

        # Should recover original angles (or equivalent)
        matrix_out = euler_to_matrix(rot_out, tilt_out, psi_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-6)

    def test_single_rotation_z(self):
        """Test rotation around Z axis only"""
        rot = torch.tensor([0.5])
        tilt = torch.tensor([0.0])
        psi = torch.tensor([0.3])

        matrix = euler_to_matrix(rot, tilt, psi)

        # When tilt=0, the rotation simplifies to rotation around Z
        # The ZYZ convention gives: R_z(psi) @ R_y(0) @ R_z(rot) = R_z(psi + rot)
        # But with the specific matrix form from the C# implementation
        total_angle = rot + psi

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
        rot = torch.rand(batch_size) * 2 * np.pi
        tilt = torch.rand(batch_size) * np.pi
        psi = torch.rand(batch_size) * 2 * np.pi

        matrices = euler_to_matrix(rot, tilt, psi)

        assert matrices.shape == (batch_size, 3, 3)

        # Check all matrices are valid rotations (determinant ~= 1)
        dets = torch.det(matrices)
        assert torch.allclose(dets, torch.ones(batch_size), atol=1e-5)

    def test_roundtrip_random(self):
        """Test roundtrip conversion with random angles"""
        rot = torch.rand(5) * 2 * np.pi
        tilt = torch.rand(5) * np.pi
        psi = torch.rand(5) * 2 * np.pi

        matrices = euler_to_matrix(rot, tilt, psi)
        rot_out, tilt_out, psi_out = matrix_to_euler(matrices)

        # Reconstruct matrices
        matrices_out = euler_to_matrix(rot_out, tilt_out, psi_out)

        # Matrices should match
        assert torch.allclose(matrices, matrices_out, atol=1e-5)

    def test_specific_angles(self):
        """Test with specific known angles"""
        # Test case from C# implementation
        rot = torch.tensor([0.1, 0.5, 1.0])
        tilt = torch.tensor([0.2, 0.7, 1.5])
        psi = torch.tensor([0.3, 0.4, 0.8])

        matrices = euler_to_matrix(rot, tilt, psi)

        # Verify orthogonality (R^T @ R = I)
        matrices_t = matrices.transpose(-2, -1)
        identity_check = torch.bmm(matrices_t, matrices)
        identity = torch.eye(3).unsqueeze(0).expand(3, -1, -1)

        assert torch.allclose(identity_check, identity, atol=1e-5)

    def test_gimbal_lock(self):
        """Test gimbal lock case (tilt = 0 or pi)"""
        # Case 1: tilt = 0
        rot = torch.tensor([0.5])
        tilt = torch.tensor([0.0])
        psi = torch.tensor([0.3])

        matrix = euler_to_matrix(rot, tilt, psi)
        rot_out, tilt_out, psi_out = matrix_to_euler(matrix)

        # Should handle gimbal lock gracefully
        matrix_out = euler_to_matrix(rot_out, tilt_out, psi_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-5)

        # Case 2: tilt = pi
        rot = torch.tensor([0.5])
        tilt = torch.tensor([np.pi])
        psi = torch.tensor([0.3])

        matrix = euler_to_matrix(rot, tilt, psi)
        rot_out, tilt_out, psi_out = matrix_to_euler(matrix)

        matrix_out = euler_to_matrix(rot_out, tilt_out, psi_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-5)


class TestEulerXYZExtrinsic:
    """Test XYZ extrinsic convention Euler angles"""

    def test_identity(self):
        """Test identity rotation"""
        k1 = torch.tensor([0.0])
        k2 = torch.tensor([0.0])
        k3 = torch.tensor([0.0])

        matrix = euler_xyz_extrinsic_to_matrix(k1, k2, k3)

        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(matrix, expected, atol=1e-6)

    def test_roundtrip_identity(self):
        """Test roundtrip for identity"""
        k1 = torch.tensor([0.0])
        k2 = torch.tensor([0.0])
        k3 = torch.tensor([0.0])

        matrix = euler_xyz_extrinsic_to_matrix(k1, k2, k3)
        k1_out, k2_out, k3_out = matrix_to_euler_xyz_extrinsic(matrix)

        matrix_out = euler_xyz_extrinsic_to_matrix(k1_out, k2_out, k3_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-6)

    def test_rotation_x(self):
        """Test pure X rotation"""
        k1 = torch.tensor([0.5])
        k2 = torch.tensor([0.0])
        k3 = torch.tensor([0.0])

        matrix = euler_xyz_extrinsic_to_matrix(k1, k2, k3)

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
        k1 = torch.tensor([0.0])
        k2 = torch.tensor([0.5])
        k3 = torch.tensor([0.0])

        matrix = euler_xyz_extrinsic_to_matrix(k1, k2, k3)

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
        k1 = torch.tensor([0.0])
        k2 = torch.tensor([0.0])
        k3 = torch.tensor([0.5])

        matrix = euler_xyz_extrinsic_to_matrix(k1, k2, k3)

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
        k1 = torch.rand(batch_size) * 2 * np.pi
        k2 = torch.rand(batch_size) * 2 * np.pi
        k3 = torch.rand(batch_size) * 2 * np.pi

        matrices = euler_xyz_extrinsic_to_matrix(k1, k2, k3)

        assert matrices.shape == (batch_size, 3, 3)

        # Check determinants
        dets = torch.det(matrices)
        assert torch.allclose(dets, torch.ones(batch_size), atol=1e-5)

    def test_roundtrip_random(self):
        """Test roundtrip with random angles"""
        k1 = (torch.rand(5) - 0.5) * np.pi  # Avoid gimbal lock region
        k2 = (torch.rand(5) - 0.5) * np.pi * 0.8  # Stay away from ±pi/2
        k3 = (torch.rand(5) - 0.5) * np.pi

        matrices = euler_xyz_extrinsic_to_matrix(k1, k2, k3)
        k1_out, k2_out, k3_out = matrix_to_euler_xyz_extrinsic(matrices)

        matrices_out = euler_xyz_extrinsic_to_matrix(k1_out, k2_out, k3_out)

        assert torch.allclose(matrices, matrices_out, atol=1e-5)

    def test_gimbal_lock_xyz(self):
        """Test gimbal lock handling for XYZ"""
        # Gimbal lock at k2 = ±pi/2
        k1 = torch.tensor([0.3])
        k2 = torch.tensor([np.pi / 2 - 0.001])  # Near gimbal lock
        k3 = torch.tensor([0.5])

        matrix = euler_xyz_extrinsic_to_matrix(k1, k2, k3)
        k1_out, k2_out, k3_out = matrix_to_euler_xyz_extrinsic(matrix)

        matrix_out = euler_xyz_extrinsic_to_matrix(k1_out, k2_out, k3_out)
        assert torch.allclose(matrix, matrix_out, atol=1e-4)


class TestGradients:
    """Test that conversions support autograd"""

    def test_euler_to_matrix_gradient(self):
        """Test gradients flow through euler_to_matrix"""
        rot = torch.tensor([0.5], requires_grad=True)
        tilt = torch.tensor([0.3], requires_grad=True)
        psi = torch.tensor([0.2], requires_grad=True)

        matrix = euler_to_matrix(rot, tilt, psi)
        loss = matrix.sum()
        loss.backward()

        assert rot.grad is not None
        assert tilt.grad is not None
        assert psi.grad is not None

    def test_matrix_to_euler_gradient(self):
        """Test gradients flow through matrix_to_euler"""
        rot = torch.tensor([0.5], requires_grad=True)
        tilt = torch.tensor([0.3], requires_grad=True)
        psi = torch.tensor([0.2], requires_grad=True)

        matrix = euler_to_matrix(rot, tilt, psi)
        rot_out, tilt_out, psi_out = matrix_to_euler(matrix)

        loss = rot_out.sum() + tilt_out.sum() + psi_out.sum()
        loss.backward()

        assert rot.grad is not None
        assert tilt.grad is not None
        assert psi.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
