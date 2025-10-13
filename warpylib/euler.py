"""
Euler angle conversions using PyTorch

Replicates Matrix3.cs Euler angle functionality using PyTorch tensors.
Supports ZYZ convention (rot, tilt, psi) used in cryo-EM.
"""

import torch
from typing import Tuple


def euler_to_matrix(rot: torch.Tensor, tilt: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to rotation matrices (ZYZ convention).

    This corresponds to: R_z(psi) @ R_y(tilt) @ R_z(rot)

    Args:
        rot: Rotation angle (alpha) in radians, shape (...,)
        tilt: Tilt angle (beta) in radians, shape (...,)
        psi: Psi angle (gamma) in radians, shape (...,)

    Returns:
        Rotation matrices of shape (..., 3, 3)

    Example:
        >>> rot = torch.tensor([0.0, 0.5])
        >>> tilt = torch.tensor([0.0, 0.3])
        >>> psi = torch.tensor([0.0, 0.2])
        >>> matrices = euler_to_matrix(rot, tilt, psi)
        >>> matrices.shape
        torch.Size([2, 3, 3])
    """
    ca = torch.cos(rot)
    cb = torch.cos(tilt)
    cg = torch.cos(psi)
    sa = torch.sin(rot)
    sb = torch.sin(tilt)
    sg = torch.sin(psi)

    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    # Build rotation matrix
    shape = rot.shape
    matrices = torch.zeros(*shape, 3, 3, dtype=rot.dtype, device=rot.device)

    # Row 1
    matrices[..., 0, 0] = cg * cc - sg * sa
    matrices[..., 0, 1] = cg * cs + sg * ca
    matrices[..., 0, 2] = -cg * sb

    # Row 2
    matrices[..., 1, 0] = -sg * cc - cg * sa
    matrices[..., 1, 1] = -sg * cs + cg * ca
    matrices[..., 1, 2] = sg * sb

    # Row 3
    matrices[..., 2, 0] = sc
    matrices[..., 2, 1] = ss
    matrices[..., 2, 2] = cb

    return matrices


def matrix_to_euler(matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Euler angles from rotation matrices (ZYZ convention).

    Args:
        matrices: Rotation matrices of shape (..., 3, 3)

    Returns:
        Tuple of (rot, tilt, psi) each of shape (...,) in radians

    Example:
        >>> matrices = torch.eye(3).unsqueeze(0)
        >>> rot, tilt, psi = matrix_to_euler(matrices)
        >>> rot.shape, tilt.shape, psi.shape
        (torch.Size([1]), torch.Size([1]), torch.Size([1]))
    """
    # Extract matrix elements
    m11 = matrices[..., 0, 0]
    m21 = matrices[..., 1, 0]
    m31 = matrices[..., 2, 0]
    m13 = matrices[..., 0, 2]
    m23 = matrices[..., 1, 2]
    m32 = matrices[..., 2, 1]
    m33 = matrices[..., 2, 2]

    abs_sb = torch.sqrt(m13 ** 2 + m23 ** 2)

    # Threshold for numerical stability
    epsilon = 16 * 1.192092896e-07

    # Initialize output tensors
    shape = matrices.shape[:-2]
    alpha = torch.zeros(shape, dtype=matrices.dtype, device=matrices.device)
    beta = torch.zeros(shape, dtype=matrices.dtype, device=matrices.device)
    gamma = torch.zeros(shape, dtype=matrices.dtype, device=matrices.device)

    # Normal case: abs_sb > epsilon
    normal_mask = abs_sb > epsilon

    if normal_mask.any():
        # Extract values for normal case
        m11_n = m11[normal_mask]
        m21_n = m21[normal_mask]
        m31_n = m31[normal_mask]
        m13_n = m13[normal_mask]
        m23_n = m23[normal_mask]
        m32_n = m32[normal_mask]
        m33_n = m33[normal_mask]
        abs_sb_n = abs_sb[normal_mask]

        gamma_n = torch.atan2(m23_n, -m13_n)
        alpha_n = torch.atan2(m32_n, m31_n)

        # Determine sign of sin(beta)
        sin_gamma = torch.sin(gamma_n)
        cos_gamma = torch.cos(gamma_n)

        # Handle small sin(gamma)
        small_sin_mask = torch.abs(sin_gamma) < 1.192092896e-07
        sign_sb = torch.where(
            small_sin_mask,
            torch.sign(-m13_n / cos_gamma),
            torch.where(sin_gamma > 0, torch.sign(m23_n), -torch.sign(m23_n))
        )

        beta_n = torch.atan2(sign_sb * abs_sb_n, m33_n)

        alpha[normal_mask] = alpha_n
        beta[normal_mask] = beta_n
        gamma[normal_mask] = gamma_n

    # Gimbal lock case: abs_sb <= epsilon
    gimbal_mask = ~normal_mask

    if gimbal_mask.any():
        m11_g = m11[gimbal_mask]
        m21_g = m21[gimbal_mask]
        m33_g = m33[gimbal_mask]

        # Check sign of m33
        positive_mask = torch.sign(m33_g) > 0

        # Positive case: rotation around Z
        alpha[gimbal_mask] = torch.where(
            positive_mask,
            torch.zeros_like(m33_g),
            torch.zeros_like(m33_g)
        )
        beta[gimbal_mask] = torch.where(
            positive_mask,
            torch.zeros_like(m33_g),
            torch.full_like(m33_g, torch.pi)
        )
        gamma[gimbal_mask] = torch.where(
            positive_mask,
            torch.atan2(-m21_g, m11_g),
            torch.atan2(m21_g, -m11_g)
        )

    return (alpha, beta, gamma)


def euler_xyz_extrinsic_to_matrix(k1: torch.Tensor, k2: torch.Tensor, k3: torch.Tensor) -> torch.Tensor:
    """
    Convert Euler angles to rotation matrices (XYZ extrinsic convention).

    This corresponds to: R_z(k3) @ R_y(k2) @ R_x(k1)

    Args:
        k1: Rotation around X axis in radians, shape (...,)
        k2: Rotation around Y axis in radians, shape (...,)
        k3: Rotation around Z axis in radians, shape (...,)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    c1, s1 = torch.cos(k1), torch.sin(k1)
    c2, s2 = torch.cos(k2), torch.sin(k2)
    c3, s3 = torch.cos(k3), torch.sin(k3)

    shape = k1.shape
    matrices = torch.zeros(*shape, 3, 3, dtype=k1.dtype, device=k1.device)

    # R_z(k3) @ R_y(k2) @ R_x(k1)
    matrices[..., 0, 0] = c2 * c3
    matrices[..., 0, 1] = s1 * s2 * c3 - c1 * s3
    matrices[..., 0, 2] = c1 * s2 * c3 + s1 * s3

    matrices[..., 1, 0] = c2 * s3
    matrices[..., 1, 1] = s1 * s2 * s3 + c1 * c3
    matrices[..., 1, 2] = c1 * s2 * s3 - s1 * c3

    matrices[..., 2, 0] = -s2
    matrices[..., 2, 1] = s1 * c2
    matrices[..., 2, 2] = c1 * c2

    return matrices


def matrix_to_euler_xyz_extrinsic(matrices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Euler angles from rotation matrices (XYZ extrinsic convention).

    This corresponds to: R_z(k3) @ R_y(k2) @ R_x(k1)

    Args:
        matrices: Rotation matrices of shape (..., 3, 3)

    Returns:
        Tuple of (k1, k2, k3) each of shape (...,) in radians
    """
    # Extract matrix elements
    m11 = matrices[..., 0, 0]
    m21 = matrices[..., 1, 0]
    m22 = matrices[..., 1, 1]
    m23 = matrices[..., 1, 2]
    m31 = matrices[..., 2, 0]
    m32 = matrices[..., 2, 1]
    m33 = matrices[..., 2, 2]

    # Direct readout of second angle
    k2 = torch.asin(-m31)

    tolerance = 1e-4

    # Initialize outputs
    shape = matrices.shape[:-2]
    k1 = torch.zeros(shape, dtype=matrices.dtype, device=matrices.device)
    k3 = torch.zeros(shape, dtype=matrices.dtype, device=matrices.device)

    # No gimbal lock case
    no_gimbal_mask = torch.abs(m11) >= tolerance

    if no_gimbal_mask.any():
        k1[no_gimbal_mask] = torch.atan2(m32[no_gimbal_mask], m33[no_gimbal_mask])
        k3[no_gimbal_mask] = torch.atan2(m21[no_gimbal_mask], m11[no_gimbal_mask])

    # Gimbal lock case (k2 == 90 degrees)
    gimbal_mask = ~no_gimbal_mask

    if gimbal_mask.any():
        k1[gimbal_mask] = torch.atan2(m23[gimbal_mask], m22[gimbal_mask])
        k3[gimbal_mask] = 0.0

    return (k1, k2, k3)


def rotate_x(angle: torch.Tensor) -> torch.Tensor:
    """
    Create rotation matrix around X axis.

    Note: This matches C# Matrix3.RotateX which uses transposed convention.

    Args:
        angle: Rotation angle in radians, shape (...,)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    c = torch.cos(angle)
    s = torch.sin(angle)

    shape = angle.shape
    matrices = torch.zeros(*shape, 3, 3, dtype=angle.dtype, device=angle.device)

    matrices[..., 0, 0] = 1
    matrices[..., 1, 1] = c
    matrices[..., 1, 2] = -s  # Transposed: was s
    matrices[..., 2, 1] = s   # Transposed: was -s
    matrices[..., 2, 2] = c

    return matrices


def rotate_y(angle: torch.Tensor) -> torch.Tensor:
    """
    Create rotation matrix around Y axis.

    Note: This matches C# Matrix3.RotateY which uses transposed convention.

    Args:
        angle: Rotation angle in radians, shape (...,)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    c = torch.cos(angle)
    s = torch.sin(angle)

    shape = angle.shape
    matrices = torch.zeros(*shape, 3, 3, dtype=angle.dtype, device=angle.device)

    matrices[..., 0, 0] = c
    matrices[..., 0, 2] = s   # Transposed: was -s
    matrices[..., 1, 1] = 1
    matrices[..., 2, 0] = -s  # Transposed: was s
    matrices[..., 2, 2] = c

    return matrices


def rotate_z(angle: torch.Tensor) -> torch.Tensor:
    """
    Create rotation matrix around Z axis.

    Note: This matches C# Matrix3.RotateZ which uses transposed convention.

    Args:
        angle: Rotation angle in radians, shape (...,)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    c = torch.cos(angle)
    s = torch.sin(angle)

    shape = angle.shape
    matrices = torch.zeros(*shape, 3, 3, dtype=angle.dtype, device=angle.device)

    matrices[..., 0, 0] = c
    matrices[..., 0, 1] = -s  # Transposed: was s
    matrices[..., 1, 0] = s   # Transposed: was -s
    matrices[..., 1, 1] = c
    matrices[..., 2, 2] = 1

    return matrices
