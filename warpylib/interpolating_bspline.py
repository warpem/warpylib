"""
Interpolating B-spline implementation in PyTorch.

This module implements cubic interpolating B-splines that match einspline's behavior.
Unlike standard B-splines which use data values as control points, interpolating B-splines
solve for coefficients that make the spline pass exactly through the data points.

This implementation supports:
- Natural boundary conditions (second derivative = 0 at boundaries)
- Uniform grids only
- Full API compliance with torch-cubic-spline-grids
"""

from functools import lru_cache
from typing import Callable, Optional, Tuple, Union
import einops
import torch
import torch.nn as nn


# Einspline's basis matrix (transposed format)
# This is multiplied by [t^3, t^2, t, 1] to get basis functions
EINSPLINE_BASIS_MATRIX = torch.tensor([
    [-1.0/6.0,  3.0/6.0, -3.0/6.0,  1.0/6.0],
    [ 3.0/6.0, -6.0/6.0,  0.0/6.0,  4.0/6.0],
    [-3.0/6.0,  3.0/6.0,  3.0/6.0,  1.0/6.0],
    [ 1.0/6.0,  0.0/6.0,  0.0/6.0,  0.0/6.0]
], dtype=torch.float32)


@lru_cache(maxsize=10000)
def _build_1d_system_matrix(M: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build and cache the (M+2)x(M+2) system matrix for 1D interpolating B-spline.

    Args:
        M: Number of data points
        dtype: Data type for the matrix
        device: Device for the matrix

    Returns:
        A: (M+2, M+2) system matrix
    """
    delta = 1.0 / (M - 1)
    delta_inv_sq = (1.0 / delta) ** 2

    A = torch.zeros(M + 2, M + 2, dtype=dtype, device=device)

    # Row 0: Left boundary condition (second derivative = 0)
    A[0, 0] = delta_inv_sq
    A[0, 1] = -2.0 * delta_inv_sq
    A[0, 2] = delta_inv_sq

    # Rows 1 to M: Interpolation equations (vectorized)
    diag_indices = torch.arange(1, M + 1, device=device)
    A[diag_indices, diag_indices - 1] = 1.0 / 6.0
    A[diag_indices, diag_indices] = 2.0 / 3.0
    A[diag_indices, diag_indices + 1] = 1.0 / 6.0

    # Row M+1: Right boundary condition (second derivative = 0)
    A[M + 1, M - 1] = delta_inv_sq
    A[M + 1, M] = -2.0 * delta_inv_sq
    A[M + 1, M + 1] = delta_inv_sq

    return A


def find_coefs_1d(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Find B-spline coefficients for 1D data with natural boundary conditions.

    Solves the linear system Ac = d where:
    - A is the (M+2)x(M+2) system matrix
    - c are the B-spline coefficients (unknowns)
    - d is the RHS vector (data values with boundary conditions)

    Natural boundary condition: second derivative = 0 at boundaries.

    Args:
        data: (M,) or (C, M) data values to interpolate.
              If 2D, solves independently for each channel.

    Returns:
        coefs: (M+2,) or (C, M+2) B-spline coefficients
    """
    if data.ndim == 1:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, M = data.shape

    # Build system matrix (same for all channels)
    A = _build_1d_system_matrix(M, dtype=data.dtype, device=data.device)

    # Build RHS matrix for all channels at once: (C, M+2)
    B = torch.zeros(C, M + 2, dtype=data.dtype, device=data.device)
    B[:, 1:M+1] = data  # Interpolation equations RHS (boundaries stay 0)

    # Batched solve: A is (M+2, M+2), B.T is (M+2, C)
    # Result is (M+2, C), transpose to (C, M+2)
    coefs = torch.linalg.solve(A, B.T).T

    if squeeze_output:
        coefs = coefs.squeeze(0)

    return coefs


def find_coefs_2d(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Find B-spline coefficients for 2D data with natural boundary conditions.

    Uses separable approach with batched solves:
    1. Solve 1D problems along X-direction for all (C, Y) positions at once
    2. Solve 1D problems along Y-direction for all (C, X) positions at once

    Args:
        data: (C, Mx, My) data values to interpolate

    Returns:
        coefs: (C, Mx+2, My+2) B-spline coefficients
    """
    if data.ndim == 2:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, Mx, My = data.shape

    # Build system matrices (reused for all solves)
    Ax = _build_1d_system_matrix(Mx, dtype=data.dtype, device=data.device)
    Ay = _build_1d_system_matrix(My, dtype=data.dtype, device=data.device)

    # Step 1: Solve along X-direction for all (C, Y) positions at once
    # Reshape data: (C, Mx, My) -> (C*My, Mx)
    data_flat = data.permute(0, 2, 1).reshape(-1, Mx)  # (C*My, Mx)

    # Build RHS matrix: (C*My, Mx+2)
    B_x = torch.zeros(data_flat.shape[0], Mx + 2, dtype=data.dtype, device=data.device)
    B_x[:, 1:Mx+1] = data_flat

    # Batched solve: Ax is (Mx+2, Mx+2), B_x.T is (Mx+2, C*My)
    coefs_x_flat = torch.linalg.solve(Ax, B_x.T).T  # (C*My, Mx+2)
    coefs_x = coefs_x_flat.reshape(C, My, Mx + 2).permute(0, 2, 1)  # (C, Mx+2, My)

    # Step 2: Solve along Y-direction for all (C, X) positions at once
    # Reshape: (C, Mx+2, My) -> (C*(Mx+2), My)
    coefs_x_flat = coefs_x.reshape(-1, My)  # (C*(Mx+2), My)

    # Build RHS matrix: (C*(Mx+2), My+2)
    B_y = torch.zeros(coefs_x_flat.shape[0], My + 2, dtype=data.dtype, device=data.device)
    B_y[:, 1:My+1] = coefs_x_flat

    # Batched solve
    coefs_flat = torch.linalg.solve(Ay, B_y.T).T  # (C*(Mx+2), My+2)
    coefs = coefs_flat.reshape(C, Mx + 2, My + 2)

    if squeeze_output:
        coefs = coefs.squeeze(0)

    return coefs


def interpolate_grid_1d(
    data: torch.Tensor,
    u: torch.Tensor,
    matrix: Optional[torch.Tensor] = None,
    monotonicity: Optional[str] = None,
) -> torch.Tensor:
    """
    Interpolate 1D grid data using interpolating B-splines.

    This function signature matches torch-cubic-spline-grids API for compatibility.

    Args:
        data: (C, M+2) padded grid data (B-spline coefficients)
        u: (B, 1) batch of coordinates in [0, 1]
        matrix: Ignored (for API compatibility - einspline uses fixed basis matrix)
        monotonicity: Optional monotonicity constraint (not yet implemented)

    Returns:
        values: (B, C) interpolated values
    """
    _ = matrix  # Unused, for API compatibility
    if monotonicity is not None:
        raise NotImplementedError("Monotonicity constraints not yet implemented for interpolating B-splines")

    # u is (B, 1), squeeze to (B,)
    u = u.squeeze(-1)

    C, M_plus_2 = data.shape
    M = M_plus_2 - 2
    B = u.shape[0]

    # Get basis matrix on correct device
    basis_matrix = EINSPLINE_BASIS_MATRIX.to(device=data.device, dtype=data.dtype)

    # Transform coordinates to grid space
    # u is in [0, 1], map to [0, M-1] in grid indices
    u_norm = u * (M - 1)

    # Find grid cell and local coordinate
    i = torch.floor(u_norm).long()
    t = u_norm - i.float()

    # Handle out-of-bounds cases
    mask_low = u_norm < 0
    i = torch.where(mask_low, torch.zeros_like(i), i)
    t = torch.where(mask_low, u_norm, t)

    mask_high = u_norm >= M - 2
    i = torch.where(mask_high, torch.full_like(i, M - 2), i)
    t = torch.where(mask_high, u_norm - (M - 2), t)

    # Build power vector [t^3, t^2, t, 1]: (B, 4)
    t_powers = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=1)

    # Compute basis functions: (4, B)
    basis_vals = torch.matmul(basis_matrix, t_powers.T)

    # Extract 4 control points for each query point across all channels
    # Use advanced indexing: data[:, i+offset] for offset in 0..3
    # Build index tensor: (4,) offsets + (B,) indices -> broadcast to (4, B)
    offsets = torch.arange(4, device=data.device)
    indices = i.unsqueeze(0) + offsets.unsqueeze(1)  # (4, B)

    # Gather control points: data is (C, M+2), indices is (4, B)
    # Result: (C, 4, B)
    control_points = data[:, indices]  # (C, 4, B)

    # Compute weighted sum: control_points * basis_vals, sum over the 4 dimension
    # control_points: (C, 4, B), basis_vals: (4, B) -> broadcast to (C, 4, B)
    # Result: (C, B) -> transpose to (B, C)
    values = torch.sum(control_points * basis_vals.unsqueeze(0), dim=1).T

    return values


def find_coefs_3d(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Find B-spline coefficients for 3D data with natural boundary conditions.

    Uses separable approach with batched solves:
    1. Solve 1D problems along X-direction for all (C, Y, Z) positions at once
    2. Solve 1D problems along Y-direction for all (C, X, Z) positions at once
    3. Solve 1D problems along Z-direction for all (C, X, Y) positions at once

    Args:
        data: (C, Mx, My, Mz) data values to interpolate

    Returns:
        coefs: (C, Mx+2, My+2, Mz+2) B-spline coefficients
    """
    if data.ndim == 3:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, Mx, My, Mz = data.shape

    # Build system matrices (reused for all solves)
    Ax = _build_1d_system_matrix(Mx, dtype=data.dtype, device=data.device)
    Ay = _build_1d_system_matrix(My, dtype=data.dtype, device=data.device)
    Az = _build_1d_system_matrix(Mz, dtype=data.dtype, device=data.device)

    # Step 1: Solve along X-direction for all (C, Y, Z) positions at once
    # Reshape data: (C, Mx, My, Mz) -> (C*My*Mz, Mx)
    data_flat = data.permute(0, 2, 3, 1).reshape(-1, Mx)  # (C*My*Mz, Mx)

    # Build RHS matrix: (C*My*Mz, Mx+2)
    B_x = torch.zeros(data_flat.shape[0], Mx + 2, dtype=data.dtype, device=data.device)
    B_x[:, 1:Mx+1] = data_flat

    # Batched solve: Ax is (Mx+2, Mx+2), B_x.T is (Mx+2, C*My*Mz)
    coefs_x_flat = torch.linalg.solve(Ax, B_x.T).T  # (C*My*Mz, Mx+2)
    coefs_x = coefs_x_flat.reshape(C, My, Mz, Mx + 2).permute(0, 3, 1, 2)  # (C, Mx+2, My, Mz)

    # Step 2: Solve along Y-direction for all (C, X, Z) positions at once
    # Reshape: (C, Mx+2, My, Mz) -> (C*(Mx+2)*Mz, My)
    coefs_x_flat = coefs_x.permute(0, 1, 3, 2).reshape(-1, My)  # (C*(Mx+2)*Mz, My)

    # Build RHS matrix: (C*(Mx+2)*Mz, My+2)
    B_y = torch.zeros(coefs_x_flat.shape[0], My + 2, dtype=data.dtype, device=data.device)
    B_y[:, 1:My+1] = coefs_x_flat

    # Batched solve
    coefs_xy_flat = torch.linalg.solve(Ay, B_y.T).T  # (C*(Mx+2)*Mz, My+2)
    coefs_xy = coefs_xy_flat.reshape(C, Mx + 2, Mz, My + 2).permute(0, 1, 3, 2)  # (C, Mx+2, My+2, Mz)

    # Step 3: Solve along Z-direction for all (C, X, Y) positions at once
    # Reshape: (C, Mx+2, My+2, Mz) -> (C*(Mx+2)*(My+2), Mz)
    coefs_xy_flat = coefs_xy.reshape(-1, Mz)  # (C*(Mx+2)*(My+2), Mz)

    # Build RHS matrix: (C*(Mx+2)*(My+2), Mz+2)
    B_z = torch.zeros(coefs_xy_flat.shape[0], Mz + 2, dtype=data.dtype, device=data.device)
    B_z[:, 1:Mz+1] = coefs_xy_flat

    # Batched solve
    coefs_flat = torch.linalg.solve(Az, B_z.T).T  # (C*(Mx+2)*(My+2), Mz+2)
    coefs = coefs_flat.reshape(C, Mx + 2, My + 2, Mz + 2)

    if squeeze_output:
        coefs = coefs.squeeze(0)

    return coefs


def interpolate_grid_2d(
    data: torch.Tensor,
    u: torch.Tensor,
    matrix: Optional[torch.Tensor] = None,
    monotonicity: Optional[str] = None,
) -> torch.Tensor:
    """
    Interpolate 2D grid data using interpolating B-splines.

    This function signature matches torch-cubic-spline-grids API for compatibility.

    Args:
        data: (C, Mx+2, My+2) padded grid data (B-spline coefficients)
        u: (B, 2) batch of coordinates in [0, 1] x [0, 1]
        matrix: Ignored (for API compatibility - einspline uses fixed basis matrix)
        monotonicity: Optional monotonicity constraint (not yet implemented)

    Returns:
        values: (B, C) interpolated values
    """
    _ = matrix  # Unused, for API compatibility
    if monotonicity is not None:
        raise NotImplementedError("Monotonicity constraints not yet implemented for interpolating B-splines")

    C, Mx_plus_2, My_plus_2 = data.shape
    Mx = Mx_plus_2 - 2
    My = My_plus_2 - 2
    B = u.shape[0]

    # Extract x and y coordinates
    ux = u[:, 0]  # (B,)
    uy = u[:, 1]  # (B,)

    # Get basis matrix on correct device
    basis_matrix = EINSPLINE_BASIS_MATRIX.to(device=data.device, dtype=data.dtype)

    # Transform coordinates to grid space
    ux_norm = ux * (Mx - 1)
    uy_norm = uy * (My - 1)

    # Find grid cells and local coordinates for X dimension
    ix = torch.floor(ux_norm).long()
    tx = ux_norm - ix.float()

    # Handle X boundary cases
    mask_low_x = ux_norm < 0
    ix = torch.where(mask_low_x, torch.zeros_like(ix), ix)
    tx = torch.where(mask_low_x, ux_norm, tx)

    mask_high_x = ux_norm >= Mx - 2
    ix = torch.where(mask_high_x, torch.full_like(ix, Mx - 2), ix)
    tx = torch.where(mask_high_x, ux_norm - (Mx - 2), tx)

    # Find grid cells and local coordinates for Y dimension
    iy = torch.floor(uy_norm).long()
    ty = uy_norm - iy.float()

    # Handle Y boundary cases
    mask_low_y = uy_norm < 0
    iy = torch.where(mask_low_y, torch.zeros_like(iy), iy)
    ty = torch.where(mask_low_y, uy_norm, ty)

    mask_high_y = uy_norm >= My - 2
    iy = torch.where(mask_high_y, torch.full_like(iy, My - 2), iy)
    ty = torch.where(mask_high_y, uy_norm - (My - 2), ty)

    # Build power vectors: (B, 4)
    tx_powers = torch.stack([tx**3, tx**2, tx, torch.ones_like(tx)], dim=1)
    ty_powers = torch.stack([ty**3, ty**2, ty, torch.ones_like(ty)], dim=1)

    # Compute basis functions: (4, B)
    basis_x = torch.matmul(basis_matrix, tx_powers.T)
    basis_y = torch.matmul(basis_matrix, ty_powers.T)

    # Build index tensors for 4x4 control point extraction
    offsets = torch.arange(4, device=data.device)
    # ix_expanded: (4, B), iy_expanded: (4, B)
    ix_expanded = ix.unsqueeze(0) + offsets.unsqueeze(1)  # (4, B)
    iy_expanded = iy.unsqueeze(0) + offsets.unsqueeze(1)  # (4, B)

    # Extract 4x4 control points for all channels at once
    # data: (C, Mx+2, My+2)
    # We want: control_grid[c, i, j, b] = data[c, ix[b]+i, iy[b]+j]
    # Result shape: (C, 4, 4, B)
    control_grid = data[:, ix_expanded[:, None, :], iy_expanded[None, :, :]]
    # This gives (C, 4, 4, B) due to broadcasting

    # Apply separable 2D cubic interpolation vectorized across all channels
    # Step 1: Apply basis in X direction
    # control_grid: (C, 4, 4, B), basis_x: (4, B)
    # For each c, j, b: sum over i of control_grid[c, i, j, b] * basis_x[i, b]
    # Result: (C, 4, B)
    intermediate = torch.einsum('cijb,ib->cjb', control_grid, basis_x)

    # Step 2: Apply basis in Y direction
    # intermediate: (C, 4, B), basis_y: (4, B)
    # For each c, b: sum over j of intermediate[c, j, b] * basis_y[j, b]
    # Result: (C, B)
    values = torch.einsum('cjb,jb->cb', intermediate, basis_y)

    # Transpose to (B, C)
    return values.T


def interpolate_grid_3d(
    data: torch.Tensor,
    u: torch.Tensor,
    matrix: Optional[torch.Tensor] = None,
    monotonicity: Optional[str] = None,
) -> torch.Tensor:
    """
    Interpolate 3D grid data using interpolating B-splines.

    This function signature matches torch-cubic-spline-grids API for compatibility.

    Args:
        data: (C, Mx+2, My+2, Mz+2) padded grid data (B-spline coefficients)
        u: (B, 3) batch of coordinates in [0, 1]^3
        matrix: Ignored (for API compatibility - einspline uses fixed basis matrix)
        monotonicity: Optional monotonicity constraint (not yet implemented)

    Returns:
        values: (B, C) interpolated values
    """
    _ = matrix  # Unused, for API compatibility
    if monotonicity is not None:
        raise NotImplementedError("Monotonicity constraints not yet implemented for interpolating B-splines")

    C, Mx_plus_2, My_plus_2, Mz_plus_2 = data.shape
    Mx = Mx_plus_2 - 2
    My = My_plus_2 - 2
    Mz = Mz_plus_2 - 2
    B = u.shape[0]

    # Extract x, y, z coordinates
    ux = u[:, 0]  # (B,)
    uy = u[:, 1]  # (B,)
    uz = u[:, 2]  # (B,)

    # Get basis matrix on correct device
    basis_matrix = EINSPLINE_BASIS_MATRIX.to(device=data.device, dtype=data.dtype)

    # Transform coordinates to grid space
    ux_norm = ux * (Mx - 1)
    uy_norm = uy * (My - 1)
    uz_norm = uz * (Mz - 1)

    # Find grid cells and local coordinates for X dimension
    ix = torch.floor(ux_norm).long()
    tx = ux_norm - ix.float()

    # Handle X boundary cases
    mask_low_x = ux_norm < 0
    ix = torch.where(mask_low_x, torch.zeros_like(ix), ix)
    tx = torch.where(mask_low_x, ux_norm, tx)

    mask_high_x = ux_norm >= Mx - 2
    ix = torch.where(mask_high_x, torch.full_like(ix, Mx - 2), ix)
    tx = torch.where(mask_high_x, ux_norm - (Mx - 2), tx)

    # Find grid cells and local coordinates for Y dimension
    iy = torch.floor(uy_norm).long()
    ty = uy_norm - iy.float()

    # Handle Y boundary cases
    mask_low_y = uy_norm < 0
    iy = torch.where(mask_low_y, torch.zeros_like(iy), iy)
    ty = torch.where(mask_low_y, uy_norm, ty)

    mask_high_y = uy_norm >= My - 2
    iy = torch.where(mask_high_y, torch.full_like(iy, My - 2), iy)
    ty = torch.where(mask_high_y, uy_norm - (My - 2), ty)

    # Find grid cells and local coordinates for Z dimension
    iz = torch.floor(uz_norm).long()
    tz = uz_norm - iz.float()

    # Handle Z boundary cases
    mask_low_z = uz_norm < 0
    iz = torch.where(mask_low_z, torch.zeros_like(iz), iz)
    tz = torch.where(mask_low_z, uz_norm, tz)

    mask_high_z = uz_norm >= Mz - 2
    iz = torch.where(mask_high_z, torch.full_like(iz, Mz - 2), iz)
    tz = torch.where(mask_high_z, uz_norm - (Mz - 2), tz)

    # Build power vectors: (B, 4)
    tx_powers = torch.stack([tx**3, tx**2, tx, torch.ones_like(tx)], dim=1)
    ty_powers = torch.stack([ty**3, ty**2, ty, torch.ones_like(ty)], dim=1)
    tz_powers = torch.stack([tz**3, tz**2, tz, torch.ones_like(tz)], dim=1)

    # Compute basis functions: (4, B)
    basis_x = torch.matmul(basis_matrix, tx_powers.T)
    basis_y = torch.matmul(basis_matrix, ty_powers.T)
    basis_z = torch.matmul(basis_matrix, tz_powers.T)

    # Build index tensors for 4x4x4 control point extraction
    offsets = torch.arange(4, device=data.device)
    # Each expanded: (4, B)
    ix_expanded = ix.unsqueeze(0) + offsets.unsqueeze(1)
    iy_expanded = iy.unsqueeze(0) + offsets.unsqueeze(1)
    iz_expanded = iz.unsqueeze(0) + offsets.unsqueeze(1)

    # Extract 4x4x4 control points for all channels at once
    # data: (C, Mx+2, My+2, Mz+2)
    # We want: control_grid[c, i, j, k, b] = data[c, ix[b]+i, iy[b]+j, iz[b]+k]
    # Use advanced indexing with broadcasting
    # ix_expanded[:, None, None, :] -> (4, 1, 1, B)
    # iy_expanded[None, :, None, :] -> (1, 4, 1, B)
    # iz_expanded[None, None, :, :] -> (1, 1, 4, B)
    control_grid = data[:,
                        ix_expanded[:, None, None, :],
                        iy_expanded[None, :, None, :],
                        iz_expanded[None, None, :, :]]
    # Result shape: (C, 4, 4, 4, B)

    # Apply separable 3D cubic interpolation vectorized across all channels
    # Step 1: Apply basis in X direction
    # control_grid: (C, 4, 4, 4, B), basis_x: (4, B)
    # For each c, j, k, b: sum over i of control_grid[c, i, j, k, b] * basis_x[i, b]
    # Result: (C, 4, 4, B)
    intermediate_yz = torch.einsum('cijkb,ib->cjkb', control_grid, basis_x)

    # Step 2: Apply basis in Y direction
    # intermediate_yz: (C, 4, 4, B), basis_y: (4, B)
    # For each c, k, b: sum over j of intermediate_yz[c, j, k, b] * basis_y[j, b]
    # Result: (C, 4, B)
    intermediate_z = torch.einsum('cjkb,jb->ckb', intermediate_yz, basis_y)

    # Step 3: Apply basis in Z direction
    # intermediate_z: (C, 4, B), basis_z: (4, B)
    # For each c, b: sum over k of intermediate_z[c, k, b] * basis_z[k, b]
    # Result: (C, B)
    values = torch.einsum('ckb,kb->cb', intermediate_z, basis_z)

    # Transpose to (B, C)
    return values.T


def coerce_to_multichannel_grid(grid: torch.Tensor, grid_ndim: int) -> torch.Tensor:
    """If missing, add a channel dimension to a multidimensional grid.

    e.g. for a 1D (M,) grid
          `M -> 1 M`
        `C M -> C M`
    """
    grid_is_multichannel = grid.ndim == grid_ndim + 1
    grid_is_single_channel = grid.ndim == grid_ndim
    if grid_is_single_channel is False and grid_is_multichannel is False:
        raise ValueError(f'expected a {grid_ndim}D grid, got {grid.ndim}')
    if grid_is_single_channel:
        grid = einops.rearrange(grid, '... -> 1 ...')
    return grid


def batch_iterator(iterable, n: int = 1):
    """Split an iterable into batches of constant length."""
    max_len = len(iterable)
    for idx in range(0, max_len, n):
        yield iterable[idx : min(idx + n, max_len)]


class InterpolatingBSpline1d(nn.Module):
    """
    1D Interpolating B-spline grid with API compliance to torch-cubic-spline-grids.

    This implements cubic B-splines that interpolate (pass through) the data points,
    matching einspline's behavior with natural boundary conditions.
    The coefficients are computed by solving a linear system.

    Args:
        resolution: Number of data points (int or tuple)
        n_channels: Number of channels
        minibatch_size: Maximum batch size for evaluation
    """

    ndim: int = 1
    _interpolation_function: Callable = staticmethod(interpolate_grid_1d)
    _interpolation_matrix: torch.Tensor = EINSPLINE_BASIS_MATRIX
    _data: nn.Parameter
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Union[int, Tuple[int]]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[str] = None,
    ):
        super().__init__()

        if resolution is None:
            resolution = 2
        if isinstance(resolution, tuple):
            resolution = resolution[0]

        # Initialize data as learnable parameters
        grid_shape = (n_channels, resolution)
        self.data = torch.zeros(size=grid_shape)

        self._minibatch_size = minibatch_size
        self._monotonicity = monotonicity

        # Register interpolation matrix as buffer
        self.register_buffer(
            name='interpolation_matrix',
            tensor=self._interpolation_matrix,
            persistent=False,
        )

        # Cache for coefficients
        self._coefs_cache = None
        self._data_version = None

    @property
    def data(self) -> torch.Tensor:
        """Get data values."""
        return self._data.detach()

    @data.setter
    def data(self, grid_data: torch.Tensor) -> None:
        """Set data values."""
        grid_data = coerce_to_multichannel_grid(grid_data, grid_ndim=self.ndim)
        self._data = nn.Parameter(grid_data)
        self._coefs_cache = None  # Invalidate cache

    @property
    def n_channels(self) -> int:
        """Number of channels in the grid."""
        return int(self._data.size(0))

    @property
    def resolution(self) -> Tuple[int, ...]:
        """Grid resolution (number of data points per dimension)."""
        return tuple(self._data.shape[1:])

    def _compute_coefs(self) -> torch.Tensor:
        """Compute B-spline coefficients from data."""
        return find_coefs_1d(self._data)

    def _get_coefs_cached(self) -> torch.Tensor:
        """Get coefficients with caching for inference."""
        current_version = self._data._version
        if self._coefs_cache is None or self._data_version != current_version:
            with torch.no_grad():
                self._coefs_cache = self._compute_coefs()
                self._data_version = current_version
        return self._coefs_cache

    def _interpolate(self, u: torch.Tensor) -> torch.Tensor:
        """Interpolate at given coordinates (internal method)."""
        # Compute coefficients (this will be part of computational graph if training)
        if self.training or torch.is_grad_enabled():
            coefs = self._compute_coefs()
        else:
            coefs = self._get_coefs_cached()

        return self._interpolation_function(
            coefs,
            u,
            matrix=self.interpolation_matrix,
            monotonicity=self._monotonicity,
        )

    def _coerce_to_batched_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        """Convert input coordinates to batched format (B, 1)."""
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim

        if self._input_is_coordinate_like is False and self.ndim == 1:
            u = einops.rearrange(u, '... -> ... 1')  # add singleton coord dimension
        else:
            u = torch.atleast_2d(u)  # add batch dimension if missing

        u, self._packed_shapes = einops.pack([u], pattern='* coords')

        if u.shape[-1] != self.ndim:
            ndim = u.shape[-1]
            raise ValueError(
                f'Cannot interpolate on a {self.ndim}D grid with {ndim}D coordinates'
            )
        return u

    def _unpack_interpolated_output(self, interpolated: torch.Tensor) -> torch.Tensor:
        """Convert batched output back to input format."""
        [interpolated] = einops.unpack(
            interpolated, packed_shapes=self._packed_shapes, pattern='* coords'
        )
        # Only squeeze the coordinate dimension if input was not coordinate-like
        # and we have a single channel (otherwise we want to keep the channel dimension)
        if self._input_is_coordinate_like is False and self.ndim == 1 and self.n_channels == 1:
            interpolated = einops.rearrange(interpolated, '... 1 -> ...')
        return interpolated

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline at given coordinates.

        Args:
            u: Coordinates in [0, 1] to evaluate. Can be:
               - (N,) for N points
               - (N, 1) for N points with explicit dimension
               - (..., 1) for arbitrary batch shapes

        Returns:
            values: Interpolated values with shape matching input:
                    - (N,) if single channel and input is (N,)
                    - (N, C) if multiple channels
                    - (..., C) for arbitrary batch shapes
        """
        u = self._coerce_to_batched_coordinates(u)  # (B, 1)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch_iterator(u, n=self._minibatch_size)
        ]  # List[Tensor[(B, C)]]
        interpolated = torch.cat(interpolated, dim=0)  # (B, C)

        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> "InterpolatingBSpline1d":
        """
        Create spline from existing data (API-compliant factory method).

        Args:
            data: (M,) or (C, M) data values

        Returns:
            Initialized InterpolatingBSpline1d instance
        """
        grid = cls()
        grid.data = data
        return grid


class InterpolatingBSpline2d(nn.Module):
    """
    2D Interpolating B-spline grid with API compliance to torch-cubic-spline-grids.

    This implements cubic B-splines that interpolate (pass through) the data points,
    matching einspline's behavior with natural boundary conditions.
    The coefficients are computed by solving separable linear systems.

    Args:
        resolution: Number of data points per dimension (tuple of 2 ints)
        n_channels: Number of channels
        minibatch_size: Maximum batch size for evaluation
    """

    ndim: int = 2
    _interpolation_function: Callable = staticmethod(interpolate_grid_2d)
    _interpolation_matrix: torch.Tensor = EINSPLINE_BASIS_MATRIX
    _data: nn.Parameter
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[str] = None,
    ):
        super().__init__()

        if resolution is None:
            resolution = (2, 2)

        # Initialize data as learnable parameters
        grid_shape = (n_channels, *resolution)
        self.data = torch.zeros(size=grid_shape)

        self._minibatch_size = minibatch_size
        self._monotonicity = monotonicity

        # Register interpolation matrix as buffer
        self.register_buffer(
            name='interpolation_matrix',
            tensor=self._interpolation_matrix,
            persistent=False,
        )

        # Cache for coefficients
        self._coefs_cache = None
        self._data_version = None

    @property
    def data(self) -> torch.Tensor:
        """Get data values."""
        return self._data.detach()

    @data.setter
    def data(self, grid_data: torch.Tensor) -> None:
        """Set data values."""
        grid_data = coerce_to_multichannel_grid(grid_data, grid_ndim=self.ndim)
        self._data = nn.Parameter(grid_data)
        self._coefs_cache = None  # Invalidate cache

    @property
    def n_channels(self) -> int:
        """Number of channels in the grid."""
        return int(self._data.size(0))

    @property
    def resolution(self) -> Tuple[int, ...]:
        """Grid resolution (number of data points per dimension)."""
        return tuple(self._data.shape[1:])

    def _compute_coefs(self) -> torch.Tensor:
        """Compute B-spline coefficients from data."""
        return find_coefs_2d(self._data)

    def _get_coefs_cached(self) -> torch.Tensor:
        """Get coefficients with caching for inference."""
        current_version = self._data._version
        if self._coefs_cache is None or self._data_version != current_version:
            with torch.no_grad():
                self._coefs_cache = self._compute_coefs()
                self._data_version = current_version
        return self._coefs_cache

    def _interpolate(self, u: torch.Tensor) -> torch.Tensor:
        """Interpolate at given coordinates (internal method)."""
        # Compute coefficients (this will be part of computational graph if training)
        if self.training or torch.is_grad_enabled():
            coefs = self._compute_coefs()
        else:
            coefs = self._get_coefs_cached()

        return self._interpolation_function(
            coefs,
            u,
            matrix=self.interpolation_matrix,
            monotonicity=self._monotonicity,
        )

    def _coerce_to_batched_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        """Convert input coordinates to batched format (B, 2)."""
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim

        if self._input_is_coordinate_like is False:
            raise ValueError(
                f'For 2D grids, coordinates must have shape (..., 2), got {u.shape}'
            )

        u = torch.atleast_2d(u)  # add batch dimension if missing
        u, self._packed_shapes = einops.pack([u], pattern='* coords')

        if u.shape[-1] != self.ndim:
            ndim = u.shape[-1]
            raise ValueError(
                f'Cannot interpolate on a {self.ndim}D grid with {ndim}D coordinates'
            )
        return u

    def _unpack_interpolated_output(self, interpolated: torch.Tensor) -> torch.Tensor:
        """Convert batched output back to input format."""
        [interpolated] = einops.unpack(
            interpolated, packed_shapes=self._packed_shapes, pattern='* coords'
        )
        return interpolated

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline at given coordinates.

        Args:
            u: Coordinates in [0, 1] x [0, 1] to evaluate. Must have shape:
               - (N, 2) for N points
               - (..., 2) for arbitrary batch shapes

        Returns:
            values: Interpolated values with shape:
                    - (N, C) for C channels
                    - (..., C) for arbitrary batch shapes
        """
        u = self._coerce_to_batched_coordinates(u)  # (B, 2)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch_iterator(u, n=self._minibatch_size)
        ]  # List[Tensor[(B, C)]]
        interpolated = torch.cat(interpolated, dim=0)  # (B, C)

        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> "InterpolatingBSpline2d":
        """
        Create spline from existing data (API-compliant factory method).

        Args:
            data: (Mx, My) or (C, Mx, My) data values

        Returns:
            Initialized InterpolatingBSpline2d instance
        """
        grid = cls()
        grid.data = data
        return grid


class InterpolatingBSpline3d(nn.Module):
    """
    3D Interpolating B-spline grid with API compliance to torch-cubic-spline-grids.

    This implements cubic B-splines that interpolate (pass through) the data points,
    matching einspline's behavior with natural boundary conditions.
    The coefficients are computed by solving separable linear systems.

    Args:
        resolution: Number of data points per dimension (tuple of 3 ints)
        n_channels: Number of channels
        minibatch_size: Maximum batch size for evaluation
    """

    ndim: int = 3
    _interpolation_function: Callable = staticmethod(interpolate_grid_3d)
    _interpolation_matrix: torch.Tensor = EINSPLINE_BASIS_MATRIX
    _data: nn.Parameter
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Tuple[int, int, int]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[str] = None,
    ):
        super().__init__()

        if resolution is None:
            resolution = (2, 2, 2)

        # Initialize data as learnable parameters
        grid_shape = (n_channels, *resolution)
        self.data = torch.zeros(size=grid_shape)

        self._minibatch_size = minibatch_size
        self._monotonicity = monotonicity

        # Register interpolation matrix as buffer
        self.register_buffer(
            name='interpolation_matrix',
            tensor=self._interpolation_matrix,
            persistent=False,
        )

        # Cache for coefficients
        self._coefs_cache = None
        self._data_version = None

    @property
    def data(self) -> torch.Tensor:
        """Get data values."""
        return self._data.detach()

    @data.setter
    def data(self, grid_data: torch.Tensor) -> None:
        """Set data values."""
        grid_data = coerce_to_multichannel_grid(grid_data, grid_ndim=self.ndim)
        self._data = nn.Parameter(grid_data)
        self._coefs_cache = None  # Invalidate cache

    @property
    def n_channels(self) -> int:
        """Number of channels in the grid."""
        return int(self._data.size(0))

    @property
    def resolution(self) -> Tuple[int, ...]:
        """Grid resolution (number of data points per dimension)."""
        return tuple(self._data.shape[1:])

    def _compute_coefs(self) -> torch.Tensor:
        """Compute B-spline coefficients from data."""
        return find_coefs_3d(self._data)

    def _get_coefs_cached(self) -> torch.Tensor:
        """Get coefficients with caching for inference."""
        current_version = self._data._version
        if self._coefs_cache is None or self._data_version != current_version:
            with torch.no_grad():
                self._coefs_cache = self._compute_coefs()
                self._data_version = current_version
        return self._coefs_cache

    def _interpolate(self, u: torch.Tensor) -> torch.Tensor:
        """Interpolate at given coordinates (internal method)."""
        # Compute coefficients (this will be part of computational graph if training)
        if self.training or torch.is_grad_enabled():
            coefs = self._compute_coefs()
        else:
            coefs = self._get_coefs_cached()

        return self._interpolation_function(
            coefs,
            u,
            matrix=self.interpolation_matrix,
            monotonicity=self._monotonicity,
        )

    def _coerce_to_batched_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        """Convert input coordinates to batched format (B, 3)."""
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim

        if self._input_is_coordinate_like is False:
            raise ValueError(
                f'For 3D grids, coordinates must have shape (..., 3), got {u.shape}'
            )

        u = torch.atleast_2d(u)  # add batch dimension if missing
        u, self._packed_shapes = einops.pack([u], pattern='* coords')

        if u.shape[-1] != self.ndim:
            ndim = u.shape[-1]
            raise ValueError(
                f'Cannot interpolate on a {self.ndim}D grid with {ndim}D coordinates'
            )
        return u

    def _unpack_interpolated_output(self, interpolated: torch.Tensor) -> torch.Tensor:
        """Convert batched output back to input format."""
        [interpolated] = einops.unpack(
            interpolated, packed_shapes=self._packed_shapes, pattern='* coords'
        )
        return interpolated

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline at given coordinates.

        Args:
            u: Coordinates in [0, 1]^3 to evaluate. Must have shape:
               - (N, 3) for N points
               - (..., 3) for arbitrary batch shapes

        Returns:
            values: Interpolated values with shape:
                    - (N, C) for C channels
                    - (..., C) for arbitrary batch shapes
        """
        u = self._coerce_to_batched_coordinates(u)  # (B, 3)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch_iterator(u, n=self._minibatch_size)
        ]  # List[Tensor[(B, C)]]
        interpolated = torch.cat(interpolated, dim=0)  # (B, C)

        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> "InterpolatingBSpline3d":
        """
        Create spline from existing data (API-compliant factory method).

        Args:
            data: (Mx, My, Mz) or (C, Mx, My, Mz) data values

        Returns:
            Initialized InterpolatingBSpline3d instance
        """
        grid = cls()
        grid.data = data
        return grid
