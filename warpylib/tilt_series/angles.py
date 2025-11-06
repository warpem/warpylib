"""
TiltSeries angle transformations

This module contains methods for transforming particle orientations
(Euler angles) across tilts.
"""

import torch
from ..euler import euler_to_matrix, matrix_to_euler, rotate_x, rotate_y, rotate_z


def get_angle_in_all_tilts_single(
    ts: "TiltSeries", coords: torch.Tensor, angles: torch.Tensor = None
) -> torch.Tensor:
    """
    Get Euler angles for coordinates across all tilts.

    Convenience method that replicates coordinates for each tilt and
    calls the main angle transformation method. Optionally applies an
    additional particle rotation if angles are provided.

    Args:
        ts: TiltSeries instance
        coords: Coordinates in volume space (Angstroms), shape (..., 3)
               where ... represents arbitrary batch dimensions
        angles: Optional particle Euler angles in radians (ZYZ convention),
               shape (..., 3). If provided, these rotations are applied
               after the tilt geometry transformations.

    Returns:
        Euler angles for all tilts, shape (..., n_tilts, 3) in radians (ZYZ convention)
    """
    # Replicate coordinates for each tilt: (..., 3) -> (..., n_tilts, 3)
    per_tilt_coords = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    # Replicate angles if provided: (..., 3) -> (..., n_tilts, 3)
    per_tilt_angles = None
    if angles is not None:
        per_tilt_angles = angles.unsqueeze(-2).expand(*angles.shape[:-1], ts.n_tilts, 3)

    # Transform using main method
    return get_angle_in_all_tilts(ts, per_tilt_coords, per_tilt_angles)


def get_angle_in_all_tilts(
    ts: "TiltSeries", coords: torch.Tensor, angles: torch.Tensor = None
) -> torch.Tensor:
    """
    Get Euler angles for coordinates across all tilts.

    This method computes the orientation (as Euler angles) that results from
    the tilt geometry and any spatially varying angle corrections from the
    grid_angle_x/y/z grids. Optionally applies an additional particle rotation
    if angles are provided.

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
               where ... represents arbitrary batch dimensions
        angles: Optional particle Euler angles in radians (ZYZ convention),
               shape (..., n_tilts, 3). If provided, these rotations are applied
               after the tilt geometry transformations.

    Returns:
        Euler angles in radians (ZYZ convention), shape (..., n_tilts, 3)
    """
    # Get device from TiltSeries
    device = ts.angles.device

    # Ensure coords are on the same device as TiltSeries
    coords = coords.to(device)

    # Store original shape and flatten batch dimensions
    original_shape = coords.shape
    batch_shape = original_shape[:-2]

    # Validate that coords has the right number of tilts
    n_tilts = coords.shape[-2]
    if n_tilts != ts.n_tilts:
        raise ValueError(f"coords has {n_tilts} tilts but TiltSeries has {ts.n_tilts}")

    # Flatten to (n_particles, n_tilts, 3)
    coords_flat = coords.reshape(-1, ts.n_tilts, 3)
    n_particles = coords_flat.shape[0]

    # Grid coordinate normalization
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    deg_to_rad = torch.pi / 180.0

    # Prepare grid coordinates for angle grids (3D: X, Y, tilt_index)
    # Normalize XY coordinates: (n_particles, n_tilts, 2)
    normalized_xy = coords_flat[..., :2] / ts.volume_dimensions_physical[:2]

    # Tilt indices: (n_tilts,)
    tilt_indices = torch.arange(ts.n_tilts, dtype=torch.float32, device=device) * grid_step

    # Expand to (n_particles, n_tilts)
    tilt_grid_indices = tilt_indices.unsqueeze(0).expand(n_particles, -1)

    # Stack to (n_particles, n_tilts, 3)
    grid_coords = torch.stack([
        normalized_xy[..., 0],
        normalized_xy[..., 1],
        tilt_grid_indices
    ], dim=-1)

    # Flatten for interpolation: (n_particles * n_tilts, 3)
    grid_coords_flat = grid_coords.reshape(-1, 3)

    # Get angle corrections from grids (in degrees)
    grid_angle_x_interp = ts.grid_angle_x.get_interpolated(grid_coords_flat)
    grid_angle_y_interp = ts.grid_angle_y.get_interpolated(grid_coords_flat)
    grid_angle_z_interp = ts.grid_angle_z.get_interpolated(grid_coords_flat)

    # Reshape back to (n_particles, n_tilts, 3) and convert to radians
    grid_angles = torch.stack([
        grid_angle_x_interp,
        grid_angle_y_interp,
        grid_angle_z_interp
    ], dim=-1).reshape(n_particles, ts.n_tilts, 3) * deg_to_rad

    # Build tilt rotation matrices for each tilt
    # Stack angles into a single tensor (n_tilts, 3)
    euler_angles = torch.stack([
        torch.zeros(ts.n_tilts, dtype=torch.float32, device=device),  # rot = 0
        (ts.angles + ts.level_angle_y) * deg_to_rad,   # tilt
        -ts.tilt_axis_angles * deg_to_rad              # psi
    ], dim=-1)

    # Get Euler matrices (n_tilts, 3, 3)
    tilt_matrices = euler_to_matrix(euler_angles)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad], device=device)).squeeze(0)  # (3, 3)
    tilt_matrices = torch.matmul(tilt_matrices, level_x_matrix)  # (n_tilts, 3, 3)

    # Build correction matrices from angle grids (XYZ extrinsic)
    # C#: Matrix3.RotateZ(...) * Matrix3.RotateY(...) * Matrix3.RotateX(...)
    # Flatten for batched rotation matrix computation: (n_particles * n_tilts, 3)
    grid_angles_flat = grid_angles.reshape(-1, 3)

    # Compute rotation matrices for all particles and tilts at once
    # rotate_x/y/z expect shape (..., 1) so we need to unsqueeze
    rot_z_matrices = rotate_z(grid_angles_flat[:, 2:3])  # (n_flat, 3, 3)
    rot_y_matrices = rotate_y(grid_angles_flat[:, 1:2])  # (n_flat, 3, 3)
    rot_x_matrices = rotate_x(grid_angles_flat[:, 0:1])  # (n_flat, 3, 3)

    # Chain them: Z @ Y @ X
    correction_matrices_flat = torch.matmul(
        torch.matmul(rot_z_matrices, rot_y_matrices),
        rot_x_matrices
    )  # (n_flat, 3, 3)

    # Reshape to (n_particles, n_tilts, 3, 3)
    correction_matrices = correction_matrices_flat.reshape(n_particles, ts.n_tilts, 3, 3)

    # Combined rotation: CorrectionMatrix @ TiltMatrix
    # result[p, t, :, :] = correction_matrices[p, t, :, :] @ tilt_matrices[t, :, :]
    # Using einsum: result[p, t, i, j] = sum_k correction[p, t, i, k] * tilt[t, k, j]
    rotation_matrices = torch.einsum('ptik,tkj->ptij', correction_matrices, tilt_matrices)

    # If particle angles are provided, apply them as an additional rotation
    if angles is not None:
        # Ensure angles are on the same device
        angles = angles.to(device)

        # Flatten angles to (n_particles, n_tilts, 3)
        angles_flat = angles.reshape(-1, ts.n_tilts, 3)

        # Convert to rotation matrices: (n_particles * n_tilts, 3, 3)
        particle_matrices_flat = euler_to_matrix(angles_flat.reshape(-1, 3))

        # Reshape to (n_particles, n_tilts, 3, 3)
        particle_matrices = particle_matrices_flat.reshape(n_particles, ts.n_tilts, 3, 3)

        # Apply particle rotation: rotation_matrices @ particle_matrices
        rotation_matrices = torch.matmul(rotation_matrices, particle_matrices)

    # Extract Euler angles from rotation matrices
    # Flatten to (n_particles * n_tilts, 3, 3)
    rotation_matrices_flat = rotation_matrices.reshape(-1, 3, 3)

    # Convert to Euler angles: (n_particles * n_tilts, 3)
    result_flat = matrix_to_euler(rotation_matrices_flat)

    # Reshape to (n_particles, n_tilts, 3)
    result = result_flat.reshape(n_particles, ts.n_tilts, 3)

    # Reshape back to original batch shape
    result = result.reshape(*batch_shape, ts.n_tilts, 3)

    return result


def get_angles_in_one_tilt(
    ts: "TiltSeries", coords: torch.Tensor, tilt_id: int, angles: torch.Tensor = None
) -> torch.Tensor:
    """
    Get particle Euler angles for a specific tilt.

    More efficient than get_angle_in_all_tilts when you only need
    results for one tilt. Optionally applies an additional particle rotation
    if angles are provided.

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (N, 3)
        tilt_id: Index of the tilt (0 to n_tilts-1)
        angles: Optional particle Euler angles in radians (ZYZ convention), shape (N, 3).
               If provided, these rotations are applied after the tilt geometry transformations.

    Returns:
        Transformed Euler angles, shape (N, 3) in radians (ZYZ convention)
    """
    if tilt_id < 0 or tilt_id >= ts.n_tilts:
        raise ValueError(f"tilt_id must be between 0 and {ts.n_tilts-1}, got {tilt_id}")

    # Get device from TiltSeries
    device = ts.angles.device

    # Ensure coords are on the same device
    coords = coords.to(device)

    n_coords = coords.shape[0]

    # Grid coordinate normalization
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    deg_to_rad = torch.pi / 180.0

    # Build tilt rotation matrix for this specific tilt
    euler_angle = torch.tensor([[
        0.0,  # rot
        (ts.angles[tilt_id] + ts.level_angle_y) * deg_to_rad,  # tilt
        -ts.tilt_axis_angles[tilt_id] * deg_to_rad  # psi
    ]], dtype=torch.float32, device=device)

    tilt_matrix = euler_to_matrix(euler_angle).squeeze(0)  # (3, 3)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad], device=device)).squeeze(0)  # (3, 3)
    tilt_matrix = torch.matmul(tilt_matrix, level_x_matrix)  # (3, 3)

    # Build grid coordinates for all coords at once: (n_coords, 3)
    grid_coords = torch.stack([
        coords[:, 0] / ts.volume_dimensions_physical[0],
        coords[:, 1] / ts.volume_dimensions_physical[1],
        torch.full((n_coords,), tilt_id * grid_step, dtype=torch.float32, device=device)
    ], dim=-1)

    # Get angle corrections from grids (in degrees) for all coords at once
    grid_angle_x_interp = ts.grid_angle_x.get_interpolated(grid_coords)  # (n_coords,)
    grid_angle_y_interp = ts.grid_angle_y.get_interpolated(grid_coords)  # (n_coords,)
    grid_angle_z_interp = ts.grid_angle_z.get_interpolated(grid_coords)  # (n_coords,)

    # Stack and convert to radians: (n_coords, 3)
    grid_angles = torch.stack([
        grid_angle_x_interp,
        grid_angle_y_interp,
        grid_angle_z_interp
    ], dim=-1) * deg_to_rad

    # Build correction matrices from angle grids (XYZ extrinsic)
    # Compute rotation matrices for all coordinates at once
    rot_z_matrices = rotate_z(grid_angles[:, 2:3].squeeze())  # (n_coords, 3, 3)
    rot_y_matrices = rotate_y(grid_angles[:, 1:2].squeeze())  # (n_coords, 3, 3)
    rot_x_matrices = rotate_x(grid_angles[:, 0:1].squeeze())  # (n_coords, 3, 3)

    # Chain them: Z @ Y @ X
    correction_matrices = torch.matmul(
        torch.matmul(rot_z_matrices, rot_y_matrices),
        rot_x_matrices
    )  # (n_coords, 3, 3)

    # Combined rotation: CorrectionMatrix @ TiltMatrix
    # rotation_matrices[i] = correction_matrices[i] @ tilt_matrix
    rotation_matrices = torch.matmul(correction_matrices, tilt_matrix)  # (n_coords, 3, 3)

    # If particle angles are provided, apply them as an additional rotation
    if angles is not None:
        # Ensure angles are on the same device
        angles = angles.to(device)

        # Convert to rotation matrices: (n_coords, 3, 3)
        particle_matrices = euler_to_matrix(angles)

        # Apply particle rotation: rotation_matrices @ particle_matrices
        rotation_matrices = torch.matmul(rotation_matrices, particle_matrices)

    # Extract Euler angles from rotation matrices: (n_coords, 3)
    result = matrix_to_euler(rotation_matrices)

    return result
