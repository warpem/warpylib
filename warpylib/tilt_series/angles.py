"""
TiltSeries angle transformations

This module contains methods for transforming particle orientations
(Euler angles) across tilts.
"""

import torch
from ..euler import euler_to_matrix, matrix_to_euler, rotate_x, rotate_y, rotate_z


def get_angle_in_all_tilts_single(ts: "TiltSeries", coords: torch.Tensor) -> torch.Tensor:
    """
    Get Euler angles for coordinates across all tilts.

    Convenience method that replicates coordinates for each tilt and
    calls the main angle transformation method.

    Args:
        ts: TiltSeries instance
        coords: Coordinates in volume space (Angstroms), shape (..., 3)
               where ... represents arbitrary batch dimensions

    Returns:
        Euler angles for all tilts, shape (..., n_tilts, 3) in radians (ZYZ convention)
    """
    # Replicate coordinates for each tilt: (..., 3) -> (..., n_tilts, 3)
    per_tilt_coords = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    # Transform using main method
    return get_angle_in_all_tilts(ts, per_tilt_coords)


def get_angle_in_all_tilts(ts: "TiltSeries", coords: torch.Tensor) -> torch.Tensor:
    """
    Get Euler angles for coordinates across all tilts.

    This method computes the orientation (as Euler angles) that results from
    the tilt geometry and any spatially varying angle corrections from the
    grid_angle_x/y/z grids.

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
               where ... represents arbitrary batch dimensions

    Returns:
        Euler angles in radians (ZYZ convention), shape (..., n_tilts, 3)
    """
    # Store original shape and flatten batch dimensions
    original_shape = coords.shape
    batch_shape = original_shape[:-2]

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
    tilt_indices = torch.arange(ts.n_tilts, dtype=torch.float32) * grid_step

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
        torch.zeros(ts.n_tilts, dtype=torch.float32),  # rot = 0
        (ts.angles + ts.level_angle_y) * deg_to_rad,   # tilt
        -ts.tilt_axis_angles * deg_to_rad              # psi
    ], dim=-1)

    # Get Euler matrices (n_tilts, 3, 3)
    tilt_matrices = euler_to_matrix(euler_angles)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad])).squeeze(0)  # (3, 3)
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


def get_particle_rotation_matrix_in_all_tilts(
    ts: "TiltSeries", coords: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """
    Get rotation matrices for particles across all tilts.

    This method computes the full rotation matrix that combines:
    1. The particle's intrinsic rotation (angles parameter)
    2. The tilt geometry
    3. Any spatially varying angle corrections

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (N, 3)
               where N must be divisible by n_tilts
        angles: Particle Euler angles in radians (ZYZ convention), shape (N, 3)

    Returns:
        Rotation matrices, shape (N, 3, 3)
    """
    n_coords = coords.shape[0]
    if n_coords % ts.n_tilts != 0:
        raise ValueError(
            f"Number of coordinates ({n_coords}) must be divisible by n_tilts ({ts.n_tilts})"
        )

    # Grid coordinate normalization
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0

    # Prepare grid coordinates for angle grids (3D: X, Y, tilt_index)
    grid_coords = torch.zeros((n_coords, 3), dtype=torch.float32)
    for i in range(n_coords):
        t = i % ts.n_tilts
        grid_coords[i, 0] = coords[i, 0] / ts.volume_dimensions_physical[0]
        grid_coords[i, 1] = coords[i, 1] / ts.volume_dimensions_physical[1]
        grid_coords[i, 2] = t * grid_step

    # Get angle corrections from grids (in degrees)
    grid_angle_x_interp = ts.grid_angle_x.get_interpolated(grid_coords)
    grid_angle_y_interp = ts.grid_angle_y.get_interpolated(grid_coords)
    grid_angle_z_interp = ts.grid_angle_z.get_interpolated(grid_coords)

    # Build tilt rotation matrices for each tilt
    deg_to_rad = torch.pi / 180.0

    # Stack angles into a single tensor (n_tilts, 3)
    euler_angles_tilt = torch.stack([
        torch.zeros(ts.n_tilts, dtype=torch.float32),  # rot = 0
        (ts.angles + ts.level_angle_y) * deg_to_rad,   # tilt
        -ts.tilt_axis_angles * deg_to_rad              # psi
    ], dim=-1)

    # Get Euler matrices
    tilt_matrices = euler_to_matrix(euler_angles_tilt)  # (n_tilts, 3, 3)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad]))  # (1, 3, 3)
    tilt_matrices = torch.matmul(tilt_matrices, level_x_matrix.squeeze(0))  # (n_tilts, 3, 3)

    # Initialize result
    result = torch.zeros((n_coords, 3, 3), dtype=torch.float32)

    # Transform each coordinate
    for i in range(n_coords):
        t = i % ts.n_tilts

        # Build particle rotation matrix from input angles (already in radians)
        particle_matrix = euler_to_matrix(angles[i].unsqueeze(0)).squeeze(0)

        # Build correction matrix from angle grids (XYZ extrinsic)
        angle_x_rad = grid_angle_x_interp[i] * deg_to_rad
        angle_y_rad = grid_angle_y_interp[i] * deg_to_rad
        angle_z_rad = grid_angle_z_interp[i] * deg_to_rad

        correction_matrix = torch.matmul(
            torch.matmul(
                rotate_z(torch.tensor([angle_z_rad])).squeeze(0),
                rotate_y(torch.tensor([angle_y_rad])).squeeze(0)
            ),
            rotate_x(torch.tensor([angle_x_rad])).squeeze(0)
        )

        # Combined rotation: CorrectionMatrix * TiltMatrix * ParticleMatrix
        rotation = torch.matmul(
            torch.matmul(correction_matrix, tilt_matrices[t]),
            particle_matrix
        )

        result[i] = rotation

    return result


def get_particle_angle_in_all_tilts_single(
    ts: "TiltSeries", coord: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    """
    Get particle Euler angles for a single coordinate and angle across all tilts.

    Convenience method that replicates a single coordinate and angle for each tilt.

    Args:
        ts: TiltSeries instance
        coord: Single coordinate in volume space (Angstroms), shape (3,)
        angle: Single Euler angle in radians (ZYZ convention), shape (3,)

    Returns:
        Transformed Euler angles for all tilts, shape (n_tilts, 3) in radians
    """
    # Replicate coordinate and angle for each tilt
    per_tilt_coords = coord.unsqueeze(0).repeat(ts.n_tilts, 1)
    per_tilt_angles = angle.unsqueeze(0).repeat(ts.n_tilts, 1)

    # Transform using main method
    return get_particle_angle_in_all_tilts(ts, per_tilt_coords, per_tilt_angles)


def get_particle_angle_in_all_tilts(
    ts: "TiltSeries", coords: torch.Tensor, angles: torch.Tensor
) -> torch.Tensor:
    """
    Get particle Euler angles across all tilts.

    This computes how a particle's orientation (given by angles) transforms
    when accounting for tilt geometry and spatially varying corrections.

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (N, 3)
        angles: Particle Euler angles in radians (ZYZ convention), shape (N, 3)

    Returns:
        Transformed Euler angles, shape (N, 3) in radians (ZYZ convention)
    """
    # Get rotation matrices
    matrices = get_particle_rotation_matrix_in_all_tilts(ts, coords, angles)

    # Convert back to Euler angles
    result = torch.zeros((matrices.shape[0], 3), dtype=torch.float32)
    for i in range(matrices.shape[0]):
        angles = matrix_to_euler(matrices[i].unsqueeze(0))
        result[i] = angles[0]

    return result


def get_angles_in_one_tilt(
    ts: "TiltSeries", coords: torch.Tensor, particle_angles: torch.Tensor, tilt_id: int
) -> torch.Tensor:
    """
    Get particle Euler angles for a specific tilt.

    More efficient than get_particle_angle_in_all_tilts when you only need
    results for one tilt.

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (N, 3)
        particle_angles: Particle Euler angles in radians (ZYZ convention), shape (N, 3)
        tilt_id: Index of the tilt (0 to n_tilts-1)

    Returns:
        Transformed Euler angles, shape (N, 3) in radians (ZYZ convention)
    """
    if tilt_id < 0 or tilt_id >= ts.n_tilts:
        raise ValueError(f"tilt_id must be between 0 and {ts.n_tilts-1}, got {tilt_id}")

    n_coords = coords.shape[0]

    # Grid coordinate normalization
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    deg_to_rad = torch.pi / 180.0

    # Build tilt rotation matrix for this specific tilt
    euler_angle = torch.tensor([[
        0.0,  # rot
        (ts.angles[tilt_id] + ts.level_angle_y) * deg_to_rad,  # tilt
        -ts.tilt_axis_angles[tilt_id] * deg_to_rad  # psi
    ]], dtype=torch.float32)

    tilt_matrix = euler_to_matrix(euler_angle).squeeze(0)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad])).squeeze(0)
    tilt_matrix = torch.matmul(tilt_matrix, level_x_matrix)

    # Initialize result
    result = torch.zeros((n_coords, 3), dtype=torch.float32)

    # Process each coordinate
    for p in range(n_coords):
        # Grid coordinates for angle grids
        grid_coords = torch.tensor([
            coords[p, 0] / ts.volume_dimensions_physical[0],
            coords[p, 1] / ts.volume_dimensions_physical[1],
            tilt_id * grid_step
        ], dtype=torch.float32).unsqueeze(0)

        # Build particle rotation matrix
        particle_matrix = euler_to_matrix(particle_angles[p].unsqueeze(0)).squeeze(0)

        # Get angle corrections from grids
        angle_x_rad = ts.grid_angle_x.get_interpolated(grid_coords)[0] * deg_to_rad
        angle_y_rad = ts.grid_angle_y.get_interpolated(grid_coords)[0] * deg_to_rad
        angle_z_rad = ts.grid_angle_z.get_interpolated(grid_coords)[0] * deg_to_rad

        # Build correction matrix
        correction_matrix = torch.matmul(
            torch.matmul(
                rotate_z(torch.tensor([angle_z_rad])).squeeze(0),
                rotate_y(torch.tensor([angle_y_rad])).squeeze(0)
            ),
            rotate_x(torch.tensor([angle_x_rad])).squeeze(0)
        )

        # Combined rotation: CorrectionMatrix * TiltMatrix * ParticleMatrix
        rotation = torch.matmul(
            torch.matmul(correction_matrix, tilt_matrix),
            particle_matrix
        )

        # Extract Euler angles
        angles = matrix_to_euler(rotation.unsqueeze(0))
        result[p] = angles[0]

    return result
