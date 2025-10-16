"""
TiltSeries position transformations

This module contains methods for transforming 3D volume coordinates
to 2D image positions across tilts.
"""

import torch
from ..euler import euler_to_matrix, rotate_x


def get_positions_in_one_tilt(ts: "TiltSeries", coords: torch.Tensor, tilt_id: int) -> torch.Tensor:
    """
    Transform 3D volume coordinates to 2D image positions for a specific tilt.

    This method applies the same geometric transformation as get_position_in_all_tilts
    but only for a single specified tilt, making it more efficient when you don't
    need results for all tilts.

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (N, 3)
        tilt_id: Index of the tilt to transform to (0 to n_tilts-1)

    Returns:
        Transformed coordinates, shape (N, 3) where:
        - X, Y are image positions in Angstroms
        - Z is defocus in micrometers
    """
    if tilt_id < 0 or tilt_id >= ts.n_tilts:
        raise ValueError(f"tilt_id must be between 0 and {ts.n_tilts-1}, got {tilt_id}")

    n_coords = coords.shape[0]

    # Volume and image centers
    volume_center = ts.volume_dimensions_physical / 2
    image_center = ts.image_dimensions_physical / 2

    # Grid coordinate normalization factors
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    dose_range = ts.max_dose - ts.min_dose
    dose_step = 1.0 / dose_range if dose_range > 0 else 0.0
    min_dose = ts.min_dose

    # Build tilt rotation matrix for this specific tilt
    deg_to_rad = torch.pi / 180.0

    euler_angle = torch.tensor([[
        0.0,  # rot
        (ts.angles[tilt_id] + ts.level_angle_y) * deg_to_rad,  # tilt
        -ts.tilt_axis_angles[tilt_id] * deg_to_rad  # psi
    ]], dtype=torch.float32)

    # Get Euler matrix
    tilt_matrix = euler_to_matrix(euler_angle).squeeze(0)  # (3, 3)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad])).squeeze(0)  # (3, 3)
    tilt_matrix = torch.matmul(tilt_matrix, level_x_matrix)

    # Build flipped matrix if angles are inverted
    tilt_matrix_flipped = None
    if ts.are_angles_inverted:
        euler_angle_flipped = torch.tensor([[
            0.0,  # rot
            -(ts.angles[tilt_id] + ts.level_angle_y) * deg_to_rad,  # tilt (flipped)
            -ts.tilt_axis_angles[tilt_id] * deg_to_rad  # psi
        ]], dtype=torch.float32)
        tilt_matrix_flipped = euler_to_matrix(euler_angle_flipped).squeeze(0)
        level_x_rad_flipped = -ts.level_angle_x * deg_to_rad
        level_x_matrix_flipped = rotate_x(torch.tensor([level_x_rad_flipped])).squeeze(0)
        tilt_matrix_flipped = torch.matmul(tilt_matrix_flipped, level_x_matrix_flipped)

    # Initialize result
    result = torch.zeros_like(coords)

    # Process each coordinate
    for p in range(n_coords):
        # Grid coordinates for 3D grids
        grid_coords = torch.tensor([
            coords[p, 0] / ts.volume_dimensions_physical[0],
            coords[p, 1] / ts.volume_dimensions_physical[1],
            tilt_id * grid_step
        ], dtype=torch.float32)

        # Center coordinate
        centered = coords[p] - volume_center

        # Prepare 4D grid coordinates for volume warping
        temporal_grid_coords = torch.tensor([
            grid_coords[0],
            grid_coords[1],
            coords[p, 2] / ts.volume_dimensions_physical[2],
            (ts.dose[tilt_id] - min_dose) * dose_step
        ], dtype=torch.float32).unsqueeze(0)  # (1, 4)

        # Get volume warp
        sample_warping = torch.tensor([
            ts.grid_volume_warp_x.get_interpolated(temporal_grid_coords)[0],
            ts.grid_volume_warp_y.get_interpolated(temporal_grid_coords)[0],
            ts.grid_volume_warp_z.get_interpolated(temporal_grid_coords)[0]
        ], dtype=torch.float32)

        centered = centered + sample_warping

        # Apply tilt rotation
        transformed = torch.matmul(tilt_matrix, centered)

        # Add tilt axis offsets (in image space)
        transformed[0] += ts.tilt_axis_offset_x[tilt_id]
        transformed[1] += ts.tilt_axis_offset_y[tilt_id]

        # Add image center
        transformed[0] += image_center[0]
        transformed[1] += image_center[1]

        # Prepare grid coordinates for movement grids
        transformed_coords = torch.tensor([
            transformed[0] / ts.image_dimensions_physical[0],
            transformed[1] / ts.image_dimensions_physical[1],
            tilt_id * grid_step
        ], dtype=torch.float32).unsqueeze(0)  # (1, 3)

        # Get movement corrections
        movement_x = ts.grid_movement_x.get_interpolated(transformed_coords)[0]
        movement_y = ts.grid_movement_y.get_interpolated(transformed_coords)[0]

        # Apply movement corrections
        transformed[0] -= movement_x
        transformed[1] -= movement_y

        # Get defocus and convert Z (Angstroms to micrometers: 1e-4)
        defocus = ts.grid_ctf_defocus.get_interpolated(grid_coords.unsqueeze(0))[0]
        transformed[2] = defocus + 1e-4 * transformed[2]

        result[p] = transformed

        # Handle inverted angles (flip Z coordinate and rotation)
        if ts.are_angles_inverted:
            centered_flipped = coords[p] - volume_center + sample_warping
            centered_flipped[2] *= -1
            transformed_flipped = torch.matmul(tilt_matrix_flipped, centered_flipped)
            result[p, 2] = defocus + 1e-4 * transformed_flipped[2]

        # Apply rounding factors
        result[p] *= ts.size_rounding_factors

    return result


def get_position_in_all_tilts_single(ts: "TiltSeries", coords: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D volume coordinates to 2D image positions for all tilts.

    Convenience method that replicates coordinates for each tilt and
    calls the main transformation method.

    Args:
        ts: TiltSeries instance
        coords: Coordinates in volume space (Angstroms), shape (..., 3)
               where ... represents arbitrary batch dimensions

    Returns:
        Transformed coordinates for all tilts, shape (..., n_tilts, 3) where:
        - X, Y are image positions in Angstroms
        - Z is defocus in micrometers
    """
    # Replicate coordinates for each tilt: (..., 3) -> (..., n_tilts, 3)
    per_tilt_coords = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    # Transform using main method
    return get_position_in_all_tilts(ts, per_tilt_coords)


def get_position_in_all_tilts(ts: "TiltSeries", coords: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D volume coordinates to 2D image positions for each tilt.

    This method applies the full geometric transformation including:
    - Volume warping (spatially and temporally varying)
    - Tilt rotation based on tilt angles
    - Tilt axis offsets
    - Stage movement corrections
    - Defocus calculation for CTF

    Args:
        ts: TiltSeries instance
        coords: Input coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
               where ... represents arbitrary batch dimensions

    Returns:
        Transformed coordinates, shape (..., n_tilts, 3) where:
        - X, Y are image positions in Angstroms
        - Z is defocus in micrometers
    """
    # Store original shape and flatten batch dimensions
    original_shape = coords.shape
    batch_shape = original_shape[:-2]

    # Flatten to (n_particles, n_tilts, 3)
    coords_flat = coords.reshape(-1, ts.n_tilts, 3)
    n_particles = coords_flat.shape[0]

    # Volume and image centers
    volume_center = ts.volume_dimensions_physical / 2  # (3,)
    image_center = ts.image_dimensions_physical / 2  # (2,)

    # Grid coordinate normalization factors
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    dose_range = ts.max_dose - ts.min_dose
    dose_step = 1.0 / dose_range if dose_range > 0 else 0.0
    min_dose = ts.min_dose

    # Prepare grid coordinates for defocus (3D grid: X, Y, tilt_index)
    # Sample at volume center (0.5, 0.5) for each tilt
    tilt_indices = torch.arange(ts.n_tilts, dtype=torch.float32) * grid_step
    grid_coords_defocus = torch.stack([
        torch.full((ts.n_tilts,), 0.5),
        torch.full((ts.n_tilts,), 0.5),
        tilt_indices
    ], dim=-1)  # (n_tilts, 3)

    # Get defocus values for each tilt
    grid_defocus_interp = ts.grid_ctf_defocus.get_interpolated(grid_coords_defocus)  # (n_tilts,)

    # Prepare 4D grid coordinates for volume warping (X, Y, Z, dose)
    # Shape: (n_particles, n_tilts, 4)
    normalized_coords = coords_flat / ts.volume_dimensions_physical  # (n_particles, n_tilts, 3)
    dose_coords = (ts.dose - min_dose) * dose_step  # (n_tilts,)
    dose_coords = dose_coords.unsqueeze(0).expand(n_particles, -1)  # (n_particles, n_tilts)

    temporal_grid_coords = torch.cat([
        normalized_coords,  # (n_particles, n_tilts, 3)
        dose_coords.unsqueeze(-1)  # (n_particles, n_tilts, 1)
    ], dim=-1)  # (n_particles, n_tilts, 4)

    # Flatten for interpolation: (n_particles * n_tilts, 4)
    temporal_grid_coords_flat = temporal_grid_coords.reshape(-1, 4)

    # Get volume warp interpolations
    grid_volume_warp_x_interp = ts.grid_volume_warp_x.get_interpolated(temporal_grid_coords_flat)
    grid_volume_warp_y_interp = ts.grid_volume_warp_y.get_interpolated(temporal_grid_coords_flat)
    grid_volume_warp_z_interp = ts.grid_volume_warp_z.get_interpolated(temporal_grid_coords_flat)

    # Reshape back to (n_particles, n_tilts, 3)
    volume_warp = torch.stack([
        grid_volume_warp_x_interp,
        grid_volume_warp_y_interp,
        grid_volume_warp_z_interp
    ], dim=-1).reshape(n_particles, ts.n_tilts, 3)

    # Build tilt rotation matrices for each tilt
    deg_to_rad = torch.pi / 180.0

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

    # Center coordinates: (n_particles, n_tilts, 3)
    centered = coords_flat - volume_center

    # Apply volume warping: (n_particles, n_tilts, 3)
    centered = centered + volume_warp

    # Apply tilt rotation using batch matrix multiplication
    # tilt_matrices: (n_tilts, 3, 3)
    # centered: (n_particles, n_tilts, 3)
    # We need to apply each tilt's rotation to all particles at that tilt
    # Compute: result[p, t, :] = tilt_matrices[t, :, :] @ centered[p, t, :]
    # In einsum: result[p, t, j] = sum_i tilt_matrices[t, j, i] * centered[p, t, i]
    transformed = torch.einsum('tji,pti->ptj', tilt_matrices, centered)  # (n_particles, n_tilts, 3)

    # Add tilt axis offsets (broadcast over particles): (n_particles, n_tilts, 2)
    transformed[..., 0] += ts.tilt_axis_offset_x.unsqueeze(0)
    transformed[..., 1] += ts.tilt_axis_offset_y.unsqueeze(0)

    # Add image center (broadcast over particles and tilts)
    transformed[..., :2] += image_center

    # Handle inverted angles (flip Z coordinate and rotation)
    if ts.are_angles_inverted:
        euler_angles_flipped = torch.stack([
            torch.zeros(ts.n_tilts, dtype=torch.float32),  # rot = 0
            -(ts.angles + ts.level_angle_y) * deg_to_rad,  # tilt (flipped)
            -ts.tilt_axis_angles * deg_to_rad              # psi
        ], dim=-1)
        tilt_matrices_flipped = euler_to_matrix(euler_angles_flipped)
        level_x_rad_flipped = -ts.level_angle_x * deg_to_rad
        level_x_matrix_flipped = rotate_x(torch.tensor([level_x_rad_flipped])).squeeze(0)
        tilt_matrices_flipped = torch.matmul(tilt_matrices_flipped, level_x_matrix_flipped)

        centered_flipped = centered.clone()
        centered_flipped[..., 2] *= -1
        transformed_flipped = torch.einsum('tji,pti->ptj', tilt_matrices_flipped, centered_flipped)
        transformed[..., 2] = transformed_flipped[..., 2]

    # Prepare grid coordinates for movement grids (3D: X, Y, tilt_index)
    # Shape: (n_particles, n_tilts, 3)
    tilt_grid_indices = tilt_indices.unsqueeze(0).expand(n_particles, -1)  # (n_particles, n_tilts)
    transformed_grid_coords = torch.stack([
        transformed[..., 0] / ts.image_dimensions_physical[0],
        transformed[..., 1] / ts.image_dimensions_physical[1],
        tilt_grid_indices
    ], dim=-1)  # (n_particles, n_tilts, 3)

    # Flatten for interpolation
    transformed_grid_coords_flat = transformed_grid_coords.reshape(-1, 3)

    # Get movement corrections
    grid_movement_x_interp = ts.grid_movement_x.get_interpolated(transformed_grid_coords_flat)
    grid_movement_y_interp = ts.grid_movement_y.get_interpolated(transformed_grid_coords_flat)

    # Reshape back: (n_particles, n_tilts)
    movement_x = grid_movement_x_interp.reshape(n_particles, ts.n_tilts)
    movement_y = grid_movement_y_interp.reshape(n_particles, ts.n_tilts)

    # Subtract movement corrections
    transformed[..., 0] -= movement_x
    transformed[..., 1] -= movement_y

    # Convert Z to defocus (Angstroms to micrometers: 1e-4)
    # grid_defocus_interp: (n_tilts,), broadcast over particles
    transformed[..., 2] = grid_defocus_interp.unsqueeze(0) + 1e-4 * transformed[..., 2]

    # Apply rounding factors (broadcast over particles and tilts)
    transformed = transformed * ts.size_rounding_factors

    # Reshape back to original batch shape
    result = transformed.reshape(*batch_shape, ts.n_tilts, 3)

    return result
