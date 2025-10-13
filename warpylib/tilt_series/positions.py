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

    rot_angle = torch.tensor([0.0], dtype=torch.float32)
    tilt_angle = torch.tensor([(ts.angles[tilt_id] + ts.level_angle_y) * deg_to_rad], dtype=torch.float32)
    psi_angle = torch.tensor([-ts.tilt_axis_angles[tilt_id] * deg_to_rad], dtype=torch.float32)

    # Get Euler matrix
    tilt_matrix = euler_to_matrix(rot_angle, tilt_angle, psi_angle).squeeze(0)  # (3, 3)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad])).squeeze(0)  # (3, 3)
    tilt_matrix = torch.matmul(tilt_matrix, level_x_matrix)

    # Build flipped matrix if angles are inverted
    tilt_matrix_flipped = None
    if ts.are_angles_inverted:
        tilt_angle_flipped = torch.tensor([-(ts.angles[tilt_id] + ts.level_angle_y) * deg_to_rad], dtype=torch.float32)
        tilt_matrix_flipped = euler_to_matrix(rot_angle, tilt_angle_flipped, psi_angle).squeeze(0)
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


def get_position_in_all_tilts_single(ts: "TiltSeries", coord: torch.Tensor) -> torch.Tensor:
    """
    Transform a single 3D volume coordinate to 2D image positions for all tilts.

    Convenience method that replicates a single coordinate for each tilt and
    calls the main transformation method.

    Args:
        ts: TiltSeries instance
        coord: Single coordinate in volume space (Angstroms), shape (3,)

    Returns:
        Transformed coordinates for all tilts, shape (n_tilts, 3) where:
        - X, Y are image positions in Angstroms
        - Z is defocus in micrometers
    """
    # Replicate coordinate for each tilt
    per_tilt_coords = coord.unsqueeze(0).repeat(ts.n_tilts, 1)

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
        coords: Input coordinates in volume space (Angstroms), shape (N, 3)
               where N must be divisible by n_tilts. Coords are grouped by tilt:
               first n_tilts entries are for each tilt at position 0,
               next n_tilts entries are for each tilt at position 1, etc.

    Returns:
        Transformed coordinates, shape (N, 3) where:
        - X, Y are image positions in Angstroms
        - Z is defocus in micrometers
    """
    n_coords = coords.shape[0]
    if n_coords % ts.n_tilts != 0:
        raise ValueError(
            f"Number of coordinates ({n_coords}) must be divisible by n_tilts ({ts.n_tilts})"
        )

    # Volume and image centers
    volume_center = ts.volume_dimensions_physical / 2
    image_center = ts.image_dimensions_physical / 2

    # Grid coordinate normalization factors
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    dose_range = ts.max_dose - ts.min_dose
    dose_step = 1.0 / dose_range if dose_range > 0 else 0.0
    min_dose = ts.min_dose

    # Prepare grid coordinates for defocus (3D grid: X, Y, tilt_index)
    # Sample at volume center (0.5, 0.5) for each tilt
    grid_coords_defocus = torch.zeros((ts.n_tilts, 3), dtype=torch.float32)
    grid_coords_defocus[:, 0] = 0.5  # X
    grid_coords_defocus[:, 1] = 0.5  # Y
    grid_coords_defocus[:, 2] = torch.arange(ts.n_tilts, dtype=torch.float32) * grid_step

    # Get defocus values for each tilt
    grid_defocus_interp = ts.grid_ctf_defocus.get_interpolated(grid_coords_defocus)

    # Prepare 4D grid coordinates for volume warping (X, Y, Z, dose)
    temporal_grid_coords = torch.zeros((n_coords, 4), dtype=torch.float32)

    for i in range(n_coords):
        t = i % ts.n_tilts
        temporal_grid_coords[i, 0] = coords[i, 0] / ts.volume_dimensions_physical[0]
        temporal_grid_coords[i, 1] = coords[i, 1] / ts.volume_dimensions_physical[1]
        temporal_grid_coords[i, 2] = coords[i, 2] / ts.volume_dimensions_physical[2]
        temporal_grid_coords[i, 3] = (ts.dose[t] - min_dose) * dose_step

    # Get volume warp interpolations
    grid_volume_warp_x_interp = ts.grid_volume_warp_x.get_interpolated(temporal_grid_coords)
    grid_volume_warp_y_interp = ts.grid_volume_warp_y.get_interpolated(temporal_grid_coords)
    grid_volume_warp_z_interp = ts.grid_volume_warp_z.get_interpolated(temporal_grid_coords)

    # Build tilt rotation matrices for each tilt
    # C#: Matrix3.Euler(0, (Angles[t] + LevelAngleY) * Helper.ToRad, -TiltAxisAngles[t] * Helper.ToRad) *
    #     Matrix3.RotateX(LevelAngleX * Helper.ToRad)
    deg_to_rad = torch.pi / 180.0

    rot_angles = torch.zeros(ts.n_tilts, dtype=torch.float32)  # First angle (rot) = 0
    tilt_angles = (ts.angles + ts.level_angle_y) * deg_to_rad  # Second angle (tilt)
    psi_angles = -ts.tilt_axis_angles * deg_to_rad  # Third angle (psi)

    # Get Euler matrices
    tilt_matrices = euler_to_matrix(rot_angles, tilt_angles, psi_angles)  # (n_tilts, 3, 3)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad]))  # (1, 3, 3)
    tilt_matrices = torch.matmul(tilt_matrices, level_x_matrix.squeeze(0))  # (n_tilts, 3, 3)

    # Build flipped matrices if angles are inverted
    tilt_matrices_flipped = None
    if ts.are_angles_inverted:
        tilt_angles_flipped = -(ts.angles + ts.level_angle_y) * deg_to_rad
        tilt_matrices_flipped = euler_to_matrix(rot_angles, tilt_angles_flipped, psi_angles)
        level_x_rad_flipped = -ts.level_angle_x * deg_to_rad
        level_x_matrix_flipped = rotate_x(torch.tensor([level_x_rad_flipped]))
        tilt_matrices_flipped = torch.matmul(tilt_matrices_flipped, level_x_matrix_flipped.squeeze(0))

    # Initialize result
    result = torch.zeros_like(coords)

    # Transform each coordinate
    for i in range(n_coords):
        t = i % ts.n_tilts

        # Center coordinate
        centered = coords[i] - volume_center

        # Apply volume warping
        sample_warping = torch.tensor([
            grid_volume_warp_x_interp[i],
            grid_volume_warp_y_interp[i],
            grid_volume_warp_z_interp[i]
        ], dtype=torch.float32)
        centered = centered + sample_warping

        # Apply tilt rotation
        rotation = tilt_matrices[t]
        transformed = torch.matmul(rotation, centered)

        # Add tilt axis offsets (in image space)
        transformed[0] += ts.tilt_axis_offset_x[t]
        transformed[1] += ts.tilt_axis_offset_y[t]

        # Add image center
        transformed[0] += image_center[0]
        transformed[1] += image_center[1]

        result[i] = transformed

        # Handle inverted angles (flip Z coordinate and rotation)
        if ts.are_angles_inverted:
            rotation_flipped = tilt_matrices_flipped[t]
            centered_flipped = coords[i] - volume_center
            centered_flipped = centered_flipped + sample_warping
            centered_flipped[2] *= -1
            transformed_flipped = torch.matmul(rotation_flipped, centered_flipped)
            result[i, 2] = transformed_flipped[2]

    # Prepare grid coordinates for movement grids (3D: X, Y, tilt_index)
    transformed_grid_coords = torch.zeros((n_coords, 3), dtype=torch.float32)
    for i in range(n_coords):
        t = i % ts.n_tilts
        transformed_grid_coords[i, 0] = result[i, 0] / ts.image_dimensions_physical[0]
        transformed_grid_coords[i, 1] = result[i, 1] / ts.image_dimensions_physical[1]
        transformed_grid_coords[i, 2] = t * grid_step

    # Get movement corrections
    grid_movement_x_interp = ts.grid_movement_x.get_interpolated(transformed_grid_coords)
    grid_movement_y_interp = ts.grid_movement_y.get_interpolated(transformed_grid_coords)

    # Apply final corrections
    for i in range(n_coords):
        t = i % ts.n_tilts

        # Subtract movement corrections
        result[i, 0] -= grid_movement_x_interp[i]
        result[i, 1] -= grid_movement_y_interp[i]

        # Convert Z to defocus (Angstroms to micrometers: 1e-4)
        result[i, 2] = grid_defocus_interp[t] + 1e-4 * result[i, 2]

        # Apply rounding factors
        result[i] *= ts.size_rounding_factors

    return result
