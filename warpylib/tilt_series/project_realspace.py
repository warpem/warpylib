"""
Real-space tomogram projection

This module contains methods for transforming volumes into tilted coordinate
frames for real-space reconstruction algorithms (e.g., SIRT).
"""

import torch
import torch.nn.functional as F
from ..euler import euler_to_matrix, rotate_x


def transform_volume(
    ts: "TiltSeries",
    volume: torch.Tensor,
    pixel_size: float,
    output_dimensions: torch.Tensor,
    tilt_ids: list[int],
    upscale_factor: int = 1
) -> torch.Tensor:
    """
    Transform a 3D volume into tilted coordinate frames for specified tilts.

    This method resamples the input volume into coordinate systems that account for:
    - Tilt rotation (tilt angle, tilt axis angle, level angles)
    - Tilt axis offsets
    - Stage movement corrections
    - Volume warping (spatially and temporally varying deformation)
    - Size rounding factors

    The transformation is applied such that summing along the Z dimension of the
    output volumes produces projections matching the tilt geometry.

    Args:
        ts: TiltSeries instance
        volume: Input volume to transform, shape (D, H, W) or (1, 1, D, H, W).
                If upscale_factor > 1, this should be pre-upscaled externally.
        pixel_size: Pixel size in Angstroms (isotropic, applies to all dimensions)
                   for the ORIGINAL (non-upscaled) volume
        output_dimensions: Target volume dimensions in voxels, shape (3,)
                                   Both input and output volumes are centered relative
                                   to each other
        tilt_ids: List of tilt indices to transform for (batch dimension)
        upscale_factor: Factor by which the input volume was upscaled externally
                       (default 1 = no upscaling). This tells transform_volume to
                       scale sampling coordinates appropriately.

    Returns:
        Transformed volumes, shape (n_tilts, D_out, H_out, W_out)
        where D_out, H_out, W_out correspond to output_dimensions
    """
    # Get device from TiltSeries tensors
    device = ts.angles.device

    # Ensure inputs are on the correct device
    volume = volume.to(device)
    output_dimensions_physical = (output_dimensions * pixel_size).to(device)

    # Ensure volume is 5D for grid_sample: (N, C, D, H, W)
    if volume.ndim == 3:
        volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    elif volume.ndim == 4:
        volume = volume.unsqueeze(1)  # (N, 1, D, H, W)

    n_tilts = len(tilt_ids)

    # Volume centers
    volume_center = ts.volume_dimensions_physical / 2  # (3,)
    output_center = output_dimensions_physical / 2  # (3,)

    # Compute output volume dimensions in pixels (ensure even numbers)
    output_dims_pixels = (output_dimensions_physical / pixel_size).round().long()
    # Round to nearest even number
    output_dims_pixels = (output_dims_pixels + 1) // 2 * 2
    D_out, H_out, W_out = output_dims_pixels[2].item(), output_dims_pixels[1].item(), output_dims_pixels[0].item()

    # Grid coordinate normalization factors
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0
    dose_range = ts.max_dose - ts.min_dose
    dose_step = 1.0 / (dose_range - 1) if dose_range > 1 else 0.0
    min_dose = ts.min_dose

    # Build rotation matrices for specified tilts
    deg_to_rad = torch.pi / 180.0

    tilt_ids_tensor = torch.tensor(tilt_ids, dtype=torch.long, device=device)

    # Get per-tilt parameters
    angles_selected = ts.angles[tilt_ids_tensor]
    tilt_axis_angles_selected = ts.tilt_axis_angles[tilt_ids_tensor]
    tilt_axis_offset_x_selected = ts.tilt_axis_offset_x[tilt_ids_tensor]
    tilt_axis_offset_y_selected = ts.tilt_axis_offset_y[tilt_ids_tensor]
    dose_selected = ts.dose[tilt_ids_tensor]

    # Build Euler angles (n_tilts, 3)
    euler_angles = torch.stack([
        torch.zeros(n_tilts, dtype=torch.float32, device=device),  # rot = 0
        (angles_selected + ts.level_angle_y) * deg_to_rad,  # tilt
        -tilt_axis_angles_selected * deg_to_rad  # psi
    ], dim=-1)

    # Get rotation matrices (n_tilts, 3, 3)
    tilt_matrices = euler_to_matrix(euler_angles)

    # Apply level angle X rotation
    level_x_rad = ts.level_angle_x * deg_to_rad
    level_x_matrix = rotate_x(torch.tensor([level_x_rad], device=device)).squeeze(0)  # (3, 3)
    tilt_matrices = torch.matmul(tilt_matrices, level_x_matrix)  # (n_tilts, 3, 3)

    # Transpose for inverse rotation
    tilt_matrices_inv = tilt_matrices.transpose(-2, -1)  # (n_tilts, 3, 3)

    # Create output coordinate grids for all tilts
    # Each grid point represents a position in the output volume
    z_coords = torch.linspace(0, output_dimensions[2].item() - 1, D_out, device=device)
    y_coords = torch.linspace(0, output_dimensions[1].item() - 1, H_out, device=device)
    x_coords = torch.linspace(0, output_dimensions[0].item() - 1, W_out, device=device)

    # Create meshgrid: (D, H, W)
    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

    # Stack into coordinate tensor: (D, H, W, 3)
    output_coords = torch.stack([xx, yy, zz], dim=-1) * pixel_size

    # Replicate for all tilts: (n_tilts, D, H, W, 3)
    output_coords = output_coords.unsqueeze(0).expand(n_tilts, -1, -1, -1, -1)

    # Flatten spatial dimensions for easier processing: (n_tilts, D*H*W, 3)
    coords_shape = output_coords.shape
    output_coords_flat = output_coords.reshape(n_tilts, -1, 3)

    # Apply inverse transformations in reverse order
    # Start with output coordinates
    coords = output_coords_flat.clone()

    # Apply rounding factors (inverse)
    coords /= ts.size_rounding_factors

    # 1. Add movement corrections (approximate: evaluate at output coords)
    # Normalize coordinates for grid interpolation
    tilt_grid_indices = tilt_ids_tensor.float() * grid_step  # (n_tilts,)

    # For each tilt, normalize by output dimensions
    normalized_coords = coords.clone()
    normalized_coords[..., 0] /= output_dimensions_physical[0]  # X
    normalized_coords[..., 1] /= output_dimensions_physical[1]  # Y

    # Add tilt index: (n_tilts, D*H*W, 3)
    tilt_indices_expanded = tilt_grid_indices.view(n_tilts, 1, 1).expand(-1, coords.shape[1], -1)
    movement_grid_coords = torch.cat([normalized_coords[..., :2], tilt_indices_expanded], dim=-1)

    # Flatten for interpolation: (n_tilts * D*H*W, 3)
    movement_grid_coords_flat = movement_grid_coords.reshape(-1, 3)

    # Get movement corrections
    movement_x_interp = ts.grid_movement_x.get_interpolated(movement_grid_coords_flat)
    movement_y_interp = ts.grid_movement_y.get_interpolated(movement_grid_coords_flat)

    # Reshape: (n_tilts, D*H*W)
    movement_x = movement_x_interp.reshape(n_tilts, -1)
    movement_y = movement_y_interp.reshape(n_tilts, -1)

    # Add movement corrections
    coords[..., 0] += movement_x
    coords[..., 1] += movement_y

    # 2. Subtract output center
    coords -= output_center

    # 3. Subtract tilt axis offsets
    coords[..., 0] -= tilt_axis_offset_x_selected.view(n_tilts, 1)
    coords[..., 1] -= tilt_axis_offset_y_selected.view(n_tilts, 1)

    # 4. Apply inverse rotation
    # coords: (n_tilts, D*H*W, 3)
    # tilt_matrices_inv: (n_tilts, 3, 3)
    # result[t, p, j] = sum_i tilt_matrices_inv[t, j, i] * coords[t, p, i]
    coords = torch.einsum('tji,tpi->tpj', tilt_matrices_inv, coords)

    # 5. Subtract volume warp (approximate: evaluate at current coords)
    # Need to normalize coordinates for 4D warp grids
    warp_coords = coords.clone()
    warp_coords[..., 0] /= ts.volume_dimensions_physical[0]  # X
    warp_coords[..., 1] /= ts.volume_dimensions_physical[1]  # Y
    warp_coords[..., 2] /= ts.volume_dimensions_physical[2]  # Z

    # Add dose coordinate: (n_tilts, D*H*W, 4)
    dose_coords = (dose_selected - min_dose) * dose_step  # (n_tilts,)
    dose_coords_expanded = dose_coords.view(n_tilts, 1, 1).expand(-1, warp_coords.shape[1], -1)
    temporal_grid_coords = torch.cat([warp_coords, dose_coords_expanded], dim=-1)

    # Flatten for interpolation: (n_tilts * D*H*W, 4)
    temporal_grid_coords_flat = temporal_grid_coords.reshape(-1, 4)

    # Get volume warp interpolations
    warp_x_interp = ts.grid_volume_warp_x.get_interpolated(temporal_grid_coords_flat)
    warp_y_interp = ts.grid_volume_warp_y.get_interpolated(temporal_grid_coords_flat)
    warp_z_interp = ts.grid_volume_warp_z.get_interpolated(temporal_grid_coords_flat)

    # Reshape and stack: (n_tilts, D*H*W, 3)
    volume_warp = torch.stack([
        warp_x_interp.reshape(n_tilts, -1),
        warp_y_interp.reshape(n_tilts, -1),
        warp_z_interp.reshape(n_tilts, -1)
    ], dim=-1)

    # Subtract volume warp
    coords -= volume_warp

    # 6. Add volume center
    coords += volume_center

    # Reshape back to spatial grid: (n_tilts, D, H, W, 3)
    sampling_coords = coords.reshape(n_tilts, D_out, H_out, W_out, 3)

    # Normalize to [-1, 1] for grid_sample
    # grid_sample expects coordinates in (x, y, z) order with range [-1, 1]
    # where -1 corresponds to the first element and +1 to the last
    volume_dims_tensor = torch.tensor(
        [volume.shape[4], volume.shape[3], volume.shape[2]],  # (W, H, D)
        dtype=torch.float32,
        device=device
    )
    volume_physical_dims = ts.volume_dimensions_physical  # (X, Y, Z)

    # Normalize: convert from physical coordinates [0, physical_size] to [-1, 1]
    normalized_sampling = sampling_coords.clone()
    normalized_sampling[..., 0] = 2 * (sampling_coords[..., 0] / volume_physical_dims[0]) - 1  # X
    normalized_sampling[..., 1] = 2 * (sampling_coords[..., 1] / volume_physical_dims[1]) - 1  # Y
    normalized_sampling[..., 2] = 2 * (sampling_coords[..., 2] / volume_physical_dims[2]) - 1  # Z

    # grid_sample expects grid in (x, y, z) order, which we already have

    # Expand volume to match batch size if needed
    if volume.shape[0] == 1:
        volume_expanded = volume.expand(n_tilts, -1, -1, -1, -1)
    else:
        volume_expanded = volume

    # Sample from volume using grid_sample
    # Input: (n_tilts, 1, D, H, W)
    # Grid: (n_tilts, D_out, H_out, W_out, 3)
    # Output: (n_tilts, 1, D_out, H_out, W_out)
    transformed = F.grid_sample(
        volume_expanded,
        normalized_sampling,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )

    # Remove channel dimension: (n_tilts, D_out, H_out, W_out)
    transformed = transformed.squeeze(1)

    return transformed