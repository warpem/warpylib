"""
TiltSeries utility functions

This module contains utility methods for manipulating tilt series geometry.
"""

import torch
from ..euler import euler_to_matrix
from . import angles


def _get_all_tilt_matrices(ts: "TiltSeries") -> torch.Tensor:
    """
    Get tilt rotation matrices for all tilts by sampling the geometry at volume center.

    Uses get_angle_in_all_tilts_single as the source of truth for the geometric model.

    Args:
        ts: TiltSeries instance

    Returns:
        Rotation matrices of shape (n_tilts, 3, 3)
    """
    device = ts.angles.device

    # Sample at volume center with no particle rotation
    volume_center = (ts.volume_dimensions_physical / 2).to(device)  # (3,)

    # Get Euler angles for all tilts at once
    euler_angles = angles.get_angle_in_all_tilts_single(ts, volume_center, angles=None)  # (n_tilts, 3)

    # Convert to rotation matrices
    tilt_matrices = euler_to_matrix(euler_angles)  # (n_tilts, 3, 3)

    return tilt_matrices


def _get_tilt_matrix(ts: "TiltSeries", tilt_id: int) -> torch.Tensor:
    """
    Get the tilt rotation matrix for a single tilt by sampling the geometry at volume center.

    Uses get_angles_in_one_tilt as the source of truth for the geometric model.

    Args:
        ts: TiltSeries instance
        tilt_id: Index of the tilt (0 to n_tilts-1)

    Returns:
        Rotation matrix of shape (3, 3)
    """
    device = ts.angles.device

    # Sample at volume center with no particle rotation
    volume_center = (ts.volume_dimensions_physical / 2).unsqueeze(0).to(device)  # (1, 3)

    # Get Euler angles from the canonical geometry model
    euler_angles = angles.get_angles_in_one_tilt(ts, volume_center, tilt_id, angles=None)  # (1, 3)

    # Convert to rotation matrix
    tilt_matrix = euler_to_matrix(euler_angles).squeeze(0)  # (3, 3)

    return tilt_matrix


def apply_tilt_shift_and_propagate(
    ts: "TiltSeries",
    source_tilt_id: int,
    shift: torch.Tensor,
    propagate_to: str = "both"
) -> None:
    """
    Apply a 2D shift to a tilt and propagate it geometrically to other tilts.

    The shift is assumed to be in the plane of the tilt image (the central
    Fourier slice for that projection), with axes aligned with the image axes.
    The shift is back-projected to 3D specimen space and then re-projected
    onto the target tilts, accounting for the 3D projection geometry.

    This function supports gradient flow - if shift.requires_grad is True,
    gradients will flow back through ts.tilt_axis_offset_x/y to the input shift.

    Args:
        ts: TiltSeries instance
        source_tilt_id: Index of the tilt being shifted (0 to n_tilts-1)
        shift: 2D shift tensor of shape (2,) as [shift_x, shift_y] in Angstroms (image space)
        propagate_to: Which tilts to propagate to by index:
            - "lower": tilts with index < source_tilt_id
            - "higher": tilts with index > source_tilt_id
            - "both": all other tilts (default)

    Returns:
        None. Modifies ts.tilt_axis_offset_x and ts.tilt_axis_offset_y in place.
    """
    if source_tilt_id < 0 or source_tilt_id >= ts.n_tilts:
        raise ValueError(f"source_tilt_id must be between 0 and {ts.n_tilts-1}, got {source_tilt_id}")

    if propagate_to not in ("lower", "higher", "both"):
        raise ValueError(f"propagate_to must be 'lower', 'higher', or 'both', got '{propagate_to}'")

    device = ts.angles.device
    shift = shift.to(device)

    # Build the source tilt's rotation matrix
    source_tilt_matrix = _get_tilt_matrix(ts, source_tilt_id)

    # The 2D shift in image space with Z=0 in the rotated frame
    shift_3d_input = torch.cat([shift, torch.zeros(1, device=device, dtype=shift.dtype)])

    # Back-project to 3D specimen space using inverse rotation (= transpose)
    shift_3d = torch.matmul(source_tilt_matrix.T, shift_3d_input)

    # Get all tilt matrices
    tilt_matrices = _get_all_tilt_matrices(ts)  # (n_tilts, 3, 3)

    # Project the 3D shift onto all tilts: (n_tilts, 3, 3) @ (3,) -> (n_tilts, 3)
    projected_shifts = torch.matmul(tilt_matrices, shift_3d)[:, :2]  # (n_tilts, 2)

    # Build mask for which tilts to update
    if propagate_to == "lower":
        mask = torch.arange(ts.n_tilts, device=device) <= source_tilt_id
    elif propagate_to == "higher":
        mask = torch.arange(ts.n_tilts, device=device) >= source_tilt_id
    else:  # "both"
        mask = torch.ones(ts.n_tilts, dtype=torch.bool, device=device)

    # Apply shifts using masked assignment (preserves gradients)
    ts.tilt_axis_offset_x = torch.where(mask, ts.tilt_axis_offset_x + projected_shifts[:, 0], ts.tilt_axis_offset_x)
    ts.tilt_axis_offset_y = torch.where(mask, ts.tilt_axis_offset_y + projected_shifts[:, 1], ts.tilt_axis_offset_y)


def apply_tomogram_shift_3d(
    ts: "TiltSeries",
    shift: torch.Tensor,
) -> None:
    """
    Apply a 3D shift to all tilts such that the tomogram volume center moves by the specified vector.

    Given a 3D shift in specimen/tomogram space, this method computes the corresponding
    2D shift for each tilt image and adds it to the tilt axis offsets. This effectively
    shifts the reconstruction origin by the specified 3D vector.

    This function supports gradient flow - if shift.requires_grad is True,
    gradients will flow back through ts.tilt_axis_offset_x/y to the input shift.

    Args:
        ts: TiltSeries instance
        shift: 3D shift tensor of shape (3,) as [shift_x, shift_y, shift_z] in Angstroms
               (specimen space, Z is along beam at 0° tilt)

    Returns:
        None. Modifies ts.tilt_axis_offset_x and ts.tilt_axis_offset_y in place.
    """
    device = ts.angles.device
    shift = shift.to(device)

    # Get all tilt matrices
    tilt_matrices = _get_all_tilt_matrices(ts)  # (n_tilts, 3, 3)

    # Project the 3D shift onto all tilts: (n_tilts, 3, 3) @ (3,) -> (n_tilts, 3)
    projected_shifts = torch.matmul(tilt_matrices, shift)

    # Add the X and Y components to the tilt offsets
    ts.tilt_axis_offset_x = ts.tilt_axis_offset_x + projected_shifts[:, 0]
    ts.tilt_axis_offset_y = ts.tilt_axis_offset_y + projected_shifts[:, 1]