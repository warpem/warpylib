"""
TiltSeries utility functions

This module contains utility methods for manipulating tilt series geometry.
"""

import torch
from ..euler import euler_to_matrix
from . import angles


def _get_tilt_matrix(ts: "TiltSeries", tilt_id: int) -> torch.Tensor:
    """
    Get the tilt rotation matrix for a given tilt by sampling the geometry at volume center.

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
    shift_x: float,
    shift_y: float,
    propagate_to: str = "both"
) -> None:
    """
    Apply a 2D shift to a tilt and propagate it geometrically to other tilts.

    The shift is assumed to be in the plane of the tilt image (the central
    Fourier slice for that projection), with axes aligned with the image axes.
    The shift is back-projected to 3D specimen space and then re-projected
    onto the target tilts, accounting for the 3D projection geometry.

    Args:
        ts: TiltSeries instance
        source_tilt_id: Index of the tilt being shifted (0 to n_tilts-1)
        shift_x: X shift in Angstroms (image space)
        shift_y: Y shift in Angstroms (image space)
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

    # Build the source tilt's rotation matrix
    source_tilt_matrix = _get_tilt_matrix(ts, source_tilt_id)

    # The 2D shift in image space with Z=0 in the rotated frame
    shift_2d = torch.tensor([shift_x, shift_y, 0.0], dtype=torch.float32, device=device)

    # Back-project to 3D specimen space using inverse rotation (= transpose)
    shift_3d = torch.matmul(source_tilt_matrix.T, shift_2d)

    # Apply the shift to the source tilt
    ts.tilt_axis_offset_x[source_tilt_id] += shift_x
    ts.tilt_axis_offset_y[source_tilt_id] += shift_y

    # Determine which tilts to propagate to
    if propagate_to == "lower":
        target_tilt_ids = range(0, source_tilt_id)
    elif propagate_to == "higher":
        target_tilt_ids = range(source_tilt_id + 1, ts.n_tilts)
    else:  # "both"
        target_tilt_ids = [t for t in range(ts.n_tilts) if t != source_tilt_id]

    # Propagate the shift to target tilts
    for target_tilt_id in target_tilt_ids:
        target_tilt_matrix = _get_tilt_matrix(ts, target_tilt_id)

        # Project the 3D shift onto the target tilt's image plane
        projected_shift = torch.matmul(target_tilt_matrix, shift_3d)

        # Add the X and Y components to the target tilt's offsets
        ts.tilt_axis_offset_x[target_tilt_id] += projected_shift[0].item()
        ts.tilt_axis_offset_y[target_tilt_id] += projected_shift[1].item()


def apply_tomogram_shift_3d(
    ts: "TiltSeries",
    shift_x: float,
    shift_y: float,
    shift_z: float
) -> None:
    """
    Apply a 3D shift to all tilts such that the tomogram volume center moves by the specified vector.

    Given a 3D shift in specimen/tomogram space, this method computes the corresponding
    2D shift for each tilt image and adds it to the tilt axis offsets. This effectively
    shifts the reconstruction origin by the specified 3D vector.

    Args:
        ts: TiltSeries instance
        shift_x: X shift in Angstroms (specimen space)
        shift_y: Y shift in Angstroms (specimen space)
        shift_z: Z shift in Angstroms (specimen space, along beam at 0° tilt)

    Returns:
        None. Modifies ts.tilt_axis_offset_x and ts.tilt_axis_offset_y in place.
    """
    device = ts.angles.device

    # Sample at volume center with no particle rotation to get all tilt matrices
    volume_center = (ts.volume_dimensions_physical / 2).to(device)  # (3,)

    # Get Euler angles for all tilts at once
    euler_angles = angles.get_angle_in_all_tilts_single(ts, volume_center, angles=None)  # (n_tilts, 3)

    # Convert to rotation matrices
    tilt_matrices = euler_to_matrix(euler_angles)  # (n_tilts, 3, 3)

    # The 3D shift in specimen space
    shift_3d = torch.tensor([shift_x, shift_y, shift_z], dtype=torch.float32, device=device)

    # Project the 3D shift onto all tilts at once: (n_tilts, 3, 3) @ (3,) -> (n_tilts, 3)
    projected_shifts = torch.matmul(tilt_matrices, shift_3d)

    # Add the X and Y components to the tilt offsets
    ts.tilt_axis_offset_x += projected_shifts[:, 0]
    ts.tilt_axis_offset_y += projected_shifts[:, 1]