"""
Import 2D alignment data from IMOD-style .xf and .tlt files.

This module provides functionality to import tilt series alignments from
external alignment software (typically IMOD).
"""

from pathlib import Path
from typing import Optional
import torch
from ..euler import matrix_to_euler


def import_alignments(
    ts: "TiltSeries",
    xf_path: str,
    binned_pixel_size_mean: float,
    tlt_path: Optional[str] = None,
) -> None:
    """
    Import 2D alignment parameters from .xf and optionally .tlt files.

    This function reads IMOD-style alignment files and updates the tilt series
    geometry parameters (tilt axis angles and offsets, and optionally tilt angles).

    Args:
        ts: TiltSeries instance to update
        xf_path: Path to .xf file containing 2D transforms
        binned_pixel_size_mean: Mean pixel size in Angstroms for the binned data
                               (used to scale shifts from pixels to Angstroms)
        tlt_path: Optional path to .tlt file containing tilt angles. If None,
                 existing angles in ts.angles are preserved.

    Raises:
        FileNotFoundError: If .xf or .tlt file doesn't exist
        ValueError: If the number of lines doesn't match expected tilt count

    Notes:
        - Only processes tilts where use_tilt[t] is True
        - .xf format: Each line has 6 values forming a 2D affine transform:
          [a11, a21, a12, a22, dx, dy] representing rotation + translation
        - .tlt format: Each line contains a single tilt angle in degrees
    """
    xf_path = Path(xf_path)

    if not xf_path.exists():
        raise FileNotFoundError(f"Could not find {xf_path}")

    n_tilts = ts.n_tilts
    n_valid = ts.use_tilt.sum().item()

    # Read and parse .xf file
    with open(xf_path, 'r') as f:
        xf_lines = [line.strip() for line in f if line.strip()]

    if len(xf_lines) != n_valid:
        raise ValueError(
            f"{n_valid} active tilts in series, but {len(xf_lines)} lines in {xf_path}"
        )

    # Process transforms for each active tilt
    line_idx = 0
    for t in range(n_tilts):
        if not ts.use_tilt[t]:
            continue

        # Parse the 6 values from the .xf line
        parts = xf_lines[line_idx].split()
        if len(parts) != 6:
            raise ValueError(
                f"Expected 6 values in .xf line {line_idx}, got {len(parts)}"
            )

        values = [float(p) for p in parts]

        # Build 3x3 rotation matrix from 2x2 transform
        # .xf format: a11 a21 a12 a22 dx dy
        # C# code constructs: VecX = (a11, a12), VecY = (a21, a22)
        # Then Matrix3(VecX.X, VecX.Y, 0, VecY.X, VecY.Y, 0, 0, 0, 1)
        # Which creates the matrix:
        # [[VecX.X, VecY.X, 0],     [[a11, a21, 0],
        #  [VecX.Y, VecY.Y, 0],  ==  [a12, a22, 0],
        #  [0,      0,      1]]      [0,   0,   1]]
        rotation = torch.tensor([
            [values[0], values[1], 0.0],
            [values[2], values[3], 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)

        # Extract Euler angles (ZYZ convention)
        euler = matrix_to_euler(rotation.unsqueeze(0)).squeeze(0)

        # Store Z rotation angle in degrees
        ts.tilt_axis_angles[t] = euler[2] * (180.0 / torch.pi)

        # Transform shift vector
        # Original shift from .xf file (negated as per C# code)
        shift = torch.tensor([-values[4], -values[5], 0.0], dtype=torch.float32)

        # Apply inverse rotation (transpose of rotation matrix)
        shift = rotation.T @ shift

        # Scale by pixel size to convert from pixels to Angstroms
        shift = shift * binned_pixel_size_mean

        # Store offsets
        ts.tilt_axis_offset_x[t] = shift[0]
        ts.tilt_axis_offset_y[t] = shift[1]

        line_idx += 1

    # Optionally read and parse .tlt file
    if tlt_path is not None:
        tlt_path = Path(tlt_path)

        if not tlt_path.exists():
            raise FileNotFoundError(f"Could not find {tlt_path}")

        with open(tlt_path, 'r') as f:
            tlt_lines = [line.strip() for line in f if line.strip()]

        # Handle .tlt file based on line count
        if len(tlt_lines) == n_valid:
            # Parse all angles first
            parsed_angles = torch.tensor([float(line) for line in tlt_lines], dtype=torch.float32)

            # Check if all angles are zero (likely an error)
            if torch.all(parsed_angles == 0):
                raise ValueError(f"All tilt angles are zero in {tlt_path}")

            # Apply angles to active tilts only
            line_idx = 0
            for t in range(n_tilts):
                if not ts.use_tilt[t]:
                    continue
                ts.angles[t] = parsed_angles[line_idx]
                line_idx += 1
        elif len(tlt_lines) == n_tilts:
            # One angle per tilt (including inactive tilts)
            parsed_angles = torch.tensor([float(line) for line in tlt_lines], dtype=torch.float32)

            # Check if all angles are zero
            if torch.all(parsed_angles == 0):
                raise ValueError(f"All tilt angles are zero in {tlt_path}")

            # Apply all angles directly
            ts.angles = parsed_angles
        else:
            raise ValueError(
                f"Expected {n_valid} or {n_tilts} lines in .tlt file, got {len(tlt_lines)}"
            )