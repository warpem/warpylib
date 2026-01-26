"""
Tilt stack creation for tilt series.

This module provides functionality to create aligned tilt stacks from
individual tilt images, along with angle files and thumbnails.
"""

from pathlib import Path
from typing import TYPE_CHECKING
import torch
import mrcfile
from PIL import Image
import numpy as np

from ..ops import norm

if TYPE_CHECKING:
    from .core import TiltSeries


def stack_tilts(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    pixel_size: float,
    create_thumbnails: bool = True,
) -> None:
    """
    Create aligned tilt stack, angle file, and thumbnails.

    This method takes preprocessed tilt images, filters to only the tilts marked
    as "used", stacks them into a single MRC file, writes the corresponding angle
    file, and optionally creates PNG thumbnails for visualization.

    Parameters
    ----------
    ts : TiltSeries
        The tilt series object containing metadata and paths.
    tilt_data : torch.Tensor
        Tilt images with shape (n_tilts, H, W).
    pixel_size : float
        Pixel size of the tilt data in Angstroms.
    create_thumbnails : bool, optional
        Whether to create PNG thumbnails for each tilt. Default is True.

    Notes
    -----
    The following files are created:
    - {tilt_stack_dir}/{root_name}.st: MRC stack of used tilts
    - {tilt_stack_dir}/{root_name}.rawtlt: Angle file with one angle per line
    - {tilt_stack_dir}/thumbnails/{movie_name}.png: PNG thumbnail for each tilt

    The thumbnail normalization uses warpylib's norm function with a circular
    region of diameter equal to half the image size, then scales to 0-255 range
    with median at 128 and +/-3 sigma spanning the full range.
    """

    # Create output directories
    Path(ts.tilt_stack_dir).mkdir(parents=True, exist_ok=True)

    # Get indices of used tilts
    used_indices = torch.nonzero(ts.use_tilt, as_tuple=True)[0]
    used_tilt_data = tilt_data[used_indices]
    used_angles = ts.angles[used_indices]

    # Write tilt stack as MRC
    # MRC format expects (Z, Y, X) which is (n_tilts, H, W)
    with mrcfile.new(ts.tilt_stack_path, overwrite=True) as mrc:
        mrc.set_data(used_tilt_data.cpu().numpy().astype("float32"))
        mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

    # Write angle file
    angle_lines = [f"{angle:.2f}" for angle in used_angles.tolist()]
    Path(ts.angle_file_path).write_text("\n".join(angle_lines))

    # Create thumbnails
    if create_thumbnails:
        Path(ts.tilt_stack_thumbnail_dir).mkdir(parents=True, exist_ok=True)

        for idx in used_indices.tolist():
            tilt_image = tilt_data[idx]  # (H, W)

            # Normalize using center circular region (diameter = half of smaller dimension)
            h, w = tilt_image.shape
            diameter = min(h, w) // 2

            # Use norm to normalize based on center circular region
            normalized = norm(tilt_image, dimensionality=2, diameter=diameter)

            # Transform to 0-255 range: median at 128, +/-3σ spans full range
            # Since norm gives (x - mean) / std, we map [-3, 3] -> [0, 255]
            # Formula: (normalized / 6 + 0.5) * 255, clamped to [0, 255]
            thumbnail = ((normalized / 6.0 + 0.5) * 255.0).clamp(0, 255)

            # Convert to uint8 numpy array
            thumbnail_np = thumbnail.cpu().numpy().astype(np.uint8)

            # Save as PNG
            thumbnail_path = ts.tilt_stack_thumbnail_path(ts.tilt_movie_paths[idx])
            Image.fromarray(thumbnail_np, mode="L").save(thumbnail_path)