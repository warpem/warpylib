"""
Full tomogram reconstruction using tiled weighted backprojection.

This module contains methods for reconstructing full tomograms by dividing the volume
into overlapping tiles, reconstructing each tile, and assembling the final volume.
"""

import torch
import math
from typing import Optional
from ..ops import norm, mask_rectangular, subtract_plane, resize
from ..ops.bandpass import bandpass
from .reconstruct_subvolumes import get_sinc2_correction


def reconstruct_full(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    pixel_size: float,
    volume_dimensions_physical: tuple,
    subvolume_size: int = 64,
    subvolume_padding: float = 2.0,
    normalize: bool = True,
    invert: bool = False,
    apply_ctf: bool = True,
    ctf_weighted: bool = True,
    correct_attenuation: bool = True,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Reconstruct full tomogram using tiled weighted backprojection.

    This method divides the volume into a regular grid of tiles, reconstructs each
    tile independently using weighted backprojection, and assembles them into the
    full tomogram. Tiles are processed in batches to manage memory usage.

    The reconstruction process:
    1. Extracts sub-images at padded size (subvolume_size * subvolume_padding)
    2. Reconstructs each tile with CTF correction
    3. Crops to central subvolume_size
    4. Assembles cropped tiles into final volume

    Args:
        ts: TiltSeries instance containing geometry and transformations
        tilt_data: Tilt images, shape (n_tilts, H, W)
        pixel_size: Pixel size of tilt_data and output reconstruction (Angstroms)
        volume_dimensions_physical: Volume size in Angstroms (X, Y, Z)
        subvolume_size: Size of sub-volumes for tiled reconstruction (pixels)
        subvolume_padding: Padding factor - extracts boxes of size (subvolume_size * subvolume_padding)
        normalize: Whether to normalize tilt images
        invert: Whether to invert contrast
        apply_ctf: Whether to apply CTF correction
        ctf_weighted: Whether to apply dose/location weighting to CTFs
        correct_attenuation: Whether to apply sinc^2 correction for interpolation attenuation (default: True)
        batch_size: Number of tiles to process simultaneously

    Returns:
        Reconstructed tomogram, shape (Z, Y, X) in pixels

    Example:
        >>> ts = TiltSeries("path/to/metadata.xml")
        >>> tilt_data = ts.load_images(original_pixel_size=0.834, desired_pixel_size=8.0)
        >>> volume = ts.reconstruct_full(
        ...     tilt_data=tilt_data,
        ...     pixel_size=8.0,
        ...     volume_dimensions_physical=(4000*0.834, 5700*0.834, 1000*0.834),
        ...     subvolume_size=64,
        ...     subvolume_padding=2.0
        ... )
        >>> volume.shape
        torch.Size([104, 594, 417])
    """
    # Store volume dimensions on TiltSeries for use by other methods
    ts.volume_dimensions_physical = torch.tensor(volume_dimensions_physical, dtype=torch.float32)

    # Calculate volume dimensions in pixels (rounded to even)
    dims_physical = torch.tensor(volume_dimensions_physical, dtype=torch.float32)
    dims_pixels = (dims_physical / pixel_size).round().to(torch.int64)
    dims_pixels = (dims_pixels / 2).round().to(torch.int64) * 2  # Ensure even

    dim_x, dim_y, dim_z = int(dims_pixels[0]), int(dims_pixels[1]), int(dims_pixels[2])

    # Preprocess tilt data
    tilt_data_processed = preprocess_tilt_data(
        tilt_data=tilt_data,
        normalize=normalize,
        invert=invert,
        subvolume_size=subvolume_size,
        subvolume_padding=subvolume_padding
    )

    # Calculate padded extraction size (even)
    size_padded = int(subvolume_size * subvolume_padding)
    size_padded = (size_padded // 2) * 2  # Ensure even

    # Calculate sinc^2 correction pattern for interpolation attenuation (if enabled)
    if correct_attenuation:
        # Generate correction for padded size with oversampling=1.0 (matching reconstruction)
        sinc2_correction_padded = get_sinc2_correction(size=size_padded, oversampling=1.0)

        # Crop to central subvolume_size (same as reconstruction cropping)
        sinc2_correction = resize(sinc2_correction_padded, size=(subvolume_size, subvolume_size, subvolume_size))

        # Take reciprocal to get correction factor: 1 / max(sinc^2, 1e-6)
        correction_factor = 1.0 / torch.clamp(sinc2_correction, min=1e-6)

    # Generate grid of tile positions
    # Grid covers volume in steps of subvolume_size, centered on tile positions
    grid_x = (dim_x + subvolume_size - 1) // subvolume_size
    grid_y = (dim_y + subvolume_size - 1) // subvolume_size
    grid_z = (dim_z + subvolume_size - 1) // subvolume_size

    # Generate tile center coordinates in pixels
    tile_coords_list = []
    for iz in range(grid_z):
        for iy in range(grid_y):
            for ix in range(grid_x):
                # Center of each tile
                cx = ix * subvolume_size + subvolume_size // 2
                cy = iy * subvolume_size + subvolume_size // 2
                cz = iz * subvolume_size + subvolume_size // 2
                tile_coords_list.append([cx, cy, cz])

    tile_coords_pixels = torch.tensor(tile_coords_list, dtype=torch.float32)
    # Convert to physical coordinates (Angstroms)
    tile_coords_physical = tile_coords_pixels * pixel_size  # (n_tiles, 3)

    n_tiles = len(tile_coords_list)

    # Initialize output volume
    output_volume = torch.zeros(dim_z, dim_y, dim_x, dtype=torch.float32)

    # Process tiles in batches
    for batch_start in range(0, n_tiles, batch_size):
        batch_end = min(batch_start + batch_size, n_tiles)
        batch_coords_physical = tile_coords_physical[batch_start:batch_end]  # (batch, 3)

        # Reconstruct this batch of tiles
        # Output: (batch, size_padded, size_padded, size_padded)
        reconstructed_batch = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data_processed,
            coords=batch_coords_physical,
            pixel_size=pixel_size,
            size=size_padded,
            oversampling=1.0,
            apply_ctf=apply_ctf,
            ctf_weighted=ctf_weighted,
            padding_mode='zeros'
        )

        # Crop each reconstruction to central subvolume_size
        # Calculate crop boundaries
        crop_start = (size_padded - subvolume_size) // 2
        crop_end = crop_start + subvolume_size

        cropped_batch = reconstructed_batch[
            :,
            crop_start:crop_end,
            crop_start:crop_end,
            crop_start:crop_end
        ]  # (batch, subvolume_size, subvolume_size, subvolume_size)

        # Apply sinc^2 correction to cropped tiles (if enabled)
        if correct_attenuation:
            cropped_batch = cropped_batch * correction_factor

        # Place each tile into output volume
        for i in range(batch_end - batch_start):
            global_idx = batch_start + i
            ix = global_idx % grid_x
            iy = (global_idx // grid_x) % grid_y
            iz = global_idx // (grid_x * grid_y)

            # Calculate position in output volume
            x_start = ix * subvolume_size
            y_start = iy * subvolume_size
            z_start = iz * subvolume_size

            # Handle boundaries - determine how much of the tile fits
            x_end = min(x_start + subvolume_size, dim_x)
            y_end = min(y_start + subvolume_size, dim_y)
            z_end = min(z_start + subvolume_size, dim_z)

            # Determine how much to copy from the cropped tile
            copy_x = x_end - x_start
            copy_y = y_end - y_start
            copy_z = z_end - z_start

            # Copy data
            output_volume[
                z_start:z_end,
                y_start:y_end,
                x_start:x_end
            ] = cropped_batch[i, :copy_z, :copy_y, :copy_x]

    return output_volume


def preprocess_tilt_data(
    tilt_data: torch.Tensor,
    normalize: bool,
    invert: bool,
    subvolume_size: int,
    subvolume_padding: float,
) -> torch.Tensor:
    """
    Preprocess tilt data before reconstruction.

    Applies normalization, edge masking, and bandpass filtering to tilt images.

    Args:
        tilt_data: Input tilt images, shape (n_tilts, H, W)
        normalize: Whether to normalize images
        invert: Whether to invert contrast
        subvolume_size: Size of sub-volumes (for bandpass calculation)
        subvolume_padding: Padding factor (for bandpass calculation)

    Returns:
        Preprocessed tilt data, same shape as input
    """
    n_tilts = tilt_data.shape[0]

    # Process each tilt independently
    processed = []

    for t in range(n_tilts):
        img = tilt_data[t].clone()

        # Fit and subtract plane to remove background gradients
        img = subtract_plane(img, fit_and_subtract=True)

        # Sophisticated bandpass filtering to avoid edge artifacts:
        # 1. Pad to 2x size with mirroring
        # 2. Apply soft rectangular mask over padded region
        # 3. Bandpass filter
        # 4. Crop back to original size

        h, w = img.shape
        h_padded, w_padded = h + 256, w + 256  # Pad by 128 pixels on each side

        # Pad with reflection mode
        img_padded = resize(img, size=(h_padded, w_padded), padding_mode='reflect')

        # Apply rectangular mask with soft edge covering the entire padded area
        # The inner region is the original size, soft edge covers the padding
        img_padded = mask_rectangular(
            img_padded,
            region=(h, w),  # Inner size
            soft_edge=128      # Soft edge covers the padded region
        )

        # Bandpass filter on padded image
        # High-pass to remove low frequencies, low-pass at Nyquist
        high_pass_freq = 1.0 / (subvolume_size / 2)
        img_padded = bandpass(
            img_padded,
            dimensionality=2,
            low_freq=high_pass_freq,
            high_freq=1.0,
            soft_edge_low=0.0,
            soft_edge_high=0.0
        )

        # Crop back to original size
        img = resize(img_padded, size=(h, w))

        # Normalize to mean=0, std=1
        if normalize:
            img = norm(img, dimensionality=2)

        if invert:
            img = -img

        processed.append(img)

    return torch.stack(processed, dim=0)
