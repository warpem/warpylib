"""
TiltSeries particle image extraction methods

This module contains methods for extracting particle images from tilt series data,
with sub-pixel precision achieved through Fourier-space shifting.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from .positions import get_position_in_all_tilts
from ..ops.shift_rft import shift_rft


def get_images_for_particles_rft(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    coords: torch.Tensor,
    pixel_size: float,
    size: int,
    mask: Optional[torch.Tensor] = None,
    padding_mode: str = 'zeros',
) -> torch.Tensor:
    """
    Extract particle images from tilt series and return as Fourier transforms.

    This function extracts image patches for particles across all tilts, applying
    sub-pixel precision through Fourier-space phase shifts. Extraction uses nearest-
    neighbor sampling for performance, with fractional positioning handled via
    phase shifts in Fourier space.

    Args:
        ts: TiltSeries instance containing geometry and transformations
        tilt_data: Tilt images, shape (n_tilts, H, W)
        coords: Particle coordinates in volume space (Angstroms),
               shape (..., n_tilts, 3) where ... represents arbitrary batch dimensions.
               Each particle can have different coordinates per tilt (for tracking).
        pixel_size: Pixel size of tilt_data in Angstroms
        size: Extraction box size in pixels (should be even)
        mask: Optional mask to apply in real space before FFT, shape (size, size)
             or broadcastable to (..., n_tilts, size, size)
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')

    Returns:
        Complex tensor of extracted particle images in RFFT format,
        shape (..., n_tilts, size, size//2+1). Images are:
        - Centered (fftshifted)
        - Normalized by 1/(size*size)
        - Sub-pixel aligned via Fourier-space phase shifts

    Examples:
        >>> # Extract images for 10 particles across 40 tilts
        >>> coords = torch.randn(10, 40, 3) * 1000  # Angstroms
        >>> tilt_data = torch.randn(40, 4096, 4096)
        >>> images_ft = ts.get_images_for_particles_rft(
        ...     tilt_data, coords, pixel_size=2.0, size=64
        ... )
        >>> images_ft.shape
        torch.Size([10, 40, 64, 33])

        >>> # With masking
        >>> from warpylib.ops.masking import mask_sphere
        >>> mask = mask_sphere(torch.ones(64, 64), diameter=60.0, soft_edge_width=4.0)
        >>> images_ft = ts.get_images_for_particles_rft(
        ...     tilt_data, coords, pixel_size=2.0, size=64, mask=mask
        ... )
    """
    # Validate size is even for proper fftshift
    if size % 2 != 0:
        raise ValueError(f"Size must be even, got {size}")

    # Get positions in Angstroms using TiltSeries transformations
    # Shape: (..., n_tilts, 3) where Z is defocus
    positions_angstrom = get_position_in_all_tilts(ts, coords)

    # Convert XY to pixel coordinates
    positions_pixels = positions_angstrom[..., :2] / pixel_size  # (..., n_tilts, 2)

    # Flatten batch dimensions for processing
    batch_shape = positions_pixels.shape[:-2]
    positions_flat = positions_pixels.reshape(-1, ts.n_tilts, 2)
    n_particles = positions_flat.shape[0]

    # Compute top-left corner of extraction region (center - size/2)
    top_left = positions_flat - size / 2  # (n_particles, n_tilts, 2)

    # Split into integer and fractional parts for sub-pixel precision
    # Integer part: where we extract from
    # Fractional part: compensated by Fourier-space shift
    top_left_int = torch.floor(top_left)
    top_left_frac = top_left - top_left_int

    # Create base sampling grid (same for all extractions)
    device = tilt_data.device
    y_offset, x_offset = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing='ij'
    )  # Both (size, size)

    # Extract patches tilt-by-tilt
    # For each tilt, extract all n_particles patches from that single tilt image
    H, W = tilt_data.shape[-2:]
    patches_list = []

    for t in range(ts.n_tilts):
        # Get integer positions for all particles at this tilt
        # Shape: (n_particles, 2)
        top_left_int_t = top_left_int[:, t, :]

        # Build grid for this tilt: (n_particles, size, size, 2)
        grid_x = top_left_int_t[:, None, None, 0] + x_offset  # (n_particles, size, size)
        grid_y = top_left_int_t[:, None, None, 1] + y_offset  # (n_particles, size, size)

        # Normalize to [-1, 1] for grid_sample
        grid_x_norm = 2 * grid_x / (W - 1) - 1
        grid_y_norm = 2 * grid_y / (H - 1) - 1
        grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)  # (n_particles, size, size, 2)

        # Extract all particles from this tilt
        # tilt_data[t] has shape (H, W), need to add batch and channel dims
        tilt_image = tilt_data[t].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # grid_sample expects input (N, C, H, W) and grid (N, H_out, W_out, 2)
        # We need to repeat the tilt image n_particles times (memory view, no copy)
        tilt_image_batched = tilt_image.expand(n_particles, -1, -1, -1)  # (n_particles, 1, H, W)

        patches_t = F.grid_sample(
            tilt_image_batched,
            grid,
            mode='nearest',
            padding_mode=padding_mode,
            align_corners=True
        )  # (n_particles, 1, size, size)

        patches_list.append(patches_t.squeeze(1))  # (n_particles, size, size)

    # Stack all tilts: (n_particles, n_tilts, size, size)
    patches = torch.stack(patches_list, dim=1)

    # Apply mask in real space if provided
    if mask is not None:
        patches = patches * mask

    # Transform to Fourier space
    patches_fft = torch.fft.rfft2(patches, dim=(-2, -1))

    # Normalize (matches C# which divides by size * size after FFT)
    patches_fft = patches_fft / (size * size)

    # Compute Fourier-space shifts for sub-pixel precision and centering
    # Shift components:
    #   1. -fractional_part: compensate for rounding down to integer position
    #   2. +size/2: center the image (equivalent to fftshift)
    decenter = size / 2
    shifts = -top_left_frac + decenter  # (n_particles, n_tilts, 2)

    # Apply shifts in Fourier space
    patches_fft_shifted = shift_rft(patches_fft, shifts)

    # Reshape back to original batch shape
    result = patches_fft_shifted.view(*batch_shape, ts.n_tilts, size, size // 2 + 1)

    return result


def get_images_for_particles_single_rft(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    coords: torch.Tensor,
    pixel_size: float,
    size: int,
    mask: Optional[torch.Tensor] = None,
    padding_mode: str = 'zeros',
) -> torch.Tensor:
    """
    Extract particle images for static particle positions (same across all tilts).

    Convenience function that replicates coordinates for each tilt and calls
    get_images_for_particles_rft. Use this when particles don't move between tilts.

    Args:
        ts: TiltSeries instance
        tilt_data: Tilt images, shape (n_tilts, H, W)
        coords: Particle coordinates in volume space (Angstroms),
               shape (..., 3) where ... represents arbitrary batch dimensions
        pixel_size: Pixel size of tilt_data in Angstroms
        size: Extraction box size in pixels (should be even)
        mask: Optional mask to apply in real space before FFT
        padding_mode: Padding mode for grid_sample

    Returns:
        Complex tensor of shape (..., n_tilts, size, size//2+1)

    Examples:
        >>> # Extract images for 100 particles at static positions
        >>> coords = torch.randn(100, 3) * 1000  # Angstroms
        >>> tilt_data = torch.randn(40, 4096, 4096)
        >>> images_ft = ts.get_images_for_particles_single_rft(
        ...     tilt_data, coords, pixel_size=2.0, size=64
        ... )
        >>> images_ft.shape
        torch.Size([100, 40, 64, 33])
    """
    # Replicate coords for each tilt: (..., 3) -> (..., n_tilts, 3)
    coords_per_tilt = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    return get_images_for_particles_rft(
        ts, tilt_data, coords_per_tilt, pixel_size, size, mask, padding_mode
    )