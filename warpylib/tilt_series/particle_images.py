"""
TiltSeries particle image extraction methods

This module contains methods for extracting particle images from tilt series data,
with sub-pixel precision achieved through Fourier-space shifting.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch_subpixel_crop import subpixel_crop_2d
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

    result = subpixel_crop_2d(
        image=tilt_data,
        positions=positions_pixels,
        sidelength=size,
        mask=mask,
        return_rfft=True,
        fftshifted=True,
    )

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