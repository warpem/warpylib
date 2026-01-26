"""
Filter operations for Fourier space processing.

This module contains various filtering operations that can be applied
to 2D and 3D data in real or Fourier space.
"""

import torch
from typing import Optional
from functools import lru_cache

@lru_cache(maxsize=2)
def get_sinc2_correction(
    size: int | tuple[int, int] | tuple[int, int, int],
    oversampling: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a 2D or 3D volume with sinc^2 interpolation correction values.

    Computes the radial sinc^2 attenuation pattern from linear interpolation in Fourier space.
    Higher oversampling reduces attenuation by effectively spreading the data over a finer grid.

    Args:
        size: Box size in pixels. Can be:
            - int: Creates cubic volume (3D) of shape (size, size, size)
            - tuple[int, int]: Creates rectangular image (2D) of shape (height, width)
            - tuple[int, int, int]: Creates cuboid volume (3D) of shape (depth, height, width)
        oversampling: Oversampling factor used during reconstruction. The radial distance
                     is divided by this factor to account for reduced attenuation. (default: 1.0)

    Returns:
        2D or 3D tensor with sinc^2 correction values:
        - For int input: shape (size, size, size)
        - For (h, w) input: shape (h, w)
        - For (d, h, w) input: shape (d, h, w)

    Examples:
        >>> # 3D cubic correction
        >>> correction_3d = get_sinc2_correction(64, oversampling=2.0)
        >>> correction_3d.shape
        torch.Size([64, 64, 64])

        >>> # 2D rectangular correction
        >>> correction_2d = get_sinc2_correction((128, 256), oversampling=1.5)
        >>> correction_2d.shape
        torch.Size([128, 256])

        >>> # 3D cuboid correction
        >>> correction_cuboid = get_sinc2_correction((32, 64, 128), oversampling=2.0)
        >>> correction_cuboid.shape
        torch.Size([32, 64, 128])
    """
    # Parse size input to determine dimensionality
    if isinstance(size, int):
        # Cubic 3D volume
        dims = (size, size, size)
        ndim = 3
    elif len(size) == 2:
        # 2D image
        dims = size
        ndim = 2
    elif len(size) == 3:
        # 3D volume (possibly non-cubic)
        dims = size
        ndim = 3
    else:
        raise ValueError(f"size must be int, tuple[int, int], or tuple[int, int, int], got {size}")

    # Create 1D coordinate arrays for each dimension, normalized to frequency units
    # Range: -0.5 to 0.5 (DC at center, Nyquist at edges)
    coords = []
    for dim_size in dims:
        coord = (torch.arange(dim_size, dtype=torch.float32, device=device) - dim_size // 2) / dim_size
        coords.append(coord)

    # Create meshgrid based on dimensionality
    if ndim == 2:
        y, x = torch.meshgrid(coords[0], coords[1], indexing='ij')
        # Calculate radial distance
        r = torch.sqrt(x**2 + y**2)
    else:  # ndim == 3
        z, y, x = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        # Calculate radial distance
        r = torch.sqrt(x**2 + y**2 + z**2)

    # Apply oversampling scaling to radial distance
    if oversampling > 1.0:
        r = r / oversampling

    # Compute sinc^2(π*r)
    correction = 1.0 / torch.clamp(torch.sinc(r) ** 2, min=1e-6)

    return correction


def get_sinc2_correction_rft(
    size: int | tuple[int, int] | tuple[int, int, int],
    oversampling: float = 1.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create a 2D or 3D volume with sinc^2 interpolation correction values in rfft format.

    Computes the radial sinc^2 attenuation pattern from linear interpolation in Fourier space,
    with coordinates arranged in rfft (half Hermitian) format. The DC component is at the
    0th position (regular rfft convention).

    Args:
        size: Box size in pixels. Can be:
            - int: Creates cubic volume (3D) of shape (size, size, size//2+1)
            - tuple[int, int]: Creates rectangular image (2D) of shape (height, width//2+1)
            - tuple[int, int, int]: Creates cuboid volume (3D) of shape (depth, height, width//2+1)
        oversampling: Oversampling factor used during reconstruction. The radial distance
                     is divided by this factor to account for reduced attenuation. (default: 1.0)
        device: Device to put the result tensor on (default: None, uses default device)

    Returns:
        2D or 3D tensor with sinc^2 correction values in rfft format:
        - For int input: shape (size, size, size//2+1)
        - For (h, w) input: shape (h, w//2+1)
        - For (d, h, w) input: shape (d, h, w//2+1)

    Examples:
        >>> # 3D cubic correction in rfft format
        >>> correction_3d = get_sinc2_correction_rft(64, oversampling=2.0)
        >>> correction_3d.shape
        torch.Size([64, 64, 33])

        >>> # 2D rectangular correction in rfft format
        >>> correction_2d = get_sinc2_correction_rft((128, 256), oversampling=1.5)
        >>> correction_2d.shape
        torch.Size([128, 129])

        >>> # 3D cuboid correction in rfft format
        >>> correction_cuboid = get_sinc2_correction_rft((32, 64, 128), oversampling=2.0)
        >>> correction_cuboid.shape
        torch.Size([32, 64, 65])
    """
    # Parse size input to determine dimensionality
    if isinstance(size, int):
        # Cubic 3D volume
        dims = (size, size, size)
        ndim = 3
    elif len(size) == 2:
        # 2D image
        dims = size
        ndim = 2
    elif len(size) == 3:
        # 3D volume (possibly non-cubic)
        dims = size
        ndim = 3
    else:
        raise ValueError(f"size must be int, tuple[int, int], or tuple[int, int, int], got {size}")

    # Create coordinate arrays in rfft format
    # For non-width dimensions: use fftfreq (includes negative frequencies)
    # For width dimension: use rfftfreq (only non-negative frequencies)
    coords = []
    for i, dim_size in enumerate(dims):
        if i == len(dims) - 1:  # Last dimension (width) uses rfftfreq
            coord = torch.fft.rfftfreq(dim_size, d=1.0, device=device)
        else:  # Other dimensions use fftfreq
            coord = torch.fft.fftfreq(dim_size, d=1.0, device=device)
        coords.append(coord)

    # Create meshgrid based on dimensionality
    if ndim == 2:
        y, x = torch.meshgrid(coords[0], coords[1], indexing='ij')
        # Calculate radial distance
        r = torch.sqrt(x**2 + y**2)
    else:  # ndim == 3
        z, y, x = torch.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        # Calculate radial distance
        r = torch.sqrt(x**2 + y**2 + z**2)

    # Apply oversampling scaling to radial distance
    if oversampling > 1.0:
        r = r / oversampling

    # Compute sinc^2(π*r)
    arg = torch.pi * r
    # Handle sinc(0) = 1
    correction = torch.ones_like(r)
    mask = r != 0
    correction[mask] = (torch.sin(arg[mask]) / arg[mask]) ** 2

    return correction