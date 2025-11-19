"""
Subtomogram reconstruction from tilt series

This module contains methods for reconstructing subtomograms at specified 3D positions
using weighted backprojection with CTF correction.
"""

import torch
import torch_projectors
import mrcfile
from typing import Optional
from ..euler import euler_to_matrix, rotate_x
from ..ops import get_sinc2_correction
from torch_grid_utils import fftfreq_grid
from functools import lru_cache


@lru_cache(maxsize=2)
def sinc2(shape, device):
    grid = fftfreq_grid(
        image_shape=shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=device
    )
    sinc2 = torch.sinc(grid) ** 2
    correction = 1.0 / torch.clamp(sinc2, min=1e-6)
    return correction


def ifftshift_and_crop_3d(real_tensor: torch.Tensor, oversampling_factor: float) -> torch.Tensor:
    """
    Apply ifftshift and crop 3D volume to remove oversampling padding.

    Args:
        real_tensor: Real-space volume, shape (..., size, size, size)
        oversampling_factor: Oversampling factor used during reconstruction

    Returns:
        Cropped volume, shape (..., original_size, original_size, original_size)
    """
    shifted = torch.fft.ifftshift(real_tensor, dim=(-3, -2, -1))
    current_size = real_tensor.shape[-3]
    original_size = int(current_size / oversampling_factor)
    crop_total = current_size - original_size
    crop_start = crop_total // 2
    crop_end = crop_start + original_size
    return shifted[..., crop_start:crop_end, crop_start:crop_end, crop_start:crop_end]


def reconstruct_subvolumes(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    coords: torch.Tensor,
    pixel_size: float,
    size: int,
    oversampling: float = 1.0,
    apply_ctf: bool = True,
    ctf_weighted: bool = True,
    padding_mode: str = 'zeros',
    tilt_ids: Optional[torch.Tensor] = None,
    angles: Optional[torch.Tensor] = None,
    correct_attenuation: bool = False,
) -> torch.Tensor:
    """
    Reconstruct subtomograms at specified 3D positions using weighted backprojection.

    This method extracts sub-images from tilt series at specified 3D coordinates,
    applies CTF correction, and performs weighted backprojection to reconstruct
    subtomograms in real space.

    Args:
        ts: TiltSeries instance containing geometry and transformations
        tilt_data: Tilt images, shape (n_tilts, H, W)
        coords: Particle coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
                where ... represents arbitrary batch dimensions. Each particle can have
                different coordinates per tilt (for tracking).
        pixel_size: Pixel size of tilt_data in Angstroms
        size: Reconstruction box size in pixels (should be even)
        oversampling: Oversampling factor for reconstruction (default: 2.0)
        apply_ctf: Whether to apply CTF correction (default: True)
        ctf_weighted: Whether to apply dose/location weighting to CTFs (default: True)
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')
        tilt_ids: Optional tensor of tilt indices to use for reconstruction, shape (n_selected_tilts,).
                  If None, all tilts are used. (default: None)
        angles: Optional Euler angles in radians (ZYZ convention) to change reconstruction orientation,
                shape (..., n_tilts, 3). If provided, these rotations are applied to change the
                coordinate system of the reconstruction. (default: None)
        correct_attenuation: Do sinc2 attenuation

    Returns:
        Reconstructed subtomograms in real space, shape (..., size, size, size)

    Example:
        >>> # Reconstruct subtomograms for particles with tracked positions
        >>> ts = TiltSeries.load_meta("path/to/metadata.xml")
        >>> tilt_images = ts.load_images(pixel_size=10.0)
        >>> coords = torch.randn(10, ts.n_tilts, 3) * 100  # 10 particles
        >>> subtomos = ts.reconstruct_subvolumes(
        ...     tilt_images, coords, pixel_size=10.0, size=64
        ... )
        >>> subtomos.shape
        torch.Size([10, 64, 64, 64])
    """
    # Store original batch shapeno,
    original_shape = coords.shape[:-2]
    n_tilts = coords.shape[-2]

    if n_tilts != ts.n_tilts:
        raise ValueError(f"coords has {n_tilts} tilts but TiltSeries has {ts.n_tilts}")

    # extraction patch size
    subtilt_patch_size = int(size * oversampling)

    # Get sub-images in Fourier space (..., n_tilts, size, size//2+1)
    images_rft = ts.get_images_for_particles_rft(
        tilt_data=tilt_data,
        coords=coords,
        pixel_size=pixel_size,
        size=subtilt_patch_size,
        padding_mode=padding_mode
    )

    # Get CTFs if requested
    if apply_ctf:
        # Get CTFs for particles (..., n_tilts)
        ctfs = ts.get_ctfs_for_particles(
            coords=coords,
            pixel_size=pixel_size,
            weighted=ctf_weighted
        )

        # Evaluate 2D CTFs in Fourier space (..., n_tilts, size, size//2+1)
        ctf_2d = ctfs.get_2d(size=subtilt_patch_size, device=images_rft.device)

        # Apply CTF correction: multiply images by CTF
        images_rft = images_rft * ctf_2d

        # We don't do deconvolution here, just divide by abs(CTF) to get weighted sum
        ctf_2d = torch.abs(ctf_2d)
    else:
        ctf_2d = torch.ones(images_rft.shape, dtype=torch.float32, device=images_rft.device)

    # Filter by tilt_ids if provided
    if tilt_ids is not None:
        # Select only the specified tilts
        images_rft = images_rft[..., tilt_ids, :, :]
        ctf_2d = ctf_2d[..., tilt_ids, :, :]
        n_tilts = len(tilt_ids)

    # Flatten batch dimensions for processing
    # (..., n_tilts, size, size//2+1) -> (n_particles, n_tilts, size, size//2+1)
    images_rft_flat = images_rft.reshape(
        -1, n_tilts, subtilt_patch_size, subtilt_patch_size // 2 + 1
    )
    n_particles = images_rft_flat.shape[0]

    ctf_2d = ctf_2d.reshape(
        -1, n_tilts, subtilt_patch_size, subtilt_patch_size // 2 + 1
    )

    # Compute rotation matrices for each tilt
    # These transform from volume space to image space
    deg_to_rad = torch.pi / 180.0

    # Stack Euler angles for all tilts (..., n_tilts, 3)
    euler_angles = ts.get_angle_in_all_tilts(coords=coords, angles=angles)

    # Filter Euler angles by tilt_ids if provided
    if tilt_ids is not None:
        euler_angles = euler_angles[..., tilt_ids, :]

    # Get Euler matrices (..., n_tilts, 3, 3)
    tilt_matrices = euler_to_matrix(euler_angles)

    # No shifts needed (sub-images are already centered)
    shifts = torch.zeros(n_particles, n_tilts, 2, dtype=torch.float32, device=images_rft.device)

    # Backproject using torch_projectors
    # Input: (n_particles, n_tilts, size, size//2+1) complex
    # Output: (n_particles, oversampled_size, oversampled_size, oversampled_size//2+1) complex
    data_rec, weight_rec = torch_projectors.backproject_2d_to_3d_forw(
        projections=images_rft_flat,
        weights=ctf_2d,
        rotations=tilt_matrices.transpose(-2, -1),
        shifts=shifts,
        interpolation='linear',
        oversampling=1.0,
    )

    if apply_ctf:
        ctf_2d = torch.ones_like(ctf_2d)

        _, interp_weight_rec = torch_projectors.backproject_2d_to_3d_forw(
            projections=images_rft_flat,
            weights=ctf_2d,
            rotations=tilt_matrices.transpose(-2, -1),
            shifts=shifts,
            interpolation='linear',
            oversampling=1.0,
        )

        interp_weight_rec = torch.clamp(interp_weight_rec, min=0.0, max=1.0)

    else:
        # If we're not applying CTF, the weights volume already contains interpolation weights only
        interp_weight_rec = torch.clamp(weight_rec, min=0.0, max=1.0)

    # We don't want to bring voxels with interpolation weight < 1 back up, so multiply by interp_weight_rec
    data_rec = data_rec * interp_weight_rec / torch.clamp(weight_rec, min=1e-6)

    # Convert reconstruction to real space
    # irfftn with norm='forward' to match the example
    real_reconstruction = torch.fft.irfftn(data_rec, dim=(-3, -2, -1), norm='backward')

    if correct_attenuation:
        # sinc2 attenutation
        real_reconstruction = real_reconstruction * sinc2(
            (subtilt_patch_size,) * 3, device=real_reconstruction.device
        )

    # ifftshift and crop to original size
    result = ifftshift_and_crop_3d(real_reconstruction, oversampling)

    # Reshape back to original batch shape
    result = result.reshape(*original_shape, size, size, size)

    return result


def reconstruct_subvolumes_single(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    coords: torch.Tensor,
    pixel_size: float,
    size: int,
    oversampling: float = 2.0,
    apply_ctf: bool = True,
    ctf_weighted: bool = True,
    padding_mode: str = 'zeros',
    tilt_ids: Optional[torch.Tensor] = None,
    angles: Optional[torch.Tensor] = None,
    correct_attenuation: bool = False,
) -> torch.Tensor:
    """
    Reconstruct subtomograms at static 3D positions (same across all tilts).

    Convenience method that replicates coordinates for each tilt and
    calls the main reconstruction method.

    Args:
        ts: TiltSeries instance containing geometry and transformations
        tilt_data: Tilt images, shape (n_tilts, H, W)
        coords: Particle coordinates in volume space (Angstroms), shape (..., 3)
                where ... represents arbitrary batch dimensions. All tilts will
                use the same coordinates (static particles).
        pixel_size: Pixel size of tilt_data in Angstroms
        size: Reconstruction box size in pixels (should be even)
        oversampling: Oversampling factor for reconstruction (default: 2.0)
        apply_ctf: Whether to apply CTF correction (default: True)
        ctf_weighted: Whether to apply dose/location weighting to CTFs (default: True)
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')
        tilt_ids: Optional tensor of tilt indices to use for reconstruction, shape (n_selected_tilts,).
                  If None, all tilts are used. (default: None)
        angles: Optional Euler angles in radians (ZYZ convention) to change reconstruction orientation,
                shape (..., 3). If provided, these rotations are applied to change the
                coordinate system of the reconstruction. (default: None)
        correct_attenuation: do sinc2 attenutation

    Returns:
        Reconstructed subtomograms in real space, shape (..., size, size, size)

    Example:
        >>> # Reconstruct subtomograms for static particle positions
        >>> ts = TiltSeries.load_meta("path/to/metadata.xml")
        >>> tilt_images = ts.load_images(pixel_size=10.0)
        >>> coords = torch.randn(10, 3) * 100  # 10 static particles
        >>> subtomos = ts.reconstruct_subvolumes_single(
        ...     tilt_images, coords, pixel_size=10.0, size=64
        ... )
        >>> subtomos.shape
        torch.Size([10, 64, 64, 64])
    """
    # Replicate coordinates for each tilt: (..., 3) -> (..., n_tilts, 3)
    per_tilt_coords = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    # Replicate angles if provided: (..., 3) -> (..., n_tilts, 3)
    per_tilt_angles = None
    if angles is not None:
        per_tilt_angles = angles.unsqueeze(-2).expand(*angles.shape[:-1], ts.n_tilts, 3)

    # Reconstruct using main method
    return reconstruct_subvolumes(
        ts=ts,
        tilt_data=tilt_data,
        coords=per_tilt_coords,
        pixel_size=pixel_size,
        size=size,
        oversampling=oversampling,
        apply_ctf=apply_ctf,
        ctf_weighted=ctf_weighted,
        padding_mode=padding_mode,
        tilt_ids=tilt_ids,
        angles=per_tilt_angles,
        correct_attenuation=correct_attenuation,
    )