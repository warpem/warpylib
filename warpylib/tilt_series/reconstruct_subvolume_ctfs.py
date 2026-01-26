"""
CTF volume reconstruction for subtomograms

This module contains methods for reconstructing CTF volumes at specified 3D positions
using weighted backprojection. The CTF volumes represent a weighted sum of CTF^2 divided
by |CTF|, which approximates |CTF|. The output volumes remain in Fourier space (rfft format,
half-Hermitian along x) and are used for CTF correction in subtomogram averaging.
"""

import torch
import torch_projectors
import mrcfile
from typing import Optional
from ..euler import euler_to_matrix, rotate_x
from ..ops import resize_ft


def reconstruct_subvolume_ctfs(
    ts: "TiltSeries",
    coords: torch.Tensor,
    pixel_size: float,
    size: int,
    oversampling: float = 1.0,
    apply_ctf: bool = True,
    ctf_weighted: bool = True,
    tilt_ids: Optional[torch.Tensor] = None,
    angles: Optional[torch.Tensor] = None,
    ctf_ignore_below_res: Optional[float] = None,
    ctf_ignore_transition_res: Optional[float] = None,
) -> torch.Tensor:
    """
    Reconstruct CTF volumes at specified 3D positions using weighted backprojection.

    This method computes CTF volumes by backprojecting CTF^2 patterns weighted by |CTF|,
    then dividing by the sum of |CTF| weights. The result approximates |CTF| in 3D Fourier
    space and is used for CTF correction in subtomogram averaging. The output remains in
    Fourier space (rfft format, half-Hermitian along x).

    Args:
        ts: TiltSeries instance containing geometry and transformations
        coords: Particle coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
                where ... represents arbitrary batch dimensions. Each particle can have
                different coordinates per tilt (for tracking).
        pixel_size: Pixel size in Angstroms
        size: Volume box size in pixels (should be even)
        oversampling: CTF patch size factor (ctf_patch_size = size * oversampling) for better
                     sampling of high-frequency CTF oscillations (default: 1.0)
        apply_ctf: Whether to use actual CTF or flat weighting (default: True)
        ctf_weighted: Whether to apply dose/location weighting to CTFs (default: True)
        tilt_ids: Optional tensor of tilt indices to use for reconstruction, shape (n_selected_tilts,).
                  If None, all tilts are used. (default: None)
        angles: Optional Euler angles in radians (ZYZ convention) to change reconstruction orientation,
                shape (..., n_tilts, 3). If provided, these rotations are applied to change the
                coordinate system of the reconstruction. (default: None)
        ctf_ignore_below_res: Resolution in Angstroms below which CTF is fully ignored (set to 1).
                              Must be greater than ctf_ignore_transition_res. (default: None)
        ctf_ignore_transition_res: Resolution in Angstroms at which CTF is fully applied.
                                   Required when ctf_ignore_below_res is set. (default: None)

    Returns:
        CTF volumes in Fourier space (rfft format), shape (..., size, size, size//2+1)

    Example:
        >>> # Reconstruct CTF volumes for particles with tracked positions
        >>> ts = TiltSeries.load_meta("path/to/metadata.xml")
        >>> coords = torch.randn(10, ts.n_tilts, 3) * 100  # 10 particles
        >>> ctf_vols = ts.reconstruct_subvolume_ctfs(
        ...     coords, pixel_size=10.0, size=64
        ... )
        >>> ctf_vols.shape
        torch.Size([10, 64, 64, 33])
    """
    # Store original batch shapeno,
    original_shape = coords.shape[:-2]
    n_tilts = coords.shape[-2]

    if n_tilts != ts.n_tilts:
        raise ValueError(f"coords has {n_tilts} tilts but TiltSeries has {ts.n_tilts}")

    # CTF patch size for better sampling
    ctf_patch_size = int(size * oversampling)

    # Get CTFs if requested
    if apply_ctf:
        # Get CTFs for particles (..., n_tilts)
        ctfs = ts.get_ctfs_for_particles(
            coords=coords,
            pixel_size=pixel_size,
            weighted=ctf_weighted
        )

        # Evaluate 2D CTFs in Fourier space (..., n_tilts, ctf_patch_size, ctf_patch_size//2+1)
        ctf_2d = ctfs.get_2d(
            size=ctf_patch_size,
            device=coords.device,
            ignore_below_res=ctf_ignore_below_res,
            ignore_transition_res=ctf_ignore_transition_res,
        )
    else:
        # Get CTFs for particles (..., n_tilts) that don't have any oscillations, just weights (if desired)
        ctfs = ts.get_ctfs_for_particles(
            coords=coords,
            pixel_size=pixel_size,
            weighted=ctf_weighted
        )

        ctfs = ctfs.make_flat()

        # Evaluate 2D CTFs in Fourier space (..., n_tilts, ctf_patch_size, ctf_patch_size//2+1)
        ctf_2d = ctfs.get_2d(
            size=ctf_patch_size,
            device=coords.device,
            ignore_below_res=ctf_ignore_below_res,
            ignore_transition_res=ctf_ignore_transition_res,
        )

    weights_2d = torch.abs(ctf_2d)
    ctf_2d = ctf_2d ** 2

    # Filter by tilt_ids if provided
    if tilt_ids is not None:
        # Select only the specified tilts
        ctf_2d = ctf_2d[..., tilt_ids, :, :]
        weights_2d = weights_2d[..., tilt_ids, :, :]
        n_tilts = len(tilt_ids)

    # Flatten batch dimensions for processing
    # (..., n_tilts, ctf_patch_size, ctf_patch_size//2+1) -> (n_particles, n_tilts, ctf_patch_size, ctf_patch_size//2+1)

    ctf_2d = ctf_2d.reshape(
        -1, n_tilts, ctf_patch_size, ctf_patch_size // 2 + 1
    )
    n_particles = ctf_2d.shape[0]

    weights_2d = weights_2d.reshape(
        -1, n_tilts, ctf_patch_size, ctf_patch_size // 2 + 1
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

    # No shifts needed for CTF backprojection
    shifts = torch.zeros(n_particles, n_tilts, 2, dtype=torch.float32, device=coords.device)

    # Backproject CTF^2 patterns weighted by |CTF| using torch_projectors
    # Input: (n_particles, n_tilts, ctf_patch_size, ctf_patch_size//2+1) complex CTF^2
    # Output: (n_particles, ctf_patch_size, ctf_patch_size, ctf_patch_size//2+1) complex
    data_rec, weight_rec = torch_projectors.backproject_2d_to_3d_forw(
        projections=torch.complex(ctf_2d, torch.zeros_like(ctf_2d)),  # CTF^2
        weights=weights_2d,  # |CTF|
        rotations=tilt_matrices.transpose(-2, -1),
        shifts=shifts,
        interpolation='linear',
        oversampling=1.0,
    )

    weights_2d = torch.ones_like(weights_2d)

    _, interp_weight_rec = torch_projectors.backproject_2d_to_3d_forw(
        projections=torch.complex(ctf_2d, torch.zeros_like(ctf_2d)),
        weights=weights_2d,
        rotations=tilt_matrices.transpose(-2, -1),
        shifts=shifts,
        interpolation='linear',
        oversampling=1.0,
    )

    interp_weight_rec = torch.clamp(interp_weight_rec, min=0.0, max=1.0)

    # We don't want to bring voxels with interpolation weight < 1 back up, so multiply by interp_weight_rec
    data_rec = data_rec * interp_weight_rec / torch.clamp(weight_rec, min=1e-6)

    if ctf_patch_size > size:
        # Crop from larger patch size by transforming to real space, cropping, then back to Fourier
        result_padded = torch.fft.irfftn(data_rec, dim=(-3, -2, -1), norm='backward')
        result_cropped = resize_ft(result_padded, size=(size, size, size))
        # Transform back to rfft format, extract real part (CTF patterns are real-valued in Fourier space)
        result = torch.real(torch.fft.rfftn(result_cropped, dim=(-3, -2, -1), norm='backward'))
    else:
        # Already at correct size, extract real part (CTF patterns are real-valued in Fourier space)
        result = torch.real(data_rec)

    # Reshape back to original batch shape
    result = result.reshape(*original_shape, size, size, size // 2 + 1)

    return result


def reconstruct_subvolume_ctfs_single(
    ts: "TiltSeries",
    coords: torch.Tensor,
    pixel_size: float,
    size: int,
    oversampling: float = 1.0,
    apply_ctf: bool = True,
    ctf_weighted: bool = True,
    tilt_ids: Optional[torch.Tensor] = None,
    angles: Optional[torch.Tensor] = None,
    ctf_ignore_below_res: Optional[float] = None,
    ctf_ignore_transition_res: Optional[float] = None,
) -> torch.Tensor:
    """
    Reconstruct CTF volumes at static 3D positions (same across all tilts).

    Convenience method that replicates coordinates for each tilt and
    calls the main CTF volume reconstruction method.

    Args:
        ts: TiltSeries instance containing geometry and transformations
        coords: Particle coordinates in volume space (Angstroms), shape (..., 3)
                where ... represents arbitrary batch dimensions. All tilts will
                use the same coordinates (static particles).
        pixel_size: Pixel size in Angstroms
        size: Volume box size in pixels (should be even)
        oversampling: CTF patch size factor (ctf_patch_size = size * oversampling) for better
                     sampling of high-frequency CTF oscillations (default: 1.0)
        apply_ctf: Whether to use actual CTF or flat weighting (default: True)
        ctf_weighted: Whether to apply dose/location weighting to CTFs (default: True)
        tilt_ids: Optional tensor of tilt indices to use for reconstruction, shape (n_selected_tilts,).
                  If None, all tilts are used. (default: None)
        angles: Optional Euler angles in radians (ZYZ convention) to change reconstruction orientation,
                shape (..., 3). If provided, these rotations are applied to change the
                coordinate system of the reconstruction. (default: None)
        ctf_ignore_below_res: Resolution in Angstroms below which CTF is fully ignored (set to 1).
                              Must be greater than ctf_ignore_transition_res. (default: None)
        ctf_ignore_transition_res: Resolution in Angstroms at which CTF is fully applied.
                                   Required when ctf_ignore_below_res is set. (default: None)

    Returns:
        CTF volumes in Fourier space (rfft format), shape (..., size, size, size//2+1)

    Example:
        >>> # Reconstruct CTF volumes for static particle positions
        >>> ts = TiltSeries.load_meta("path/to/metadata.xml")
        >>> coords = torch.randn(10, 3) * 100  # 10 static particles
        >>> ctf_vols = ts.reconstruct_subvolume_ctfs_single(
        ...     coords, pixel_size=10.0, size=64
        ... )
        >>> ctf_vols.shape
        torch.Size([10, 64, 64, 33])
    """
    # Replicate coordinates for each tilt: (..., 3) -> (..., n_tilts, 3)
    per_tilt_coords = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    # Replicate angles if provided: (..., 3) -> (..., n_tilts, 3)
    per_tilt_angles = None
    if angles is not None:
        per_tilt_angles = angles.unsqueeze(-2).expand(*angles.shape[:-1], ts.n_tilts, 3)

    # Reconstruct using main method
    return reconstruct_subvolume_ctfs(
        ts=ts,
        coords=per_tilt_coords,
        pixel_size=pixel_size,
        size=size,
        oversampling=oversampling,
        apply_ctf=apply_ctf,
        ctf_weighted=ctf_weighted,
        tilt_ids=tilt_ids,
        angles=per_tilt_angles,
        ctf_ignore_below_res=ctf_ignore_below_res,
        ctf_ignore_transition_res=ctf_ignore_transition_res,
    )