"""
CTF volume reconstruction for subtomograms

This module contains methods for reconstructing solid CTF volumes at specified 3D positions
using Fourier insertion. The CTF volumes represent a weighted sum of CTF^2 divided
by |CTF|, which approximates |CTF|. The output volumes remain in Fourier space (rfft format,
half-Hermitian along x) and are used for CTF correction in subtomogram averaging.
Unlike traditional CTF volumes, these solid CTF volumes do not have the small missing wedges
between adjacent tilts. The gaps between adjacent tilts are filled with CTF values rendered
from interpolated CTF parameters of the adjacent tilts.
"""

import torch
import torch_projectors
import mrcfile
from typing import Optional
from ..ctf import CTF
from ..euler import euler_to_matrix
from ..ops import get_sinc2_correction, resize_ft


def _get_param_at_tilt(param, tilt_idx, batch_shape, device):
    """Extract parameter value at a specific tilt index.

    Args:
        param: Parameter value (scalar, 1D tensor for per-tilt, or ND tensor for per-particle-per-tilt)
        tilt_idx: Tilt index to extract
        batch_shape: Original batch shape for scalar expansion
        device: Device for tensor creation

    Returns:
        Parameter value at the specified tilt, shape (*batch_shape,) or scalar
    """
    if isinstance(param, torch.Tensor):
        if param.ndim == 0:
            # Scalar tensor
            return param
        elif param.ndim == 1:
            # Per-tilt tensor (n_tilts,)
            return param[tilt_idx]
        else:
            # Per-particle-per-tilt tensor (..., n_tilts)
            return param[..., tilt_idx]
    else:
        # Python scalar
        return param


def _normalize_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Normalize a matrix to be a valid rotation matrix using SVD.

    Given an arbitrary 3x3 matrix, finds the closest orthogonal matrix
    with determinant +1 (proper rotation).

    Args:
        matrix: Input matrices of shape (..., 3, 3)

    Returns:
        Normalized rotation matrices of shape (..., 3, 3)
    """
    # SVD: M = U @ S @ V^T
    # Closest orthogonal matrix is U @ V^T
    U, S, Vh = torch.linalg.svd(matrix)
    R = U @ Vh

    # Ensure proper rotation (det = +1) by flipping sign if needed
    det = torch.linalg.det(R)
    # Create a diagonal correction matrix that flips the last column of U if det < 0
    correction = torch.ones_like(det).unsqueeze(-1).expand(*det.shape, 3)
    correction = torch.diag_embed(correction)
    correction[..., 2, 2] = torch.sign(det)

    R = U @ correction @ Vh

    return R

def reconstruct_subvolume_solid_ctfs(
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
    Reconstruct solid CTF volumes by filling gaps between tilt angles.

    Unlike reconstruct_subvolume_ctfs which only inserts thin slices at discrete
    tilt angles, this method fills the gaps in Fourier space by interpolating
    CTF parameters between adjacent tilts to ensure continuous coverage.

    The method calculates the angular step needed for continuous Fourier space
    coverage (approximately 2/size radians at Nyquist), then inserts interpolated
    CTF slices between adjacent included tilts.

    Edge handling:
    - Continuous blocks of excluded tilts at the low or high end of the tilt
      series are NOT filled (we don't extrapolate beyond the data).
    - Excluded tilts flanked by included tilts ARE filled by interpolating
      between the closest included tilts (maintaining the same slice density
      as between immediate included neighbors).

    Args:
        ts: TiltSeries instance containing geometry and transformations
        coords: Particle coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
                where ... represents arbitrary batch dimensions. Each particle can have
                different coordinates per tilt (for tracking).
        pixel_size: Pixel size in Angstroms
        size: Volume box size in pixels (should be even)
        oversampling: CTF patch size factor (ctf_patch_size = size * oversampling) for better
                     sampling of high-frequency CTF oscillations (default: 1.0)
        apply_ctf: Whether to use actual CTF or flat ones (default: True)
        ctf_weighted: Whether to apply dose/location weighting to CTFs (default: True)
        padding_mode: Padding mode for grid_sample ('zeros', 'border', 'reflection')
        tilt_ids: Optional tensor of tilt indices to use for reconstruction, shape (n_selected_tilts,).
                  If None, uses ts.use_tilt mask. (default: None)
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
        >>> # Reconstruct solid CTF volumes for particles with tracked positions
        >>> ts = TiltSeries.load_meta("path/to/metadata.xml")
        >>> coords = torch.randn(10, ts.n_tilts, 3) * 100  # 10 particles
        >>> ctf_vols = ts.reconstruct_subvolume_solid_ctfs(
        ...     coords, pixel_size=10.0, size=64
        ... )
        >>> ctf_vols.shape
        torch.Size([10, 64, 64, 33])
    """
    device = coords.device

    # Store original batch shape
    original_shape = coords.shape[:-2]
    n_tilts = coords.shape[-2]

    if n_tilts != ts.n_tilts:
        raise ValueError(f"coords has {n_tilts} tilts but TiltSeries has {ts.n_tilts}")

    # Determine which tilts are included
    if tilt_ids is not None:
        included_mask = torch.zeros(ts.n_tilts, dtype=torch.bool, device=device)
        included_mask[tilt_ids] = True
    else:
        included_mask = ts.use_tilt.bool().to(device)

    included_indices = torch.where(included_mask)[0]

    if len(included_indices) == 0:
        raise ValueError("No tilts are included for reconstruction")

    # Sort included tilts by angle
    included_angles = ts.angles[included_indices].to(device)
    sorted_order = torch.argsort(included_angles)
    sorted_included_indices = included_indices[sorted_order]
    sorted_included_angles = included_angles[sorted_order]

    # Calculate angular step for continuous coverage
    # At Nyquist, consecutive slices should be separated by ≤1 voxel
    # Δθ ≈ 2/size radians ensures coverage
    angular_step_rad = 2.0 / size
    angular_step_deg = angular_step_rad * (180.0 / torch.pi)

    # Get original CTFs and Euler angles for all tilts
    ctfs = ts.get_ctfs_for_particles(
        coords=coords,
        pixel_size=pixel_size,
        weighted=ctf_weighted
    )

    if not apply_ctf:
        ctfs = ctfs.make_flat()

    euler_angles = ts.get_angle_in_all_tilts(coords=coords, angles=angles)

    # Convert Euler angles to rotation matrices for proper interpolation
    # (Euler angles can have multiple equivalent representations, so interpolating
    # them directly doesn't work reliably)
    rotation_matrices = euler_to_matrix(euler_angles)  # (..., n_tilts, 3, 3)

    # Build lists of all tilt data (original + interpolated)
    all_rotation_matrices = []
    all_defocus = []
    all_defocus_delta = []
    all_defocus_angle = []
    all_phase_shift = []
    all_bfactor = []
    all_bfactor_delta = []
    all_bfactor_angle = []
    all_scale = []

    # Process each pair of adjacent included tilts
    n_included = len(sorted_included_indices)
    for i in range(n_included):
        idx = sorted_included_indices[i].item()

        # Add the original tilt (using rotation matrix instead of Euler angles)
        all_rotation_matrices.append(rotation_matrices[..., idx, :, :])
        all_defocus.append(_get_param_at_tilt(ctfs.defocus, idx, original_shape, device))
        all_defocus_delta.append(_get_param_at_tilt(ctfs.defocus_delta, idx, original_shape, device))
        all_defocus_angle.append(_get_param_at_tilt(ctfs.defocus_angle, idx, original_shape, device))
        all_phase_shift.append(_get_param_at_tilt(ctfs.phase_shift, idx, original_shape, device))
        all_bfactor.append(_get_param_at_tilt(ctfs.bfactor, idx, original_shape, device))
        all_bfactor_delta.append(_get_param_at_tilt(ctfs.bfactor_delta, idx, original_shape, device))
        all_bfactor_angle.append(_get_param_at_tilt(ctfs.bfactor_angle, idx, original_shape, device))
        all_scale.append(_get_param_at_tilt(ctfs.scale, idx, original_shape, device))

        # Add interpolated tilts between this and next included tilt
        if i < n_included - 1:
            next_idx = sorted_included_indices[i + 1].item()
            angle_gap = (sorted_included_angles[i + 1] - sorted_included_angles[i]).item()

            # Number of interpolated tilts needed to fill the gap
            # We want gaps of at most angular_step_deg between consecutive slices
            n_interp = max(0, int(torch.ceil(torch.tensor(angle_gap / angular_step_deg)).item()) - 1)

            if n_interp > 0:
                # Get rotation matrices at both endpoints
                rot_a = rotation_matrices[..., idx, :, :]
                rot_b = rotation_matrices[..., next_idx, :, :]

                defocus_a = _get_param_at_tilt(ctfs.defocus, idx, original_shape, device)
                defocus_b = _get_param_at_tilt(ctfs.defocus, next_idx, original_shape, device)

                defocus_delta_a = _get_param_at_tilt(ctfs.defocus_delta, idx, original_shape, device)
                defocus_delta_b = _get_param_at_tilt(ctfs.defocus_delta, next_idx, original_shape, device)

                defocus_angle_a = _get_param_at_tilt(ctfs.defocus_angle, idx, original_shape, device)
                defocus_angle_b = _get_param_at_tilt(ctfs.defocus_angle, next_idx, original_shape, device)

                phase_shift_a = _get_param_at_tilt(ctfs.phase_shift, idx, original_shape, device)
                phase_shift_b = _get_param_at_tilt(ctfs.phase_shift, next_idx, original_shape, device)

                bfactor_a = _get_param_at_tilt(ctfs.bfactor, idx, original_shape, device)
                bfactor_b = _get_param_at_tilt(ctfs.bfactor, next_idx, original_shape, device)

                bfactor_delta_a = _get_param_at_tilt(ctfs.bfactor_delta, idx, original_shape, device)
                bfactor_delta_b = _get_param_at_tilt(ctfs.bfactor_delta, next_idx, original_shape, device)

                bfactor_angle_a = _get_param_at_tilt(ctfs.bfactor_angle, idx, original_shape, device)
                bfactor_angle_b = _get_param_at_tilt(ctfs.bfactor_angle, next_idx, original_shape, device)

                scale_a = _get_param_at_tilt(ctfs.scale, idx, original_shape, device)
                scale_b = _get_param_at_tilt(ctfs.scale, next_idx, original_shape, device)

                for j in range(1, n_interp + 1):
                    t = j / (n_interp + 1)  # Interpolation factor (0 < t < 1)

                    # Interpolate rotation matrices and normalize to ensure valid rotation
                    interp_rot = torch.lerp(rot_a, rot_b, t)
                    interp_rot = _normalize_rotation_matrix(interp_rot)
                    all_rotation_matrices.append(interp_rot)

                    # Interpolate CTF parameters
                    all_defocus.append(torch.lerp(defocus_a, defocus_b, t) if isinstance(defocus_a, torch.Tensor) else defocus_a + t * (defocus_b - defocus_a))
                    all_defocus_delta.append(torch.lerp(defocus_delta_a, defocus_delta_b, t) if isinstance(defocus_delta_a, torch.Tensor) else defocus_delta_a + t * (defocus_delta_b - defocus_delta_a))
                    all_defocus_angle.append(torch.lerp(defocus_angle_a, defocus_angle_b, t) if isinstance(defocus_angle_a, torch.Tensor) else defocus_angle_a + t * (defocus_angle_b - defocus_angle_a))
                    all_phase_shift.append(torch.lerp(phase_shift_a, phase_shift_b, t) if isinstance(phase_shift_a, torch.Tensor) else phase_shift_a + t * (phase_shift_b - phase_shift_a))
                    all_bfactor.append(torch.lerp(bfactor_a, bfactor_b, t) if isinstance(bfactor_a, torch.Tensor) else bfactor_a + t * (bfactor_b - bfactor_a))
                    all_bfactor_delta.append(torch.lerp(bfactor_delta_a, bfactor_delta_b, t) if isinstance(bfactor_delta_a, torch.Tensor) else bfactor_delta_a + t * (bfactor_delta_b - bfactor_delta_a))
                    all_bfactor_angle.append(torch.lerp(bfactor_angle_a, bfactor_angle_b, t) if isinstance(bfactor_angle_a, torch.Tensor) else bfactor_angle_a + t * (bfactor_angle_b - bfactor_angle_a))
                    all_scale.append(torch.lerp(scale_a, scale_b, t) if isinstance(scale_a, torch.Tensor) else scale_a + t * (scale_b - scale_a))

    # Stack all parameters into tensors
    n_total_tilts = len(all_rotation_matrices)

    # Stack rotation matrices: (..., n_total_tilts, 3, 3)
    tilt_matrices = torch.stack(all_rotation_matrices, dim=-3)

    # Stack CTF parameters
    # Handle mixed scalar/tensor cases by converting to tensors
    def stack_params(param_list, batch_shape):
        """Stack parameters, handling mixed scalar/tensor cases."""
        tensors = []
        for p in param_list:
            if isinstance(p, torch.Tensor):
                if p.ndim == 0:
                    # Scalar tensor - keep as scalar or expand to batch shape
                    if len(batch_shape) == 0:
                        tensors.append(p)
                    else:
                        tensors.append(p.expand(*batch_shape))
                else:
                    tensors.append(p)
            else:
                # Python scalar -> convert to tensor (scalar or batch shape)
                if len(batch_shape) == 0:
                    tensors.append(torch.tensor(p, device=device, dtype=torch.float32))
                else:
                    tensors.append(torch.full(batch_shape, p, device=device, dtype=torch.float32))
        return torch.stack(tensors, dim=-1)

    stacked_defocus = stack_params(all_defocus, original_shape)
    stacked_defocus_delta = stack_params(all_defocus_delta, original_shape)
    stacked_defocus_angle = stack_params(all_defocus_angle, original_shape)
    stacked_phase_shift = stack_params(all_phase_shift, original_shape)
    stacked_bfactor = stack_params(all_bfactor, original_shape)
    stacked_bfactor_delta = stack_params(all_bfactor_delta, original_shape)
    stacked_bfactor_angle = stack_params(all_bfactor_angle, original_shape)
    stacked_scale = stack_params(all_scale, original_shape)

    # Create interpolated CTF object with stacked parameters
    interp_ctf = CTF()
    interp_ctf.pixel_size = pixel_size
    interp_ctf.voltage = ctfs.voltage
    interp_ctf.cs = ctfs.cs
    interp_ctf.amplitude = ctfs.amplitude
    interp_ctf.defocus = stacked_defocus
    interp_ctf.defocus_delta = stacked_defocus_delta
    interp_ctf.defocus_angle = stacked_defocus_angle
    interp_ctf.phase_shift = stacked_phase_shift
    interp_ctf.bfactor = stacked_bfactor
    interp_ctf.bfactor_delta = stacked_bfactor_delta
    interp_ctf.bfactor_angle = stacked_bfactor_angle
    interp_ctf.scale = stacked_scale

    # CTF patch size for better sampling
    ctf_patch_size = int(size * oversampling)

    # Evaluate 2D CTFs in Fourier space (..., n_total_tilts, ctf_patch_size, ctf_patch_size//2+1)
    ctf_2d = interp_ctf.get_2d(
        size=ctf_patch_size,
        device=device,
        ignore_below_res=ctf_ignore_below_res,
        ignore_transition_res=ctf_ignore_transition_res,
    )

    weights_2d = torch.ones_like(ctf_2d)
    ctf_2d = ctf_2d.abs()

    # Flatten batch dimensions for processing
    ctf_2d = ctf_2d.reshape(-1, n_total_tilts, ctf_patch_size, ctf_patch_size // 2 + 1)
    n_particles = ctf_2d.shape[0]

    weights_2d = weights_2d.reshape(-1, n_total_tilts, ctf_patch_size, ctf_patch_size // 2 + 1)

    # Reshape rotation matrices for backprojection
    # tilt_matrices is already (..., n_total_tilts, 3, 3), reshape to (n_particles, n_total_tilts, 3, 3)
    tilt_matrices = tilt_matrices.reshape(n_particles, n_total_tilts, 3, 3)

    # No shifts needed for CTF backprojection
    shifts = torch.zeros(n_particles, n_total_tilts, 2, dtype=torch.float32, device=device)

    # Backproject CTF^2 patterns weighted by |CTF| using torch_projectors
    data_rec, weight_rec = torch_projectors.backproject_2d_to_3d_forw(
        projections=torch.complex(ctf_2d, torch.zeros_like(ctf_2d)),
        weights=weights_2d,
        rotations=tilt_matrices.transpose(-2, -1),
        shifts=shifts,
        interpolation='linear',
        oversampling=1.0,
    )

    weight_rec = torch.clamp(weight_rec, min=1e-6)

    # For the solid CTF volumes, we want to bring components with interpolation
    # weights < 1 back up to 1
    data_rec = data_rec / weight_rec

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


def reconstruct_subvolume_solid_ctfs_single(
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
    Reconstruct solid CTF volumes at static 3D positions (same across all tilts).

    Convenience method that replicates coordinates for each tilt and
    calls the main solid CTF volume reconstruction method.

    Args:
        ts: TiltSeries instance containing geometry and transformations
        coords: Particle coordinates in volume space (Angstroms), shape (..., 3)
                where ... represents arbitrary batch dimensions. All tilts will
                use the same coordinates (static particles).
        pixel_size: Pixel size in Angstroms
        size: Volume box size in pixels (should be even)
        oversampling: CTF patch size factor (ctf_patch_size = size * oversampling) for better
                     sampling of high-frequency CTF oscillations (default: 1.0)
        apply_ctf: Whether to use actual CTF or flat ones (default: True)
        ctf_weighted: Whether to apply dose/location weighting to CTFs (default: True)
        tilt_ids: Optional tensor of tilt indices to use for reconstruction, shape (n_selected_tilts,).
                  If None, uses ts.use_tilt mask. (default: None)
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
        >>> # Reconstruct solid CTF volumes for static particle positions
        >>> ts = TiltSeries.load_meta("path/to/metadata.xml")
        >>> coords = torch.randn(10, 3) * 100  # 10 static particles
        >>> ctf_vols = ts.reconstruct_subvolume_solid_ctfs_single(
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
    return reconstruct_subvolume_solid_ctfs(
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