"""
TiltSeries CTF generation methods

This module contains methods for generating batched CTF objects for particles across tilts.
Replicates functionality from WarpLib's TiltSeries.cs GetCTFsForOneTilt and
GetCTFsForOneParticle methods.
"""

import torch
from ..ctf import CTF
from .positions import get_position_in_all_tilts


def get_ctfs_for_particles_single(
    ts: "TiltSeries",
    coords: torch.Tensor,
    pixel_size: float,
    weighted: bool = True,
    weights_only: bool = False,
    use_global_weights: bool = False,
) -> CTF:
    """
    Generate batched CTF for particles across all tilts (single position per particle).

    Convenience method that replicates coordinates for each tilt and calls
    get_ctfs_for_particles.

    Args:
        ts: TiltSeries instance
        coords: Coordinates in volume space (Angstroms), shape (..., 3)
               where ... represents arbitrary batch dimensions
        pixel_size: Pixel size in Angstroms for the CTF
        weighted: If True, apply dose and location weighting
        weights_only: If True, zero out defocus/Cs/amplitude (for weight-only CTF)
        use_global_weights: If True, include global B-factor and weight

    Returns:
        CTF object with batched parameters, shape (..., n_tilts)
    """
    # Replicate coordinates for each tilt: (..., 3) -> (..., n_tilts, 3)
    per_tilt_coords = coords.unsqueeze(-2).expand(*coords.shape[:-1], ts.n_tilts, 3)

    # Transform using main method
    return get_ctfs_for_particles(ts, per_tilt_coords, pixel_size, weighted, weights_only, use_global_weights)


def get_ctfs_for_particles(
    ts: "TiltSeries",
    coords: torch.Tensor,
    pixel_size: float,
    weighted: bool = True,
    weights_only: bool = False,
    use_global_weights: bool = False,
) -> CTF:
    """
    Generate batched CTF for particles across all tilts.

    This method transforms particle positions to get defocus at each tilt,
    then creates a CTF object with batched parameters.

    Args:
        ts: TiltSeries instance
        coords: Coordinates in volume space (Angstroms), shape (..., n_tilts, 3)
               where ... represents arbitrary batch dimensions
        pixel_size: Pixel size in Angstroms for the CTF
        weighted: If True, apply dose and location weighting
        weights_only: If True, zero out defocus/Cs/amplitude (for weight-only CTF)
        use_global_weights: If True, include global B-factor and weight

    Returns:
        CTF object with batched parameters, shape (..., n_tilts)
    """
    # Get device from TiltSeries
    device = ts.angles.device

    # Ensure coords are on the same device as TiltSeries
    coords = coords.to(device)

    # Store original batch shape
    batch_shape = coords.shape[:-2]

    # Get transformed positions: (..., n_tilts, 3)
    image_positions = get_position_in_all_tilts(ts, coords)

    # Flatten batch dimensions for processing
    image_positions_flat = image_positions.reshape(-1, ts.n_tilts, 3)
    coords_flat = coords.reshape(-1, ts.n_tilts, 3)
    n_particles = coords_flat.shape[0]

    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0

    # Create batched CTF
    ctf = ts.ctf.get_copy()
    ctf.pixel_size = pixel_size

    # Defocus from transformed positions (Z coordinate is in micrometers)
    # Shape: (n_particles, n_tilts)
    defocus = image_positions_flat[..., 2]

    # Get tilt-specific CTF parameters
    defocus_delta = torch.zeros(ts.n_tilts, dtype=torch.float32, device=device)
    defocus_angle = torch.zeros(ts.n_tilts, dtype=torch.float32, device=device)
    phase_shift = torch.zeros(ts.n_tilts, dtype=torch.float32, device=device)

    for t in range(ts.n_tilts):
        defocus_delta[t] = ts.get_tilt_defocus_delta(t)
        defocus_angle[t] = ts.get_tilt_defocus_angle(t)
        phase_shift[t] = ts.get_tilt_phase(t)

    if not weights_only:
        # Reshape to (..., n_tilts)
        ctf.defocus = defocus.reshape(*batch_shape, ts.n_tilts)
        ctf.defocus_delta = defocus_delta  # (n_tilts,) will broadcast
        ctf.defocus_angle = defocus_angle  # (n_tilts,) will broadcast
        ctf.phase_shift = phase_shift  # (n_tilts,) will broadcast
    else:
        ctf.defocus = torch.zeros((*batch_shape, ts.n_tilts), dtype=torch.float32)
        ctf.defocus_delta = 0.0
        ctf.cs = 0.0
        ctf.amplitude = 1.0

    if weighted:
        # Grid coordinates for interpolation
        tilt_indices = torch.arange(ts.n_tilts, dtype=torch.float32, device=device) * grid_step

        # Prepare coordinates for dose grids (at volume center)
        # Shape: (n_tilts, 3)
        dose_coords = torch.stack([
            torch.full((ts.n_tilts,), 0.5, device=device),
            torch.full((ts.n_tilts,), 0.5, device=device),
            tilt_indices
        ], dim=-1)

        # Prepare coordinates for location grids
        # Shape: (n_particles, n_tilts, 3)
        location_coords = torch.stack([
            coords_flat[..., 0] / ts.volume_dimensions_physical[0],
            coords_flat[..., 1] / ts.volume_dimensions_physical[1],
            tilt_indices.unsqueeze(0).expand(n_particles, -1)
        ], dim=-1)

        # Get dose weighting
        grid_is_trivial = all(d == 1 for d in ts.grid_dose_weights.dimensions)
        if grid_is_trivial:
            # Simple cosine weighting: (n_tilts,)
            dose_weight = torch.cos(torch.deg2rad(ts.angles))
        else:
            dose_weight = ts.grid_dose_weights.get_interpolated(dose_coords)

        # Get location weighting: (n_particles * n_tilts,)
        location_coords_flat = location_coords.reshape(-1, 3)
        location_weight = ts.grid_location_weights.get_interpolated(location_coords_flat)
        location_weight = location_weight.reshape(n_particles, ts.n_tilts)

        # Combine weights: (n_particles, n_tilts)
        scale = dose_weight.unsqueeze(0) * location_weight

        # Apply UseTilt mask
        use_tilt_mask = ts.use_tilt.float()
        use_tilt_mask = torch.where(use_tilt_mask > 0, 1.0, 0.0001)
        scale = scale * use_tilt_mask.unsqueeze(0)

        # Get B-factor
        grid_is_trivial = all(d == 1 for d in ts.grid_dose_bfacs.dimensions)
        if grid_is_trivial:
            # Simple dose-based B-factor: (n_tilts,)
            dose_bfac = -ts.dose * 4
        else:
            dose_bfac = ts.grid_dose_bfacs.get_interpolated(dose_coords)
            dose_bfac = torch.minimum(dose_bfac, -ts.dose * 3)

        # Get location B-factor: (n_particles * n_tilts,)
        location_bfac = ts.grid_location_bfacs.get_interpolated(location_coords_flat)
        location_bfac = location_bfac.reshape(n_particles, ts.n_tilts)

        # Combine B-factors: (n_particles, n_tilts)
        bfactor = dose_bfac.unsqueeze(0) + location_bfac

        # B-factor anisotropy (same for all particles)
        bfactor_delta = ts.grid_dose_bfacs_delta.get_interpolated(dose_coords)
        bfactor_angle = ts.grid_dose_bfacs_angle.get_interpolated(dose_coords)

        # Apply global weights if requested
        if use_global_weights:
            bfactor = bfactor + ts.global_bfactor
            scale = scale * ts.global_weight

        # Reshape to original batch shape
        ctf.bfactor = bfactor.reshape(*batch_shape, ts.n_tilts)
        ctf.bfactor_delta = bfactor_delta  # (n_tilts,) will broadcast
        ctf.bfactor_angle = bfactor_angle  # (n_tilts,) will broadcast
        ctf.scale = scale.reshape(*batch_shape, ts.n_tilts)

    return ctf


def get_ctfs_for_one_tilt(
    ts: "TiltSeries",
    tilt_id: int,
    defoci: torch.Tensor,
    coords: torch.Tensor,
    pixel_size: float,
    weighted: bool = True,
    weights_only: bool = False,
    use_global_weights: bool = False,
) -> CTF:
    """
    Generate batched CTF for multiple particles at a single tilt.

    This method creates a CTF object with batched parameters for many particles,
    useful for efficient batch processing of particles in a single tilt image.

    Args:
        ts: TiltSeries instance
        tilt_id: Index of the tilt (0 to n_tilts-1)
        defoci: Defocus values in micrometers for each particle, shape (...,)
        coords: Particle coordinates in volume space (Angstroms), shape (..., 3)
        pixel_size: Pixel size in Angstroms for the CTF
        weighted: If True, apply dose and location weighting
        weights_only: If True, zero out defocus/Cs/amplitude (for weight-only CTF)
        use_global_weights: If True, include global B-factor and weight

    Returns:
        CTF object with batched parameters, shape (...,)
    """
    if tilt_id < 0 or tilt_id >= ts.n_tilts:
        raise ValueError(f"tilt_id must be between 0 and {ts.n_tilts-1}, got {tilt_id}")

    # Store original batch shape
    batch_shape = coords.shape[:-1]

    # Flatten batch dimensions
    coords_flat = coords.reshape(-1, 3)
    defoci_flat = defoci.reshape(-1)
    n_particles = coords_flat.shape[0]

    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0

    # Get tilt-specific CTF parameters (scalars, will be broadcast)
    defocus_delta = ts.get_tilt_defocus_delta(tilt_id)
    defocus_angle = ts.get_tilt_defocus_angle(tilt_id)
    phase_shift = ts.get_tilt_phase(tilt_id)

    # Create CTF
    ctf = ts.ctf.get_copy()
    ctf.pixel_size = pixel_size

    if not weights_only:
        ctf.defocus = defoci  # (...,) original shape
        ctf.defocus_delta = defocus_delta  # scalar, will broadcast
        ctf.defocus_angle = defocus_angle  # scalar, will broadcast
        ctf.phase_shift = phase_shift  # scalar, will broadcast
    else:
        ctf.defocus = torch.zeros_like(defoci)
        ctf.defocus_delta = 0.0
        ctf.cs = 0.0
        ctf.amplitude = 1.0

    if weighted:
        # Get dose-based weights and B-factors (scalars, same for all particles)
        dose_coords = torch.tensor([[0.5, 0.5, tilt_id * grid_step]], dtype=torch.float32)

        # B-factor
        grid_is_trivial = all(d == 1 for d in ts.grid_dose_bfacs.dimensions)
        if grid_is_trivial:
            bfac = -ts.dose[tilt_id] * 4
        else:
            bfac = torch.minimum(
                ts.grid_dose_bfacs.get_interpolated(dose_coords)[0],
                torch.tensor(-ts.dose[tilt_id] * 3)
            )

        # Weight
        grid_is_trivial = all(d == 1 for d in ts.grid_dose_weights.dimensions)
        if grid_is_trivial:
            weight = torch.cos(torch.deg2rad(ts.angles[tilt_id]))
        else:
            weight = ts.grid_dose_weights.get_interpolated(dose_coords)[0]

        # Apply UseTilt mask
        weight = weight * (1.0 if ts.use_tilt[tilt_id] else 0.0001)

        # Global weights
        if use_global_weights:
            bfac = bfac + ts.global_bfactor
            weight = weight * ts.global_weight

        # B-factor anisotropy (scalars)
        bfac_delta = ts.grid_dose_bfacs_delta.get_interpolated(dose_coords)[0]
        bfac_angle = ts.grid_dose_bfacs_angle.get_interpolated(dose_coords)[0]

        # Get location-based corrections (batched per particle)
        # Shape: (n_particles, 3)
        location_coords = torch.stack([
            coords_flat[:, 0] / ts.volume_dimensions_physical[0],
            coords_flat[:, 1] / ts.volume_dimensions_physical[1],
            torch.full((n_particles,), 0.5)  # Z at volume center
        ], dim=-1)

        location_bfac = ts.grid_location_bfacs.get_interpolated(location_coords)  # (n_particles,)
        location_weight = ts.grid_location_weights.get_interpolated(location_coords)  # (n_particles,)

        # Combine dose and location contributions
        # bfac and weight are scalars, will broadcast with location tensors
        bfactor_flat = bfac + location_bfac  # (n_particles,)
        scale_flat = weight * location_weight  # (n_particles,)

        # Reshape back to original batch shape
        ctf.bfactor = bfactor_flat.reshape(*batch_shape)
        ctf.bfactor_delta = bfac_delta  # scalar, will broadcast
        ctf.bfactor_angle = bfac_angle  # scalar, will broadcast
        ctf.scale = scale_flat.reshape(*batch_shape)

    return ctf
