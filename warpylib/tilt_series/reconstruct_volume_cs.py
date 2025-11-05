"""
Real-space tomogram reconstruction using gradient optimization.

This module implements compressed sensing / iterative real-space reconstruction
that refines an initial Fourier-space reconstruction using a forward model with
CTF application and normalized cross-correlation loss.
"""

import torch
import mrcfile
from typing import Tuple, Optional
from pathlib import Path
from .reconstruct_volume import reconstruct_full, preprocess_tilt_data
from ..ops.rescale import rescale
from ..ops.norm import norm
from ..ops.filters import get_sinc2_correction_rft
from ..euler import euler_to_matrix


def calculate_rotated_bounding_box_z(
    ts: "TiltSeries",
    volume_dims_physical: torch.Tensor,
    tilt_ids: list[int],
    device: torch.device
) -> float:
    """
    Calculate the Z dimension needed for rotated volume bounding box.

    Uses TiltSeries angle methods to get proper rotation matrices accounting
    for tilt angle, tilt axis angle, level angles, and any spatially-varying
    angle corrections at the volume center.

    Args:
        ts: TiltSeries instance
        volume_dims_physical: Volume dimensions in Angstroms (X, Y, Z)
        tilt_ids: List of tilt indices
        device: Torch device

    Returns:
        Maximum Z dimension in Angstroms across all tilts in the batch
    """
    # Get volume center coordinates
    volume_center = volume_dims_physical * 0.5

    # Get Euler angles for volume center across all tilts
    # Shape: (n_tilts, 3) in radians
    euler_angles_all = ts.get_angle_in_all_tilts_single(coords=volume_center)

    # Select only the tilts we need
    euler_angles = euler_angles_all[tilt_ids]  # (batch_size, 3)

    # Convert Euler angles to rotation matrices
    rotation_matrices = euler_to_matrix(euler_angles)  # (batch_size, 3, 3)

    # Define 8 corners of the volume bounding box (centered at origin)
    half_dims = volume_dims_physical / 2.0
    corners = torch.tensor([
        [-half_dims[0], -half_dims[1], -half_dims[2]],
        [ half_dims[0], -half_dims[1], -half_dims[2]],
        [-half_dims[0],  half_dims[1], -half_dims[2]],
        [ half_dims[0],  half_dims[1], -half_dims[2]],
        [-half_dims[0], -half_dims[1],  half_dims[2]],
        [ half_dims[0], -half_dims[1],  half_dims[2]],
        [-half_dims[0],  half_dims[1],  half_dims[2]],
        [ half_dims[0],  half_dims[1],  half_dims[2]],
    ], dtype=torch.float32, device=device)  # (8, 3)

    # Rotate corners for each tilt and find maximum Z extent
    max_z_extent = 0.0
    for i in range(len(tilt_ids)):
        # Rotate all 8 corners: (8, 3) @ (3, 3).T -> (8, 3)
        rotated_corners = torch.matmul(corners, rotation_matrices[i].T)

        # Find Z extent (min to max in Z)
        z_min = rotated_corners[:, 2].min()
        z_max = rotated_corners[:, 2].max()
        z_extent = z_max - z_min

        max_z_extent = max(max_z_extent, z_extent.item())

    return max_z_extent


def reconstruct_full_cs(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    pixel_size: float,
    volume_dimensions_physical: Tuple[float, float, float],
    normalize: bool = True,
    invert: bool = False,
    n_iterations: int = 100,
    learning_rate: float = 1e-4,
    tilt_batch_size: int = 4,
    debug_output_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Reconstruct tomogram using real-space gradient optimization.

    This method starts with an initial Fourier-space reconstruction, then refines
    it iteratively using a real-space forward model with CTF application and
    normalized cross-correlation loss.

    The optimization process:
    1. Initial reconstruction via weighted backprojection (reconstruct_full)
    2. Upsample by 2x for better interpolation during rotation
    3. Iteratively refine using gradient descent:
       - Transform volume to each tilt's coordinate frame
       - Project along Z and apply CTF
       - Compute NCC with observed images
       - Backpropagate and update volume
    4. Downsample back to original resolution

    Args:
        ts: TiltSeries instance containing geometry and transformations
        tilt_data: Tilt images, shape (n_tilts, H, W)
        pixel_size: Pixel size of tilt_data and output reconstruction (Angstroms)
        volume_dimensions_physical: Volume size in Angstroms (X, Y, Z)
        normalize: Whether to normalize tilt images (default: True)
        invert: Whether to invert contrast (default: False)
        n_iterations: Number of optimization iterations (default: 100)
        learning_rate: Adam optimizer learning rate (default: 1e-4)
        tilt_batch_size: Number of tilts to process per iteration (default: 4)
        debug_output_dir: Optional directory to write intermediate reconstructions
                          after each iteration (default: None)

    Returns:
        Reconstructed tomogram, shape (Z, Y, X) in pixels

    Example:
        >>> ts = TiltSeries("path/to/metadata.xml")
        >>> tilt_data = ts.load_images(original_pixel_size=0.834, desired_pixel_size=10.0)
        >>> volume = ts.reconstruct_full_cs(
        ...     tilt_data=tilt_data,
        ...     pixel_size=10.0,
        ...     volume_dimensions_physical=(4600*0.834, 6000*0.834, 1000*0.834),
        ...     n_iterations=100,
        ...     learning_rate=1e-4,
        ...     tilt_batch_size=4
        ... )
    """
    device = tilt_data.device

    # Convert volume dimensions to tensor
    volume_dims_physical = torch.tensor(volume_dimensions_physical, dtype=torch.float32, device=device)

    print("Starting initial Fourier-space reconstruction...")
    # Step 1: Initial reconstruction using Fourier-space method
    with torch.no_grad():
        reconstructed_volume = reconstruct_full(
            ts=ts,
            tilt_data=tilt_data,
            pixel_size=pixel_size,
            volume_dimensions_physical=volume_dimensions_physical,
            subvolume_size=64,
            subvolume_oversampling=2.0,
            normalize=normalize,
            invert=invert,
            apply_ctf=True,
            ctf_weighted=True,
            batch_size=8
        )

        print(f"Initial reconstruction shape: {reconstructed_volume.shape}")

        # Step 2: Preprocess tilt data
        print("Preprocessing tilt data...")
        tilt_data_processed = preprocess_tilt_data(
            tilt_data=tilt_data,
            normalize=normalize,
            invert=invert,
            subvolume_size=64,
            subvolume_padding=2.0
        )

        # Step 3: Upsample reconstruction by 2x
        print("Upsampling reconstruction by 2x...")
        upscale_factor = 2
        D, H, W = reconstructed_volume.shape
        upscaled_size = (D * upscale_factor, H * upscale_factor, W * upscale_factor)
        upscaled_volume = rescale(reconstructed_volume, size=upscaled_size)
        print(f"Upscaled volume shape: {upscaled_volume.shape}")

        # Step 4: Prepare CTFs for all tilts
        print("Preparing CTF tensors...")
        # Get CTFs at volume center
        volume_center = volume_dims_physical * 0.5
        ctfs = ts.get_ctfs_for_particles_single(
            coords=volume_center,
            pixel_size=pixel_size,
            weighted=True,
            weights_only=False,
            use_global_weights=False
        )

        # Generate 2D CTF patterns matching tilt image dimensions
        # Shape: (n_tilts, H, W//2+1)
        ctf_2d = ctfs.get_2d(
            size=(tilt_data_processed.shape[1], tilt_data_processed.shape[2]),
            device=device
        )
        print(f"CTF patterns shape: {ctf_2d.shape}")

        sinc2_2d = get_sinc2_correction_rft(size=(tilt_data_processed.shape[1], tilt_data_processed.shape[2]), oversampling=2.0)

        # Pre-multiply tilt_data by CTFs for weighting
        # Also pre-multiply by sinc^2 to account for linear interpolation in reconstruction loop
        tilt_data_processed = torch.fft.irfft2(torch.fft.rfft2(tilt_data_processed) * ctf_2d * sinc2_2d)
        tilt_data_processed = norm(tilt_data_processed, dimensionality=2)

        # Use unweighted CTFs in forward model
        ctfs = ts.get_ctfs_for_particles_single(
            coords=volume_center,
            pixel_size=pixel_size,
            weighted=False,
            weights_only=False,
            use_global_weights=False
        )
        ctf_2d = ctfs.get_2d(
            size=(tilt_data_processed.shape[1], tilt_data_processed.shape[2]),
            device=device
        )

        # Phase-flip CTFs for forward model
        ctf_2d = torch.abs(ctf_2d)

    # Step 5: Detach and enable gradients
    print("\nStarting optimization...")
    volume_optimizable = upscaled_volume.detach().clone().requires_grad_(True)

    # Setup optimizer
    optimizer = torch.optim.Adam([volume_optimizable], lr=learning_rate)

    # Setup debug output directory if requested
    if debug_output_dir is not None:
        debug_dir = Path(debug_output_dir)
        debug_dir.mkdir(exist_ok=True, parents=True)
        print(f"Debug outputs will be written to: {debug_dir}")

    # Calculate output dimensions for transform_volume
    tilt_image_dims = torch.tensor([
        tilt_data_processed.shape[2],  # W
        tilt_data_processed.shape[1],  # H
    ], dtype=torch.float32, device=device)

    # Pre-calculate Z dimensions for all tilts based on rotated bounding boxes
    print("Pre-calculating Z dimensions for rotated volumes...")
    z_dims_physical = torch.zeros(ts.n_tilts, dtype=torch.float32, device=device)
    for t in range(ts.n_tilts):
        z_dims_physical[t] = calculate_rotated_bounding_box_z(
            ts, volume_dims_physical, [t], device
        )
    print(f"  Z dimension range: {z_dims_physical.min():.1f} - {z_dims_physical.max():.1f} Å")

    # Step 6: Optimization loop
    for iteration in range(n_iterations):
        optimizer.zero_grad()

        total_ncc = 0.0
        n_batches = 0

        # Collect projections for debug output
        all_projections = [] if debug_output_dir is not None else None

        # Process tilts in batches
        for batch_start in range(0, ts.n_tilts, tilt_batch_size):
            batch_end = min(batch_start + tilt_batch_size, ts.n_tilts)
            tilt_ids = list(range(batch_start, batch_end))
            batch_size = len(tilt_ids)

            # Get pre-calculated Z dimension for this batch
            z_physical_max = z_dims_physical[tilt_ids].max()
            z_pixels = (z_physical_max / pixel_size).round().long().item()
            # Ensure even
            z_pixels = (z_pixels + 1) // 2 * 2

            # Output dimensions: (X, Y, Z)
            output_dimensions = torch.tensor([
                tilt_image_dims[0],  # X
                tilt_image_dims[1],  # Y
                z_pixels             # Z
            ], dtype=torch.float32, device=device)

            # Transform volume to tilted coordinate frames
            # Output: (batch_size, Z, H, W)
            transformed = ts.transform_volume(
                volume=volume_optimizable,
                pixel_size=pixel_size,
                output_dimensions=output_dimensions,
                tilt_ids=tilt_ids,
                upscale_factor=upscale_factor
            )

            # Project by summing along Z
            # Output: (batch_size, H, W)
            projections = transformed.sum(dim=1)

            # Apply CTF in Fourier space
            projections_ft = torch.fft.rfft2(projections, dim=(-2, -1))
            projections_ctf_ft = projections_ft * ctf_2d[tilt_ids]
            projections_ctf = torch.fft.irfft2(
                projections_ctf_ft,
                s=(tilt_data_processed.shape[1], tilt_data_processed.shape[2]),
                dim=(-2, -1)
            )

            # Normalize projections
            projections_normalized = norm(projections_ctf, dimensionality=2)

            # Collect projections for debug output
            if all_projections is not None:
                all_projections.append(projections_normalized.detach())

            # Compute NCC with observed images
            ncc = (projections_normalized * tilt_data_processed[tilt_ids]).mean()
            total_ncc += ncc.item() * batch_size

            # Loss: 1 - NCC
            loss = 1.0 - ncc
            loss.backward()

            n_batches += batch_size

        # Optimization step
        optimizer.step()

        # Print progress
        avg_ncc = total_ncc / n_batches
        print(f"Iteration {iteration + 1}/{n_iterations}: NCC = {avg_ncc:.6f}")

        # Write debug output if requested
        if debug_output_dir is not None and (iteration + 1) % 10 == 0:
            with torch.no_grad():
                # Downsample current state to original resolution
                debug_volume = rescale(volume_optimizable.detach(), size=(D, H, W))

                # Write volume to MRC file
                debug_filename = debug_dir / f"iteration_{iteration + 1:04d}.mrc"
                with mrcfile.new(str(debug_filename), overwrite=True) as mrc:
                    mrc.set_data(debug_volume.cpu().numpy().astype('float32'))
                    mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

                # Stack projections and write to MRC file
                projections_stack = torch.cat(all_projections, dim=0)  # (n_tilts, H, W)
                projections_filename = debug_dir / f"projections_{iteration + 1:04d}.mrc"
                with mrcfile.new(str(projections_filename), overwrite=True) as mrc:
                    mrc.set_data(projections_stack.cpu().numpy().astype('float32'))
                    mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

    # Step 7: Downsample back to original resolution
    print("\nDownsampling optimized volume...")
    with torch.no_grad():
        final_volume = rescale(volume_optimizable.detach(), size=(D, H, W))

    print(f"Final volume shape: {final_volume.shape}")
    print("Reconstruction complete!")

    return final_volume