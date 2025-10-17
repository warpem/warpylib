"""
Normalization for RFFT-formatted tensors (half FFT, decentered).
"""

import torch
from typing import Literal


def norm_rft(
    tensor: torch.Tensor,
    dimensionality: int,
    diameter: float = 0,
    mode: Literal["inner", "outer"] = "inner",
) -> torch.Tensor:
    """
    Normalize an RFFT-formatted tensor (half FFT, decentered).

    Calculates statistics in Fourier space on a circular/spherical region. For real-valued
    tensors, subtracts mean and divides by std. For complex-valued tensors, only divides
    by std (no mean subtraction). Properly handles RFFT Hermitian symmetry by not
    double-counting x=0 components.

    Parameters
    ----------
    tensor : torch.Tensor
        RFFT-formatted tensor (real or complex), decentered
    dimensionality : int
        Number of spatial dimensions (2 or 3)
    diameter : float, optional
        Diameter of circular/spherical region in frequency space. If 0, use all components.
    mode : {'inner', 'outer'}, optional
        Whether to calculate statistics inside or outside the circle/sphere.
        Default is 'inner'.

    Returns
    -------
    torch.Tensor
        Normalized tensor (real: mean-subtracted and std-divided; complex: std-divided only)
    """
    from .masking import mask_sphere_rft

    if dimensionality > tensor.ndim:
        raise ValueError(
            f"Dimensionality ({dimensionality}) cannot exceed tensor dimensions ({tensor.ndim})"
        )

    if dimensionality not in (2, 3):
        raise ValueError(f"Dimensionality must be 2 or 3, got {dimensionality}")

    # Determine batch and spatial dimensions
    n_batch_dims = tensor.ndim - dimensionality
    spatial_shape = tensor.shape[n_batch_dims:]

    # Reshape to (batch, *spatial)
    if n_batch_dims > 0:
        batch_shape = tensor.shape[:n_batch_dims]
        tensor_reshaped = tensor.reshape(-1, *spatial_shape)
    else:
        batch_shape = None
        tensor_reshaped = tensor.unsqueeze(0)

    batch_size = tensor_reshaped.shape[0]

    is_complex = torch.is_complex(tensor_reshaped)

    # Get values for statistics (magnitude for complex, raw values for real)
    if is_complex:
        stat_values = torch.abs(tensor_reshaped)
    else:
        stat_values = tensor_reshaped

    # Create mask (handles Hermitian symmetry automatically)
    if diameter > 0:
        dummy = torch.ones(spatial_shape, device=tensor.device, dtype=stat_values.dtype)
        mask_float = mask_sphere_rft(dummy, diameter=diameter, soft_edge_width=0.0)

        if mode == "outer":
            mask = mask_float <= 0.5  # Convert to boolean
        else:  # mode == "inner"
            mask = mask_float > 0.5  # Convert to boolean

        # Expand mask for batch dimension
        mask = mask.unsqueeze(0).expand(batch_size, *spatial_shape)

    else:
        # No mask - just handle Hermitian symmetry by creating a simple mask
        # that zeros out half the x=0 components
        mask = torch.ones(spatial_shape, device=tensor.device, dtype=torch.bool)

        if dimensionality == 2:
            # For 2D, zero out the bottom half of the x=0 column (last dimension index 0)
            h = spatial_shape[0]
            h_half = h // 2
            mask[h_half + 1:, 0] = False
        elif dimensionality == 3:
            # For 3D, zero out half of the x=0 plane (last dimension index 0)
            h = spatial_shape[1]
            h_half = h // 2
            mask[:, h_half + 1:, 0] = False

        # Expand mask for batch dimension
        mask = mask.unsqueeze(0).expand(batch_size, *spatial_shape)

    # Calculate statistics on masked region (vectorized)
    spatial_dims = tuple(range(1, tensor_reshaped.ndim))

    if is_complex:
        # Complex: only calculate std from magnitudes, no mean subtraction
        sum_sq = ((stat_values ** 2) * mask).sum(dim=spatial_dims)
        count = mask.sum(dim=spatial_dims).float()
        variance = sum_sq / count
        std_val = torch.sqrt(variance)

        # Expand std for broadcasting
        std_val_expanded = std_val.reshape(batch_size, *([1] * dimensionality))

        # Normalize by dividing by std only
        result = torch.where(std_val_expanded > 0, tensor_reshaped / std_val_expanded, tensor_reshaped)

    else:
        # Real: subtract mean and divide by std
        masked_sum = (stat_values * mask).sum(dim=spatial_dims)
        count = mask.sum(dim=spatial_dims).float()
        mean_val = masked_sum / count

        # Expand mean for broadcasting
        mean_val_expanded = mean_val.reshape(batch_size, *([1] * dimensionality))

        # Calculate std
        masked_squared_diff = ((stat_values - mean_val_expanded) ** 2) * mask
        variance = masked_squared_diff.sum(dim=spatial_dims) / count
        std_val = torch.sqrt(variance)

        # Expand std for broadcasting
        std_val_expanded = std_val.reshape(batch_size, *([1] * dimensionality))

        # Normalize
        result = tensor_reshaped - mean_val_expanded
        result = torch.where(std_val_expanded > 0, result / std_val_expanded, result)

    # Reshape back to original shape
    if batch_shape is not None:
        result = result.reshape(*batch_shape, *spatial_shape)
    else:
        result = result.squeeze(0)

    return result