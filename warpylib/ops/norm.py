"""
Normalization for real-space tensors.
"""

import torch
from typing import Literal


def norm(
    tensor: torch.Tensor,
    dimensionality: int,
    diameter: float = 0,
    mode: Literal["inner", "outer"] = "inner",
) -> torch.Tensor:
    """
    Normalize a real-valued tensor.

    Calculates statistics (mean and std) on a circular/spherical region and normalizes
    the entire tensor. Supports batch dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Real-valued input tensor
    dimensionality : int
        Number of spatial dimensions (2 or 3). If less than tensor dimensionality,
        remaining leading dimensions are treated as batch dimensions.
    diameter : float, optional
        Diameter of circular/spherical region in pixels. If 0, use all components.
    mode : {'inner', 'outer'}, optional
        Whether to calculate statistics inside or outside the circle/sphere.
        Default is 'inner'.

    Returns
    -------
    torch.Tensor
        Normalized tensor (mean-subtracted and std-divided)
    """
    from .masking import mask_sphere

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

    # Calculate mean and std
    if diameter > 0:
        # Create hard-edged boolean mask using existing mask_sphere
        # Use a dummy tensor to get the mask shape
        dummy = torch.ones(spatial_shape, device=tensor.device, dtype=tensor.dtype)
        mask_float = mask_sphere(dummy, diameter=diameter, soft_edge_width=0.0)

        if mode == "outer":
            mask = mask_float <= 0.5  # Convert to boolean
        else:  # mode == "inner"
            mask = mask_float > 0.5  # Convert to boolean

        # Expand mask for batch dimension
        mask = mask.unsqueeze(0).expand(batch_size, *spatial_shape)

        # Calculate statistics on masked region (vectorized)
        spatial_dims = tuple(range(1, tensor_reshaped.ndim))
        masked_sum = (tensor_reshaped * mask).sum(dim=spatial_dims)
        mask_count = mask.sum(dim=spatial_dims).float()
        mean_val = masked_sum / mask_count

        # Expand mean for broadcasting
        mean_val_expanded = mean_val.reshape(batch_size, *([1] * dimensionality))

        # Calculate std
        masked_squared_diff = ((tensor_reshaped - mean_val_expanded) ** 2) * mask
        variance = masked_squared_diff.sum(dim=spatial_dims) / mask_count
        std_val = torch.sqrt(variance)

        # Expand std for broadcasting
        std_val_expanded = std_val.reshape(batch_size, *([1] * dimensionality))

        # Normalize entire tensor
        result = tensor_reshaped - mean_val_expanded
        result = torch.where(std_val_expanded > 0, result / std_val_expanded, result)

    else:
        # No mask - use all components
        spatial_dims = tuple(range(1, tensor_reshaped.ndim))
        mean_val = tensor_reshaped.mean(dim=spatial_dims, keepdim=True)
        std_val = tensor_reshaped.std(dim=spatial_dims, keepdim=True, unbiased=False)

        # Normalize
        result = tensor_reshaped - mean_val
        result = torch.where(std_val > 0, result / std_val, result)

    # Reshape back to original shape
    if batch_shape is not None:
        result = result.reshape(*batch_shape, *spatial_shape)
    else:
        result = result.squeeze(0)

    return result