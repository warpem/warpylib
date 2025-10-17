import torch
import math


def mask_sphere(
    tensor: torch.Tensor,
    diameter: float,
    soft_edge_width: float,
) -> torch.Tensor:
    """
    Apply a circular (2D) or spherical (3D) mask to a tensor with soft edges.

    The mask is centered on the tensor and uses a raised cosine falloff at the edges
    for smooth transitions. The diameter parameter specifies the inner diameter
    where the mask value is 1.0, with the soft edge extending outward from there.

    Dimensionality is inferred from the input tensor (2D or 3D).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape:
        - 2D: (H, W)
        - 3D: (D, H, W)
        No batch dimensions are supported.
    diameter : float
        Inner diameter of the mask in pixels/voxels. Within radius = diameter/2,
        the mask value is 1.0.
    soft_edge_width : float
        Width of the soft edge falloff in pixels/voxels. The mask transitions
        from 1.0 to 0.0 using a raised cosine over this distance.

    Returns
    -------
    torch.Tensor
        Masked tensor (tensor * mask) with same shape and dtype as input.

    Raises
    ------
    ValueError
        If tensor is not 2D or 3D.

    Examples
    --------
    >>> # 2D circular mask
    >>> x = torch.randn(128, 128)
    >>> masked = mask_sphere(x, diameter=100.0, soft_edge_width=10.0)
    >>> masked.shape
    torch.Size([128, 128])

    >>> # 3D spherical mask with sharp edge
    >>> x = torch.randn(64, 64, 64)
    >>> masked = mask_sphere(x, diameter=50.0, soft_edge_width=0.0)
    >>> masked.shape
    torch.Size([64, 64, 64])

    >>> # Soft mask with wide falloff
    >>> x = torch.randn(256, 256)
    >>> masked = mask_sphere(x, diameter=200.0, soft_edge_width=25.0)
    """
    # Infer dimensionality from tensor
    dimensionality = tensor.ndim

    if dimensionality not in (2, 3):
        raise ValueError(
            f"Tensor must be 2D or 3D, got {dimensionality}D tensor"
        )

    # Get spatial dimensions
    spatial_shape = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    # Create coordinate grids centered at the middle of the tensor
    if dimensionality == 2:
        h, w = spatial_shape
        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0

        y = torch.arange(h, device=device, dtype=torch.float32) - center_y
        x = torch.arange(w, device=device, dtype=torch.float32) - center_x

        yy, xx = torch.meshgrid(y, x, indexing='ij')
        distance = torch.sqrt(xx**2 + yy**2)

    else:  # dimensionality == 3
        d, h, w = spatial_shape
        center_z = (d - 1) / 2.0
        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0

        z = torch.arange(d, device=device, dtype=torch.float32) - center_z
        y = torch.arange(h, device=device, dtype=torch.float32) - center_y
        x = torch.arange(w, device=device, dtype=torch.float32) - center_x

        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        distance = torch.sqrt(xx**2 + yy**2 + zz**2)

    # Calculate mask with raised cosine falloff
    radius = diameter / 2.0
    mask = torch.ones_like(distance)

    if soft_edge_width > 0:
        # Transition region: radius <= r < radius + soft_edge_width
        in_transition = (distance >= radius) & (distance < radius + soft_edge_width)

        # Raised cosine: 0.5 * (1 + cos(pi * (r - radius) / soft_edge_width))
        transition_dist = distance[in_transition] - radius
        mask[in_transition] = 0.5 * (
            1.0 + torch.cos(math.pi * transition_dist / soft_edge_width)
        )

        # Outside region: r >= radius + soft_edge_width
        mask[distance >= radius + soft_edge_width] = 0.0
    else:
        # Hard edge: just zero out everything beyond radius
        mask[distance >= radius] = 0.0

    # Apply mask and convert back to original dtype
    return (tensor * mask.to(dtype))


def mask_sphere_ft(
    tensor: torch.Tensor,
    diameter: float,
    soft_edge_width: float,
) -> torch.Tensor:
    """
    Apply a circular (2D) or spherical (3D) mask in FFT format with soft edges.

    This function creates a centered mask using mask_sphere, then applies
    ifftshift to convert it to decentered FFT format (DC at index 0).

    Dimensionality is inferred from the input tensor (2D or 3D).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in FFT format with shape:
        - 2D: (H, W)
        - 3D: (D, H, W)
        No batch dimensions are supported.
    diameter : float
        Inner diameter of the mask in pixels/voxels. Within radius = diameter/2,
        the mask value is 1.0.
    soft_edge_width : float
        Width of the soft edge falloff in pixels/voxels. The mask transitions
        from 1.0 to 0.0 using a raised cosine over this distance.

    Returns
    -------
    torch.Tensor
        Masked tensor in FFT format (tensor * mask) with same shape and dtype as input.
        The mask is in decentered format (DC at index 0) suitable for FFT data.

    Raises
    ------
    ValueError
        If tensor is not 2D or 3D.

    Examples
    --------
    >>> # 2D circular mask for FFT data
    >>> x_fft = torch.fft.fft2(torch.randn(128, 128))
    >>> masked_fft = mask_sphere_ft(x_fft, diameter=100.0, soft_edge_width=10.0)
    >>> masked_fft.shape
    torch.Size([128, 128])

    >>> # 3D spherical mask for FFT data
    >>> x_fft = torch.fft.fftn(torch.randn(64, 64, 64))
    >>> masked_fft = mask_sphere_ft(x_fft, diameter=50.0, soft_edge_width=5.0)
    >>> masked_fft.shape
    torch.Size([64, 64, 64])
    """
    # Apply centered mask
    masked = mask_sphere(tensor, diameter, soft_edge_width)

    # Convert to decentered FFT format
    dimensionality = tensor.ndim
    if dimensionality == 2:
        return torch.fft.ifftshift(masked, dim=(-2, -1))
    else:  # dimensionality == 3
        return torch.fft.ifftshift(masked, dim=(-3, -2, -1))


def mask_sphere_rft(
    tensor: torch.Tensor,
    diameter: float,
    soft_edge_width: float,
) -> torch.Tensor:
    """
    Apply a circular (2D) or spherical (3D) mask in RFFT format with soft edges.

    This function creates a mask suitable for RFFT (half FFT) data by calling
    mask_sphere_ft and then taking only the positive x half. To avoid double-counting
    the x=0 line/plane (which has Hermitian symmetry), half of the x=0 components
    are zeroed out.

    Dimensionality is inferred from the input tensor (2D or 3D).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor in RFFT format with shape:
        - 2D: (H, W//2+1)
        - 3D: (D, H, W//2+1)
        No batch dimensions are supported.
    diameter : float
        Inner diameter of the mask in pixels/voxels. Within radius = diameter/2,
        the mask value is 1.0.
    soft_edge_width : float
        Width of the soft edge falloff in pixels/voxels. The mask transitions
        from 1.0 to 0.0 using a raised cosine over this distance.

    Returns
    -------
    torch.Tensor
        Masked tensor in RFFT format (tensor * mask) with same shape and dtype as input.

    Raises
    ------
    ValueError
        If tensor is not 2D or 3D.

    Examples
    --------
    >>> # 2D circular mask for RFFT data
    >>> x_rfft = torch.fft.rfft2(torch.randn(128, 128))
    >>> masked_rfft = mask_sphere_rft(x_rfft, diameter=100.0, soft_edge_width=10.0)
    >>> masked_rfft.shape
    torch.Size([128, 65])

    >>> # 3D spherical mask for RFFT data
    >>> x_rfft = torch.fft.rfftn(torch.randn(64, 64, 64))
    >>> masked_rfft = mask_sphere_rft(x_rfft, diameter=50.0, soft_edge_width=5.0)
    >>> masked_rfft.shape
    torch.Size([64, 64, 33])
    """
    dimensionality = tensor.ndim

    if dimensionality not in (2, 3):
        raise ValueError(
            f"Tensor must be 2D or 3D, got {dimensionality}D tensor"
        )

    # Reconstruct full FFT shape from RFFT shape
    if dimensionality == 2:
        h, w_half = tensor.shape
        w_full = (w_half - 1) * 2
        full_shape = (h, w_full)
    else:  # dimensionality == 3
        d, h, w_half = tensor.shape
        w_full = (w_half - 1) * 2
        full_shape = (d, h, w_full)

    # Create a dummy tensor with full FFT shape to get the full mask
    dummy_full = torch.ones(full_shape, device=tensor.device, dtype=tensor.dtype)
    full_mask = mask_sphere_ft(dummy_full, diameter, soft_edge_width)

    # Extract RFFT half (positive x half)
    if dimensionality == 2:
        mask_rft = full_mask[:, :w_half]
    else:  # dimensionality == 3
        mask_rft = full_mask[:, :, :w_half]

    # Apply mask and return
    return tensor * mask_rft.to(tensor.dtype)
