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


def mask_rectangular(
    tensor: torch.Tensor,
    region: tuple,
    soft_edge: float = 0.0,
) -> torch.Tensor:
    """
    Apply a rectangular mask to a tensor with optional soft edges.

    The mask is centered on the tensor. The region parameter specifies the inner
    region where the mask value is 1.0. Outside this region, the mask falls off
    to 0.0 using a raised cosine if soft_edge > 0, or a hard edge if soft_edge = 0.

    Dimensionality is inferred from the input tensor (2D or 3D).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape:
        - 2D: (H, W)
        - 3D: (D, H, W)
        No batch dimensions are supported.
    region : tuple
        Inner region size (Z, Y, X) for 3D or (Y, X) for 2D.
        Within this region, the mask value is 1.0.
    soft_edge : float, optional
        Width of the soft edge falloff in pixels/voxels. The mask transitions
        from 1.0 to 0.0 using a raised cosine over this distance.
        If 0, creates a hard edge. Default: 0.0

    Returns
    -------
    torch.Tensor
        Masked tensor (tensor * mask) with same shape and dtype as input.

    Raises
    ------
    ValueError
        If tensor is not 2D or 3D, or if region dimensions don't match.

    Examples
    --------
    >>> # 2D rectangular mask with hard edge
    >>> x = torch.randn(128, 128)
    >>> masked = mask_rectangular(x, region=(96, 96), soft_edge=0.0)
    >>> masked.shape
    torch.Size([128, 128])

    >>> # 3D rectangular mask with soft edge
    >>> x = torch.randn(64, 64, 64)
    >>> masked = mask_rectangular(x, region=(48, 48, 48), soft_edge=8.0)
    >>> masked.shape
    torch.Size([64, 64, 64])

    >>> # 2D mask used for apodizing image edges
    >>> x = torch.randn(256, 256)
    >>> masked = mask_rectangular(x, region=(224, 224), soft_edge=16.0)
    """
    dimensionality = tensor.ndim

    if dimensionality not in (2, 3):
        raise ValueError(
            f"Tensor must be 2D or 3D, got {dimensionality}D tensor"
        )

    if len(region) != dimensionality:
        raise ValueError(
            f"Region dimensions ({len(region)}) must match tensor dimensionality ({dimensionality})"
        )

    device = tensor.device
    dtype = tensor.dtype
    dims = tensor.shape

    # Convert region to tensor for easier math
    region_tensor = torch.tensor(region, device=device, dtype=torch.float32)
    dims_tensor = torch.tensor(dims, device=device, dtype=torch.float32)

    if soft_edge <= 0:
        # Hard edge mask
        # Calculate margins: (dims - region) / 2
        margin = ((dims_tensor - region_tensor) / 2).to(torch.int64)
        region_int = region_tensor.to(torch.int64)

        # Create mask initialized to zeros
        mask = torch.zeros_like(tensor, dtype=torch.float32)

        # Set inner region to 1
        if dimensionality == 2:
            mask[
                margin[0]:margin[0] + region_int[0],
                margin[1]:margin[1] + region_int[1]
            ] = 1.0
        else:  # dimensionality == 3
            mask[
                margin[0]:margin[0] + region_int[0],
                margin[1]:margin[1] + region_int[1],
                margin[2]:margin[2] + region_int[2]
            ] = 1.0

    else:
        # Soft edge mask with raised cosine falloff
        # Calculate margins: (dims - region) / 2
        margin = (dims_tensor - region_tensor) / 2

        if dimensionality == 2:
            h, w = dims
            y = torch.arange(h, device=device, dtype=torch.float32)
            x = torch.arange(w, device=device, dtype=torch.float32)

            # Distance from edge for each dimension
            # max(margin - pos, pos - (margin + region - 1))
            # This is 0 inside the region, positive outside
            yy_dist = torch.maximum(margin[0] - y, y - (margin[0] + region_tensor[0] - 1))
            xx_dist = torch.maximum(margin[1] - x, x - (margin[1] + region_tensor[1] - 1))

            # Normalize by soft_edge and clamp to [0, 1]
            yy_norm = torch.clamp(yy_dist / soft_edge, 0.0, 1.0)
            xx_norm = torch.clamp(xx_dist / soft_edge, 0.0, 1.0)

            # Create 2D grids
            yy_grid, xx_grid = torch.meshgrid(yy_norm, xx_norm, indexing='ij')

            # Combined distance: sqrt(xx^2 + yy^2), clamped to [0, 1]
            r = torch.clamp(torch.sqrt(xx_grid**2 + yy_grid**2), 0.0, 1.0)

        else:  # dimensionality == 3
            d, h, w = dims
            z = torch.arange(d, device=device, dtype=torch.float32)
            y = torch.arange(h, device=device, dtype=torch.float32)
            x = torch.arange(w, device=device, dtype=torch.float32)

            # Distance from edge for each dimension
            zz_dist = torch.maximum(margin[0] - z, z - (margin[0] + region_tensor[0] - 1))
            yy_dist = torch.maximum(margin[1] - y, y - (margin[1] + region_tensor[1] - 1))
            xx_dist = torch.maximum(margin[2] - x, x - (margin[2] + region_tensor[2] - 1))

            # Normalize by soft_edge and clamp to [0, 1]
            zz_norm = torch.clamp(zz_dist / soft_edge, 0.0, 1.0)
            yy_norm = torch.clamp(yy_dist / soft_edge, 0.0, 1.0)
            xx_norm = torch.clamp(xx_dist / soft_edge, 0.0, 1.0)

            # Create 3D grids
            zz_grid, yy_grid, xx_grid = torch.meshgrid(zz_norm, yy_norm, xx_norm, indexing='ij')

            # Combined distance: sqrt(xx^2 + yy^2 + zz^2), clamped to [0, 1]
            r = torch.clamp(torch.sqrt(xx_grid**2 + yy_grid**2 + zz_grid**2), 0.0, 1.0)

        # Raised cosine: 0.5 * (1 + cos(r * pi))
        mask = 0.5 * (1.0 + torch.cos(r * math.pi))

    # Apply mask and convert back to original dtype
    return tensor * mask.to(dtype)
