"""
Plane fitting and subtraction for 2D tensors.
"""

import torch


def fit_plane(tensor: torch.Tensor) -> torch.Tensor:
    """
    Fit a plane to 2D tensor using least squares.

    Fits a plane of the form z = a*x + b*y + c to the input data.

    Parameters
    ----------
    tensor : torch.Tensor
        2D input tensor with shape (H, W)

    Returns
    -------
    torch.Tensor
        Plane parameters as tensor [a, b, c] where:
        - a: slope in x direction
        - b: slope in y direction
        - c: z-intercept

    Examples
    --------
    >>> # Fit plane to image
    >>> image = torch.randn(128, 128)
    >>> params = fit_plane(image)
    >>> params.shape
    torch.Size([3])
    """
    if tensor.ndim != 2:
        raise ValueError(f"fit_plane only works on 2D tensors, got {tensor.ndim}D")

    h, w = tensor.shape
    device = tensor.device

    # Create coordinate grids
    y = torch.arange(h, device=device, dtype=torch.float64)
    x = torch.arange(w, device=device, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Flatten everything
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    z_flat = tensor.reshape(-1).to(torch.float64)

    # Compute sums needed for least squares solution
    # Using double precision to match C# code
    D = (x_flat * x_flat).sum()
    E = (x_flat * y_flat).sum()
    F = x_flat.sum()
    G = (y_flat * y_flat).sum()
    H = y_flat.sum()
    I = torch.tensor(h * w, device=device, dtype=torch.float64)
    J = (x_flat * z_flat).sum()
    K = (y_flat * z_flat).sum()
    L = z_flat.sum()

    # Compute denominator
    denom = F * F * G - 2 * E * F * H + D * H * H + E * E * I - D * G * I

    # Solve for plane parameters
    # X axis slope
    plane_a = (H * H * J - G * I * J + E * I * K + F * G * L - H * (F * K + E * L)) / denom
    # Y axis slope
    plane_b = (E * I * J + F * F * K - D * I * K + D * H * L - F * (H * J + E * L)) / denom
    # Z axis intercept
    plane_c = (F * G * J - E * H * J - E * F * K + D * H * K + E * E * L - D * G * L) / denom

    return torch.tensor([plane_a, plane_b, plane_c], device=device, dtype=tensor.dtype)


def subtract_plane(tensor: torch.Tensor, fit_and_subtract: bool = True) -> torch.Tensor:
    """
    Subtract a fitted plane from a 2D tensor.

    This removes background gradients by fitting and subtracting a plane
    of the form z = a*x + b*y + c.

    Parameters
    ----------
    tensor : torch.Tensor
        2D input tensor with shape (H, W)
    fit_and_subtract : bool, optional
        If True (default), fits and subtracts the plane.
        If False, just returns the input unchanged.

    Returns
    -------
    torch.Tensor
        Tensor with plane subtracted, same shape and dtype as input.

    Examples
    --------
    >>> # Remove background gradient from image
    >>> image = torch.randn(128, 128)
    >>> corrected = subtract_plane(image)
    >>> corrected.shape
    torch.Size([128, 128])
    """
    if not fit_and_subtract:
        return tensor

    if tensor.ndim != 2:
        raise ValueError(f"subtract_plane only works on 2D tensors, got {tensor.ndim}D")

    # Fit plane
    params = fit_plane(tensor)
    a, b, c = params[0], params[1], params[2]

    # Create coordinate grids
    h, w = tensor.shape
    device = tensor.device
    y = torch.arange(h, device=device, dtype=tensor.dtype)
    x = torch.arange(w, device=device, dtype=tensor.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Compute plane: a*x + b*y + c
    plane = a * xx + b * yy + c

    # Subtract plane
    return tensor - plane