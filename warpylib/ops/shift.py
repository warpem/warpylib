import torch
from .shift_rft import shift_rft


def shift(
    tensor: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """
    Shift a tensor by applying phase shifts in Fourier space.

    This function performs sub-pixel translation using Fourier-space phase shifts.
    The operation is circular (periodic boundaries). For integer shifts, results
    match torch.roll exactly.

    **Note on reversibility**: Fractional shifts are not perfectly reversible due to
    the Nyquist frequency constraint. When shifting by fractional amounts, the
    Nyquist frequency becomes complex, but real-valued signals require it to be real.
    This causes information loss. For perfect reversibility with fractional shifts,
    use `shift_rft` and stay in Fourier space.

    Parameters
    ----------
    tensor : torch.Tensor
        Input real-space tensor with shape:
        - 1D: (..., W)
        - 2D: (..., H, W)
        - 3D: (..., D, H, W)
        where ... represents batch dimensions.
    shifts : torch.Tensor
        Shift amounts with shape (..., ndim) where ndim ∈ {1, 2, 3}.
        Ordering follows X, Y, Z axes:
        - 1D: [shift_x]
        - 2D: [shift_x, shift_y]
        - 3D: [shift_x, shift_y, shift_z]
        Positive values shift data in the positive direction along that axis.
        Supports fractional shifts for sub-pixel translation.
        Batch dimensions must be broadcastable with tensor.

    Returns
    -------
    torch.Tensor
        Shifted tensor with same shape and dtype as input.

    Raises
    ------
    ValueError
        If spatial dimensions are odd, or if dimensionality doesn't match.

    Examples
    --------
    >>> # 1D shift
    >>> x = torch.randn(10, 64)  # 10 batches, 64 samples
    >>> shifts = torch.randn(10, 1)  # per-batch shifts in X
    >>> y = shift(x, shifts)
    >>> y.shape
    torch.Size([10, 64])

    >>> # 2D integer shift (equivalent to torch.roll)
    >>> x = torch.randn(128, 128)
    >>> shifts = torch.tensor([[5.0, -3.0]])  # shift by 5 in X, -3 in Y
    >>> y = shift(x, shifts)
    >>> # Equivalent to: torch.roll(x, shifts=(5, -3), dims=(1, 0))

    >>> # 3D fractional shift
    >>> x = torch.randn(2, 64, 64, 64)  # 2 batches, 64³ volumes
    >>> shifts = torch.tensor([[1.5, 0.0, -2.3], [0.5, 1.0, 0.5]])
    >>> y = shift(x, shifts)
    >>> y.shape
    torch.Size([2, 64, 64, 64])
    """
    # Determine dimensionality from shifts
    ndim = shifts.shape[-1]
    if ndim not in (1, 2, 3):
        raise ValueError(f"shifts must have 1, 2, or 3 components, got {ndim}")

    # Validate tensor has correct number of spatial dimensions
    if tensor.ndim < ndim:
        raise ValueError(
            f"Tensor has {tensor.ndim} dimensions but shifts specify "
            f"{ndim} spatial dimensions"
        )

    # Get spatial dimensions
    spatial_shape = tensor.shape[-ndim:]

    # Validate all dimensions are even
    for i, dim_size in enumerate(spatial_shape):
        if dim_size % 2 != 0:
            raise ValueError(
                f"Spatial dimension {i} has odd size {dim_size}. "
                f"All spatial dimensions must be even."
            )

    # Apply RFFT
    if ndim == 1:
        tensor_rfft = torch.fft.rfft(tensor, dim=-1)
    elif ndim == 2:
        tensor_rfft = torch.fft.rfft2(tensor, dim=(-2, -1))
    else:  # ndim == 3
        tensor_rfft = torch.fft.rfftn(tensor, dim=(-3, -2, -1))

    # Apply shift in Fourier space
    shifted_rfft = shift_rft(tensor_rfft, shifts)

    # Apply inverse RFFT
    if ndim == 1:
        result = torch.fft.irfft(shifted_rfft, n=spatial_shape[0], dim=-1)
    elif ndim == 2:
        result = torch.fft.irfft2(shifted_rfft, s=spatial_shape, dim=(-2, -1))
    else:  # ndim == 3
        result = torch.fft.irfftn(shifted_rfft, s=spatial_shape, dim=(-3, -2, -1))

    return result.to(tensor.dtype)
