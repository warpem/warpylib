import torch
import math


def shift_rft(
    tensor_rfft: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """
    Apply phase shifts in Fourier space (RFFT format) to translate data in real space.

    This function multiplies the RFFT coefficients by exp(2πi * k · shift) to achieve
    translation in real space. The operation is circular (periodic boundaries).

    Parameters
    ----------
    tensor_rfft : torch.Tensor
        Input tensor in RFFT format (complex-valued), with shape:
        - 1D: (..., W//2+1)
        - 2D: (..., H, W//2+1)
        - 3D: (..., D, H, W//2+1)
        where ... represents batch dimensions.
    shifts : torch.Tensor
        Shift amounts with shape (..., ndim) where ndim ∈ {1, 2, 3}.
        Ordering follows X, Y, Z axes:
        - 1D: [shift_x]
        - 2D: [shift_x, shift_y]
        - 3D: [shift_x, shift_y, shift_z]
        Positive values shift in the positive direction.
        Supports fractional shifts for sub-pixel translation.
        Batch dimensions must be broadcastable with tensor_rfft.

    Returns
    -------
    torch.Tensor
        Shifted tensor in RFFT format with same shape as input.

    Raises
    ------
    ValueError
        If spatial dimensions are odd, or if dimensionality doesn't match,
        or if tensor_rfft is not complex-valued.

    Examples
    --------
    >>> # 2D shift
    >>> x_rfft = torch.fft.rfft2(x)  # x is (..., H, W)
    >>> shifts = torch.tensor([[2.5, -1.0]])  # shift by 2.5 pixels in X, -1 in Y
    >>> y_rfft = shift_rft(x_rfft, shifts)
    >>> y = torch.fft.irfft2(y_rfft, s=x.shape[-2:])
    """
    if not tensor_rfft.is_complex():
        raise ValueError("tensor_rfft must be complex-valued")

    # Determine dimensionality from shifts
    ndim = shifts.shape[-1]
    if ndim not in (1, 2, 3):
        raise ValueError(f"shifts must have 1, 2, or 3 components, got {ndim}")

    # Validate tensor has correct number of spatial dimensions
    if tensor_rfft.ndim < ndim:
        raise ValueError(
            f"Tensor has {tensor_rfft.ndim} dimensions but shifts specify "
            f"{ndim} spatial dimensions"
        )

    # Get spatial dimensions
    # For RFFT: last dim is W//2+1, other dims are full size
    if ndim == 1:
        W_rfft = tensor_rfft.shape[-1]
        W = (W_rfft - 1) * 2
        spatial_shape = (W,)
    elif ndim == 2:
        H = tensor_rfft.shape[-2]
        W_rfft = tensor_rfft.shape[-1]
        W = (W_rfft - 1) * 2
        spatial_shape = (H, W)
    else:  # ndim == 3
        D = tensor_rfft.shape[-3]
        H = tensor_rfft.shape[-2]
        W_rfft = tensor_rfft.shape[-1]
        W = (W_rfft - 1) * 2
        spatial_shape = (D, H, W)

    # Validate all dimensions are even
    for i, dim_size in enumerate(spatial_shape):
        if dim_size % 2 != 0:
            raise ValueError(
                f"Spatial dimension {i} has odd size {dim_size}. "
                f"All spatial dimensions must be even."
            )

    # Create frequency grids
    # Frequencies for each dimension following FFT convention
    device = tensor_rfft.device
    dtype_real = tensor_rfft.real.dtype

    # Ensure shifts has the correct dtype and device
    shifts = shifts.to(dtype=dtype_real, device=device)

    if ndim == 1:
        # X dimension: only positive frequencies (RFFT)
        # Normalize by dimension size for phase shift
        freq_x = torch.arange(W_rfft, device=device, dtype=dtype_real) / W
        freqs = freq_x.unsqueeze(-1)  # (W_rfft, 1)

        # Compute phase using einsum: sum over component dimension
        # freqs: (W_rfft, 1), shifts: (..., 1) -> phase: (..., W_rfft)
        phase = torch.einsum('wc,...c->...w', freqs, shifts)

    elif ndim == 2:
        # Y dimension: full FFT frequencies (fftshift ordering for H)
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype_real)
        # X dimension: only positive frequencies (RFFT)
        # Normalize by dimension size
        freq_x = torch.arange(W_rfft, device=device, dtype=dtype_real) / W

        # Create 2D grids
        freq_y_grid, freq_x_grid = torch.meshgrid(freq_y, freq_x, indexing='ij')
        # Stack: (H, W_rfft, 2) with ordering [X, Y]
        freqs = torch.stack([freq_x_grid, freq_y_grid], dim=-1)

        # Compute phase using einsum
        # freqs: (H, W_rfft, 2), shifts: (..., 2) -> phase: (..., H, W_rfft)
        phase = torch.einsum('hwc,...c->...hw', freqs, shifts)

    else:  # ndim == 3
        # Z, Y dimensions: full FFT frequencies
        freq_z = torch.fft.fftfreq(D, device=device, dtype=dtype_real)
        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype_real)
        # X dimension: only positive frequencies (RFFT)
        # Normalize by dimension size
        freq_x = torch.arange(W_rfft, device=device, dtype=dtype_real) / W

        # Create 3D grids
        freq_z_grid, freq_y_grid, freq_x_grid = torch.meshgrid(
            freq_z, freq_y, freq_x, indexing='ij'
        )
        # Stack: (D, H, W_rfft, 3) with ordering [X, Y, Z]
        freqs = torch.stack([freq_x_grid, freq_y_grid, freq_z_grid], dim=-1)

        # Compute phase using einsum
        # freqs: (D, H, W_rfft, 3), shifts: (..., 3) -> phase: (..., D, H, W_rfft)
        phase = torch.einsum('dhwc,...c->...dhw', freqs, shifts)

    # Apply phase shift: -2π for forward shift
    phase = -2 * math.pi * phase

    # Create phase shift operator
    phase_shift = torch.exp(1j * phase.to(tensor_rfft.dtype))

    # Apply phase shift
    # Need to broadcast phase_shift with tensor_rfft's batch dimensions
    # phase_shift has shape matching spatial dims of tensor_rfft
    # tensor_rfft has shape (...batch, *spatial)
    # shifts_expanded already handles batch broadcasting
    return tensor_rfft * phase_shift
