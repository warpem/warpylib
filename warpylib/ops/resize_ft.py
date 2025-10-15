import torch
from typing import Tuple, Union


def resize_ft(
    tensor: torch.Tensor,
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    padding_value: Union[float, complex] = 0.0
) -> torch.Tensor:
    """
    Resize a batched FFT-formatted 2D or 3D tensor by cropping or padding.
    Operates in Fourier space with DC at index 0 (standard FFT convention).

    Unlike resize_rft which handles RFFT format (one Friedel-symmetric half),
    this function handles full FFT format with all frequencies represented.

    For all dimensions, preserves low frequencies (DC and nearby) by cropping/
    padding with DC at index 0 (unshifted FFT convention).

    Parameters
    ----------
    tensor : torch.Tensor
        Input FFT-formatted tensor with shape (..., H, W) for 2D or
        (..., D, H, W) for 3D, where ... represents any number of batch
        dimensions. Can be complex or real-valued.
    size : tuple of int
        Target spatial dimensions. Use (H, W) for 2D or (D, H, W) for 3D.
        All dimensions must be even. Follows torch convention: (height, width)
        or (depth, height, width).
    padding_value : float or complex, default=0.0
        Fill value for constant padding. Can be real or complex.

    Returns
    -------
    torch.Tensor
        Resized tensor in FFT format with shape (..., *size).

    Raises
    ------
    ValueError
        If any dimension is odd in either current or target size, or if the
        number of spatial dimensions doesn't match the size tuple length.

    Notes
    -----
    - Uses standard FFT convention with DC at index 0 (not fftshifted)
    - Only constant padding mode is supported (zero-padding in Fourier space)
    - Preserves low frequencies when cropping, discards high frequencies
    - For padding, adds zeros at high frequencies
    - Works with both complex (FFT output) and real-valued (e.g., CTF) tensors
    - All dimensions must be even to ensure symmetric frequency splits

    Examples
    --------
    >>> # 2D example: resize FFT of 128x128 image to 96x96
    >>> x = torch.fft.fft2(torch.randn(2, 128, 128))  # shape (2, 128, 128)
    >>> y = resize_ft(x, size=(96, 96))                # shape (2, 96, 96)
    >>> y_real = torch.fft.ifft2(y).real              # shape (2, 96, 96)

    >>> # 3D example with real-valued CTF
    >>> ctf_ft = torch.randn(4, 64, 64, 64)  # Real-valued CTF in Fourier space
    >>> ctf_resized = resize_ft(ctf_ft, size=(32, 32, 32))  # shape (4, 32, 32, 32)
    """
    # Determine dimensionality from size tuple
    ndim_spatial = len(size)
    if ndim_spatial not in (2, 3):
        raise ValueError(f"size must have 2 or 3 elements, got {ndim_spatial}")

    # Validate tensor has enough dimensions
    if tensor.ndim < ndim_spatial:
        raise ValueError(
            f"Tensor has {tensor.ndim} dimensions but size specifies "
            f"{ndim_spatial} spatial dimensions"
        )

    # Get current spatial dimensions (rightmost ndim_spatial dimensions)
    current_size = tensor.shape[-ndim_spatial:]

    # Validate all dimensions are even for both current and target
    for i in range(ndim_spatial):
        dim_size = current_size[i]
        if dim_size % 2 != 0:
            raise ValueError(
                f"Current spatial dimension {i} has odd size {dim_size}. "
                f"All dimensions must be even."
            )

    for i in range(ndim_spatial):
        dim_size = size[i]
        if dim_size % 2 != 0:
            raise ValueError(
                f"Target spatial dimension {i} has odd size {dim_size}. "
                f"All dimensions must be even."
            )

    # Early return if dimensions already match
    if current_size == tuple(size):
        return tensor

    result = tensor

    # Process each spatial dimension using unshifted FFT crop/pad
    for i in range(ndim_spatial):
        current = current_size[i]
        target = size[i]

        if current == target:
            continue

        # Dimension index (negative indexing from right)
        dim = -(ndim_spatial - i)

        # All dimensions use unshifted FFT crop/pad (DC at index 0)
        if current > target:
            # Crop: keep low frequencies (DC and nearby)
            # Keep indices [0:target//2] and [-target//2:]
            pos_part = torch.narrow(result, dim, 0, target // 2)
            neg_part = torch.narrow(result, dim, current - target // 2, target // 2)
            result = torch.cat([pos_part, neg_part], dim=dim)
        else:
            # Pad: insert zeros in middle (at high frequencies)
            # Split at current//2, insert zeros, concatenate
            pos_part = torch.narrow(result, dim, 0, current // 2)
            neg_part = torch.narrow(result, dim, current // 2, current - current // 2)

            pad_size = list(result.shape)
            pad_size[dim] = target - current
            zeros = torch.zeros(pad_size, dtype=result.dtype, device=result.device)
            if padding_value != 0:
                zeros = zeros + padding_value

            result = torch.cat([pos_part, zeros, neg_part], dim=dim)

    return result
