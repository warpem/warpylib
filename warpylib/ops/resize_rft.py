import torch
from typing import Tuple, Union


def resize_rft(
    tensor: torch.Tensor,
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    padding_value: Union[float, complex] = 0.0
) -> torch.Tensor:
    """
    Resize a batched RFFT-formatted 2D or 3D tensor by cropping or padding.
    Operates in Fourier space with DC at index 0 (standard RFFT convention).

    The size parameter specifies dimensions in real space, which are automatically
    converted to Fourier space (last dimension becomes size[-1]//2+1).

    For non-RFFT dimensions, preserves low frequencies (DC and nearby) by cropping/
    padding with DC at index 0. For the RFFT dimension, simply crops/pads from right.

    Parameters
    ----------
    tensor : torch.Tensor
        Input RFFT-formatted tensor with shape (..., H, W//2+1) for 2D or
        (..., D, H, W//2+1) for 3D, where ... represents any number of batch
        dimensions. Can be complex or real-valued.
    size : tuple of int
        Target spatial dimensions in real space. Use (H, W) for 2D or (D, H, W)
        for 3D. Follows torch convention: (height, width) or (depth, height, width).
        The last dimension will be converted to W//2+1 for RFFT format.
    padding_value : float or complex, default=0.0
        Fill value for constant padding. Can be real or complex.

    Returns
    -------
    torch.Tensor
        Resized tensor in RFFT format with shape (..., *size_ft) where size_ft
        is size with the last dimension converted to size[-1]//2+1.

    Raises
    ------
    ValueError
        If non-RFFT dimensions (all but last) are odd in either current or target size,
        or if the number of spatial dimensions doesn't match the size tuple length.

    Notes
    -----
    - Uses standard FFT convention with DC at index 0 (not fftshifted)
    - Only constant padding mode is supported (zero-padding in Fourier space)
    - Preserves low frequencies when cropping, discards high frequencies
    - For padding, adds zeros at high frequencies
    - Works with both complex (RFFT output) and real-valued (e.g., CTF) tensors

    Examples
    --------
    >>> # 2D example: resize RFFT of 128x128 image to 96x96
    >>> x = torch.fft.rfft2(torch.randn(2, 128, 128))  # shape (2, 128, 65)
    >>> y = resizeft(x, size=(96, 96))                  # shape (2, 96, 49)
    >>> y_real = torch.fft.irfft2(y, s=(96, 96))       # shape (2, 96, 96)

    >>> # 3D example with real-valued CTF
    >>> ctf_ft = torch.randn(4, 64, 64, 33)  # Real-valued CTF in Fourier space
    >>> ctf_resized = resizeft(ctf_ft, size=(32, 32, 32))  # shape (4, 32, 32, 17)
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

    # Convert size to Fourier space (last dimension is RFFT dimension)
    size_ft = list(size)
    size_ft[-1] = size[-1] // 2 + 1

    # Get current spatial dimensions (rightmost ndim_spatial dimensions)
    current_size = tensor.shape[-ndim_spatial:]

    # Validate non-RFFT dimensions (all but last) are even
    for i in range(ndim_spatial - 1):
        dim_size = current_size[i]
        if dim_size % 2 != 0:
            raise ValueError(
                f"Current spatial dimension {i} has odd size {dim_size}. "
                f"All non-RFFT dimensions must be even."
            )

    for i in range(ndim_spatial - 1):
        dim_size = size[i]
        if dim_size % 2 != 0:
            raise ValueError(
                f"Target spatial dimension {i} has odd size {dim_size}. "
                f"All non-RFFT dimensions must be even."
            )

    # Early return if dimensions already match
    if current_size == tuple(size_ft):
        return tensor

    result = tensor

    # Process each spatial dimension
    for i in range(ndim_spatial):
        current = current_size[i]
        target = size_ft[i]

        if current == target:
            continue

        # Dimension index (negative indexing from right)
        dim = -(ndim_spatial - i)

        if i == ndim_spatial - 1:
            # RFFT dimension (last): simple crop/pad from right
            if current > target:
                # Crop: keep DC and low frequencies
                result = torch.narrow(result, dim, 0, target)
            else:
                # Pad: add zeros at high frequencies
                pad_size = list(result.shape)
                pad_size[dim] = target - current
                zeros = torch.zeros(pad_size, dtype=result.dtype, device=result.device)
                if padding_value != 0:
                    zeros = zeros + padding_value
                result = torch.cat([result, zeros], dim=dim)
        else:
            # Non-RFFT dimensions: unshifted FFT crop/pad (DC at index 0)
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
