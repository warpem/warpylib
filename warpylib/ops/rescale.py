import torch
from typing import Tuple, Union
from .resize_rft import resize_rft


def rescale(
    tensor: torch.Tensor,
    size: Union[Tuple[int, int], Tuple[int, int, int]]
) -> torch.Tensor:
    """
    Rescale a batched 2D or 3D tensor by resizing in Fourier space.
    Performs RFFT, resizes in Fourier space, then inverse RFFT back to real space.

    This is bandwidth-limited rescaling: low frequencies are preserved when downscaling,
    and high frequencies are zero-padded when upscaling.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (..., H, W) for 2D or (..., D, H, W) for 3D,
        where ... represents any number of batch dimensions.
        Must be real-valued (not complex).
    size : tuple of int
        Target spatial dimensions. Use (H, W) for 2D or (D, H, W) for 3D.
        Follows torch convention: (height, width) or (depth, height, width).

    Returns
    -------
    torch.Tensor
        Rescaled tensor with shape (..., *size) where ... are the same batch
        dimensions as the input. Output is real-valued.

    Raises
    ------
    ValueError
        If tensor is complex-valued, or if current or target spatial dimensions
        are odd, or if the number of spatial dimensions doesn't match the size
        tuple length.

    Examples
    --------
    >>> # 2D example: downscale from 128x128 to 64x64
    >>> x = torch.randn(2, 3, 128, 128)
    >>> y = rescale(x, size=(64, 64))
    >>> y.shape
    torch.Size([2, 3, 64, 64])

    >>> # 3D example: upscale from 32x32x32 to 64x64x64
    >>> x = torch.randn(4, 32, 32, 32)
    >>> y = rescale(x, size=(64, 64, 64))
    >>> y.shape
    torch.Size([4, 64, 64, 64])
    """
    # Validate tensor is real-valued
    if tensor.is_complex():
        raise ValueError(
            "Input tensor must be real-valued, not complex. "
            "rescale operates on real-space tensors."
        )

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

    # Get current spatial dimensions
    current_size = tensor.shape[-ndim_spatial:]

    # Early return if dimensions already match
    if current_size == tuple(size):
        return tensor

    # Step 1: Transform to Fourier space
    if ndim_spatial == 2:
        tensor_ft = torch.fft.rfft2(tensor, dim=(-2, -1), norm='forward')
    else:  # ndim_spatial == 3
        tensor_ft = torch.fft.rfftn(tensor, dim=(-3, -2, -1), norm='forward')

    # Step 2: Resize in Fourier space
    tensor_ft_resized = resize_rft(tensor_ft, size=size, padding_value=0.0)

    # Step 3: Transform back to real space
    if ndim_spatial == 2:
        result = torch.fft.irfft2(tensor_ft_resized, s=size, dim=(-2, -1), norm='forward')
    else:  # ndim_spatial == 3
        result = torch.fft.irfftn(tensor_ft_resized, s=size, dim=(-3, -2, -1), norm='forward')

    return result
