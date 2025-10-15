import torch
import torch.nn.functional as F
from typing import Tuple, Union, Literal


def resize(
    tensor: torch.Tensor,
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    padding_mode: Literal['constant', 'replicate', 'reflect'] = 'constant',
    padding_value: float = 0.0
) -> torch.Tensor:
    """
    Resize a batched 2D or 3D tensor by cropping or padding to target dimensions.
    Original data remains centered after the operation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (..., H, W) for 2D or (..., D, H, W) for 3D,
        where ... represents any number of batch dimensions.
    size : tuple of int
        Target spatial dimensions. Use (H, W) for 2D or (D, H, W) for 3D.
        Follows torch convention: (height, width) or (depth, height, width).
    padding_mode : {'constant', 'replicate', 'reflect'}, default='constant'
        Padding mode when expanding dimensions:
        - 'constant': fill with padding_value
        - 'replicate': replicate edge values
        - 'reflect': mirror reflection without repeating edge
    padding_value : float, default=0.0
        Fill value when padding_mode='constant'.

    Returns
    -------
    torch.Tensor
        Resized tensor with shape (..., *size) where ... are the same batch
        dimensions as the input.

    Raises
    ------
    ValueError
        If current or target spatial dimensions are odd, or if the number of
        spatial dimensions doesn't match the size tuple length.

    Examples
    --------
    >>> # 2D example: crop and pad
    >>> x = torch.randn(2, 3, 128, 128)  # batch of 2, 3 channels, 128x128
    >>> y = resize(x, size=(96, 160))     # crop height, pad width
    >>> y.shape
    torch.Size([2, 3, 96, 160])

    >>> # 3D example
    >>> x = torch.randn(4, 64, 64, 64)   # batch of 4, 64x64x64
    >>> y = resize(x, size=(32, 64, 96)) # crop depth, keep height, pad width
    >>> y.shape
    torch.Size([4, 32, 64, 96])
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

    # Validate all dimensions are even
    for i, dim_size in enumerate(current_size):
        if dim_size % 2 != 0:
            raise ValueError(
                f"Current spatial dimension {i} has odd size {dim_size}. "
                f"All spatial dimensions must be even."
            )

    for i, dim_size in enumerate(size):
        if dim_size % 2 != 0:
            raise ValueError(
                f"Target spatial dimension {i} has odd size {dim_size}. "
                f"All spatial dimensions must be even."
            )

    # Early return if dimensions already match
    if current_size == tuple(size):
        return tensor

    result = tensor

    # Process each spatial dimension
    # We'll handle cropping first, then padding

    # Step 1: Crop dimensions that need to shrink
    crop_slices = [slice(None)] * (result.ndim - ndim_spatial)  # Keep all batch dims
    needs_crop = False

    for i, (current, target) in enumerate(zip(current_size, size)):
        dim_idx = -(ndim_spatial - i)  # Index from the right

        if current > target:
            # Need to crop this dimension
            needs_crop = True
            crop_amount = current - target
            start = crop_amount // 2
            end = start + target
            crop_slices.append(slice(start, end))
        else:
            # Keep full dimension (will pad later if needed)
            crop_slices.append(slice(None))

    if needs_crop:
        result = result[tuple(crop_slices)]

    # Step 2: Pad dimensions that need to grow
    # torch.nn.functional.pad expects padding in reverse order: (left, right, top, bottom, front, back)
    # For ndim_spatial dimensions, we need to specify padding for each from last to first
    pad_amounts = []
    needs_pad = False

    for i in reversed(range(ndim_spatial)):
        current = result.shape[-(ndim_spatial - i)]
        target = size[i]

        if current < target:
            needs_pad = True
            pad_total = target - current
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_amounts.extend([pad_before, pad_after])
        else:
            # No padding needed for this dimension
            pad_amounts.extend([0, 0])

    if needs_pad:
        if padding_mode == 'constant':
            result = F.pad(result, pad_amounts, mode='constant', value=padding_value)
        else:
            result = F.pad(result, pad_amounts, mode=padding_mode)

    return result
