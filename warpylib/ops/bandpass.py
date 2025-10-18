"""
Bandpass filtering for real-space tensors.
"""

import torch
from typing import Optional
from .bandpass_rft import bandpass_rft


def bandpass(
    tensor: torch.Tensor,
    dimensionality: int,
    low_freq: Optional[float] = None,
    high_freq: Optional[float] = None,
    soft_edge_low: float = 0.0,
    soft_edge_high: float = 0.0,
) -> torch.Tensor:
    """
    Apply bandpass filter to real-space tensor.

    Transforms input to Fourier space, applies bandpass filter, and transforms back.

    Parameters
    ----------
    tensor : torch.Tensor
        Real-valued input tensor
    dimensionality : int
        Number of spatial dimensions (2 or 3). If less than tensor dimensionality,
        remaining leading dimensions are treated as batch dimensions.
    low_freq : float, optional
        High-pass frequency as fraction of Nyquist [0, 1]. If None, no high-pass filtering.
    high_freq : float, optional
        Low-pass frequency as fraction of Nyquist [0, 1]. If None, no low-pass filtering.
    soft_edge_low : float, optional
        Width of raised cosine transition for high-pass, as fraction of Nyquist.
        Transition occurs from (low_freq - soft_edge_low) to low_freq.
    soft_edge_high : float, optional
        Width of raised cosine transition for low-pass, as fraction of Nyquist.
        Transition occurs from high_freq to (high_freq + soft_edge_high).

    Returns
    -------
    torch.Tensor
        Filtered real-valued tensor with same shape as input

    Raises
    ------
    ValueError
        If dimensionality is invalid or exceeds tensor dimensions

    Examples
    --------
    >>> # Low-pass filter at 0.5 Nyquist
    >>> filtered = bandpass(image, dimensionality=2, high_freq=0.5)
    >>>
    >>> # High-pass filter at 0.1 Nyquist
    >>> filtered = bandpass(image, dimensionality=2, low_freq=0.1)
    >>>
    >>> # Band-pass filter with soft edges
    >>> filtered = bandpass(
    ...     volume,
    ...     dimensionality=3,
    ...     low_freq=0.1,
    ...     high_freq=0.5,
    ...     soft_edge_low=0.05,
    ...     soft_edge_high=0.05
    ... )
    """
    if dimensionality > tensor.ndim:
        raise ValueError(
            f"Dimensionality ({dimensionality}) cannot exceed tensor dimensions ({tensor.ndim})"
        )

    if dimensionality not in (2, 3):
        raise ValueError(f"Dimensionality must be 2 or 3, got {dimensionality}")

    # Determine batch and spatial dimensions
    n_batch_dims = tensor.ndim - dimensionality
    spatial_shape = tensor.shape[n_batch_dims:]

    # Transform to Fourier space
    # rfftn operates on the last `dim` dimensions
    tensor_fft = torch.fft.rfftn(tensor, dim=tuple(range(-dimensionality, 0)))

    # Apply bandpass filter in Fourier space
    filtered_fft = bandpass_rft(
        tensor_fft,
        dimensionality=dimensionality,
        low_freq=low_freq,
        high_freq=high_freq,
        soft_edge_low=soft_edge_low,
        soft_edge_high=soft_edge_high,
    )

    # Transform back to real space
    # Need to specify original spatial shape for proper irfftn
    result = torch.fft.irfftn(
        filtered_fft, s=spatial_shape, dim=tuple(range(-dimensionality, 0))
    )

    return result