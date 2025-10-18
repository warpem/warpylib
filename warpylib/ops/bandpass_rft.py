"""
Bandpass filtering for rfft-formatted tensors.
"""

import torch
from typing import Optional


def bandpass_rft(
    tensor: torch.Tensor,
    dimensionality: int,
    low_freq: Optional[float] = None,
    high_freq: Optional[float] = None,
    soft_edge_low: float = 0.0,
    soft_edge_high: float = 0.0,
) -> torch.Tensor:
    """
    Apply bandpass filter to rfft-formatted tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Complex-valued rfft-formatted input tensor (output of torch.fft.rfftn)
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
        Filtered complex tensor with same shape as input

    Raises
    ------
    ValueError
        If dimensionality is invalid or exceeds tensor dimensions
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

    # Create normalized frequency grids for each dimension
    # Last dimension is rfft dimension, others are full FFT dimensions
    freq_grids = []

    for i, size in enumerate(spatial_shape):
        if i == dimensionality - 1:
            # Last dimension: rfft dimension (0 to Nyquist)
            # Infer original size: rfft of size N gives N//2 + 1 bins
            # Assuming even N: original_size = 2 * (rfft_size - 1)
            original_size = 2 * (size - 1)
            freq = torch.fft.rfftfreq(original_size, device=tensor.device)
        else:
            # Other dimensions: full FFT (includes negative frequencies)
            freq = torch.fft.fftfreq(size, device=tensor.device)

        # Convert from cycles/sample to fraction of Nyquist
        # fftfreq gives [-0.5, 0.5), rfftfreq gives [0, 0.5]
        # Multiply by 2 to get fraction of Nyquist [0, 1]
        freq = torch.abs(freq) * 2.0
        freq_grids.append(freq)

    # Create meshgrid and calculate radial frequency
    # Need to reshape for broadcasting
    if dimensionality == 2:
        fy, fx = freq_grids
        fy = fy.reshape(-1, 1)
        fx = fx.reshape(1, -1)
        # Normalize by dimension size to handle non-square images
        freq_rad = torch.sqrt(fy**2 + fx**2)
    else:  # dimensionality == 3
        fz, fy, fx = freq_grids
        fz = fz.reshape(-1, 1, 1)
        fy = fy.reshape(1, -1, 1)
        fx = fx.reshape(1, 1, -1)
        freq_rad = torch.sqrt(fz**2 + fy**2 + fx**2)

    # Initialize filter as ones
    filter_mask = torch.ones_like(freq_rad, dtype=tensor.dtype.to_real())

    # Apply high-pass filter (low_freq)
    if low_freq is not None:
        if soft_edge_low > 0:
            # Raised cosine transition from (low_freq - soft_edge_low) to low_freq
            transition_start = low_freq - soft_edge_low
            # Create smooth transition using raised cosine
            # Where freq < transition_start: 0
            # Where freq > low_freq: 1
            # In between: raised cosine
            mask = torch.where(
                freq_rad <= transition_start,
                torch.tensor(0.0, device=tensor.device, dtype=filter_mask.dtype),
                torch.where(
                    freq_rad >= low_freq,
                    torch.tensor(1.0, device=tensor.device, dtype=filter_mask.dtype),
                    0.5 * (1.0 - torch.cos(torch.pi * (freq_rad - transition_start) / soft_edge_low))
                )
            )
            filter_mask = filter_mask * mask
        else:
            # Hard edge
            filter_mask = torch.where(freq_rad >= low_freq, filter_mask, torch.tensor(0.0, device=tensor.device, dtype=filter_mask.dtype))

    # Apply low-pass filter (high_freq)
    if high_freq is not None:
        if soft_edge_high > 0:
            # Raised cosine transition from high_freq to (high_freq + soft_edge_high)
            transition_end = high_freq + soft_edge_high
            # Where freq < high_freq: 1
            # Where freq > transition_end: 0
            # In between: raised cosine
            mask = torch.where(
                freq_rad <= high_freq,
                torch.tensor(1.0, device=tensor.device, dtype=filter_mask.dtype),
                torch.where(
                    freq_rad >= transition_end,
                    torch.tensor(0.0, device=tensor.device, dtype=filter_mask.dtype),
                    0.5 * (1.0 + torch.cos(torch.pi * (freq_rad - high_freq) / soft_edge_high))
                )
            )
            filter_mask = filter_mask * mask
        else:
            # Hard edge
            filter_mask = torch.where(freq_rad <= high_freq, filter_mask, torch.tensor(0.0, device=tensor.device, dtype=filter_mask.dtype))

    # Broadcast filter to match batch dimensions
    if n_batch_dims > 0:
        filter_shape = (1,) * n_batch_dims + filter_mask.shape
        filter_mask = filter_mask.reshape(filter_shape)

    # Convert to complex for multiplication
    filter_mask = filter_mask.to(tensor.dtype)

    # Apply filter
    result = tensor * filter_mask

    return result