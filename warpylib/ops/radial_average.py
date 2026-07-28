import torch
from typing import Tuple, Union


def radial_average_rft(
    values: torch.Tensor,
    image_shape: Union[Tuple[int, int], Tuple[int, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotationally average an RFFT-formatted tensor into a 1D radial profile.

    Bins Fourier pixels by integer frequency-shell radius (rounded Euclidean
    distance from DC) and averages ``values`` within each shell. Uses the
    standard RFFT layout with DC at index 0 (not fftshifted): non-RFFT axes run
    in ``fftfreq`` order and the last axis is the half-spectrum ``0..W//2``.

    Shells run from 0 (DC) to ``W//2`` inclusive. Pixels whose rounded radius
    exceeds ``W//2`` (spectrum corners beyond Nyquist) are dropped, so the
    highest shells may have small counts.

    Parameters
    ----------
    values : torch.Tensor
        Real-valued RFFT-formatted tensor with shape (..., H, W//2+1) for 2D or
        (..., D, H, W//2+1) for 3D, where ... are batch dimensions. Typically a
        power spectrum.
    image_shape : tuple of int
        Full real-space spatial shape: (H, W) for 2D or (D, H, W) for 3D. All
        dimensions must be even.

    Returns
    -------
    profile : torch.Tensor
        Mean of ``values`` per shell, shape (..., W//2+1). Shells with no
        contributing pixels are zero.
    counts : torch.Tensor
        Number of Fourier pixels contributing to each shell, shape (W//2+1),
        dtype int64. Batch-independent (geometry only).

    Raises
    ------
    ValueError
        If ``image_shape`` is not length 2 or 3, any dimension is odd, or the
        tensor's trailing shape does not match the RFFT of ``image_shape``.

    Examples
    --------
    >>> power = (torch.fft.rfft2(torch.randn(8, 128, 128)).abs() ** 2)
    >>> profile, counts = radial_average_rft(power, image_shape=(128, 128))
    >>> profile.shape, counts.shape
    (torch.Size([8, 65]), torch.Size([65]))
    """
    ndim_spatial = len(image_shape)
    if ndim_spatial not in (2, 3):
        raise ValueError(f"image_shape must have 2 or 3 elements, got {ndim_spatial}")
    for i, dim_size in enumerate(image_shape):
        if dim_size % 2 != 0:
            raise ValueError(f"image_shape dimension {i} must be even, got {dim_size}")

    expected_trailing = (*image_shape[:-1], image_shape[-1] // 2 + 1)
    if tuple(values.shape[-ndim_spatial:]) != expected_trailing:
        raise ValueError(
            f"values trailing shape {tuple(values.shape[-ndim_spatial:])} does not match "
            f"RFFT of image_shape {expected_trailing}"
        )

    device = values.device
    w_half = image_shape[-1] // 2 + 1
    n_shells = image_shape[-1] // 2 + 1

    # Frequency coordinates in cycles-per-box (integer for even sizes): full
    # fftfreq order on non-RFFT axes, half-spectrum on the last axis.
    axes = [torch.fft.fftfreq(n, device=device) * n for n in image_shape[:-1]]
    axes.append(torch.arange(w_half, device=device, dtype=torch.float32))
    grids = torch.meshgrid(*axes, indexing="ij")
    radius = torch.sqrt(sum(g**2 for g in grids))
    shell = torch.round(radius).to(torch.int64)

    flat_shell = shell.reshape(-1)
    valid = flat_shell < n_shells
    flat_shell_valid = flat_shell[valid]
    counts = torch.bincount(flat_shell_valid, minlength=n_shells)

    batch_shape = values.shape[:-ndim_spatial]
    flat_values = values.reshape(-1, flat_shell.numel())[:, valid]
    sums = torch.zeros(flat_values.shape[0], n_shells, dtype=values.dtype, device=device)
    sums.index_add_(1, flat_shell_valid, flat_values)

    counts_f = counts.to(values.dtype)
    profile = torch.where(counts_f > 0, sums / counts_f, torch.zeros_like(sums))
    profile = profile.reshape(*batch_shape, n_shells)
    return profile, counts
