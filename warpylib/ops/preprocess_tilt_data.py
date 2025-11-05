import torch
from ..ops import norm, mask_rectangular, subtract_plane, resize
from ..ops.bandpass import bandpass

def preprocess_tilt_data(
    tilt_data: torch.Tensor,
    normalize: bool,
    invert: bool,
    subvolume_size: int
) -> torch.Tensor:
    """
    Preprocess tilt data before reconstruction.

    Applies normalization, edge masking, and bandpass filtering to tilt images.

    Args:
        tilt_data: Input tilt images, shape (n_tilts, H, W)
        normalize: Whether to normalize images
        invert: Whether to invert contrast
        subvolume_size: Size of sub-volumes (for bandpass calculation)

    Returns:
        Preprocessed tilt data, same shape as input
    """
    n_tilts = tilt_data.shape[0]

    # Process each tilt independently
    processed = []

    for t in range(n_tilts):
        img = tilt_data[t].clone()

        # Fit and subtract plane to remove background gradients
        img = subtract_plane(img, fit_and_subtract=True)

        # Sophisticated bandpass filtering to avoid edge artifacts:
        # 1. Pad to 2x size with mirroring
        # 2. Apply soft rectangular mask over padded region
        # 3. Bandpass filter
        # 4. Crop back to original size

        h, w = img.shape
        h_padded, w_padded = h + 256, w + 256  # Pad by 128 pixels on each side

        # Pad with reflection mode
        img_padded = resize(img, size=(h_padded, w_padded), padding_mode='reflect')

        # Apply rectangular mask with soft edge covering the entire padded area
        # The inner region is the original size, soft edge covers the padding
        img_padded = mask_rectangular(
            img_padded,
            region=(h, w),  # Inner size
            soft_edge=128      # Soft edge covers the padded region
        )

        # Bandpass filter on padded image
        # High-pass to remove low frequencies, low-pass at Nyquist
        high_pass_freq = 1.0 / (subvolume_size / 2)
        img_padded = bandpass(
            img_padded,
            dimensionality=2,
            low_freq=high_pass_freq,
            high_freq=1.0,
            soft_edge_low=0.0,
            soft_edge_high=0.0
        )

        # Crop back to original size
        img = resize(img_padded, size=(h, w))

        # Normalize to mean=0, std=1
        if normalize:
            img = norm(img, dimensionality=2)

        if invert:
            img = -img

        processed.append(img)

    return torch.stack(processed, dim=0)