"""
Image loading for tilt series.

This module provides functionality to load and rescale tilt series images
(movie averages) to a target pixel size.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
import mrcfile
from ..movie.core import Movie
from ..ops.rescale import rescale


def load_image_dimensions(
    tilt_series: "TiltSeries",
    original_pixel_size: float,
) -> None:
    """
    Load image dimensions from the first tilt and set physical dimensions.

    This is a lightweight alternative to load_images() when you only need to
    initialize the physical dimensions without loading the actual image data.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series object to update with dimension information.
    original_pixel_size : float
        Original pixel size of the images in Angstroms.

    Raises
    ------
    ValueError
        If the number of movie paths doesn't match the number of tilts.
    FileNotFoundError
        If the first image file is not found.
    """
    from .core import TiltSeries

    n_tilts = tilt_series.n_tilts

    # Validate movie paths
    if len(tilt_series.tilt_movie_paths) != n_tilts:
        raise ValueError(
            f"Number of movie paths ({len(tilt_series.tilt_movie_paths)}) "
            f"does not match number of tilts ({n_tilts})"
        )

    # Get first movie path
    first_tilt_path = tilt_series.tilt_movie_paths[0]
    full_path = str(
        Path(tilt_series.data_directory_name or tilt_series.processing_directory_name)
        / first_tilt_path
    )
    first_movie = Movie(path=full_path)

    # Read first image to determine dimensions
    with mrcfile.open(first_movie.average_path, mode="r", permissive=True) as mrc:
        original_shape = mrc.data.shape

    # Handle 2D vs 3D data (should be 2D for averages)
    if len(original_shape) == 3:
        if original_shape[0] != 1:
            raise ValueError(
                f"Average image has {original_shape[0]} layers, expected 1. "
                "This should be a 2D average."
            )
        original_height, original_width = original_shape[1], original_shape[2]
    else:
        original_height, original_width = original_shape[0], original_shape[1]

    # Get device from TiltSeries tensors
    device = tilt_series.angles.device

    # Update physical dimensions
    tilt_series.image_dimensions_physical = torch.tensor(
        [original_width * original_pixel_size, original_height * original_pixel_size],
        dtype=torch.float32,
        device=device,
    )


def load_images(
    tilt_series: "TiltSeries",
    original_pixel_size: float,
    desired_pixel_size: float,
    use_denoised: bool = False,
    load_averages: bool = True,
    load_half_averages: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Load tilt series images (movie averages) and rescale to desired pixel size.

    This method loads the movie averages for each tilt in the series, rescales them
    to the desired pixel size using bandwidth-limited Fourier rescaling, and returns
    them as stacked tensors. Optionally, odd/even half-averages can also be loaded.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series object containing movie paths and metadata.
    original_pixel_size : float
        Original pixel size of the images in Angstroms.
    desired_pixel_size : float
        Desired pixel size after rescaling in Angstroms.
    use_denoised : bool, optional
        If True, load denoised averages instead of regular averages. Default is False.
    load_half_averages : bool, optional
        If True, also load odd and even half-averages. Default is False.

    Returns
    -------
    images : torch.Tensor
        Stacked images with shape (n_tilts, height, width).
    images_odd : torch.Tensor or None
        Stacked odd half-images with shape (n_tilts, height, width), or None if
        load_half_averages is False.
    images_even : torch.Tensor or None
        Stacked even half-images with shape (n_tilts, height, width), or None if
        load_half_averages is False.

    Raises
    ------
    ValueError
        If the number of movie paths doesn't match the number of tilts.
    FileNotFoundError
        If required image files are not found.

    Notes
    -----
    - All output dimensions are guaranteed to be even numbers
    - Size rounding factors are updated on the tilt_series object to account for
      dimension rounding
    - Physical image dimensions are updated on the tilt_series object
    """
    from .core import TiltSeries

    n_tilts = tilt_series.n_tilts

    # Validate movie paths
    if len(tilt_series.tilt_movie_paths) != n_tilts:
        raise ValueError(
            f"Number of movie paths ({len(tilt_series.tilt_movie_paths)}) "
            f"does not match number of tilts ({n_tilts})"
        )

    # Create Movie objects from paths
    movies = []
    for tilt_path in tilt_series.tilt_movie_paths:
        full_path = str(
            Path(tilt_series.data_directory_name or tilt_series.processing_directory_name)
            / tilt_path
        )
        movies.append(Movie(path=full_path))

    # Read first image to determine dimensions
    first_movie = movies[0]
    first_path = (
        first_movie.average_denoised_path if use_denoised else first_movie.average_path
    )

    with mrcfile.open(first_path, mode="r", permissive=True) as mrc:
        original_shape = mrc.data.shape

    # Handle 2D vs 3D data (should be 2D for averages)
    if len(original_shape) == 3:
        if original_shape[0] != 1:
            raise ValueError(
                f"Average image has {original_shape[0]} layers, expected 1. "
                "This should be a 2D average."
            )
        original_height, original_width = original_shape[1], original_shape[2]
    else:
        original_height, original_width = original_shape[0], original_shape[1]

    # Get device from TiltSeries tensors
    device = tilt_series.angles.device

    # Update physical dimensions
    tilt_series.image_dimensions_physical = torch.tensor(
        [original_width * original_pixel_size, original_height * original_pixel_size],
        dtype=torch.float32,
        device=device,
    )

    # Calculate downsampling factor
    downsample_factor = desired_pixel_size / original_pixel_size

    # Calculate scaled dimensions, ensuring they are even
    # Round to nearest even by: round(dim / factor / 2) * 2
    scaled_width = int(round(original_width / downsample_factor / 2)) * 2
    scaled_height = int(round(original_height / downsample_factor / 2)) * 2
    scaled_shape = (scaled_height, scaled_width)

    # Calculate size rounding factors
    # These account for the rounding that was done to ensure even dimensions
    tilt_series.size_rounding_factors = torch.tensor(
        [
            scaled_width / (original_width / downsample_factor),
            scaled_height / (original_height / downsample_factor),
            1.0,  # Z dimension (not used for 2D images)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Determine if rescaling is needed
    needs_rescaling = (scaled_height, scaled_width) != (original_height, original_width)

    # Allocate output tensors
    images = (
        torch.zeros(n_tilts, scaled_height, scaled_width, dtype=torch.float32)
        if load_averages
        else None
    )
    images_odd = (
        torch.zeros(n_tilts, scaled_height, scaled_width, dtype=torch.float32)
        if load_half_averages
        else None
    )
    images_even = (
        torch.zeros(n_tilts, scaled_height, scaled_width, dtype=torch.float32)
        if load_half_averages
        else None
    )

    # Load and process each tilt
    for t, movie in enumerate(movies):

        if load_averages:
            # Load main average
            average_path = (
                movie.average_denoised_path if use_denoised else movie.average_path
            )

            with mrcfile.open(average_path, mode="r", permissive=True) as mrc:
                image_data = torch.from_numpy(mrc.data.copy()).float()

            # Remove singleton dimension if present
            if image_data.ndim == 3:
                image_data = image_data.squeeze(0)

            # Rescale if needed
            if needs_rescaling:
                image_data = rescale(image_data, size=scaled_shape)

            images[t] = image_data

        # Load half-averages if requested
        if load_half_averages:
            # Load odd average
            if not Path(movie.average_odd_path).exists():
                raise FileNotFoundError(
                    f"Odd half-average not found: {movie.average_odd_path}. "
                    "Please re-process the movies with half-average export enabled."
                )

            with mrcfile.open(movie.average_odd_path, mode="r", permissive=True) as mrc:
                image_odd = torch.from_numpy(mrc.data.copy()).float()

            if image_odd.ndim == 3:
                image_odd = image_odd.squeeze(0)

            if needs_rescaling:
                image_odd = rescale(image_odd, size=scaled_shape)

            images_odd[t] = image_odd

            # Load even average
            if not Path(movie.average_even_path).exists():
                raise FileNotFoundError(
                    f"Even half-average not found: {movie.average_even_path}. "
                    "Please re-process the movies with half-average export enabled."
                )

            with mrcfile.open(movie.average_even_path, mode="r", permissive=True) as mrc:
                image_even = torch.from_numpy(mrc.data.copy()).float()

            if image_even.ndim == 3:
                image_even = image_even.squeeze(0)

            if needs_rescaling:
                image_even = rescale(image_even, size=scaled_shape)

            images_even[t] = image_even

    if load_averages and load_half_averages:
        return images, images_odd, images_even
    elif load_averages:
        return images, None, None
    elif load_half_averages:
        return None, images_odd, images_even
    else:
        return None, None, None