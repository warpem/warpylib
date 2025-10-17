"""
Movie - Micrograph/movie metadata and path handling

Partial port of WarpLib's Movie class, focusing on path management
and basic metadata operations.
"""

from .core import Movie
from . import io
from . import paths

# Bind I/O methods to Movie
Movie.load_meta = io.load_meta
Movie.save_meta = io.save_meta

# Bind directory path properties to Movie
Movie.power_spectrum_dir = paths.power_spectrum_dir
Movie.motion_track_dir = paths.motion_track_dir
Movie.average_dir = paths.average_dir
Movie.average_odd_dir = paths.average_odd_dir
Movie.average_even_dir = paths.average_even_dir
Movie.average_denoised_dir = paths.average_denoised_dir
Movie.deconvolved_dir = paths.deconvolved_dir
Movie.denoise_training_dir = paths.denoise_training_dir
Movie.denoise_training_dir_odd = paths.denoise_training_dir_odd
Movie.denoise_training_dir_even = paths.denoise_training_dir_even
Movie.denoise_training_dir_ctf = paths.denoise_training_dir_ctf
Movie.denoise_training_dir_model = paths.denoise_training_dir_model
Movie.mask_dir = paths.mask_dir
Movie.segmentation_dir = paths.segmentation_dir
Movie.membrane_segmentation_dir = paths.membrane_segmentation_dir
Movie.particles_dir = paths.particles_dir
Movie.particles_denoising_odd_dir = paths.particles_denoising_odd_dir
Movie.particles_denoising_even_dir = paths.particles_denoising_even_dir
Movie.matching_dir = paths.matching_dir
Movie.thumbnails_dir = paths.thumbnails_dir

# Bind file path properties to Movie
Movie.power_spectrum_path = paths.power_spectrum_path
Movie.average_path = paths.average_path
Movie.average_odd_path = paths.average_odd_path
Movie.average_even_path = paths.average_even_path
Movie.average_denoised_path = paths.average_denoised_path
Movie.deconvolved_path = paths.deconvolved_path
Movie.denoise_training_odd_path = paths.denoise_training_odd_path
Movie.denoise_training_even_path = paths.denoise_training_even_path
Movie.denoise_training_ctf_path = paths.denoise_training_ctf_path
Movie.mask_path = paths.mask_path
Movie.thumbnails_path = paths.thumbnails_path
Movie.motion_tracks_path = paths.motion_tracks_path

# Bind static path helper methods to Movie
Movie.to_power_spectrum_path = paths.to_power_spectrum_path
Movie.to_average_path = paths.to_average_path
Movie.to_average_odd_path = paths.to_average_odd_path
Movie.to_average_even_path = paths.to_average_even_path
Movie.to_average_denoised_path = paths.to_average_denoised_path
Movie.to_deconvolved_path = paths.to_deconvolved_path
Movie.to_denoise_training_odd_path = paths.to_denoise_training_odd_path
Movie.to_denoise_training_even_path = paths.to_denoise_training_even_path
Movie.to_denoise_training_ctf_path = paths.to_denoise_training_ctf_path
Movie.to_mask_path = paths.to_mask_path
Movie.to_thumbnails_path = paths.to_thumbnails_path
Movie.to_motion_tracks_path = paths.to_motion_tracks_path
Movie.to_xml_path = paths.to_xml_path

__all__ = ["Movie"]
