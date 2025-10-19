"""
TiltSeries - Tilt series geometry and grid fields

Partial port of WarpLib's TiltSeries and Movie classes, focusing on
geometry definitions and grid-based fields.
"""

from .core import TiltSeries
from . import io
from . import positions
from . import angles
from . import ctf
from . import particle_images
from . import paths
from . import load_images
from . import reconstruct_subvolumes
from . import reconstruct_volume

# Bind I/O methods to TiltSeries
TiltSeries.initialize_from_tomo_star = io.initialize_from_tomo_star
TiltSeries.load_meta = io.load_meta
TiltSeries.save_meta = io.save_meta

# Bind position transformation methods to TiltSeries
TiltSeries.get_positions_in_one_tilt = positions.get_positions_in_one_tilt
TiltSeries.get_position_in_all_tilts_single = positions.get_position_in_all_tilts_single
TiltSeries.get_position_in_all_tilts = positions.get_position_in_all_tilts

# Bind angle transformation methods to TiltSeries
TiltSeries.get_angle_in_all_tilts_single = angles.get_angle_in_all_tilts_single
TiltSeries.get_angle_in_all_tilts = angles.get_angle_in_all_tilts
TiltSeries.get_particle_rotation_matrix_in_all_tilts = angles.get_particle_rotation_matrix_in_all_tilts
TiltSeries.get_particle_angle_in_all_tilts_single = angles.get_particle_angle_in_all_tilts_single
TiltSeries.get_particle_angle_in_all_tilts = angles.get_particle_angle_in_all_tilts
TiltSeries.get_angles_in_one_tilt = angles.get_angles_in_one_tilt

# Bind CTF generation methods to TiltSeries
TiltSeries.get_ctfs_for_particles_single = ctf.get_ctfs_for_particles_single
TiltSeries.get_ctfs_for_particles = ctf.get_ctfs_for_particles
TiltSeries.get_ctfs_for_one_tilt = ctf.get_ctfs_for_one_tilt

# Bind particle image extraction methods to TiltSeries
TiltSeries.get_images_for_particles_rft = particle_images.get_images_for_particles_rft
TiltSeries.get_images_for_particles_single_rft = particle_images.get_images_for_particles_single_rft

# Bind image loading methods to TiltSeries
TiltSeries.load_images = load_images.load_images

# Bind subtomogram reconstruction methods to TiltSeries
TiltSeries.reconstruct_subvolumes = reconstruct_subvolumes.reconstruct_subvolumes
TiltSeries.reconstruct_subvolumes_single = reconstruct_subvolumes.reconstruct_subvolumes_single

# Bind full volume reconstruction methods to TiltSeries
TiltSeries.reconstruct_full = reconstruct_volume.reconstruct_full

# Bind path properties to TiltSeries
TiltSeries.name = paths.name
TiltSeries.data_or_processing_directory_name = paths.data_or_processing_directory_name
TiltSeries.mask_path = paths.mask_path
TiltSeries.mask_dir = paths.mask_dir
TiltSeries.tilt_stack_dir = paths.tilt_stack_dir
TiltSeries.tilt_stack_thumbnail_dir = paths.tilt_stack_thumbnail_dir
TiltSeries.tilt_stack_path = paths.tilt_stack_path
TiltSeries.angle_file_path = paths.angle_file_path
TiltSeries.reconstruction_dir = paths.reconstruction_dir
TiltSeries.reconstruction_deconv_dir = paths.reconstruction_deconv_dir
TiltSeries.reconstruction_odd_dir = paths.reconstruction_odd_dir
TiltSeries.reconstruction_even_dir = paths.reconstruction_even_dir
TiltSeries.reconstruction_denoised_dir = paths.reconstruction_denoised_dir
TiltSeries.reconstruction_ctf_dir = paths.reconstruction_ctf_dir
TiltSeries.subtomo_dir = paths.subtomo_dir
TiltSeries.particle_series_dir = paths.particle_series_dir
TiltSeries.tilt_stack_thumbnail_path = paths.tilt_stack_thumbnail_path

# Bind static path helper methods to TiltSeries
TiltSeries.to_tilt_stack_path = paths.to_tilt_stack_path
TiltSeries.to_tilt_stack_thumbnail_path = paths.to_tilt_stack_thumbnail_path
TiltSeries.to_angle_file_path = paths.to_angle_file_path
TiltSeries.to_tomogram_with_pixel_size = paths.to_tomogram_with_pixel_size
TiltSeries.to_reconstruction_tomogram_path = paths.to_reconstruction_tomogram_path
TiltSeries.to_reconstruction_thumbnail_path = paths.to_reconstruction_thumbnail_path
TiltSeries.to_reconstruction_deconv_path = paths.to_reconstruction_deconv_path
TiltSeries.to_reconstruction_odd_path = paths.to_reconstruction_odd_path
TiltSeries.to_reconstruction_even_path = paths.to_reconstruction_even_path
TiltSeries.to_reconstruction_denoised_tomogram_path = paths.to_reconstruction_denoised_tomogram_path
TiltSeries.to_reconstruction_denoised_thumbnail_path = paths.to_reconstruction_denoised_thumbnail_path
TiltSeries.to_reconstruction_ctf_path = paths.to_reconstruction_ctf_path
TiltSeries.to_subtomo_dir_path = paths.to_subtomo_dir_path
TiltSeries.to_particle_series_dir_path = paths.to_particle_series_dir_path
TiltSeries.to_particle_series_average_path = paths.to_particle_series_average_path
TiltSeries.to_particle_series_file_path = paths.to_particle_series_file_path
TiltSeries.to_mask_path = paths.to_mask_path

__all__ = ["TiltSeries"]
