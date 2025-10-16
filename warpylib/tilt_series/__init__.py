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

__all__ = ["TiltSeries"]
