"""
Path properties and helper methods for TiltSeries.

This module contains all path-related properties from WarpLib's TiltSeries and Movie classes,
including directory paths, file paths, and static path construction helpers.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import TiltSeries


# Helper function equivalent to Helper.PathToName() in C#
def _path_to_name(path: str) -> str:
    """Remove extension and get basename (equivalent to Helper.PathToName)"""
    return Path(path).stem


def _path_to_name_with_extension(path: str) -> str:
    """Get basename with extension (equivalent to Helper.PathToNameWithExtension)"""
    return Path(path).name


# ============================================================================
# Movie base class properties (missing from core.py)
# ============================================================================

@property
def name(self: "TiltSeries") -> str:
    """Get filename with extension from path"""
    if not self.path:
        return ""
    return _path_to_name_with_extension(self.path)


@property
def data_or_processing_directory_name(self: "TiltSeries") -> str:
    """Get data directory if set, otherwise processing directory"""
    if self.data_directory_name:
        return self.data_directory_name
    return self.processing_directory_name


@property
def mask_path(self: "TiltSeries") -> str:
    """Get path to mask file"""
    return str(Path(self.mask_dir) / f"{self.root_name}.tif")


@property
def mask_dir(self: "TiltSeries") -> str:
    """Get mask directory"""
    return str(Path(self.processing_directory_name) / MASK_DIR_NAME)


# ============================================================================
# TiltSeries directory constants and paths
# ============================================================================

# Directory name constants
TILT_STACK_DIR_NAME = "tiltstack"
TILT_STACK_THUMBNAIL_DIR_NAME = "thumbnails"
RECONSTRUCTION_DIR_NAME = "reconstruction"
RECONSTRUCTION_DECONV_DIR_NAME = str(Path(RECONSTRUCTION_DIR_NAME) / "deconv")
RECONSTRUCTION_ODD_DIR_NAME = str(Path(RECONSTRUCTION_DIR_NAME) / "odd")
RECONSTRUCTION_EVEN_DIR_NAME = str(Path(RECONSTRUCTION_DIR_NAME) / "even")
RECONSTRUCTION_DENOISED_DIR_NAME = "denoised"
RECONSTRUCTION_CTF_DIR_NAME = str(Path(RECONSTRUCTION_DIR_NAME) / "ctf")
SUBTOMO_DIR_NAME = "subtomo"
PARTICLE_SERIES_DIR_NAME = "particleseries"
MASK_DIR_NAME = "mask"  # From Movie.cs


# ============================================================================
# Instance path properties
# ============================================================================

@property
def tilt_stack_dir(self: "TiltSeries") -> str:
    """Get tilt stack directory for this series"""
    return str(Path(self.processing_directory_name) / TILT_STACK_DIR_NAME / self.root_name)


@property
def tilt_stack_thumbnail_dir(self: "TiltSeries") -> str:
    """Get tilt stack thumbnail directory"""
    return str(Path(self.tilt_stack_dir) / TILT_STACK_THUMBNAIL_DIR_NAME)


@property
def tilt_stack_path(self: "TiltSeries") -> str:
    """Get tilt stack (.st) file path"""
    return str(Path(self.tilt_stack_dir) / f"{self.root_name}.st")


@property
def angle_file_path(self: "TiltSeries") -> str:
    """Get angle file (.rawtlt) path"""
    return str(Path(self.tilt_stack_dir) / f"{self.root_name}.rawtlt")


@property
def reconstruction_dir(self: "TiltSeries") -> str:
    """Get reconstruction directory"""
    return str(Path(self.processing_directory_name) / RECONSTRUCTION_DIR_NAME)


@property
def reconstruction_deconv_dir(self: "TiltSeries") -> str:
    """Get deconvolved reconstruction directory"""
    return str(Path(self.processing_directory_name) / RECONSTRUCTION_DECONV_DIR_NAME)


@property
def reconstruction_odd_dir(self: "TiltSeries") -> str:
    """Get odd half-map reconstruction directory"""
    return str(Path(self.processing_directory_name) / RECONSTRUCTION_ODD_DIR_NAME)


@property
def reconstruction_even_dir(self: "TiltSeries") -> str:
    """Get even half-map reconstruction directory"""
    return str(Path(self.processing_directory_name) / RECONSTRUCTION_EVEN_DIR_NAME)


@property
def reconstruction_denoised_dir(self: "TiltSeries") -> str:
    """Get denoised reconstruction directory"""
    return str(Path(self.processing_directory_name) / RECONSTRUCTION_DENOISED_DIR_NAME)


@property
def reconstruction_ctf_dir(self: "TiltSeries") -> str:
    """Get CTF reconstruction directory"""
    return str(Path(self.processing_directory_name) / RECONSTRUCTION_CTF_DIR_NAME)


@property
def subtomo_dir(self: "TiltSeries") -> str:
    """Get subtomogram directory for this series"""
    return str(Path(self.processing_directory_name) / SUBTOMO_DIR_NAME / self.root_name)


@property
def particle_series_dir(self: "TiltSeries") -> str:
    """Get particle series directory for this series"""
    return str(Path(self.processing_directory_name) / PARTICLE_SERIES_DIR_NAME / self.root_name)


def tilt_stack_thumbnail_path(self: "TiltSeries", tilt_name: str) -> str:
    """Get thumbnail path for a specific tilt"""
    return str(Path(self.tilt_stack_thumbnail_dir) / f"{_path_to_name(tilt_name)}.png")


# ============================================================================
# Static helper methods for path construction
# ============================================================================

@staticmethod
def to_tilt_stack_path(name: str) -> str:
    """Construct tilt stack path from series name"""
    return str(Path(TILT_STACK_DIR_NAME) / f"{_path_to_name(name)}.st")


@staticmethod
def to_tilt_stack_thumbnail_path(series_name: str, tilt_name: str) -> str:
    """Construct thumbnail path from series and tilt names"""
    return str(
        Path(TILT_STACK_DIR_NAME)
        / _path_to_name(series_name)
        / TILT_STACK_THUMBNAIL_DIR_NAME
        / f"{_path_to_name(tilt_name)}.png"
    )


@staticmethod
def to_angle_file_path(name: str) -> str:
    """Construct angle file path from series name"""
    return str(Path(TILT_STACK_DIR_NAME) / f"{_path_to_name(name)}.rawtlt")


@staticmethod
def to_tomogram_with_pixel_size(name: str, pixel_size: float) -> str:
    """Construct tomogram name with pixel size suffix"""
    return f"{_path_to_name(name)}_{pixel_size:.2f}Apx"


@staticmethod
def to_reconstruction_tomogram_path(name: str, pixel_size: float) -> str:
    """Construct reconstruction tomogram path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_DIR_NAME) / f"{tomogram_name}.mrc")


@staticmethod
def to_reconstruction_thumbnail_path(name: str, pixel_size: float) -> str:
    """Construct reconstruction thumbnail path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_DIR_NAME) / f"{tomogram_name}.png")


@staticmethod
def to_reconstruction_deconv_path(name: str, pixel_size: float) -> str:
    """Construct deconvolved reconstruction path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_DECONV_DIR_NAME) / f"{tomogram_name}.mrc")


@staticmethod
def to_reconstruction_odd_path(name: str, pixel_size: float) -> str:
    """Construct odd half-map reconstruction path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_ODD_DIR_NAME) / f"{tomogram_name}.mrc")


@staticmethod
def to_reconstruction_even_path(name: str, pixel_size: float) -> str:
    """Construct even half-map reconstruction path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_EVEN_DIR_NAME) / f"{tomogram_name}.mrc")


@staticmethod
def to_reconstruction_denoised_tomogram_path(name: str, pixel_size: float) -> str:
    """Construct denoised tomogram path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_DENOISED_DIR_NAME) / f"{tomogram_name}.mrc")


@staticmethod
def to_reconstruction_denoised_thumbnail_path(name: str, pixel_size: float) -> str:
    """Construct denoised thumbnail path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_DENOISED_DIR_NAME) / f"{tomogram_name}.png")


@staticmethod
def to_reconstruction_ctf_path(name: str, pixel_size: float) -> str:
    """Construct CTF reconstruction path"""
    tomogram_name = to_tomogram_with_pixel_size(name, pixel_size)
    return str(Path(RECONSTRUCTION_CTF_DIR_NAME) / f"{tomogram_name}.mrc")


@staticmethod
def to_subtomo_dir_path(name: str) -> str:
    """Construct subtomogram directory path from series name"""
    return str(Path(SUBTOMO_DIR_NAME) / _path_to_name(name))


@staticmethod
def to_particle_series_dir_path(path: str) -> str:
    """Construct particle series directory path"""
    return str(Path(PARTICLE_SERIES_DIR_NAME) / _path_to_name(path))


@staticmethod
def to_particle_series_average_path(path: str, angpix: float) -> str:
    """Construct particle series average path"""
    name = _path_to_name(path)
    series_dir = to_particle_series_dir_path(path)
    return str(Path(series_dir) / f"{name}_{angpix:.2f}A_average.mrcs")


@staticmethod
def to_particle_series_file_path(path: str, angpix: float, particle_id: int) -> str:
    """Construct particle series file path for a specific particle"""
    name = _path_to_name(path)
    series_dir = to_particle_series_dir_path(name)
    return str(Path(series_dir) / f"{name}_{angpix:.2f}A_{particle_id:06d}.mrcs")


@staticmethod
def to_mask_path(name: str) -> str:
    """Construct mask path from name"""
    return str(Path(MASK_DIR_NAME) / f"{_path_to_name(name)}.tif")