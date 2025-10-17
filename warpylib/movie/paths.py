"""
Path properties and helper methods for Movie.

This module contains all path-related properties from WarpLib's Movie class,
including directory paths, file paths, and static path construction helpers.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Movie


# Helper function equivalent to Helper.PathToName() in C#
def _path_to_name(path: str) -> str:
    """Remove extension and get basename (equivalent to Helper.PathToName)"""
    return Path(path).stem


# ============================================================================
# Directory name constants
# ============================================================================

POWER_SPECTRUM_DIR_NAME = "powerspectrum"
MOTION_TRACK_DIR_NAME = "motion"
AVERAGE_DIR_NAME = "average"
AVERAGE_ODD_DIR_NAME = str(Path(AVERAGE_DIR_NAME) / "odd")
AVERAGE_EVEN_DIR_NAME = str(Path(AVERAGE_DIR_NAME) / "even")
AVERAGE_DENOISED_DIR_NAME = str(Path(AVERAGE_DIR_NAME) / "denoised")
DECONVOLVED_DIR_NAME = "deconv"
DENOISE_TRAINING_DIR_NAME = "denoising"
DENOISE_TRAINING_ODD_DIR_NAME = str(Path(DENOISE_TRAINING_DIR_NAME) / "odd")
DENOISE_TRAINING_EVEN_DIR_NAME = str(Path(DENOISE_TRAINING_DIR_NAME) / "even")
DENOISE_TRAINING_CTF_DIR_NAME = str(Path(DENOISE_TRAINING_DIR_NAME) / "ctf")
DENOISE_TRAINING_MODEL_NAME = str(Path(DENOISE_TRAINING_DIR_NAME) / "model.pt")
MASK_DIR_NAME = "mask"
SEGMENTATION_DIR_NAME = "segmentation"
MEMBRANE_SEGMENTATION_DIR_NAME = str(Path(SEGMENTATION_DIR_NAME) / "membranes")
PARTICLES_DIR_NAME = "particles"
PARTICLES_DENOISING_ODD_DIR_NAME = str(Path(PARTICLES_DIR_NAME) / "odd")
PARTICLES_DENOISING_EVEN_DIR_NAME = str(Path(PARTICLES_DIR_NAME) / "even")
MATCHING_DIR_NAME = "matching"
THUMBNAILS_DIR_NAME = "thumbnails"


# ============================================================================
# Instance directory properties
# ============================================================================

@property
def power_spectrum_dir(self: "Movie") -> str:
    """Get power spectrum directory"""
    return str(Path(self.processing_directory_name) / POWER_SPECTRUM_DIR_NAME)


@property
def motion_track_dir(self: "Movie") -> str:
    """Get motion track directory"""
    return str(Path(self.processing_directory_name) / MOTION_TRACK_DIR_NAME)


@property
def average_dir(self: "Movie") -> str:
    """Get average directory"""
    return str(Path(self.processing_directory_name) / AVERAGE_DIR_NAME)


@property
def average_odd_dir(self: "Movie") -> str:
    """Get odd half-map average directory"""
    return str(Path(self.processing_directory_name) / AVERAGE_ODD_DIR_NAME)


@property
def average_even_dir(self: "Movie") -> str:
    """Get even half-map average directory"""
    return str(Path(self.processing_directory_name) / AVERAGE_EVEN_DIR_NAME)


@property
def average_denoised_dir(self: "Movie") -> str:
    """Get denoised average directory"""
    return str(Path(self.processing_directory_name) / AVERAGE_DENOISED_DIR_NAME)


@property
def deconvolved_dir(self: "Movie") -> str:
    """Get deconvolved directory"""
    return str(Path(self.processing_directory_name) / DECONVOLVED_DIR_NAME)


@property
def denoise_training_dir(self: "Movie") -> str:
    """Get denoise training directory"""
    return str(Path(self.processing_directory_name) / DENOISE_TRAINING_DIR_NAME)


@property
def denoise_training_dir_odd(self: "Movie") -> str:
    """Get odd denoise training directory"""
    return str(Path(self.processing_directory_name) / DENOISE_TRAINING_ODD_DIR_NAME)


@property
def denoise_training_dir_even(self: "Movie") -> str:
    """Get even denoise training directory"""
    return str(Path(self.processing_directory_name) / DENOISE_TRAINING_EVEN_DIR_NAME)


@property
def denoise_training_dir_ctf(self: "Movie") -> str:
    """Get CTF denoise training directory"""
    return str(Path(self.processing_directory_name) / DENOISE_TRAINING_CTF_DIR_NAME)


@property
def denoise_training_dir_model(self: "Movie") -> str:
    """Get denoise training model path"""
    return str(Path(self.processing_directory_name) / DENOISE_TRAINING_MODEL_NAME)


@property
def mask_dir(self: "Movie") -> str:
    """Get mask directory"""
    return str(Path(self.processing_directory_name) / MASK_DIR_NAME)


@property
def segmentation_dir(self: "Movie") -> str:
    """Get segmentation directory"""
    return str(Path(self.processing_directory_name) / SEGMENTATION_DIR_NAME)


@property
def membrane_segmentation_dir(self: "Movie") -> str:
    """Get membrane segmentation directory"""
    return str(Path(self.processing_directory_name) / MEMBRANE_SEGMENTATION_DIR_NAME)


@property
def particles_dir(self: "Movie") -> str:
    """Get particles directory"""
    return str(Path(self.processing_directory_name) / PARTICLES_DIR_NAME)


@property
def particles_denoising_odd_dir(self: "Movie") -> str:
    """Get odd particles denoising directory"""
    return str(Path(self.processing_directory_name) / PARTICLES_DENOISING_ODD_DIR_NAME)


@property
def particles_denoising_even_dir(self: "Movie") -> str:
    """Get even particles denoising directory"""
    return str(Path(self.processing_directory_name) / PARTICLES_DENOISING_EVEN_DIR_NAME)


@property
def matching_dir(self: "Movie") -> str:
    """Get matching directory"""
    return str(Path(self.processing_directory_name) / MATCHING_DIR_NAME)


@property
def thumbnails_dir(self: "Movie") -> str:
    """Get thumbnails directory"""
    return str(Path(self.processing_directory_name) / THUMBNAILS_DIR_NAME)


# ============================================================================
# Instance file path properties
# ============================================================================

@property
def power_spectrum_path(self: "Movie") -> str:
    """Get power spectrum file path"""
    return str(Path(self.power_spectrum_dir) / f"{self.root_name}.mrc")


@property
def average_path(self: "Movie") -> str:
    """Get average file path"""
    return str(Path(self.average_dir) / f"{self.root_name}.mrc")


@property
def average_odd_path(self: "Movie") -> str:
    """Get odd average file path"""
    return str(Path(self.average_odd_dir) / f"{self.root_name}.mrc")


@property
def average_even_path(self: "Movie") -> str:
    """Get even average file path"""
    return str(Path(self.average_even_dir) / f"{self.root_name}.mrc")


@property
def average_denoised_path(self: "Movie") -> str:
    """Get denoised average file path"""
    return str(Path(self.average_denoised_dir) / f"{self.root_name}.mrc")


@property
def deconvolved_path(self: "Movie") -> str:
    """Get deconvolved file path"""
    return str(Path(self.deconvolved_dir) / f"{self.root_name}.mrc")


@property
def denoise_training_odd_path(self: "Movie") -> str:
    """Get odd denoise training file path"""
    return str(Path(self.denoise_training_dir_odd) / f"{self.root_name}.mrc")


@property
def denoise_training_even_path(self: "Movie") -> str:
    """Get even denoise training file path"""
    return str(Path(self.denoise_training_dir_even) / f"{self.root_name}.mrc")


@property
def denoise_training_ctf_path(self: "Movie") -> str:
    """Get CTF denoise training file path"""
    return str(Path(self.denoise_training_dir_ctf) / f"{self.root_name}.mrc")


@property
def mask_path(self: "Movie") -> str:
    """Get mask file path"""
    return str(Path(self.mask_dir) / f"{self.root_name}.tif")


@property
def thumbnails_path(self: "Movie") -> str:
    """Get thumbnails file path"""
    return str(Path(self.thumbnails_dir) / f"{self.root_name}.png")


@property
def motion_tracks_path(self: "Movie") -> str:
    """Get motion tracks file path"""
    return str(Path(self.average_dir) / f"{self.root_name}_motion.json")


# ============================================================================
# Static helper methods for path construction
# ============================================================================

@staticmethod
def to_power_spectrum_path(name: str) -> str:
    """Construct power spectrum path from name"""
    return str(Path(POWER_SPECTRUM_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_average_path(name: str) -> str:
    """Construct average path from name"""
    return str(Path(AVERAGE_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_average_odd_path(name: str) -> str:
    """Construct odd average path from name"""
    return str(Path(AVERAGE_ODD_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_average_even_path(name: str) -> str:
    """Construct even average path from name"""
    return str(Path(AVERAGE_EVEN_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_average_denoised_path(name: str) -> str:
    """Construct denoised average path from name"""
    return str(Path(AVERAGE_DENOISED_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_deconvolved_path(name: str) -> str:
    """Construct deconvolved path from name"""
    return str(Path(DECONVOLVED_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_denoise_training_odd_path(name: str) -> str:
    """Construct odd denoise training path from name"""
    return str(Path(DENOISE_TRAINING_ODD_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_denoise_training_even_path(name: str) -> str:
    """Construct even denoise training path from name"""
    return str(Path(DENOISE_TRAINING_EVEN_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_denoise_training_ctf_path(name: str) -> str:
    """Construct CTF denoise training path from name"""
    return str(Path(DENOISE_TRAINING_CTF_DIR_NAME) / f"{_path_to_name(name)}.mrc")


@staticmethod
def to_mask_path(name: str) -> str:
    """Construct mask path from name"""
    return str(Path(MASK_DIR_NAME) / f"{_path_to_name(name)}.tif")


@staticmethod
def to_thumbnails_path(name: str) -> str:
    """Construct thumbnails path from name"""
    return str(Path(THUMBNAILS_DIR_NAME) / f"{_path_to_name(name)}.png")


@staticmethod
def to_motion_tracks_path(name: str) -> str:
    """Construct motion tracks path from name"""
    return str(Path(AVERAGE_DIR_NAME) / f"{_path_to_name(name)}_motion.json")


@staticmethod
def to_xml_path(name: str) -> str:
    """Construct XML path from name"""
    return f"{_path_to_name(name)}.xml"