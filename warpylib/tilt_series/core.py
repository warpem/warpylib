"""
TiltSeries core class - Geometry and grid fields

This module contains the core TiltSeries class with field definitions,
initialization, and properties.
"""

from typing import Optional, List
from pathlib import Path
import torch
from ..cubic_grid import CubicGrid
from ..ctf import CTF


class TiltSeries:
    """
    Tilt series geometry and deformation grids.

    This class contains the essential geometry parameters and grid-based
    deformation fields from WarpLib's TiltSeries and Movie classes.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        data_directory_name: Optional[str] = None,
        n_tilts: int = 1,
        image_dimensions_physical: Optional[torch.Tensor] = None,
        volume_dimensions_physical: Optional[torch.Tensor] = None,
    ):
        """
        Initialize TiltSeries.

        Args:
            path: Path to the tilt series XML file (if loading from existing)
            data_directory_name: Optional data directory name
            n_tilts: Number of tilts in the series (used if not loading from file)
            image_dimensions_physical: (2,) tensor with image dimensions in Angstroms
            volume_dimensions_physical: (3,) tensor with volume dimensions in Angstroms
        """
        # Path handling (mimics Movie.cs constructor)
        self.path: str = path if path is not None else ""
        self.data_directory_name: str = data_directory_name if data_directory_name is not None else ""
        # Runtime dimensions from Movie
        self.image_dimensions_physical: torch.Tensor = (
            image_dimensions_physical
            if image_dimensions_physical is not None
            else torch.zeros(2, dtype=torch.float32)
        )
        self.n_frames: int = 1
        self.fraction_frames: float = 1.0

        # Runtime dimensions from TiltSeries
        self.volume_dimensions_physical: torch.Tensor = (
            volume_dimensions_physical
            if volume_dimensions_physical is not None
            else torch.zeros(3, dtype=torch.float32)
        )
        self.size_rounding_factors: torch.Tensor = torch.ones(3, dtype=torch.float32)

        # Geometry parameters
        self.plane_normal: torch.Tensor = torch.zeros(3, dtype=torch.float32)
        self.level_angle_x: float = 0.0
        self.level_angle_y: float = 0.0
        self.are_angles_inverted: bool = False

        # Global parameters
        self.global_bfactor: float = 0.0
        self.global_weight: float = 1.0
        self.magnification_correction: torch.Tensor = torch.eye(2, dtype=torch.float32)
        self.unselect_filter: bool = False
        self.unselect_manual: Optional[bool] = None
        self.ctf_resolution_estimate: float = 0.0

        # Per-tilt parameters
        self.angles: torch.Tensor = torch.zeros(n_tilts, dtype=torch.float32)
        self.dose: torch.Tensor = torch.zeros(n_tilts, dtype=torch.float32)
        self.use_tilt: torch.Tensor = torch.ones(n_tilts, dtype=torch.bool)
        self.tilt_axis_angles: torch.Tensor = torch.zeros(n_tilts, dtype=torch.float32)
        self.tilt_axis_offset_x: torch.Tensor = torch.zeros(n_tilts, dtype=torch.float32)
        self.tilt_axis_offset_y: torch.Tensor = torch.zeros(n_tilts, dtype=torch.float32)
        self.tilt_movie_paths: List[str] = [""] * n_tilts
        self.fov_fraction: torch.Tensor = torch.ones(n_tilts, dtype=torch.float32)

        # CTF object
        self.ctf: CTF = CTF()

        # CTF-related 3D grids
        self.grid_ctf_defocus: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_defocus_delta: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_defocus_angle: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_cs: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_phase: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_doming: CubicGrid = CubicGrid((1, 1, 1))

        # Motion-related 3D grids
        self.grid_movement_x: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_movement_y: CubicGrid = CubicGrid((1, 1, 1))

        # Beam-tilt/geometry 3D grids
        self.grid_angle_x: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_angle_y: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_angle_z: CubicGrid = CubicGrid((1, 1, 1))

        # Dose-related 3D grids
        self.grid_dose_bfacs: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_dose_bfacs_delta: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_dose_bfacs_angle: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_dose_weights: CubicGrid = CubicGrid((1, 1, 1))

        # Location-related 3D grids
        self.grid_location_bfacs: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_location_weights: CubicGrid = CubicGrid((1, 1, 1))

        # Volume warp 4D grids
        self.grid_volume_warp_x: CubicGrid = CubicGrid((1, 1, 1, 1))
        self.grid_volume_warp_y: CubicGrid = CubicGrid((1, 1, 1, 1))
        self.grid_volume_warp_z: CubicGrid = CubicGrid((1, 1, 1, 1))

        # Load metadata if path is provided (mimics Movie.cs constructor behavior)
        if path is not None:
            from .io import load_meta
            load_meta(self, self.xml_path)

    @property
    def root_name(self) -> str:
        """Get root name from path (filename without extension)"""
        if not self.path:
            return ""
        return Path(self.path).stem

    @property
    def xml_name(self) -> str:
        """Get XML filename"""
        if not self.root_name:
            return ""
        return f"{self.root_name}.xml"

    @property
    def processing_directory_name(self) -> str:
        """Get processing directory from path"""
        if not self.path:
            return ""
        return str(Path(self.path).parent)

    @property
    def xml_path(self) -> str:
        """Get full XML path"""
        if not self.processing_directory_name or not self.xml_name:
            return ""
        return str(Path(self.processing_directory_name) / self.xml_name)

    @property
    def data_path(self) -> str:
        """Get data path (either in data directory or processing directory)"""
        if self.data_directory_name:
            name = Path(self.path).name if self.path else ""
            return str(Path(self.data_directory_name) / name)
        return self.path

    @property
    def n_tilts(self) -> int:
        """Get number of tilts"""
        return len(self.angles)

    @property
    def min_tilt(self) -> float:
        """Get minimum tilt angle"""
        return float(self.angles.min())

    @property
    def max_tilt(self) -> float:
        """Get maximum tilt angle"""
        return float(self.angles.max())

    @property
    def min_dose(self) -> float:
        """Get minimum dose"""
        return float(self.dose.min())

    @property
    def max_dose(self) -> float:
        """Get maximum dose"""
        return float(self.dose.max())

    def get_tilt_defocus(self, tilt_id: int) -> float:
        """Get defocus value for a specific tilt"""
        if (
            self.grid_ctf_defocus is not None
            and len(self.grid_ctf_defocus.flat_values) > tilt_id
        ):
            return float(self.grid_ctf_defocus.flat_values[tilt_id])
        return 0.0

    def get_tilt_defocus_delta(self, tilt_id: int) -> float:
        """Get defocus delta (astigmatism) for a specific tilt"""
        if (
            self.grid_ctf_defocus_delta is not None
            and len(self.grid_ctf_defocus_delta.flat_values) > tilt_id
        ):
            return float(self.grid_ctf_defocus_delta.flat_values[tilt_id])
        return 0.0

    def get_tilt_defocus_angle(self, tilt_id: int) -> float:
        """Get defocus angle (astigmatism angle) for a specific tilt"""
        if (
            self.grid_ctf_defocus_angle is not None
            and len(self.grid_ctf_defocus_angle.flat_values) > tilt_id
        ):
            return float(self.grid_ctf_defocus_angle.flat_values[tilt_id])
        return 0.0

    def get_tilt_phase(self, tilt_id: int) -> float:
        """Get phase shift for a specific tilt"""
        if (
            self.grid_ctf_phase is not None
            and len(self.grid_ctf_phase.flat_values) > tilt_id
        ):
            return float(self.grid_ctf_phase.flat_values[tilt_id])
        return 0.0

    def indices_sorted_angle(self) -> torch.Tensor:
        """Get indices sorted by tilt angle"""
        return torch.argsort(self.angles)

    def indices_sorted_absolute_angle(self) -> torch.Tensor:
        """Get indices sorted by absolute tilt angle"""
        return torch.argsort(torch.abs(self.angles))

    def indices_sorted_dose(self) -> torch.Tensor:
        """Get indices sorted by dose"""
        return torch.argsort(self.dose)
