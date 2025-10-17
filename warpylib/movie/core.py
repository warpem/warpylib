"""
Movie core class - Basic movie/micrograph handling

This module contains the core Movie class with basic path properties
and initialization, ported from WarpLib's Movie class.
"""

from typing import Optional, List
from pathlib import Path
import torch
from ..cubic_grid import CubicGrid
from ..ctf import CTF


class Movie:
    """
    Movie/micrograph metadata and path handling.

    This class contains the essential path properties and metadata
    from WarpLib's Movie class, focused on path management and basic metadata.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        data_directory_name: Optional[str] = None,
    ):
        """
        Initialize Movie.

        Args:
            path: Path to the raw movie file (e.g., movie.mrc). The XML metadata
                  file path is automatically derived from this by changing the extension.
            data_directory_name: Optional data directory name
        """
        # Path handling (mimics Movie.cs constructor)
        self.path: str = path if path is not None else ""
        self.data_directory_name: str = data_directory_name if data_directory_name is not None else ""

        # Runtime dimensions
        self.image_dimensions_physical: torch.Tensor = torch.zeros(2, dtype=torch.float32)
        self.n_frames: int = 1
        self.fraction_frames: float = 1.0

        # Global parameters
        self.global_bfactor: float = 0.0
        self.global_weight: float = 1.0
        self.magnification_correction: torch.Tensor = torch.eye(2, dtype=torch.float32)
        self.unselect_filter: bool = False
        self.unselect_manual: Optional[bool] = None
        self.ctf_resolution_estimate: float = 0.0
        self.mean_frame_movement: float = 0.0
        self.mask_percentage: float = -1.0

        # CTF object
        self.ctf: CTF = CTF()

        # CTF-related grids (3D: X, Y spatial + Z for frames/tilts)
        self.grid_ctf_defocus: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_defocus_delta: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_defocus_angle: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_cs: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_phase: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_ctf_doming: CubicGrid = CubicGrid((1, 1, 1))

        # Motion-related grids (3D: X, Y spatial + Z for frames)
        self.grid_movement_x: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_movement_y: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_local_x: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_local_y: CubicGrid = CubicGrid((1, 1, 1))

        # Pyramid motion grids (lists of grids for multi-scale motion)
        self.pyramid_shift_x: List[CubicGrid] = []
        self.pyramid_shift_y: List[CubicGrid] = []

        # Angle grids
        self.grid_angle_x: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_angle_y: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_angle_z: CubicGrid = CubicGrid((1, 1, 1))

        # Dose weighting-related grids
        self.grid_dose_bfacs: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_dose_bfacs_delta: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_dose_bfacs_angle: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_dose_weights: CubicGrid = CubicGrid((1, 1, 1))

        # Spatial weighting-related grids
        self.grid_location_bfacs: CubicGrid = CubicGrid((1, 1, 1))
        self.grid_location_weights: CubicGrid = CubicGrid((1, 1, 1))

        # Load metadata if path is provided (mimics Movie.cs constructor behavior)
        if path is not None:
            from .io import load_meta
            load_meta(self, self.xml_path)

    @property
    def name(self) -> str:
        """Get filename with extension from path"""
        if not self.path:
            return ""
        return Path(self.path).name

    @property
    def root_name(self) -> str:
        """Get root name from path (filename without extension)"""
        if not self.path:
            return ""
        return Path(self.path).stem

    @property
    def processing_directory_name(self) -> str:
        """Get processing directory from path"""
        if not self.path:
            return ""
        return str(Path(self.path).parent)

    @property
    def data_or_processing_directory_name(self) -> str:
        """Get data directory if set, otherwise processing directory"""
        if self.data_directory_name:
            return self.data_directory_name
        return self.processing_directory_name

    @property
    def data_path(self) -> str:
        """Get data path (either in data directory or processing directory)"""
        if self.data_directory_name:
            return str(Path(self.data_directory_name) / self.name)
        return self.path

    @property
    def xml_name(self) -> str:
        """Get XML filename"""
        if not self.root_name:
            return ""
        return f"{self.root_name}.xml"

    @property
    def xml_path(self) -> str:
        """Get full XML path"""
        if not self.processing_directory_name or not self.xml_name:
            return ""
        return str(Path(self.processing_directory_name) / self.xml_name)