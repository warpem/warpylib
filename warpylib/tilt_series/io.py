"""
TiltSeries I/O - XML and STAR file operations

This module contains methods for loading and saving TiltSeries metadata
from XML and STAR files.
"""

from typing import Union
from pathlib import Path
import torch
from lxml import etree
import pandas as pd
import starfile
from ..cubic_grid import CubicGrid
from ..ctf import CTF


def initialize_from_tomo_star(ts: "TiltSeries", star_table: Union[str, pd.DataFrame]) -> None:
    """
    Initialize tilt series parameters from a STAR file (commonly used in cryo-EM).

    This method reads per-tilt metadata from a STAR table and populates the
    corresponding fields (angles, dose, axis angles, offsets, and movie paths).

    Args:
        ts: TiltSeries instance to initialize
        star_table: Either a path to a STAR file or a pandas DataFrame already
                   loaded from a STAR file using the starfile package

    Raises:
        ValueError: If required columns are missing or if the table is empty
    """
    # Load STAR file if path is provided
    if isinstance(star_table, str):
        star_table = starfile.read(star_table)

    # Check for required columns
    if "wrpDose" not in star_table.columns or "wrpAngleTilt" not in star_table.columns:
        raise ValueError("STAR file must contain 'wrpDose' and 'wrpAngleTilt' columns")

    n_rows = len(star_table)
    if n_rows == 0:
        raise ValueError("STAR table is empty")

    # Extract required columns
    angles_list = star_table["wrpAngleTilt"].tolist()
    dose_list = star_table["wrpDose"].tolist()

    # Extract optional columns with defaults
    axis_angles_list = (
        star_table["wrpAxisAngle"].tolist()
        if "wrpAxisAngle" in star_table.columns
        else [0.0] * n_rows
    )

    offset_x_list = (
        star_table["wrpAxisOffsetX"].tolist()
        if "wrpAxisOffsetX" in star_table.columns
        else [0.0] * n_rows
    )

    offset_y_list = (
        star_table["wrpAxisOffsetY"].tolist()
        if "wrpAxisOffsetY" in star_table.columns
        else [0.0] * n_rows
    )

    movie_paths_list = (
        star_table["wrpMovieName"].tolist()
        if "wrpMovieName" in star_table.columns
        else [""] * n_rows
    )

    # Validate that we have at least the essential metadata
    if not angles_list or not dose_list:
        raise ValueError(
            "Metadata must contain at least tilt angles and accumulated dose"
        )

    # Update instance attributes
    ts.angles = torch.tensor(angles_list, dtype=torch.float32)
    ts.dose = torch.tensor(dose_list, dtype=torch.float32)
    ts.tilt_axis_angles = torch.tensor(axis_angles_list, dtype=torch.float32)
    ts.tilt_axis_offset_x = torch.tensor(offset_x_list, dtype=torch.float32)
    ts.tilt_axis_offset_y = torch.tensor(offset_y_list, dtype=torch.float32)
    ts.tilt_movie_paths = movie_paths_list
    ts.use_tilt = torch.ones(n_rows, dtype=torch.bool)
    ts.fov_fraction = torch.ones(n_rows, dtype=torch.float32)


def load_meta(ts: "TiltSeries", xml_path: str) -> None:
    """
    Load metadata from XML file.

    Args:
        ts: TiltSeries instance to load into
        xml_path: Path to XML file
    """
    if not Path(xml_path).exists():
        return

    try:
        tree = etree.parse(xml_path)
        root = tree.getroot()

        # Load data directory if present
        data_dir = root.get("DataDirectory")
        if data_dir:
            ts.data_directory_name = data_dir

        # Load geometry attributes
        ts.are_angles_inverted = root.get("AreAnglesInverted", "False") == "True"

        plane_normal_str = root.get("PlaneNormal")
        if plane_normal_str:
            values = [float(x) for x in plane_normal_str.split(",")]
            ts.plane_normal = torch.tensor(values, dtype=torch.float32)

        level_x = root.get("LevelAngleX")
        if level_x:
            ts.level_angle_x = float(level_x)

        level_y = root.get("LevelAngleY")
        if level_y:
            ts.level_angle_y = float(level_y)

        # Load global parameters
        bfactor = root.get("Bfactor")
        if bfactor:
            ts.global_bfactor = float(bfactor)

        weight = root.get("Weight")
        if weight:
            ts.global_weight = float(weight)

        mag_correction = root.get("MagnificationCorrection")
        if mag_correction:
            # Parse as Matrix2 (4 values)
            values = [float(x.strip()) for x in mag_correction.split(",")]
            if len(values) == 4:
                ts.magnification_correction = torch.tensor([[values[0], values[1]],
                                                           [values[2], values[3]]], dtype=torch.float32)

        volume_dims = root.get("VolumeDimensionsAngstrom")
        if volume_dims:
            values = [float(x) for x in volume_dims.split(",")]
            ts.volume_dimensions_physical = torch.tensor(values, dtype=torch.float32)

        image_dims = root.get("ImageDimensionsAngstrom")
        if image_dims:
            values = [float(x) for x in image_dims.split(",")]
            ts.image_dimensions_physical = torch.tensor(values, dtype=torch.float32)

        unselect_filter = root.get("UnselectFilter")
        if unselect_filter:
            ts.unselect_filter = unselect_filter == "True"

        unselect_manual = root.get("UnselectManual")
        if unselect_manual and unselect_manual != "null":
            ts.unselect_manual = unselect_manual == "True"

        ctf_res = root.get("CTFResolutionEstimate")
        if ctf_res:
            ts.ctf_resolution_estimate = float(ctf_res)

        # Load per-tilt parameters
        angles_elem = root.find("Angles")
        if angles_elem is not None and angles_elem.text:
            angles = [float(x) for x in angles_elem.text.strip().split("\n") if x.strip()]
            ts.angles = torch.tensor(angles, dtype=torch.float32)

        dose_elem = root.find("Dose")
        if dose_elem is not None and dose_elem.text:
            doses = [float(x) for x in dose_elem.text.strip().split("\n") if x.strip()]
            ts.dose = torch.tensor(doses, dtype=torch.float32)
        else:
            ts.dose = torch.zeros(len(ts.angles), dtype=torch.float32)

        use_tilt_elem = root.find("UseTilt")
        if use_tilt_elem is not None and use_tilt_elem.text:
            use_tilts = [x.strip() == "True" for x in use_tilt_elem.text.strip().split("\n") if x.strip()]
            ts.use_tilt = torch.tensor(use_tilts, dtype=torch.bool)
        else:
            ts.use_tilt = torch.ones(len(ts.angles), dtype=torch.bool)

        axis_angle_elem = root.find("AxisAngle")
        if axis_angle_elem is not None and axis_angle_elem.text:
            axis_angles = [float(x) for x in axis_angle_elem.text.strip().split("\n") if x.strip()]
            ts.tilt_axis_angles = torch.tensor(axis_angles, dtype=torch.float32)
        else:
            ts.tilt_axis_angles = torch.zeros(len(ts.angles), dtype=torch.float32)

        axis_offset_x_elem = root.find("AxisOffsetX")
        if axis_offset_x_elem is not None and axis_offset_x_elem.text:
            offsets_x = [float(x) for x in axis_offset_x_elem.text.strip().split("\n") if x.strip()]
            ts.tilt_axis_offset_x = torch.tensor(offsets_x, dtype=torch.float32)
        else:
            ts.tilt_axis_offset_x = torch.zeros(len(ts.angles), dtype=torch.float32)

        axis_offset_y_elem = root.find("AxisOffsetY")
        if axis_offset_y_elem is not None and axis_offset_y_elem.text:
            offsets_y = [float(x) for x in axis_offset_y_elem.text.strip().split("\n") if x.strip()]
            ts.tilt_axis_offset_y = torch.tensor(offsets_y, dtype=torch.float32)
        else:
            ts.tilt_axis_offset_y = torch.zeros(len(ts.angles), dtype=torch.float32)

        movie_path_elem = root.find("MoviePath")
        if movie_path_elem is not None and movie_path_elem.text:
            ts.tilt_movie_paths = [x.strip() for x in movie_path_elem.text.strip().split("\n") if x.strip()]
        else:
            ts.tilt_movie_paths = [""] * len(ts.angles)

        fov_elem = root.find("FOVFraction")
        if fov_elem is not None and fov_elem.text:
            fov_fractions = [float(x) for x in fov_elem.text.strip().split("\n") if x.strip()]
            ts.fov_fraction = torch.tensor(fov_fractions, dtype=torch.float32)
        else:
            ts.fov_fraction = torch.ones(len(ts.angles), dtype=torch.float32)

        # Load CTF
        ctf_elem = root.find("CTF")
        if ctf_elem is not None:
            ts.ctf = CTF.load_from_xml(ctf_elem)

        # Load grids
        grid_ctf = root.find("GridCTF")
        if grid_ctf is not None:
            ts.grid_ctf_defocus = CubicGrid.load_from_xml(grid_ctf)

        grid_ctf_delta = root.find("GridCTFDefocusDelta")
        if grid_ctf_delta is not None:
            ts.grid_ctf_defocus_delta = CubicGrid.load_from_xml(grid_ctf_delta)

        grid_ctf_angle = root.find("GridCTFDefocusAngle")
        if grid_ctf_angle is not None:
            ts.grid_ctf_defocus_angle = CubicGrid.load_from_xml(grid_ctf_angle)

        grid_ctf_phase = root.find("GridCTFPhase")
        if grid_ctf_phase is not None:
            ts.grid_ctf_phase = CubicGrid.load_from_xml(grid_ctf_phase)

        grid_move_x = root.find("GridMovementX")
        if grid_move_x is not None:
            ts.grid_movement_x = CubicGrid.load_from_xml(grid_move_x)

        grid_move_y = root.find("GridMovementY")
        if grid_move_y is not None:
            ts.grid_movement_y = CubicGrid.load_from_xml(grid_move_y)

        grid_vol_warp_x = root.find("GridVolumeWarpX")
        if grid_vol_warp_x is not None:
            ts.grid_volume_warp_x = CubicGrid.load_from_xml(grid_vol_warp_x)

        grid_vol_warp_y = root.find("GridVolumeWarpY")
        if grid_vol_warp_y is not None:
            ts.grid_volume_warp_y = CubicGrid.load_from_xml(grid_vol_warp_y)

        grid_vol_warp_z = root.find("GridVolumeWarpZ")
        if grid_vol_warp_z is not None:
            ts.grid_volume_warp_z = CubicGrid.load_from_xml(grid_vol_warp_z)

        grid_angle_x = root.find("GridAngleX")
        if grid_angle_x is not None:
            ts.grid_angle_x = CubicGrid.load_from_xml(grid_angle_x)

        grid_angle_y = root.find("GridAngleY")
        if grid_angle_y is not None:
            ts.grid_angle_y = CubicGrid.load_from_xml(grid_angle_y)

        grid_angle_z = root.find("GridAngleZ")
        if grid_angle_z is not None:
            ts.grid_angle_z = CubicGrid.load_from_xml(grid_angle_z)

        grid_dose_bfacs = root.find("GridDoseBfacs")
        if grid_dose_bfacs is not None:
            ts.grid_dose_bfacs = CubicGrid.load_from_xml(grid_dose_bfacs)

        grid_dose_bfacs_delta = root.find("GridDoseBfacsDelta")
        if grid_dose_bfacs_delta is not None:
            ts.grid_dose_bfacs_delta = CubicGrid.load_from_xml(grid_dose_bfacs_delta)

        grid_dose_bfacs_angle = root.find("GridDoseBfacsAngle")
        if grid_dose_bfacs_angle is not None:
            ts.grid_dose_bfacs_angle = CubicGrid.load_from_xml(grid_dose_bfacs_angle)

        grid_dose_weights = root.find("GridDoseWeights")
        if grid_dose_weights is not None:
            ts.grid_dose_weights = CubicGrid.load_from_xml(grid_dose_weights)

        grid_location_bfacs = root.find("GridLocationBfacs")
        if grid_location_bfacs is not None:
            ts.grid_location_bfacs = CubicGrid.load_from_xml(grid_location_bfacs)

        grid_location_weights = root.find("GridLocationWeights")
        if grid_location_weights is not None:
            ts.grid_location_weights = CubicGrid.load_from_xml(grid_location_weights)

    except Exception as e:
        print(f"Error loading metadata: {e}")
        return


def save_meta(ts: "TiltSeries", xml_path: str = None) -> None:
    """
    Save metadata to XML file.

    Args:
        ts: TiltSeries instance to save
        xml_path: Path to XML file
    """

    if xml_path is None:
        xml_path = ts.xml_path

    # Create parent directory if needed
    Path(xml_path).parent.mkdir(parents=True, exist_ok=True)

    root = etree.Element("TiltSeries")

    # Save data directory if present
    if ts.data_directory_name:
        root.set("DataDirectory", ts.data_directory_name)

    # Save geometry attributes
    root.set("AreAnglesInverted", str(ts.are_angles_inverted))
    root.set("PlaneNormal", f"{ts.plane_normal[0]:.9g}, {ts.plane_normal[1]:.9g}, {ts.plane_normal[2]:.9g}")
    root.set("LevelAngleX", f"{ts.level_angle_x:.9g}")
    root.set("LevelAngleY", f"{ts.level_angle_y:.9g}")

    # Save global parameters
    root.set("Bfactor", f"{ts.global_bfactor:.9g}")
    root.set("Weight", f"{ts.global_weight:.9g}")

    # Save magnification correction as Matrix2 (4 values)
    mag = ts.magnification_correction
    root.set("MagnificationCorrection", f"{mag[0,0]:.9g}, {mag[0,1]:.9g}, {mag[1,0]:.9g}, {mag[1,1]:.9g}")

    root.set("VolumeDimensionsAngstrom",
             f"{ts.volume_dimensions_physical[0]:.9g}, {ts.volume_dimensions_physical[1]:.9g}, {ts.volume_dimensions_physical[2]:.9g}")
    root.set("ImageDimensionsAngstrom",
             f"{ts.image_dimensions_physical[0]:.9g}, {ts.image_dimensions_physical[1]:.9g}")

    root.set("UnselectFilter", str(ts.unselect_filter))
    if ts.unselect_manual is not None:
        root.set("UnselectManual", str(ts.unselect_manual))
    else:
        root.set("UnselectManual", "")
    root.set("CTFResolutionEstimate", f"{ts.ctf_resolution_estimate:.9g}")

    # Save per-tilt parameters
    angles_elem = etree.SubElement(root, "Angles")
    angles_elem.text = "\n".join([f"{x:.9g}" for x in ts.angles.tolist()])

    dose_elem = etree.SubElement(root, "Dose")
    dose_elem.text = "\n".join([f"{x:.9g}" for x in ts.dose.tolist()])

    use_tilt_elem = etree.SubElement(root, "UseTilt")
    use_tilt_elem.text = "\n".join([str(bool(x)) for x in ts.use_tilt.tolist()])

    axis_angle_elem = etree.SubElement(root, "AxisAngle")
    axis_angle_elem.text = "\n".join([f"{x:.9g}" for x in ts.tilt_axis_angles.tolist()])

    axis_offset_x_elem = etree.SubElement(root, "AxisOffsetX")
    axis_offset_x_elem.text = "\n".join([f"{x:.9g}" for x in ts.tilt_axis_offset_x.tolist()])

    axis_offset_y_elem = etree.SubElement(root, "AxisOffsetY")
    axis_offset_y_elem.text = "\n".join([f"{x:.9g}" for x in ts.tilt_axis_offset_y.tolist()])

    movie_path_elem = etree.SubElement(root, "MoviePath")
    movie_path_elem.text = "\n".join(ts.tilt_movie_paths)

    fov_elem = etree.SubElement(root, "FOVFraction")
    fov_elem.text = "\n".join([f"{x:.9g}" for x in ts.fov_fraction.tolist()])

    # Save CTF
    ctf_elem = etree.SubElement(root, "CTF")
    ts.ctf.save_to_xml(ctf_elem)

    # Save grids
    grid_ctf = etree.SubElement(root, "GridCTF")
    ts.grid_ctf_defocus.save_to_xml(grid_ctf)

    grid_ctf_delta = etree.SubElement(root, "GridCTFDefocusDelta")
    ts.grid_ctf_defocus_delta.save_to_xml(grid_ctf_delta)

    grid_ctf_angle = etree.SubElement(root, "GridCTFDefocusAngle")
    ts.grid_ctf_defocus_angle.save_to_xml(grid_ctf_angle)

    grid_ctf_phase = etree.SubElement(root, "GridCTFPhase")
    ts.grid_ctf_phase.save_to_xml(grid_ctf_phase)

    grid_move_x = etree.SubElement(root, "GridMovementX")
    ts.grid_movement_x.save_to_xml(grid_move_x)

    grid_move_y = etree.SubElement(root, "GridMovementY")
    ts.grid_movement_y.save_to_xml(grid_move_y)

    grid_vol_warp_x = etree.SubElement(root, "GridVolumeWarpX")
    ts.grid_volume_warp_x.save_to_xml(grid_vol_warp_x)

    grid_vol_warp_y = etree.SubElement(root, "GridVolumeWarpY")
    ts.grid_volume_warp_y.save_to_xml(grid_vol_warp_y)

    grid_vol_warp_z = etree.SubElement(root, "GridVolumeWarpZ")
    ts.grid_volume_warp_z.save_to_xml(grid_vol_warp_z)

    grid_angle_x = etree.SubElement(root, "GridAngleX")
    ts.grid_angle_x.save_to_xml(grid_angle_x)

    grid_angle_y = etree.SubElement(root, "GridAngleY")
    ts.grid_angle_y.save_to_xml(grid_angle_y)

    grid_angle_z = etree.SubElement(root, "GridAngleZ")
    ts.grid_angle_z.save_to_xml(grid_angle_z)

    grid_dose_bfacs = etree.SubElement(root, "GridDoseBfacs")
    ts.grid_dose_bfacs.save_to_xml(grid_dose_bfacs)

    grid_dose_bfacs_delta = etree.SubElement(root, "GridDoseBfacsDelta")
    ts.grid_dose_bfacs_delta.save_to_xml(grid_dose_bfacs_delta)

    grid_dose_bfacs_angle = etree.SubElement(root, "GridDoseBfacsAngle")
    ts.grid_dose_bfacs_angle.save_to_xml(grid_dose_bfacs_angle)

    grid_dose_weights = etree.SubElement(root, "GridDoseWeights")
    ts.grid_dose_weights.save_to_xml(grid_dose_weights)

    grid_location_bfacs = etree.SubElement(root, "GridLocationBfacs")
    ts.grid_location_bfacs.save_to_xml(grid_location_bfacs)

    grid_location_weights = etree.SubElement(root, "GridLocationWeights")
    ts.grid_location_weights.save_to_xml(grid_location_weights)

    # Write to file with pretty formatting
    tree = etree.ElementTree(root)
    tree.write(
        xml_path,
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8",
    )
