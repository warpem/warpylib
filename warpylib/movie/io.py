"""
Movie I/O - XML operations

This module contains methods for loading and saving Movie metadata
from XML files.
"""

from pathlib import Path
from lxml import etree
from ..cubic_grid import CubicGrid
from ..ctf import CTF


def load_meta(movie: "Movie", xml_path: str) -> None:
    """
    Load metadata from XML file.

    Args:
        movie: Movie instance to load into
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
            movie.data_directory_name = data_dir

        # Load global parameters
        unselect_manual_str = root.get("UnselectManual")
        if unselect_manual_str and unselect_manual_str != "null":
            movie.unselect_manual = unselect_manual_str == "True"
        else:
            movie.unselect_manual = None

        unselect_filter = root.get("UnselectFilter")
        if unselect_filter:
            movie.unselect_filter = unselect_filter == "True"

        ctf_res = root.get("CTFResolutionEstimate")
        if ctf_res:
            movie.ctf_resolution_estimate = float(ctf_res)

        mean_movement = root.get("MeanFrameMovement")
        if mean_movement:
            movie.mean_frame_movement = float(mean_movement)

        mask_pct = root.get("MaskPercentage")
        if mask_pct:
            movie.mask_percentage = float(mask_pct)

        bfactor = root.get("Bfactor")
        if bfactor:
            movie.global_bfactor = float(bfactor)

        weight = root.get("Weight")
        if weight:
            movie.global_weight = float(weight)

        mag_correction = root.get("MagnificationCorrection")
        if mag_correction:
            # Parse as Matrix2 (4 values)
            import torch
            values = [float(x.strip()) for x in mag_correction.strip("()").split(",")]
            if len(values) == 4:
                movie.magnification_correction = torch.tensor([[values[0], values[1]],
                                                               [values[2], values[3]]], dtype=torch.float32)

        # Load CTF
        ctf_elem = root.find("CTF")
        if ctf_elem is not None:
            movie.ctf = CTF.load_from_xml(ctf_elem)

        # Load CTF grids
        grid_ctf = root.find("GridCTF")
        if grid_ctf is not None:
            movie.grid_ctf_defocus = CubicGrid.load_from_xml(grid_ctf)

        grid_ctf_delta = root.find("GridCTFDefocusDelta")
        if grid_ctf_delta is not None:
            movie.grid_ctf_defocus_delta = CubicGrid.load_from_xml(grid_ctf_delta)

        grid_ctf_angle = root.find("GridCTFDefocusAngle")
        if grid_ctf_angle is not None:
            movie.grid_ctf_defocus_angle = CubicGrid.load_from_xml(grid_ctf_angle)

        grid_ctf_cs = root.find("GridCTFCs")
        if grid_ctf_cs is not None:
            movie.grid_ctf_cs = CubicGrid.load_from_xml(grid_ctf_cs)

        grid_ctf_phase = root.find("GridCTFPhase")
        if grid_ctf_phase is not None:
            movie.grid_ctf_phase = CubicGrid.load_from_xml(grid_ctf_phase)

        grid_ctf_doming = root.find("GridCTFDoming")
        if grid_ctf_doming is not None:
            movie.grid_ctf_doming = CubicGrid.load_from_xml(grid_ctf_doming)

        # Load motion grids
        grid_move_x = root.find("GridMovementX")
        if grid_move_x is not None:
            movie.grid_movement_x = CubicGrid.load_from_xml(grid_move_x)

        grid_move_y = root.find("GridMovementY")
        if grid_move_y is not None:
            movie.grid_movement_y = CubicGrid.load_from_xml(grid_move_y)

        grid_local_x = root.find("GridLocalMovementX")
        if grid_local_x is not None:
            movie.grid_local_x = CubicGrid.load_from_xml(grid_local_x)

        grid_local_y = root.find("GridLocalMovementY")
        if grid_local_y is not None:
            movie.grid_local_y = CubicGrid.load_from_xml(grid_local_y)

        # Load pyramid shift grids (multiple elements with same name)
        movie.pyramid_shift_x = []
        for pyramid_x_elem in root.findall("PyramidShiftX"):
            movie.pyramid_shift_x.append(CubicGrid.load_from_xml(pyramid_x_elem))

        movie.pyramid_shift_y = []
        for pyramid_y_elem in root.findall("PyramidShiftY"):
            movie.pyramid_shift_y.append(CubicGrid.load_from_xml(pyramid_y_elem))

        # Load angle grids
        grid_angle_x = root.find("GridAngleX")
        if grid_angle_x is not None:
            movie.grid_angle_x = CubicGrid.load_from_xml(grid_angle_x)

        grid_angle_y = root.find("GridAngleY")
        if grid_angle_y is not None:
            movie.grid_angle_y = CubicGrid.load_from_xml(grid_angle_y)

        grid_angle_z = root.find("GridAngleZ")
        if grid_angle_z is not None:
            movie.grid_angle_z = CubicGrid.load_from_xml(grid_angle_z)

        # Load dose grids
        grid_dose_bfacs = root.find("GridDoseBfacs")
        if grid_dose_bfacs is not None:
            movie.grid_dose_bfacs = CubicGrid.load_from_xml(grid_dose_bfacs)

        grid_dose_bfacs_delta = root.find("GridDoseBfacsDelta")
        if grid_dose_bfacs_delta is not None:
            movie.grid_dose_bfacs_delta = CubicGrid.load_from_xml(grid_dose_bfacs_delta)

        grid_dose_bfacs_angle = root.find("GridDoseBfacsAngle")
        if grid_dose_bfacs_angle is not None:
            movie.grid_dose_bfacs_angle = CubicGrid.load_from_xml(grid_dose_bfacs_angle)

        grid_dose_weights = root.find("GridDoseWeights")
        if grid_dose_weights is not None:
            movie.grid_dose_weights = CubicGrid.load_from_xml(grid_dose_weights)

        # Load location grids
        grid_location_bfacs = root.find("GridLocationBfacs")
        if grid_location_bfacs is not None:
            movie.grid_location_bfacs = CubicGrid.load_from_xml(grid_location_bfacs)

        grid_location_weights = root.find("GridLocationWeights")
        if grid_location_weights is not None:
            movie.grid_location_weights = CubicGrid.load_from_xml(grid_location_weights)

    except Exception as e:
        print(f"Error loading metadata: {e}")
        return


def save_meta(movie: "Movie", xml_path: str) -> None:
    """
    Save metadata to XML file.

    Args:
        movie: Movie instance to save
        xml_path: Path to XML file
    """
    # Create parent directory if needed
    Path(xml_path).parent.mkdir(parents=True, exist_ok=True)

    root = etree.Element("Movie")

    # Save data directory
    root.set("DataDirectory", movie.data_directory_name if movie.data_directory_name else "")

    # Save global parameters
    root.set("UnselectFilter", str(movie.unselect_filter))
    if movie.unselect_manual is not None:
        root.set("UnselectManual", str(movie.unselect_manual))
    else:
        root.set("UnselectManual", "null")

    root.set("CTFResolutionEstimate", f"{movie.ctf_resolution_estimate:.9g}")
    root.set("MeanFrameMovement", f"{movie.mean_frame_movement:.9g}")
    root.set("MaskPercentage", f"{movie.mask_percentage:.9g}")
    root.set("Bfactor", f"{movie.global_bfactor:.9g}")
    root.set("Weight", f"{movie.global_weight:.9g}")

    # Save magnification correction as Matrix2 (4 values)
    mag = movie.magnification_correction
    root.set("MagnificationCorrection", f"{mag[0,0]:.9g}, {mag[0,1]:.9g}, {mag[1,0]:.9g}, {mag[1,1]:.9g}")

    # Save CTF
    ctf_elem = etree.SubElement(root, "CTF")
    movie.ctf.save_to_xml(ctf_elem)

    # Save CTF grids
    grid_ctf = etree.SubElement(root, "GridCTF")
    movie.grid_ctf_defocus.save_to_xml(grid_ctf)

    grid_ctf_delta = etree.SubElement(root, "GridCTFDefocusDelta")
    movie.grid_ctf_defocus_delta.save_to_xml(grid_ctf_delta)

    grid_ctf_angle = etree.SubElement(root, "GridCTFDefocusAngle")
    movie.grid_ctf_defocus_angle.save_to_xml(grid_ctf_angle)

    grid_ctf_cs = etree.SubElement(root, "GridCTFCs")
    movie.grid_ctf_cs.save_to_xml(grid_ctf_cs)

    grid_ctf_phase = etree.SubElement(root, "GridCTFPhase")
    movie.grid_ctf_phase.save_to_xml(grid_ctf_phase)

    grid_ctf_doming = etree.SubElement(root, "GridCTFDoming")
    movie.grid_ctf_doming.save_to_xml(grid_ctf_doming)

    # Save motion grids
    grid_move_x = etree.SubElement(root, "GridMovementX")
    movie.grid_movement_x.save_to_xml(grid_move_x)

    grid_move_y = etree.SubElement(root, "GridMovementY")
    movie.grid_movement_y.save_to_xml(grid_move_y)

    grid_local_x = etree.SubElement(root, "GridLocalMovementX")
    movie.grid_local_x.save_to_xml(grid_local_x)

    grid_local_y = etree.SubElement(root, "GridLocalMovementY")
    movie.grid_local_y.save_to_xml(grid_local_y)

    # Save pyramid shift grids
    for grid in movie.pyramid_shift_x:
        pyramid_x_elem = etree.SubElement(root, "PyramidShiftX")
        grid.save_to_xml(pyramid_x_elem)

    for grid in movie.pyramid_shift_y:
        pyramid_y_elem = etree.SubElement(root, "PyramidShiftY")
        grid.save_to_xml(pyramid_y_elem)

    # Save angle grids
    grid_angle_x = etree.SubElement(root, "GridAngleX")
    movie.grid_angle_x.save_to_xml(grid_angle_x)

    grid_angle_y = etree.SubElement(root, "GridAngleY")
    movie.grid_angle_y.save_to_xml(grid_angle_y)

    grid_angle_z = etree.SubElement(root, "GridAngleZ")
    movie.grid_angle_z.save_to_xml(grid_angle_z)

    # Save dose grids
    grid_dose_bfacs = etree.SubElement(root, "GridDoseBfacs")
    movie.grid_dose_bfacs.save_to_xml(grid_dose_bfacs)

    grid_dose_bfacs_delta = etree.SubElement(root, "GridDoseBfacsDelta")
    movie.grid_dose_bfacs_delta.save_to_xml(grid_dose_bfacs_delta)

    grid_dose_bfacs_angle = etree.SubElement(root, "GridDoseBfacsAngle")
    movie.grid_dose_bfacs_angle.save_to_xml(grid_dose_bfacs_angle)

    grid_dose_weights = etree.SubElement(root, "GridDoseWeights")
    movie.grid_dose_weights.save_to_xml(grid_dose_weights)

    # Save location grids
    grid_location_bfacs = etree.SubElement(root, "GridLocationBfacs")
    movie.grid_location_bfacs.save_to_xml(grid_location_bfacs)

    grid_location_weights = etree.SubElement(root, "GridLocationWeights")
    movie.grid_location_weights.save_to_xml(grid_location_weights)

    # Write to file with pretty formatting
    tree = etree.ElementTree(root)
    tree.write(
        xml_path,
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8",
    )