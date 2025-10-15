"""Tests for CTF class."""

import torch
from lxml import etree
import pytest
from warpylib import CTF


def test_ctf_default_initialization():
    """Test that CTF initializes with correct default values."""
    ctf = CTF()

    assert ctf.pixel_size == 1.0
    assert ctf.pixel_size_delta_percent == 0.0
    assert ctf.pixel_size_angle == 0.0
    assert ctf.cs == 2.7
    assert ctf.cc == 2.7
    assert ctf.voltage == 300.0
    assert ctf.defocus == 1.0
    assert ctf.defocus_delta == 0.0
    assert ctf.defocus_angle == 0.0
    assert ctf.amplitude == 0.07
    assert ctf.phase_shift == 0.0
    assert ctf.bfactor == 0.0
    assert ctf.bfactor_delta == 0.0
    assert ctf.bfactor_angle == 0.0
    assert ctf.scale == 1.0
    assert ctf.thickness == 0.0

    # Check tensor shapes and values
    assert ctf.distortion.shape == (4,)
    assert torch.allclose(ctf.distortion, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert ctf.beam_tilt.shape == (2,)
    assert torch.allclose(ctf.beam_tilt, torch.zeros(2))
    assert ctf.beam_tilt2.shape == (3,)
    assert torch.allclose(ctf.beam_tilt2, torch.tensor([1.0, 0.0, 1.0]))
    assert len(ctf.zernike_coeffs_odd) == 0
    assert len(ctf.zernike_coeffs_even) == 0


def test_ctf_copy():
    """Test that get_copy creates an independent copy."""
    ctf1 = CTF()
    ctf1.pixel_size = 2.5
    ctf1.voltage = 200.0
    ctf1.defocus = 3.0
    ctf1.zernike_coeffs_odd = torch.tensor([0.1, 0.2, 0.3])

    ctf2 = ctf1.get_copy()

    # Check values are copied
    assert ctf2.pixel_size == 2.5
    assert ctf2.voltage == 200.0
    assert ctf2.defocus == 3.0
    assert torch.allclose(ctf2.zernike_coeffs_odd, torch.tensor([0.1, 0.2, 0.3]))

    # Modify copy and ensure original is unchanged
    ctf2.pixel_size = 5.0
    ctf2.zernike_coeffs_odd[0] = 999.0

    assert ctf1.pixel_size == 2.5
    assert ctf1.zernike_coeffs_odd[0] == 0.1


def test_ctf_xml_serialization():
    """Test saving and loading CTF to/from XML."""
    # Create a CTF with non-default values
    ctf1 = CTF()
    ctf1.pixel_size = 2.5
    ctf1.pixel_size_delta_percent = 1.5
    ctf1.pixel_size_angle = 45.0
    ctf1.cs = 2.0
    ctf1.cc = 2.5
    ctf1.voltage = 200.0
    ctf1.defocus = 3.0
    ctf1.defocus_delta = 0.5
    ctf1.defocus_angle = 90.0
    ctf1.amplitude = 0.1
    ctf1.phase_shift = 0.5
    ctf1.bfactor = 100.0
    ctf1.bfactor_delta = 50.0
    ctf1.bfactor_angle = 45.0
    ctf1.scale = 0.8
    ctf1.thickness = 200.0
    ctf1.distortion = torch.tensor([1.1, 0.1, 0.05, 1.05])
    ctf1.beam_tilt = torch.tensor([0.5, 0.3])
    ctf1.beam_tilt2 = torch.tensor([1.1, 0.2, 0.9])
    ctf1.zernike_coeffs_odd = torch.tensor([0.1, 0.2, 0.3])
    ctf1.zernike_coeffs_even = torch.tensor([0.4, 0.5])

    # Save to XML element
    root = etree.Element("CTF")
    ctf1.save_to_xml(root)

    # Load from XML element
    ctf2 = CTF.load_from_xml(root)

    # Check all values match (using approximate equality for floats)
    assert abs(ctf2.pixel_size - 2.5) < 1e-6
    assert abs(ctf2.pixel_size_delta_percent - 1.5) < 1e-6
    assert abs(ctf2.pixel_size_angle - 45.0) < 1e-6
    assert abs(ctf2.cs - 2.0) < 1e-6
    assert abs(ctf2.cc - 2.5) < 1e-6
    assert abs(ctf2.voltage - 200.0) < 1e-6
    assert abs(ctf2.defocus - 3.0) < 1e-6
    assert abs(ctf2.defocus_delta - 0.5) < 1e-6
    assert abs(ctf2.defocus_angle - 90.0) < 1e-6
    assert abs(ctf2.amplitude - 0.1) < 1e-6
    assert abs(ctf2.phase_shift - 0.5) < 1e-6
    assert abs(ctf2.bfactor - 100.0) < 1e-6
    assert abs(ctf2.bfactor_delta - 50.0) < 1e-6
    assert abs(ctf2.bfactor_angle - 45.0) < 1e-6
    assert abs(ctf2.scale - 0.8) < 1e-6
    assert abs(ctf2.thickness - 200.0) < 1e-6

    assert torch.allclose(ctf2.distortion, ctf1.distortion, atol=1e-6)
    assert torch.allclose(ctf2.beam_tilt, ctf1.beam_tilt, atol=1e-6)
    assert torch.allclose(ctf2.beam_tilt2, ctf1.beam_tilt2, atol=1e-6)
    assert torch.allclose(ctf2.zernike_coeffs_odd, ctf1.zernike_coeffs_odd, atol=1e-6)
    assert torch.allclose(ctf2.zernike_coeffs_even, ctf1.zernike_coeffs_even, atol=1e-6)


def test_ctf_xml_partial_loading():
    """Test loading CTF with missing optional attributes."""
    # Create minimal XML element with only some Param elements
    root = etree.Element("CTF")

    # Add Param elements (not attributes)
    param1 = etree.SubElement(root, "Param")
    param1.set("Name", "PixelSize")
    param1.set("Value", "2.5")

    param2 = etree.SubElement(root, "Param")
    param2.set("Name", "Voltage")
    param2.set("Value", "200")

    param3 = etree.SubElement(root, "Param")
    param3.set("Name", "Defocus")
    param3.set("Value", "3.0")

    ctf = CTF.load_from_xml(root)

    # Check loaded values
    assert abs(ctf.pixel_size - 2.5) < 1e-6
    assert abs(ctf.voltage - 200.0) < 1e-6
    assert abs(ctf.defocus - 3.0) < 1e-6

    # Check defaults for missing values
    assert ctf.pixel_size_delta_percent == 0.0
    assert ctf.cs == 2.7
    assert ctf.amplitude == 0.07


def test_ctf_repr():
    """Test string representation of CTF."""
    ctf = CTF()
    repr_str = repr(ctf)

    assert "CTF(" in repr_str
    assert "pixel_size" in repr_str
    assert "voltage" in repr_str
    assert "defocus" in repr_str
    assert "amplitude" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
