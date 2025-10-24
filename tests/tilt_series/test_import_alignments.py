"""
Test import_alignments functionality by comparing against reference XML
"""

import torch
import pytest
from pathlib import Path
from warpylib.tilt_series import TiltSeries


def test_import_alignments_matches_csharp():
    """
    Test that import_alignments produces the same results as the C# code.

    The test data contains:
    - 00254.xf and 00254.tlt: Input alignment files
    - 00254.xml: Reference XML produced by C# ImportAlignments

    We create a fresh TiltSeries, run import_alignments, and compare
    against the values loaded from the reference XML.
    """
    test_dir = Path(__file__).parent.parent.parent / "testdata" / "xf_import"
    xf_path = test_dir / "00254.xf"
    tlt_path = test_dir / "00254.tlt"
    xml_path = test_dir / "00254.xml"

    if not xf_path.exists():
        pytest.skip(f"Test data not found: {xf_path}")

    # Load reference values from XML (produced by C# code)
    ts_reference = TiltSeries()
    ts_reference.load_meta(str(xml_path))

    print(f"\nReference (from C# import):")
    print(f"  Number of tilts: {ts_reference.n_tilts}")
    print(f"  First 3 angles: {ts_reference.angles[:3].tolist()}")
    print(f"  First 3 tilt axis angles: {ts_reference.tilt_axis_angles[:3].tolist()}")
    print(f"  First 3 offset X: {ts_reference.tilt_axis_offset_x[:3].tolist()}")
    print(f"  First 3 offset Y: {ts_reference.tilt_axis_offset_y[:3].tolist()}")

    # Create a fresh TiltSeries with the same number of tilts
    n_tilts = ts_reference.n_tilts
    ts_python = TiltSeries(n_tilts=n_tilts)

    # Copy use_tilt from reference (should all be True, but let's be sure)
    ts_python.use_tilt = ts_reference.use_tilt.clone()

    # Need to determine binned_pixel_size_mean
    # Looking at the offsets in the XML and .xf file, we can reverse-engineer this
    # From C# code: Shift *= (float)options.BinnedPixelSizeMean;
    # Let's try a common value for EMPIAR-10499 dataset
    binned_pixel_size = 10.0  # Angstroms (typical for 4x binned data)

    # Run Python import_alignments
    ts_python.import_alignments(str(xf_path), binned_pixel_size, tlt_path=str(tlt_path))

    print(f"\nPython import_alignments result:")
    print(f"  Number of tilts: {ts_python.n_tilts}")
    print(f"  First 3 angles: {ts_python.angles[:3].tolist()}")
    print(f"  First 3 tilt axis angles: {ts_python.tilt_axis_angles[:3].tolist()}")
    print(f"  First 3 offset X: {ts_python.tilt_axis_offset_x[:3].tolist()}")
    print(f"  First 3 offset Y: {ts_python.tilt_axis_offset_y[:3].tolist()}")

    # Compare angles (should match exactly)
    print("\n=== Comparing angles ===")
    angle_diff = torch.abs(ts_python.angles - ts_reference.angles)
    print(f"Max angle difference: {angle_diff.max().item():.6f} degrees")
    assert torch.allclose(ts_python.angles, ts_reference.angles, atol=0.01), \
        f"Angles don't match! Max diff: {angle_diff.max()}"

    # Compare tilt axis angles (rotation from .xf)
    print("\n=== Comparing tilt axis angles ===")
    axis_angle_diff = torch.abs(ts_python.tilt_axis_angles - ts_reference.tilt_axis_angles)
    print(f"Max axis angle difference: {axis_angle_diff.max().item():.6f} degrees")
    print(f"Mean axis angle difference: {axis_angle_diff.mean().item():.6f} degrees")
    assert torch.allclose(ts_python.tilt_axis_angles, ts_reference.tilt_axis_angles, atol=0.01), \
        f"Axis angles don't match! Max diff: {axis_angle_diff.max()}"

    # Compare offsets (these depend on binned_pixel_size being correct)
    print("\n=== Comparing offsets ===")
    offset_x_diff = torch.abs(ts_python.tilt_axis_offset_x - ts_reference.tilt_axis_offset_x)
    offset_y_diff = torch.abs(ts_python.tilt_axis_offset_y - ts_reference.tilt_axis_offset_y)
    print(f"Max offset X difference: {offset_x_diff.max().item():.3f} Å")
    print(f"Max offset Y difference: {offset_y_diff.max().item():.3f} Å")
    print(f"Mean offset X difference: {offset_x_diff.mean().item():.3f} Å")
    print(f"Mean offset Y difference: {offset_y_diff.mean().item():.3f} Å")

    # Offsets should match within a reasonable tolerance
    # Allow slightly larger tolerance for offsets as they involve more computation
    assert torch.allclose(ts_python.tilt_axis_offset_x, ts_reference.tilt_axis_offset_x, atol=0.1), \
        f"Offset X doesn't match! Max diff: {offset_x_diff.max()}"
    assert torch.allclose(ts_python.tilt_axis_offset_y, ts_reference.tilt_axis_offset_y, atol=0.1), \
        f"Offset Y doesn't match! Max diff: {offset_y_diff.max()}"

    print("\n✓ All values match! Python import_alignments produces identical results to C#")


def test_import_alignments_without_tlt():
    """Test that we can import only .xf without .tlt"""
    test_dir = Path(__file__).parent.parent.parent / "testdata" / "xf_import"
    xf_path = test_dir / "00254.xf"
    xml_path = test_dir / "00254.xml"

    if not xf_path.exists():
        pytest.skip(f"Test data not found: {xf_path}")

    # Load reference to get the number of tilts
    ts_reference = TiltSeries()
    ts_reference.load_meta(str(xml_path))

    # Create fresh TiltSeries with custom angles
    n_tilts = ts_reference.n_tilts
    ts = TiltSeries(n_tilts=n_tilts)
    custom_angles = torch.linspace(-60, 60, n_tilts)
    ts.angles = custom_angles.clone()

    # Import only transforms (no tlt_path)
    ts.import_alignments(str(xf_path), 10.0)

    # Angles should be unchanged
    assert torch.allclose(ts.angles, custom_angles), \
        "Angles should not change when tlt_path is not provided"

    # But transforms should be loaded
    assert not torch.all(ts.tilt_axis_angles == 0), \
        "Tilt axis angles should be loaded from .xf"
    assert torch.allclose(ts.tilt_axis_angles, ts_reference.tilt_axis_angles, atol=0.01), \
        "Tilt axis angles should match reference"

    print("✓ Import without .tlt works correctly")


if __name__ == "__main__":
    test_import_alignments_matches_csharp()
    test_import_alignments_without_tlt()
    print("\nAll tests passed!")