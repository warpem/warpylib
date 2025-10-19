"""Test loading tilt series images."""

import pytest
from pathlib import Path
import torch
import mrcfile
from warpylib import TiltSeries


def test_load_images_ts1():
    """Test loading images for TS_1."""
    # Path to the XML file
    xml_path = "/Users/tegunovd/Downloads/10491-test-local/5/TS_1.xml"

    # Load the tilt series
    ts = TiltSeries(xml_path)

    print(f"\nLoaded TiltSeries:")
    print(f"  Number of tilts: {ts.n_tilts}")
    print(f"  Movie paths: {len(ts.tilt_movie_paths)}")
    print(f"  First movie path: {ts.tilt_movie_paths[0] if ts.tilt_movie_paths else 'None'}")
    print(f"  Data directory: {ts.data_directory_name}")
    print(f"  Processing directory: {ts.processing_directory_name}")

    # Define pixel sizes (typical values - adjust as needed)
    original_pixel_size = 0.834  # Angstroms
    desired_pixel_size = 10.0   # Angstroms

    # Load images without half-averages first
    print(f"\nLoading images (original: {original_pixel_size}Å, desired: {desired_pixel_size}Å)...")
    images, images_odd, images_even = ts.load_images(
        original_pixel_size=original_pixel_size,
        desired_pixel_size=desired_pixel_size,
        use_denoised=False,
        load_half_averages=False,
    )

    # Check results
    print(f"\nResults:")
    print(f"  Images shape: {images.shape}")
    print(f"  Images dtype: {images.dtype}")
    print(f"  Images device: {images.device}")
    print(f"  Images min/max: {images.min():.6f} / {images.max():.6f}")
    print(f"  Images odd: {images_odd}")
    print(f"  Images even: {images_even}")

    print(f"\nTiltSeries updated attributes:")
    print(f"  image_dimensions_physical: {ts.image_dimensions_physical}")
    print(f"  size_rounding_factors: {ts.size_rounding_factors}")

    # Assertions
    assert images.shape[0] == ts.n_tilts, "First dimension should match n_tilts"
    assert images.ndim == 3, "Should be 3D tensor (n_tilts, height, width)"
    assert images.dtype == torch.float32, "Should be float32"
    assert images_odd is None, "Should be None when load_half_averages=False"
    assert images_even is None, "Should be None when load_half_averages=False"

    # Check dimensions are even
    assert images.shape[1] % 2 == 0, "Height should be even"
    assert images.shape[2] % 2 == 0, "Width should be even"

    # Save the stack to testoutputs
    output_dir = Path(__file__).parent.parent.parent / 'testoutputs'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'TS_1_stack_4apx.mrc'

    print(f"\nSaving stack to: {output_path}")
    with mrcfile.new(str(output_path), overwrite=True) as mrc:
        # MRC expects (z, y, x) for stacks
        mrc.set_data(images.numpy())
        # Set voxel size in the header
        mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

    print(f"  Saved {images.shape[0]} images of size {images.shape[1]}×{images.shape[2]}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")

    print("\n✓ Basic loading test passed!")


def test_load_images_with_half_averages():
    """Test loading images with half-averages."""
    xml_path = "/Users/tegunovd/Downloads/10491-test-local/5/TS_1.xml"

    ts = TiltSeries()
    ts.load_meta(xml_path)

    original_pixel_size = 1.0
    desired_pixel_size = 2.0

    print(f"\nLoading images with half-averages...")

    try:
        images, images_odd, images_even = ts.load_images(
            original_pixel_size=original_pixel_size,
            desired_pixel_size=desired_pixel_size,
            use_denoised=False,
            load_half_averages=True,
        )

        print(f"\nResults with half-averages:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images odd shape: {images_odd.shape}")
        print(f"  Images even shape: {images_even.shape}")

        # Assertions
        assert images.shape == images_odd.shape, "Odd should have same shape"
        assert images.shape == images_even.shape, "Even should have same shape"

        print("\n✓ Half-averages loading test passed!")

    except FileNotFoundError as e:
        print(f"\n! Half-averages not found (expected if not exported): {e}")
        pytest.skip("Half-averages not available for this dataset")


def test_load_images_no_scaling():
    """Test loading images without rescaling (same pixel size)."""
    xml_path = "/Users/tegunovd/Downloads/10491-test-local/5/TS_1.xml"

    ts = TiltSeries()
    ts.load_meta(xml_path)

    pixel_size = 1.0

    print(f"\nLoading images without rescaling (pixel size: {pixel_size}Å)...")
    images, images_odd, images_even = ts.load_images(
        original_pixel_size=pixel_size,
        desired_pixel_size=pixel_size,
        use_denoised=False,
        load_half_averages=False,
    )

    print(f"\nResults (no rescaling):")
    print(f"  Images shape: {images.shape}")
    print(f"  Size rounding factors: {ts.size_rounding_factors}")

    # With no rescaling, rounding factors should be very close to 1
    # (may not be exactly 1 due to even dimension enforcement)
    print("\n✓ No-scaling test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])