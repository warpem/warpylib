"""
Tests for TiltSeries.reconstruct_full method
"""

import pytest
import torch
import mrcfile
from pathlib import Path

from warpylib.tilt_series import TiltSeries


class TestReconstructVolume:
    """Test reconstruct_full method"""

    def test_reconstruct_from_10491(self):
        """Write out full tomogram reconstruction to MRC file for visual inspection using real data"""
        # Setup test outputs directory
        testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs'
        testoutputs_dir.mkdir(exist_ok=True)

        # Load real tilt series metadata
        xml_path = "/Users/tegunovd/Downloads/10491-test-local/5/TS_1.xml"
        ts = TiltSeries(xml_path)

        print(f"\nLoaded TiltSeries from {xml_path}")
        print(f"  Number of tilts: {ts.n_tilts}")
        print(f"  Tilt angles: {ts.angles}")

        # Set volume dimensions based on original image size
        original_pixel_size = 0.834  # Angstroms
        volume_x = 4000 * original_pixel_size  # Angstroms
        volume_y = 5700 * original_pixel_size  # Angstroms
        volume_z = 1000 * original_pixel_size  # Angstroms

        print(f"  Volume dimensions (Å): [{volume_x}, {volume_y}, {volume_z}]")

        # Load images at desired pixel size
        desired_pixel_size = 10.0  # Angstroms
        print(f"\nLoading images at {desired_pixel_size} Å/pixel...")

        tilt_data, _, _ = ts.load_images(
            original_pixel_size=original_pixel_size,
            desired_pixel_size=desired_pixel_size,
            use_denoised=False,
            load_half_averages=False,
        )

        print(f"  Loaded images shape: {tilt_data.shape}")
        print(f"  Image dimensions (Å): {ts.image_dimensions_physical}")

        # Test 1: Full reconstruction without CTF
        print(f"\nReconstructing full tomogram (no CTF, no normalization)...")
        volume_no_ctf = ts.reconstruct_full(
            tilt_data=tilt_data,
            pixel_size=desired_pixel_size,
            volume_dimensions_physical=(volume_x, volume_y, volume_z),
            subvolume_size=64,
            subvolume_padding=2.0,
            normalize=True,
            invert=True,
            apply_ctf=False,
            ctf_weighted=False,
            batch_size=8
        )

        print(f"  Reconstruction shape: {volume_no_ctf.shape}")
        print(f"  Value range: {volume_no_ctf.min():.3f} to {volume_no_ctf.max():.3f}")

        output_path = testoutputs_dir / 'full_reconstruction_10491_no_ctf.mrc'
        print(f"  Writing to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            # MRC expects (Z, Y, X)
            mrc.set_data(volume_no_ctf.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Test 2: Full reconstruction with CTF and normalization
        print(f"\nReconstructing full tomogram (with CTF and normalization)...")
        volume_with_ctf = ts.reconstruct_full(
            tilt_data=tilt_data,
            pixel_size=desired_pixel_size,
            volume_dimensions_physical=(volume_x, volume_y, volume_z),
            subvolume_size=64,
            subvolume_padding=2.0,
            normalize=True,
            invert=True,
            apply_ctf=True,
            ctf_weighted=True,
            batch_size=8
        )

        print(f"  Reconstruction shape: {volume_with_ctf.shape}")
        print(f"  Value range: {volume_with_ctf.min():.3f} to {volume_with_ctf.max():.3f}")

        output_path = testoutputs_dir / 'full_reconstruction_10491_with_ctf.mrc'
        print(f"  Writing to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(volume_with_ctf.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Test 3: Smaller reconstruction for faster testing
        print(f"\nReconstructing smaller tomogram (normalized, with CTF)...")
        volume_small = ts.reconstruct_full(
            tilt_data=tilt_data,
            pixel_size=desired_pixel_size,
            volume_dimensions_physical=(volume_x / 2, volume_y / 2, volume_z / 2),
            subvolume_size=64,
            subvolume_padding=2.0,
            normalize=True,
            invert=True,
            apply_ctf=True,
            ctf_weighted=True,
            batch_size=8
        )

        print(f"  Reconstruction shape: {volume_small.shape}")
        print(f"  Value range: {volume_small.min():.3f} to {volume_small.max():.3f}")

        output_path = testoutputs_dir / 'full_reconstruction_10491_small.mrc'
        print(f"  Writing to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(volume_small.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        print(f"\nAll reconstructions written to {testoutputs_dir}")

        # Basic sanity checks
        assert volume_no_ctf.ndim == 3, "Volume should be 3D"
        assert volume_with_ctf.ndim == 3, "Volume should be 3D"
        assert volume_small.ndim == 3, "Volume should be 3D"

        assert torch.all(torch.isfinite(volume_no_ctf)), "All values should be finite"
        assert torch.all(torch.isfinite(volume_with_ctf)), "All values should be finite"
        assert torch.all(torch.isfinite(volume_small)), "All values should be finite"

        # Check dimensions are even
        for dim in volume_no_ctf.shape:
            assert dim % 2 == 0, "All dimensions should be even"
        for dim in volume_with_ctf.shape:
            assert dim % 2 == 0, "All dimensions should be even"
        for dim in volume_small.shape:
            assert dim % 2 == 0, "All dimensions should be even"

        # Small volume should be roughly half the size of the large ones
        assert volume_small.shape[0] < volume_no_ctf.shape[0]
        assert volume_small.shape[1] < volume_no_ctf.shape[1]
        assert volume_small.shape[2] < volume_no_ctf.shape[2]

        print("\n✓ All tests passed!")


    def test_reconstruct_from_10499(self):
        """Write out full tomogram reconstruction to MRC file for visual inspection using real data"""
        # Setup test outputs directory
        testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs'
        testoutputs_dir.mkdir(exist_ok=True)

        # Load real tilt series metadata
        xml_path = "/Users/tegunovd/Downloads/10499-test-local/6/00254.xml"
        ts = TiltSeries(xml_path)

        print(f"\nLoaded TiltSeries from {xml_path}")
        print(f"  Number of tilts: {ts.n_tilts}")
        print(f"  Tilt angles: {ts.angles}")

        # Set volume dimensions based on original image size
        original_pixel_size = 1.7005  # Angstroms
        volume_x = 4000 * original_pixel_size  # Angstroms
        volume_y = 4000 * original_pixel_size  # Angstroms
        volume_z = 1600 * original_pixel_size  # Angstroms

        print(f"  Volume dimensions (Å): [{volume_x}, {volume_y}, {volume_z}]")

        # Load images at desired pixel size
        desired_pixel_size = 10.0  # Angstroms
        print(f"\nLoading images at {desired_pixel_size} Å/pixel...")

        tilt_data, _, _ = ts.load_images(
            original_pixel_size=original_pixel_size,
            desired_pixel_size=desired_pixel_size,
            use_denoised=False,
            load_half_averages=False,
        )

        print(f"  Loaded images shape: {tilt_data.shape}")
        print(f"  Image dimensions (Å): {ts.image_dimensions_physical}")


        # Test 2: Full reconstruction with CTF and normalization
        print(f"\nReconstructing full tomogram (with CTF and normalization)...")
        volume_with_ctf = ts.reconstruct_full(
            tilt_data=tilt_data,
            pixel_size=desired_pixel_size,
            volume_dimensions_physical=(volume_x, volume_y, volume_z),
            subvolume_size=64,
            subvolume_padding=2.0,
            normalize=True,
            invert=True,
            apply_ctf=True,
            ctf_weighted=True,
            batch_size=8
        )

        print(f"  Reconstruction shape: {volume_with_ctf.shape}")
        print(f"  Value range: {volume_with_ctf.min():.3f} to {volume_with_ctf.max():.3f}")

        output_path = testoutputs_dir / 'full_reconstruction_10499.mrc'
        print(f"  Writing to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(volume_with_ctf.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        print("\n✓ All tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])