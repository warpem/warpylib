"""
Tests for TiltSeries.transform_volume method
"""

import pytest
import torch
import mrcfile
from pathlib import Path

from warpylib.tilt_series import TiltSeries
from warpylib.ops.rescale import rescale


class TestTransformVolume:
    """Test transform_volume method"""

    def test_reconstruct_and_project_10491(self):
        """Reconstruct volume from 10491 data, transform to central tilt, and compare projection"""
        # Setup test outputs directory
        testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs_10491'
        testoutputs_dir.mkdir(exist_ok=True)

        # Load real tilt series metadata
        xml_path = "/Users/tegunovd/Downloads/10491-test-local/5/TS_1.xml"
        ts = TiltSeries(xml_path)

        print(f"\nLoaded TiltSeries from {xml_path}")
        print(f"  Number of tilts: {ts.n_tilts}")
        print(f"  Tilt angles: {ts.angles}")

        # Find central tilt (closest to 0 degrees)
        central_tilt_id = torch.argmin(torch.abs(ts.angles)).item()
        print(f"  Central tilt ID: {central_tilt_id}, angle: {ts.angles[central_tilt_id]:.2f}°")

        # Set volume dimensions based on original image size
        original_pixel_size = 0.834  # Angstroms
        volume_x = 4600 * original_pixel_size  # Angstroms
        volume_y = 6000 * original_pixel_size  # Angstroms
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

        # Reconstruct full tomogram with CTF
        print(f"\nReconstructing full tomogram (with CTF and normalization)...")
        reconstructed_volume = ts.reconstruct_full(
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

        print(f"  Reconstruction shape: {reconstructed_volume.shape}")
        print(f"  Value range: {reconstructed_volume.min():.3f} to {reconstructed_volume.max():.3f}")

        # Upscale volume by 2x for better interpolation quality
        print(f"\nUpscaling reconstructed volume by 2x...")
        upscale_factor = 4
        D, H, W = reconstructed_volume.shape
        upscaled_size = (D * upscale_factor, H * upscale_factor, W * upscale_factor)
        upscaled_volume = rescale(reconstructed_volume, size=upscaled_size)

        print(f"  Upscaled volume shape: {upscaled_volume.shape}")
        print(f"  Value range: {upscaled_volume.min():.3f} to {upscaled_volume.max():.3f}")

        # Transform volume to central tilt's coordinate frame
        print(f"\nTransforming volume to central tilt coordinate frame...")

        # Output dimensions should match the image dimensions in XY
        output_dimensions_physical = torch.tensor([
            ts.image_dimensions_physical[0].item(),  # X matches image
            ts.image_dimensions_physical[1].item(),  # Y matches image
            volume_z  # Keep Z the same
        ])

        print(f"  Output dimensions (Å): {output_dimensions_physical}")

        transformed_volume = ts.transform_volume(
            volume=upscaled_volume,
            pixel_size=desired_pixel_size,
            output_dimensions_physical=output_dimensions_physical,
            tilt_ids=[central_tilt_id],
            upscale_factor=upscale_factor
        )

        print(f"  Transformed volume shape: {transformed_volume.shape}")

        # Project by summing along Z
        print(f"\nProjecting transformed volume (sum along Z)...")
        projection = transformed_volume[0].sum(dim=0)  # Sum along Z dimension

        print(f"  Projection shape: {projection.shape}")
        print(f"  Projection value range: {projection.min():.3f} to {projection.max():.3f}")

        # Get the original central tilt image
        original_image = tilt_data[central_tilt_id]
        print(f"  Original image shape: {original_image.shape}")
        print(f"  Original image value range: {original_image.min():.3f} to {original_image.max():.3f}")

        # Write both slices to a single MRC file
        output_path = testoutputs_dir / 'projection_comparison.mrc'
        print(f"\nWriting comparison to: {output_path}")

        # Stack original and projection as two Z slices
        comparison_stack = torch.stack([original_image, projection * (-1)], dim=0)

        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            # MRC expects (Z, Y, X)
            mrc.set_data(comparison_stack.cpu().numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)
            mrc.header.nz = 2

        print(f"  Wrote 2 slices:")
        print(f"    Slice 0: Original central tilt image")
        print(f"    Slice 1: Projection from transformed volume")

        # Basic sanity checks
        assert projection.shape == original_image.shape, "Projection and original should have same shape"
        assert torch.all(torch.isfinite(projection)), "All projection values should be finite"
        assert torch.all(torch.isfinite(original_image)), "All original values should be finite"

        # Check that projection has reasonable values (not all zeros)
        assert projection.abs().sum() > 0, "Projection should not be all zeros"

        print("\n✓ Test completed successfully!")
        print(f"  Open {output_path} in a viewer to compare slices 0 (original) and 1 (projection)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
