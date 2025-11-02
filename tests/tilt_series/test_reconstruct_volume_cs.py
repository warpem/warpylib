"""
Tests for TiltSeries.reconstruct_full_cs method
"""

import pytest
import torch
import mrcfile
from pathlib import Path

from warpylib.tilt_series import TiltSeries


class TestReconstructVolumeCS:
    """Test reconstruct_full_cs method"""

    def test_reconstruct_cs_from_10491(self):
        """Reconstruct tomogram using CS optimization and write to MRC file"""
        # Setup test outputs directory
        testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs_10491'
        testoutputs_dir.mkdir(exist_ok=True)

        # Load real tilt series metadata
        xml_path = "/Users/tegunovd/Downloads/10491-test-local/5/TS_1.xml"
        ts = TiltSeries(xml_path)

        print(f"\nLoaded TiltSeries from {xml_path}")
        print(f"  Number of tilts: {ts.n_tilts}")
        print(f"  Tilt angles: {ts.angles}")

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

        # Reconstruct using CS optimization
        print(f"\nReconstructing tomogram using CS optimization...")
        volume_cs = ts.reconstruct_full_cs(
            tilt_data=tilt_data,
            pixel_size=desired_pixel_size,
            volume_dimensions_physical=(volume_x, volume_y, volume_z),
            normalize=True,
            invert=True,
            n_iterations=500,
            learning_rate=1e-3,
            tilt_batch_size=4,
            debug_output_dir=str(testoutputs_dir / 'cs_iterations')
        )

        print(f"\n  Reconstruction shape: {volume_cs.shape}")
        print(f"  Value range: {volume_cs.min():.3f} to {volume_cs.max():.3f}")

        output_path = testoutputs_dir / 'full_reconstruction_10491_cs.mrc'
        print(f"\n  Writing to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            # MRC expects (Z, Y, X)
            mrc.set_data(volume_cs.cpu().numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Basic sanity checks
        assert volume_cs.ndim == 3, "Volume should be 3D"
        assert torch.all(torch.isfinite(volume_cs)), "All values should be finite"

        # Check dimensions are even
        for dim in volume_cs.shape:
            assert dim % 2 == 0, "All dimensions should be even"

        print(f"\n✓ Test completed successfully!")
        print(f"  Output written to {output_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])