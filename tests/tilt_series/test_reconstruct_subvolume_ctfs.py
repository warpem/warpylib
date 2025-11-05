"""
Tests for TiltSeries.reconstruct_subvolume_ctfs method
"""

import pytest
import torch
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path

from warpylib.tilt_series import TiltSeries


class TestReconstructSubvolumeCTFs:
    """Test reconstruct_subvolume_ctfs method"""

    def test_basic_ctf_volume_reconstruction(self):
        """Test basic CTF volume reconstruction with flat weighting"""
        # Create simple tilt series with 3 tilts
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Create test coordinates: shape (1, 3, 3) - 1 particle, 3 tilts, 3 coords
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],  # Volume center, tilt 0
            [50.0, 50.0, 25.0],  # Volume center, tilt 1
            [50.0, 50.0, 25.0],  # Volume center, tilt 2
        ]])

        # Reconstruct CTF volume with flat weighting (no CTF oscillations)
        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,  # Use flat weighting for basic test
            ctf_weighted=True
        )

        # Check shape: (1, 32, 32, 17) - rfft format (size//2+1 in last dimension)
        assert result.shape == (1, 32, 32, 17)

        # Result should be real-valued (not complex)
        assert not result.is_complex()
        assert result.dtype == torch.float32

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_batched_particles(self):
        """Test CTF volume reconstruction with multiple particles"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Shape: (2, 3, 3) - 2 particles, 3 tilts, 3 coords
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],  # Particle 0
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],  # Particle 1
        ])

        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Check shape: (2, 32, 32, 17)
        assert result.shape == (2, 32, 32, 17)

        # Result should be real-valued (not complex)
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_different_oversamplings(self):
        """Test CTF volume reconstruction with different oversampling factors"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-10.0, 10.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Test with oversampling=1.0
        result_1x = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            oversampling=1.0,
            apply_ctf=True
        )
        assert result_1x.shape == (1, 32, 32, 17)

        # Test with oversampling=2.0
        result_2x = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            oversampling=2.0,
            apply_ctf=True
        )
        assert result_2x.shape == (1, 32, 32, 17)

        # Both should be finite and real-valued
        assert torch.all(torch.isfinite(result_1x))
        assert torch.all(torch.isfinite(result_2x))
        assert not result_1x.is_complex()
        assert not result_2x.is_complex()

    def test_single_coordinate_convenience_method(self):
        """Test convenience method for single coordinate"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate: (3,)
        coord = torch.tensor([50.0, 50.0, 25.0])

        result = ts.reconstruct_subvolume_ctfs_single(
            coords=coord,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )

        # Should return one CTF volume: (32, 32, 17)
        assert result.shape == (32, 32, 17)

        # Result should be real-valued
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with manual replication
        coords_manual = coord.unsqueeze(0).unsqueeze(0).expand(1, 3, 3)  # (1, 3, 3)
        result_manual = ts.reconstruct_subvolume_ctfs(
            coords=coords_manual,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )
        assert torch.allclose(result, result_manual.squeeze(0), atol=1e-5)

    def test_batched_convenience_method(self):
        """Test convenience method with batched coordinates"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Batched coordinates: (2, 3) - 2 particles
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [75.0, 75.0, 25.0],
        ])

        result = ts.reconstruct_subvolume_ctfs_single(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )

        # Should return: (2, 32, 32, 17) - 2 particles
        assert result.shape == (2, 32, 32, 17)

        # Result should be real-valued
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_invalid_tilt_count(self):
        """Test error handling for coordinate/tilt mismatch"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Coords with wrong number of tilts
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])  # Only 2 tilts, but ts has 3

        with pytest.raises(ValueError, match="coords has 2 tilts but TiltSeries has 3"):
            ts.reconstruct_subvolume_ctfs(
                coords=coords,
                pixel_size=10.0,
                size=32
            )

    def test_apply_ctf_false(self):
        """Test CTF volume reconstruction with apply_ctf=False (flat weighting)"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct with flat weighting (no CTF oscillations)
        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            ctf_weighted=True  # Still applies dose weighting
        )

        # Check shape
        assert result.shape == (1, 32, 32, 17)

        # Result should be real-valued
        assert not result.is_complex()

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_tilt_ids_subset(self):
        """Test CTF volume reconstruction using a subset of tilts via tilt_ids"""
        ts = TiltSeries(n_tilts=5)
        ts.angles = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
        ts.dose = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Coordinates for all 5 tilts
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct using only tilts 0, 2, 4 (every other tilt)
        tilt_ids = torch.tensor([0, 2, 4])
        result_subset = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,  # Use flat weighting (no CTF parameters set)
            ctf_weighted=False,  # Disable dose weighting
            tilt_ids=tilt_ids
        )

        # Check shape is still correct
        assert result_subset.shape == (1, 32, 32, 17)
        assert torch.all(torch.isfinite(result_subset))
        assert not result_subset.is_complex()

        # Result should be different from using all tilts
        result_all = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            ctf_weighted=False  # Disable dose weighting
        )
        assert not torch.allclose(result_subset, result_all, atol=1e-5)

    def test_tilt_ids_all_tilts(self):
        """Test that using all tilt_ids gives same result as no tilt_ids"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct without tilt_ids
        result_no_ids = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True
        )

        # Reconstruct with all tilt_ids
        tilt_ids = torch.tensor([0, 1, 2])
        result_with_ids = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        # Results should be identical
        assert torch.allclose(result_no_ids, result_with_ids, atol=1e-5)

    def test_tilt_ids_single_convenience_method(self):
        """Test tilt_ids with single coordinate convenience method"""
        ts = TiltSeries(n_tilts=4)
        ts.angles = torch.tensor([-30.0, -10.0, 10.0, 30.0])
        ts.dose = torch.tensor([0.0, 33.0, 66.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Single coordinate
        coord = torch.tensor([50.0, 50.0, 25.0])

        # Use only first and last tilts
        tilt_ids = torch.tensor([0, 3])
        result = ts.reconstruct_subvolume_ctfs_single(
            coords=coord,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        assert result.shape == (32, 32, 17)
        assert torch.all(torch.isfinite(result))
        assert not result.is_complex()

    def test_tilt_ids_batched(self):
        """Test tilt_ids with multiple particles"""
        ts = TiltSeries(n_tilts=5)
        ts.angles = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
        ts.dose = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # 2 particles, 5 tilts each
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0], [25.0, 25.0, 25.0],
             [25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0], [75.0, 75.0, 25.0],
             [75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],
        ])

        # Use tilts 1, 2, 3
        tilt_ids = torch.tensor([1, 2, 3])
        result = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        assert result.shape == (2, 32, 32, 17)
        assert torch.all(torch.isfinite(result))
        assert not result.is_complex()

    def test_write_ctf_volumes_to_mrc(self):
        """Write out CTF volumes to MRC files for visual inspection using real data"""
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
        # Original image: 5760 x 4092 pixels at 0.834 Å/pixel
        original_pixel_size = 0.834  # Angstroms
        volume_x = 4000 * original_pixel_size  # Angstroms
        volume_y = 5700 * original_pixel_size  # Angstroms
        volume_z = 1000 * original_pixel_size  # Angstroms (reasonable thickness for tilt series)

        ts.volume_dimensions_physical = torch.tensor([volume_x, volume_y, volume_z])
        print(f"  Volume dimensions (Å): {ts.volume_dimensions_physical}")

        # Pick some coordinates within the volume
        desired_pixel_size = 8.0  # Angstroms
        coords_list = [
            torch.tensor([volume_x / 2, volume_y / 2, volume_z / 2]),  # Center
            torch.tensor([volume_x / 3, volume_y / 3, volume_z / 2]),  # Off-center 1
            torch.tensor([2 * volume_x / 3, 2 * volume_y / 3, volume_z / 2]),  # Off-center 2
        ]

        # Test 1: Single particle at volume center, with CTF
        print(f"\nReconstructing CTF volume at center (with CTF)...")
        coords_single = coords_list[0]
        result_with_ctf = ts.reconstruct_subvolume_ctfs_single(
            coords=coords_single,
            pixel_size=desired_pixel_size,
            size=64,
            apply_ctf=True,
            ctf_weighted=True,
            oversampling=2.0
        )

        # For MRC output, we need to convert complex Fourier data to real
        # We'll save the magnitude (|CTF|) which is what these volumes approximate
        ctf_magnitude = torch.abs(result_with_ctf)
        output_path = testoutputs_dir / 'ctf_volume_with_ctf_magnitude.mrc'
        print(f"Writing CTF volume magnitude (with CTF) to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            # Note: rfft format is (size, size, size//2+1), we save it as-is
            mrc.set_data(ctf_magnitude.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Also save the real component
        ctf_real = torch.real(result_with_ctf)
        output_path = testoutputs_dir / 'ctf_volume_with_ctf_real.mrc'
        print(f"Writing CTF volume real component (with CTF) to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(ctf_real.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Test 2: Single particle with flat weighting (no CTF oscillations)
        print(f"\nReconstructing CTF volume at center (flat weighting)...")
        result_no_ctf = ts.reconstruct_subvolume_ctfs_single(
            coords=coords_single,
            pixel_size=desired_pixel_size,
            size=64,
            apply_ctf=False,
            ctf_weighted=True,
            oversampling=2.0
        )

        ctf_magnitude = torch.abs(result_no_ctf)
        output_path = testoutputs_dir / 'ctf_volume_flat_magnitude.mrc'
        print(f"Writing CTF volume magnitude (flat) to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(ctf_magnitude.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Test 3: Multiple particles at different locations
        print(f"\nReconstructing CTF volumes for multiple particles...")
        coords_multi = torch.stack(coords_list)
        result_multi = ts.reconstruct_subvolume_ctfs_single(
            coords=coords_multi,
            pixel_size=desired_pixel_size,
            size=64,
            apply_ctf=True,
            ctf_weighted=True,
            oversampling=2.0
        )

        # Write each particle as a separate file
        for i in range(len(coords_list)):
            ctf_magnitude = torch.abs(result_multi[i])
            output_path = testoutputs_dir / f'ctf_volume_particle_{i}_magnitude.mrc'
            print(f"Writing CTF volume {i} (position: {coords_list[i].tolist()}) to: {output_path}")
            with mrcfile.new(str(output_path), overwrite=True) as mrc:
                mrc.set_data(ctf_magnitude.numpy().astype('float32'))
                mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Also write magnitude stack
        output_path = testoutputs_dir / 'ctf_volume_stack_magnitude.mrc'
        print(f"Writing CTF volume magnitude stack to: {output_path}")
        ctf_magnitude_stack = torch.abs(result_multi)
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            # MRC stack format: (n_particles, z, y, x)
            mrc.set_data(ctf_magnitude_stack.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        print(f"\nAll CTF volumes written to {testoutputs_dir}")

        # Basic sanity checks
        assert result_with_ctf.shape == (64, 64, 33)  # rfft format: 64//2+1 = 33
        assert result_no_ctf.shape == (64, 64, 33)
        assert result_multi.shape == (3, 64, 64, 33)
        assert torch.all(torch.isfinite(result_with_ctf))
        assert torch.all(torch.isfinite(result_no_ctf))
        assert torch.all(torch.isfinite(result_multi))

        # Create a central slice visualization
        print(f"\nCreating visualizations...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot central slices (z-slice through middle)
        for i in range(3):
            # Magnitude
            mag = torch.abs(result_multi[i])
            central_slice = mag[0, :, :]  # Middle of rfft dimension
            axes[0, i].imshow(central_slice.numpy(), cmap='viridis')
            axes[0, i].set_title(f'Particle {i} - |CTF| (central slice)')
            axes[0, i].axis('off')

            # Real component
            real = torch.real(result_multi[i])
            central_slice = real[0, :, :]
            axes[1, i].imshow(central_slice.numpy(), cmap='RdBu_r')
            axes[1, i].set_title(f'Particle {i} - Real(CTF) (central slice)')
            axes[1, i].axis('off')

        plt.tight_layout()
        plot_path = testoutputs_dir / 'ctf_volumes_visualization.png'
        print(f"Saving visualization to: {plot_path}")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])