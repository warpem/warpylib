"""
Tests for TiltSeries.reconstruct_subvolumes method
"""

import pytest
import torch
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path

from warpylib.tilt_series import TiltSeries
from warpylib.tilt_series.reconstruct_subvolumes import get_sinc2_correction


class TestReconstructSubvolumes:
    """Test reconstruct_subvolumes method"""

    def test_basic_reconstruction(self):
        """Test basic subtomogram reconstruction"""
        # Create simple tilt series with 3 tilts
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Create fake tilt data
        tilt_data = torch.randn(3, 128, 128)

        # Create test coordinates: shape (1, 3, 3) - 1 particle, 3 tilts, 3 coords
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],  # Volume center, tilt 0
            [50.0, 50.0, 25.0],  # Volume center, tilt 1
            [50.0, 50.0, 25.0],  # Volume center, tilt 2
        ]])

        # Reconstruct
        result = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False  # No CTF for basic test
        )

        # Check shape: (1, 32, 32, 32)
        assert result.shape == (1, 32, 32, 32)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_batched_particles(self):
        """Test reconstruction with multiple particles"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(3, 128, 128)

        # Shape: (2, 3, 3) - 2 particles, 3 tilts, 3 coords
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],  # Particle 0
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],  # Particle 1
        ])

        result = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False
        )

        # Check shape: (2, 32, 32, 32)
        assert result.shape == (2, 32, 32, 32)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_with_ctf(self):
        """Test reconstruction with CTF correction"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(3, 128, 128)

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct with CTF
        result = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Check shape
        assert result.shape == (1, 32, 32, 32)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_different_oversamplings(self):
        """Test reconstruction with different oversampling factors"""
        ts = TiltSeries(n_tilts=2)
        ts.angles = torch.tensor([-10.0, 10.0])
        ts.dose = torch.tensor([0.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(2, 128, 128)

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Test with oversampling=1.0
        result_1x = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            oversampling=1.0,
            apply_ctf=False
        )
        assert result_1x.shape == (1, 32, 32, 32)

        # Test with oversampling=2.0
        result_2x = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            oversampling=2.0,
            apply_ctf=False
        )
        assert result_2x.shape == (1, 32, 32, 32)

        # Both should be finite
        assert torch.all(torch.isfinite(result_1x))
        assert torch.all(torch.isfinite(result_2x))

    def test_single_coordinate_convenience_method(self):
        """Test convenience method for single coordinate"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(3, 128, 128)

        # Single coordinate: (3,)
        coord = torch.tensor([50.0, 50.0, 25.0])

        result = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data,
            coords=coord,
            pixel_size=10.0,
            size=32,
            apply_ctf=False
        )

        # Should return one reconstruction: (32, 32, 32)
        assert result.shape == (32, 32, 32)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Compare with manual replication
        coords_manual = coord.unsqueeze(0).unsqueeze(0).expand(1, 3, 3)  # (1, 3, 3)
        result_manual = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords_manual,
            pixel_size=10.0,
            size=32,
            apply_ctf=False
        )
        assert torch.allclose(result, result_manual.squeeze(0), atol=1e-5)

    def test_batched_convenience_method(self):
        """Test convenience method with batched coordinates"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(3, 128, 128)

        # Batched coordinates: (2, 3) - 2 particles
        coords = torch.tensor([
            [25.0, 25.0, 25.0],
            [75.0, 75.0, 25.0],
        ])

        result = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False
        )

        # Should return: (2, 32, 32, 32) - 2 particles
        assert result.shape == (2, 32, 32, 32)

        # All results should be finite
        assert torch.all(torch.isfinite(result))

    def test_invalid_tilt_count(self):
        """Test error handling for coordinate/tilt mismatch"""
        ts = TiltSeries(n_tilts=3)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(3, 128, 128)

        # Coords with wrong number of tilts
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])  # Only 2 tilts, but ts has 3

        with pytest.raises(ValueError, match="coords has 2 tilts but TiltSeries has 3"):
            ts.reconstruct_subvolumes(
                tilt_data=tilt_data,
                coords=coords,
                pixel_size=10.0,
                size=32
            )

    def test_gradient_flow(self):
        """Test that gradients flow through the reconstruction to all grids"""
        # Create TiltSeries with grids requiring gradients
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Initialize grids with requires_grad=True
        # We'll use simple 3x3x3 grids for testing
        from warpylib.cubic_grid import CubicGrid
        grid_size = (3, 3, 3)
        ctf_grid_size = (1, 1, 3)

        # CTF grids - these should receive gradients when apply_ctf=True
        # Create leaf tensors and keep references to check gradients later
        defocus_leaf = torch.ones(3, requires_grad=True) * 2.0
        ts.grid_ctf_defocus = CubicGrid(ctf_grid_size, defocus_leaf)
        # Call retain_grad() to keep gradients on the grid values
        ts.grid_ctf_defocus.values.retain_grad()

        defocus_delta_leaf = torch.ones(3, requires_grad=True) * 0.5
        ts.grid_ctf_defocus_delta = CubicGrid(ctf_grid_size, defocus_delta_leaf)
        ts.grid_ctf_defocus_delta.values.retain_grad()

        defocus_angle_leaf = torch.zeros(3, requires_grad=True)
        ts.grid_ctf_defocus_angle = CubicGrid(ctf_grid_size, defocus_angle_leaf)
        ts.grid_ctf_defocus_angle.values.retain_grad()

        # Movement grids - these should receive gradients
        movement_x_leaf = torch.zeros(27, requires_grad=True)
        ts.grid_movement_x = CubicGrid(grid_size, movement_x_leaf)
        ts.grid_movement_x.values.retain_grad()

        movement_y_leaf = torch.zeros(27, requires_grad=True)
        ts.grid_movement_y = CubicGrid(grid_size, movement_y_leaf)
        ts.grid_movement_y.values.retain_grad()

        # Angle grids - these should receive gradients
        # Use small non-zero values to avoid gimbal lock singularities in gradient computation
        angle_x_leaf = torch.randn(3, requires_grad=True) * 0.1
        ts.grid_angle_x = CubicGrid(ctf_grid_size, angle_x_leaf)
        ts.grid_angle_x.values.retain_grad()

        angle_y_leaf = torch.randn(3, requires_grad=True) * 0.1
        ts.grid_angle_y = CubicGrid(ctf_grid_size, angle_y_leaf)
        ts.grid_angle_y.values.retain_grad()

        angle_z_leaf = torch.randn(3, requires_grad=True) * 0.1
        ts.grid_angle_z = CubicGrid(ctf_grid_size, angle_z_leaf)
        ts.grid_angle_z.values.retain_grad()

        # Create tilt data with gradients
        tilt_data = torch.randn(3, 128, 128, requires_grad=True)

        # Create coordinates
        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct with CTF to ensure CTF grids are used
        result = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            ctf_weighted=False,  # Use simple CTF correction
            oversampling=1.0
        )

        # Compute an arbitrary loss (sum of squares)
        loss = (result ** 2).sum()

        # Backpropagate
        loss.backward()

        # Check that tilt_data received gradients
        assert tilt_data.grad is not None, "tilt_data should have gradients"
        assert torch.any(tilt_data.grad != 0), "tilt_data gradients should be non-zero"

        # Check that CTF grids received gradients (when using CTF)
        assert ts.grid_ctf_defocus.values.grad is not None, "grid_ctf_defocus should have gradients"
        assert torch.any(ts.grid_ctf_defocus.values.grad != 0), "grid_ctf_defocus gradients should be non-zero"

        # Check movement grids received gradients
        assert ts.grid_movement_x.values.grad is not None, "grid_movement_x should have gradients"
        assert torch.any(ts.grid_movement_x.values.grad != 0), "grid_movement_x gradients should be non-zero"

        assert ts.grid_movement_y.values.grad is not None, "grid_movement_y should have gradients"
        assert torch.any(ts.grid_movement_y.values.grad != 0), "grid_movement_y gradients should be non-zero"

        # Check angle grids received gradients
        assert ts.grid_angle_x.values.grad is not None, "grid_angle_x should have gradients"
        assert torch.any(ts.grid_angle_x.values.grad != 0), "grid_angle_x gradients should be non-zero"

        assert ts.grid_angle_y.values.grad is not None, "grid_angle_y should have gradients"
        assert torch.any(ts.grid_angle_y.values.grad != 0), "grid_angle_y gradients should be non-zero"

        print("\nGradient flow test passed!")
        print(f"  tilt_data grad norm: {tilt_data.grad.norm():.6f}")
        print(f"  grid_ctf_defocus grad norm: {ts.grid_ctf_defocus.values.grad.norm():.6f}")
        print(f"  grid_movement_x grad norm: {ts.grid_movement_x.values.grad.norm():.6f}")
        print(f"  grid_movement_y grad norm: {ts.grid_movement_y.values.grad.norm():.6f}")
        print(f"  grid_angle_x grad norm: {ts.grid_angle_x.values.grad.norm():.6f}")
        print(f"  grid_angle_y grad norm: {ts.grid_angle_y.values.grad.norm():.6f}")
        print(f"  grid_angle_z grad norm: {ts.grid_angle_z.values.grad.norm():.6f}")

    def test_tilt_ids_subset(self):
        """Test reconstruction using a subset of tilts via tilt_ids"""
        ts = TiltSeries(n_tilts=5)
        ts.angles = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
        ts.dose = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(5, 128, 128)

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
        result_subset = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            tilt_ids=tilt_ids
        )

        # Check shape is still correct
        assert result_subset.shape == (1, 32, 32, 32)
        assert torch.all(torch.isfinite(result_subset))

        # Result should be different from using all tilts
        result_all = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False
        )
        assert not torch.allclose(result_subset, result_all, atol=1e-5)

    def test_tilt_ids_all_tilts(self):
        """Test that using all tilt_ids gives same result as no tilt_ids"""
        ts = TiltSeries(n_tilts=3)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0])
        ts.dose = torch.tensor([0.0, 50.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(3, 128, 128)

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct without tilt_ids
        result_no_ids = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False
        )

        # Reconstruct with all tilt_ids
        tilt_ids = torch.tensor([0, 1, 2])
        result_with_ids = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            tilt_ids=tilt_ids
        )

        # Results should be identical
        assert torch.allclose(result_no_ids, result_with_ids, atol=1e-5)

    def test_tilt_ids_with_ctf(self):
        """Test reconstruction with tilt_ids and CTF correction"""
        ts = TiltSeries(n_tilts=4)
        ts.angles = torch.tensor([-30.0, -10.0, 10.0, 30.0])
        ts.dose = torch.tensor([0.0, 33.0, 66.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(4, 128, 128)

        coords = torch.tensor([[
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
            [50.0, 50.0, 25.0],
        ]])

        # Reconstruct using only tilts 1 and 2 with CTF
        tilt_ids = torch.tensor([1, 2])
        result = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=True,
            ctf_weighted=True,
            tilt_ids=tilt_ids
        )

        assert result.shape == (1, 32, 32, 32)
        assert torch.all(torch.isfinite(result))

    def test_tilt_ids_single_convenience_method(self):
        """Test tilt_ids with single coordinate convenience method"""
        ts = TiltSeries(n_tilts=4)
        ts.angles = torch.tensor([-30.0, -10.0, 10.0, 30.0])
        ts.dose = torch.tensor([0.0, 33.0, 66.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(4, 128, 128)

        # Single coordinate
        coord = torch.tensor([50.0, 50.0, 25.0])

        # Use only first and last tilts
        tilt_ids = torch.tensor([0, 3])
        result = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data,
            coords=coord,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            tilt_ids=tilt_ids
        )

        assert result.shape == (32, 32, 32)
        assert torch.all(torch.isfinite(result))

    def test_tilt_ids_batched(self):
        """Test tilt_ids with multiple particles"""
        ts = TiltSeries(n_tilts=5)
        ts.angles = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
        ts.dose = torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0])
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        tilt_data = torch.randn(5, 128, 128)

        # 2 particles, 5 tilts each
        coords = torch.tensor([
            [[25.0, 25.0, 25.0], [25.0, 25.0, 25.0], [25.0, 25.0, 25.0],
             [25.0, 25.0, 25.0], [25.0, 25.0, 25.0]],
            [[75.0, 75.0, 25.0], [75.0, 75.0, 25.0], [75.0, 75.0, 25.0],
             [75.0, 75.0, 25.0], [75.0, 75.0, 25.0]],
        ])

        # Use tilts 1, 2, 3
        tilt_ids = torch.tensor([1, 2, 3])
        result = ts.reconstruct_subvolumes(
            tilt_data=tilt_data,
            coords=coords,
            pixel_size=10.0,
            size=32,
            apply_ctf=False,
            tilt_ids=tilt_ids
        )

        assert result.shape == (2, 32, 32, 32)
        assert torch.all(torch.isfinite(result))

    def test_sinc2_correction(self):
        """Test sinc^2 correction volume generation and visualization"""
        # Setup test outputs directory
        testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs'
        testoutputs_dir.mkdir(exist_ok=True)

        # Test different sizes and oversampling factors
        test_cases = [
            (64, 1.0, 'sinc2_correction_size64_os1.0'),
            (64, 2.0, 'sinc2_correction_size64_os2.0'),
            (128, 1.0, 'sinc2_correction_size128_os1.0'),
            (128, 2.0, 'sinc2_correction_size128_os2.0'),
        ]

        for size, oversampling, name in test_cases:
            print(f"\nGenerating sinc^2 correction: size={size}, oversampling={oversampling}")

            # Generate correction volume
            correction = get_sinc2_correction(size=size, oversampling=oversampling)

            # Check shape and values
            assert correction.shape == (size, size, size)
            assert torch.all(torch.isfinite(correction))
            assert torch.all(correction > 0)  # sinc^2 should be positive
            assert torch.all(correction <= 1.0)  # sinc^2(0) = 1 is the maximum

            # Check that center value is 1.0 (DC component)
            center_idx = size // 2
            assert torch.isclose(correction[center_idx, center_idx, center_idx],
                               torch.tensor(1.0), atol=1e-5)

            # Write to MRC
            output_path = testoutputs_dir / f'{name}.mrc'
            print(f"  Writing to: {output_path}")
            with mrcfile.new(str(output_path), overwrite=True) as mrc:
                mrc.set_data(correction.numpy().astype('float32'))
                mrc.voxel_size = (1.0, 1.0, 1.0)

            # Extract central X line (through center of volume)
            central_line = correction[center_idx, center_idx, :]

            # Create 1D plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x_coords = torch.arange(size) - center_idx  # Center at 0
            ax.plot(x_coords.numpy(), central_line.numpy(), 'b-', linewidth=2)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='DC (no attenuation)')
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Distance from center (pixels)')
            ax.set_ylabel('sinc^2 correction factor')
            ax.set_title(f'sinc^2 Correction (size={size}, oversampling={oversampling})\nCentral X line')
            ax.legend()

            # Save plot
            plot_path = testoutputs_dir / f'{name}_central_line.png'
            print(f"  Saving plot to: {plot_path}")
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Print some statistics
            print(f"  Center value: {correction[center_idx, center_idx, center_idx]:.6f}")
            print(f"  Min value: {correction.min():.6f}")
            print(f"  Max value: {correction.max():.6f}")
            print(f"  Edge value: {correction[0, center_idx, center_idx]:.6f}")

    def test_write_reconstructions_to_mrc(self):
        """Write out reconstructions to MRC files for visual inspection using real data"""
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

        # Load images at 10 Angstrom/pixel
        desired_pixel_size = 8.0  # Angstroms
        print(f"\nLoading images at {desired_pixel_size} Å/pixel...")

        tilt_data, _, _ = ts.load_images(
            original_pixel_size=original_pixel_size,
            desired_pixel_size=desired_pixel_size,
            use_denoised=False,
            load_half_averages=False,
        )

        print(f"  Loaded images shape: {tilt_data.shape}")
        print(f"  Image dimensions (Å): {ts.image_dimensions_physical}")

        # Pick some coordinates within the volume
        # Let's pick positions near the center and at different depths
        coords_list = [
            torch.tensor([volume_x / 2, volume_y / 2, volume_z / 2]),  # Center
            torch.tensor([volume_x / 3, volume_y / 3, volume_z / 2]),  # Off-center 1
            torch.tensor([2 * volume_x / 3, 2 * volume_y / 3, volume_z / 2]),  # Off-center 2
        ]

        # Test 1: Single particle at volume center, no CTF
        print(f"\nReconstructing particle at center (no CTF)...")
        coords_single = coords_list[0]
        result_no_ctf = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data,
            coords=coords_single,
            pixel_size=desired_pixel_size,
            size=64,
            apply_ctf=False,
            oversampling=2.0
        )

        output_path = testoutputs_dir / 'reconstruction_no_ctf.mrc'
        print(f"Writing reconstruction (no CTF) to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(result_no_ctf.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Test 2: Single particle with CTF correction
        print(f"\nReconstructing particle at center (with CTF)...")
        result_with_ctf = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data,
            coords=coords_single,
            pixel_size=desired_pixel_size,
            size=64,
            apply_ctf=True,
            ctf_weighted=True,
            oversampling=2.0
        )

        output_path = testoutputs_dir / 'reconstruction_with_ctf.mrc'
        print(f"Writing reconstruction (with CTF) to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(result_with_ctf.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Test 3: Multiple particles at different locations
        print(f"\nReconstructing multiple particles...")
        coords_multi = torch.stack(coords_list)
        result_multi = ts.reconstruct_subvolumes_single(
            tilt_data=tilt_data,
            coords=coords_multi,
            pixel_size=desired_pixel_size,
            size=64,
            apply_ctf=False,
            oversampling=2.0
        )

        # Write each particle as a separate file
        for i in range(len(coords_list)):
            output_path = testoutputs_dir / f'reconstruction_particle_{i}.mrc'
            print(f"Writing particle {i} (position: {coords_list[i].tolist()}) to: {output_path}")
            with mrcfile.new(str(output_path), overwrite=True) as mrc:
                mrc.set_data(result_multi[i].numpy().astype('float32'))
                mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        # Also write as a stack
        output_path = testoutputs_dir / 'reconstruction_stack.mrc'
        print(f"Writing particle stack to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            # MRC stack format: (n_particles, z, y, x)
            mrc.set_data(result_multi.numpy().astype('float32'))
            mrc.voxel_size = (desired_pixel_size, desired_pixel_size, desired_pixel_size)

        print(f"\nAll reconstructions written to {testoutputs_dir}")

        # Basic sanity checks
        assert result_no_ctf.shape == (64, 64, 64)
        assert result_with_ctf.shape == (64, 64, 64)
        assert result_multi.shape == (3, 64, 64, 64)
        assert torch.all(torch.isfinite(result_no_ctf))
        assert torch.all(torch.isfinite(result_with_ctf))
        assert torch.all(torch.isfinite(result_multi))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])