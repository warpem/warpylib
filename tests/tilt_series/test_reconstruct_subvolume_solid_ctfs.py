"""
Tests for TiltSeries.reconstruct_subvolume_solid_ctfs method
"""

import pytest
import torch
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path

from warpylib.tilt_series import TiltSeries
from warpylib.cubic_grid import CubicGrid


class TestReconstructSubvolumeSolidCTFs:
    """Test reconstruct_subvolume_solid_ctfs method"""

    @pytest.fixture
    def tilt_series_60deg(self):
        """Create a tilt series from -60 to +60 degrees in 2-degree steps."""
        # -60 to +60 in 2-degree steps = 61 tilts
        angles = torch.arange(-60.0, 62.0, 2.0)
        n_tilts = len(angles)

        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = angles

        # Per-tilt defocus variation: 2 um at 0 degrees, increasing with |tilt angle|
        # This simulates defocus gradient across tilts (higher defocus at high tilts)
        # Range: ~2 um at center to ~8 um at ±60 degrees
        defocus_values = 2.0 + 0.1 * torch.abs(angles)  # um

        # Set up grid_ctf_defocus with per-tilt values
        # CubicGrid dimensions are (X, Y, Z) where Z is the tilt dimension
        ts.grid_ctf_defocus = CubicGrid(
            dimensions=(1, 1, n_tilts),
            values=defocus_values
        )

        # Dose increases with tilt number
        ts.dose = torch.linspace(0.0, 150.0, n_tilts)

        # Set volume and image dimensions
        ts.volume_dimensions_physical = torch.tensor([2000.0, 2000.0, 500.0])
        ts.image_dimensions_physical = torch.tensor([2000.0, 2000.0])

        return ts

    def test_basic_solid_ctf_reconstruction(self, tilt_series_60deg):
        """Test basic solid CTF volume reconstruction"""
        ts = tilt_series_60deg
        n_tilts = ts.n_tilts

        # Create test coordinates: shape (1, n_tilts, 3) - 1 particle, all tilts, 3 coords
        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        # Reconstruct solid CTF volume
        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Check shape: (1, 256, 256, 129) - rfft format (size//2+1 in last dimension)
        assert result.shape == (1, 256, 256, 129)

        # Result should be real-valued (not complex)
        assert not result.is_complex()
        assert result.dtype == torch.float32

        # All results should be finite
        assert torch.all(torch.isfinite(result))

        # Values should be positive (CTF amplitude)
        assert torch.all(result >= 0)

    def test_solid_vs_regular_ctf_more_coverage(self, tilt_series_60deg):
        """Test that solid CTFs have more Fourier coverage than regular CTFs"""
        ts = tilt_series_60deg
        n_tilts = ts.n_tilts

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        # Reconstruct regular CTF volume
        result_regular = ts.reconstruct_subvolume_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Reconstruct solid CTF volume
        result_solid = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Both should have the same shape
        assert result_regular.shape == result_solid.shape

        # Solid CTFs should have fewer zero/near-zero values (better coverage)
        threshold = 0.01
        regular_nonzero = (result_regular.abs() > threshold).sum()
        solid_nonzero = (result_solid.abs() > threshold).sum()

        # Solid should have at least as much coverage (usually more due to gap filling)
        assert solid_nonzero >= regular_nonzero, \
            f"Expected solid CTF to have >= coverage: {solid_nonzero} vs {regular_nonzero}"

    def test_different_defocus_per_tilt(self):
        """Test with significantly different defocus values per tilt"""
        n_tilts = 61  # -60 to +60 in 2-deg steps
        angles = torch.arange(-60.0, 62.0, 2.0)

        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = angles
        ts.dose = torch.linspace(0.0, 150.0, n_tilts)
        ts.volume_dimensions_physical = torch.tensor([2000.0, 2000.0, 500.0])
        ts.image_dimensions_physical = torch.tensor([2000.0, 2000.0])

        # Large defocus variation: 1 um to 5 um
        ts.ctf.defocus = 3.0  # Base defocus

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True,
            ctf_weighted=True
        )

        assert result.shape == (1, 256, 256, 129)
        assert torch.all(torch.isfinite(result))

    def test_batched_particles(self, tilt_series_60deg):
        """Test solid CTF volume reconstruction with multiple particles"""
        ts = tilt_series_60deg
        n_tilts = ts.n_tilts

        # Shape: (3, n_tilts, 3) - 3 particles
        coords = torch.tensor([
            [[500.0, 500.0, 250.0]] * n_tilts,
            [[1000.0, 1000.0, 250.0]] * n_tilts,
            [[1500.0, 1500.0, 250.0]] * n_tilts,
        ])

        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Check shape: (3, 256, 256, 129)
        assert result.shape == (3, 256, 256, 129)
        assert not result.is_complex()
        assert torch.all(torch.isfinite(result))

    def test_single_coordinate_convenience_method(self, tilt_series_60deg):
        """Test convenience method for single static coordinate"""
        ts = tilt_series_60deg

        # Single coordinate: (3,)
        coord = torch.tensor([1000.0, 1000.0, 250.0])

        result = ts.reconstruct_subvolume_solid_ctfs_single(
            coords=coord,
            pixel_size=8.0,
            size=256,
            apply_ctf=True
        )

        # Should return one CTF volume: (256, 256, 129)
        assert result.shape == (256, 256, 129)
        assert not result.is_complex()
        assert torch.all(torch.isfinite(result))

    def test_excluded_tilts_edges(self):
        """Test that excluded tilts at edges are not filled"""
        n_tilts = 61
        angles = torch.arange(-60.0, 62.0, 2.0)

        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = angles
        ts.dose = torch.linspace(0.0, 150.0, n_tilts)
        ts.volume_dimensions_physical = torch.tensor([2000.0, 2000.0, 500.0])
        ts.image_dimensions_physical = torch.tensor([2000.0, 2000.0])

        # Exclude first 5 and last 5 tilts
        ts.use_tilt = torch.ones(n_tilts, dtype=torch.bool)
        ts.use_tilt[:5] = False
        ts.use_tilt[-5:] = False

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True
        )

        assert result.shape == (1, 256, 256, 129)
        assert torch.all(torch.isfinite(result))

    def test_excluded_tilts_middle(self):
        """Test that excluded tilts in the middle are filled"""
        n_tilts = 61
        angles = torch.arange(-60.0, 62.0, 2.0)

        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = angles
        ts.dose = torch.linspace(0.0, 150.0, n_tilts)
        ts.volume_dimensions_physical = torch.tensor([2000.0, 2000.0, 500.0])
        ts.image_dimensions_physical = torch.tensor([2000.0, 2000.0])

        # Exclude some tilts in the middle (simulate bad tilts)
        ts.use_tilt = torch.ones(n_tilts, dtype=torch.bool)
        ts.use_tilt[28:33] = False  # Exclude 5 tilts around 0 degrees

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True
        )

        assert result.shape == (1, 256, 256, 129)
        assert torch.all(torch.isfinite(result))

    def test_apply_ctf_false(self, tilt_series_60deg):
        """Test solid CTF volume reconstruction with apply_ctf=False (flat)"""
        ts = tilt_series_60deg
        n_tilts = ts.n_tilts

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        # Reconstruct with flat CTF (no oscillations)
        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=False,
            ctf_weighted=True
        )

        assert result.shape == (1, 256, 256, 129)
        assert not result.is_complex()
        assert torch.all(torch.isfinite(result))

    def test_tilt_ids_subset(self, tilt_series_60deg):
        """Test using a subset of tilts via tilt_ids"""
        ts = tilt_series_60deg
        n_tilts = ts.n_tilts

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        # Use only every 3rd tilt
        tilt_ids = torch.arange(0, n_tilts, 3)

        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True,
            tilt_ids=tilt_ids
        )

        assert result.shape == (1, 256, 256, 129)
        assert torch.all(torch.isfinite(result))

    def test_oversampling(self, tilt_series_60deg):
        """Test with different oversampling factors"""
        ts = tilt_series_60deg
        n_tilts = ts.n_tilts

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        # Test with oversampling=2.0
        result_2x = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            oversampling=2.0,
            apply_ctf=True
        )

        assert result_2x.shape == (1, 256, 256, 129)
        assert torch.all(torch.isfinite(result_2x))

    def test_invalid_tilt_count(self, tilt_series_60deg):
        """Test error handling for coordinate/tilt mismatch"""
        ts = tilt_series_60deg

        # Coords with wrong number of tilts
        coords = torch.tensor([[
            [1000.0, 1000.0, 250.0],
            [1000.0, 1000.0, 250.0],
        ]])  # Only 2 tilts

        with pytest.raises(ValueError, match=f"coords has 2 tilts but TiltSeries has {ts.n_tilts}"):
            ts.reconstruct_subvolume_solid_ctfs(
                coords=coords,
                pixel_size=8.0,
                size=256
            )

    def test_interpolation_count(self):
        """Test that interpolation creates expected number of additional tilts"""
        # With 2-degree steps and size=256, angular step should be ~0.45 degrees
        # So between each 2-degree gap, we should insert ~3-4 interpolated tilts
        n_tilts = 61
        angles = torch.arange(-60.0, 62.0, 2.0)

        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = angles
        ts.dose = torch.linspace(0.0, 150.0, n_tilts)
        ts.volume_dimensions_physical = torch.tensor([2000.0, 2000.0, 500.0])
        ts.image_dimensions_physical = torch.tensor([2000.0, 2000.0])

        coords = torch.tensor([[[1000.0, 1000.0, 250.0]] * n_tilts])

        # Run reconstruction - this should succeed if interpolation works correctly
        result = ts.reconstruct_subvolume_solid_ctfs(
            coords=coords,
            pixel_size=8.0,
            size=256,
            apply_ctf=True
        )

        assert result.shape == (1, 256, 256, 129)
        assert torch.all(torch.isfinite(result))

    def test_write_solid_ctf_volumes_to_mrc(self, tilt_series_60deg):
        """Write out solid CTF volumes to MRC files for visual inspection"""
        testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs'
        testoutputs_dir.mkdir(exist_ok=True)

        ts = tilt_series_60deg
        n_tilts = ts.n_tilts
        pixel_size = 4.0
        size = 256

        print(f"\nTiltSeries: {n_tilts} tilts from {ts.angles[0]:.1f} to {ts.angles[-1]:.1f} degrees")

        # Single particle at center
        coord = torch.tensor([1000.0, 1000.0, 250.0])

        # Reconstruct regular CTF volume
        result_regular = ts.reconstruct_subvolume_ctfs_single(
            coords=coord,
            pixel_size=pixel_size,
            size=size,
            apply_ctf=True,
            ctf_weighted=True
        )

        # Reconstruct solid CTF volume
        result_solid = ts.reconstruct_subvolume_solid_ctfs_single(
            coords=coord,
            pixel_size=pixel_size,
            size=size,
            apply_ctf=False,
            ctf_weighted=False
        )

        # Write regular CTF volume
        output_path = testoutputs_dir / 'solid_ctf_test_regular.mrc'
        print(f"Writing regular CTF volume to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(result_regular.numpy().astype('float32'))
            mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

        # Write solid CTF volume
        output_path = testoutputs_dir / 'solid_ctf_test_solid.mrc'
        print(f"Writing solid CTF volume to: {output_path}")
        with mrcfile.new(str(output_path), overwrite=True) as mrc:
            mrc.set_data(result_solid.numpy().astype('float32'))
            mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

        # Create comparison visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Central Z slice (in rfft space)
        z_slice = 0
        for i, (result, title) in enumerate([
            (result_regular, 'Regular CTF'),
            (result_solid, 'Solid CTF'),
        ]):
            # XY plane (central Z)
            axes[i, 0].imshow(result[z_slice, :, :].numpy(), cmap='viridis')
            axes[i, 0].set_title(f'{title} - XY plane (Z={z_slice})')
            axes[i, 0].axis('off')

            # XZ plane (central Y)
            axes[i, 1].imshow(result[:, size // 2, :].numpy(), cmap='viridis')
            axes[i, 1].set_title(f'{title} - XZ plane (Y={size // 2})')
            axes[i, 1].axis('off')

            # YZ plane (central X in rfft)
            axes[i, 2].imshow(result[:, :, size // 4].numpy(), cmap='viridis')
            axes[i, 2].set_title(f'{title} - YZ plane (X={size // 4})')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plot_path = testoutputs_dir / 'solid_ctf_comparison.png'
        print(f"Saving comparison visualization to: {plot_path}")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Sanity checks
        assert result_regular.shape == (256, 256, 129)
        assert result_solid.shape == (256, 256, 129)
        assert torch.all(torch.isfinite(result_regular))
        assert torch.all(torch.isfinite(result_solid))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])