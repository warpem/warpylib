"""
Tests for TiltSeries STAR file I/O
"""

import pytest
import torch
import pandas as pd

from warpylib.tilt_series import TiltSeries


class TestTiltSeriesStarIO:
    """Test STAR file I/O for TiltSeries"""

    def test_initialize_from_dataframe_required_columns(self):
        """Test initialization from DataFrame with required columns only"""
        # Create minimal STAR table
        data = {
            "wrpAngleTilt": [-60.0, -30.0, 0.0, 30.0, 60.0],
            "wrpDose": [0.0, 25.0, 50.0, 75.0, 100.0],
        }
        df = pd.DataFrame(data)

        # Initialize TiltSeries
        ts = TiltSeries()
        ts.initialize_from_tomo_star(df)

        # Check that values were loaded correctly
        assert ts.n_tilts == 5
        assert torch.allclose(
            ts.angles, torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0])
        )
        assert torch.allclose(ts.dose, torch.tensor([0.0, 25.0, 50.0, 75.0, 100.0]))

        # Check defaults for optional columns
        assert torch.allclose(ts.tilt_axis_angles, torch.zeros(5))
        assert torch.allclose(ts.tilt_axis_offset_x, torch.zeros(5))
        assert torch.allclose(ts.tilt_axis_offset_y, torch.zeros(5))
        assert ts.tilt_movie_paths == [""] * 5
        assert torch.all(ts.use_tilt == True)
        assert torch.allclose(ts.fov_fraction, torch.ones(5))

    def test_initialize_from_dataframe_all_columns(self):
        """Test initialization from DataFrame with all columns"""
        data = {
            "wrpAngleTilt": [-60.0, -30.0, 0.0],
            "wrpDose": [0.0, 50.0, 100.0],
            "wrpAxisAngle": [85.0, 85.5, 86.0],
            "wrpAxisOffsetX": [10.0, 20.0, 30.0],
            "wrpAxisOffsetY": [-5.0, -10.0, -15.0],
            "wrpMovieName": ["tilt_001.mrc", "tilt_002.mrc", "tilt_003.mrc"],
        }
        df = pd.DataFrame(data)

        ts = TiltSeries()
        ts.initialize_from_tomo_star(df)

        assert ts.n_tilts == 3
        assert torch.allclose(ts.angles, torch.tensor([-60.0, -30.0, 0.0]))
        assert torch.allclose(ts.dose, torch.tensor([0.0, 50.0, 100.0]))
        assert torch.allclose(ts.tilt_axis_angles, torch.tensor([85.0, 85.5, 86.0]))
        assert torch.allclose(ts.tilt_axis_offset_x, torch.tensor([10.0, 20.0, 30.0]))
        assert torch.allclose(ts.tilt_axis_offset_y, torch.tensor([-5.0, -10.0, -15.0]))
        assert ts.tilt_movie_paths == ["tilt_001.mrc", "tilt_002.mrc", "tilt_003.mrc"]

    def test_missing_required_columns(self):
        """Test error handling for missing required columns"""
        # Missing wrpDose
        df1 = pd.DataFrame({"wrpAngleTilt": [-60.0, 0.0, 60.0]})

        ts = TiltSeries()
        with pytest.raises(ValueError, match="must contain 'wrpDose' and 'wrpAngleTilt'"):
            ts.initialize_from_tomo_star(df1)

        # Missing wrpAngleTilt
        df2 = pd.DataFrame({"wrpDose": [0.0, 50.0, 100.0]})

        with pytest.raises(ValueError, match="must contain 'wrpDose' and 'wrpAngleTilt'"):
            ts.initialize_from_tomo_star(df2)

    def test_empty_dataframe(self):
        """Test error handling for empty DataFrame"""
        df = pd.DataFrame({"wrpAngleTilt": [], "wrpDose": []})

        ts = TiltSeries()
        with pytest.raises(ValueError, match="STAR table is empty"):
            ts.initialize_from_tomo_star(df)

    def test_partial_optional_columns(self):
        """Test with only some optional columns present"""
        # Only wrpAxisAngle, no offsets or movie names
        data = {
            "wrpAngleTilt": [-45.0, 0.0, 45.0],
            "wrpDose": [0.0, 60.0, 120.0],
            "wrpAxisAngle": [84.0, 85.0, 86.0],
        }
        df = pd.DataFrame(data)

        ts = TiltSeries()
        ts.initialize_from_tomo_star(df)

        assert torch.allclose(ts.tilt_axis_angles, torch.tensor([84.0, 85.0, 86.0]))
        assert torch.allclose(ts.tilt_axis_offset_x, torch.zeros(3))
        assert torch.allclose(ts.tilt_axis_offset_y, torch.zeros(3))
        assert ts.tilt_movie_paths == [""] * 3

    def test_data_types(self):
        """Test that data types are correctly converted to tensors"""
        data = {
            "wrpAngleTilt": [-60, -30, 0, 30, 60],  # integers
            "wrpDose": [0, 25, 50, 75, 100],  # integers
        }
        df = pd.DataFrame(data)

        ts = TiltSeries()
        ts.initialize_from_tomo_star(df)

        # Check tensor dtypes
        assert ts.angles.dtype == torch.float32
        assert ts.dose.dtype == torch.float32
        assert ts.tilt_axis_angles.dtype == torch.float32
        assert ts.tilt_axis_offset_x.dtype == torch.float32
        assert ts.tilt_axis_offset_y.dtype == torch.float32
        assert ts.use_tilt.dtype == torch.bool
        assert ts.fov_fraction.dtype == torch.float32

    def test_sorting_methods_after_star_load(self):
        """Test that sorting methods work after loading from STAR"""
        data = {
            "wrpAngleTilt": [30.0, -60.0, 0.0, 60.0, -30.0],
            "wrpDose": [75.0, 0.0, 50.0, 100.0, 25.0],
        }
        df = pd.DataFrame(data)

        ts = TiltSeries()
        ts.initialize_from_tomo_star(df)

        # Test angle sorting (angles: [30.0, -60.0, 0.0, 60.0, -30.0])
        # Sorted: -60.0(idx=1), -30.0(idx=4), 0.0(idx=2), 30.0(idx=0), 60.0(idx=3)
        sorted_by_angle = ts.indices_sorted_angle()
        assert torch.equal(sorted_by_angle, torch.tensor([1, 4, 2, 0, 3]))

        # Test absolute angle sorting (abs angles: [30.0, 60.0, 0.0, 60.0, 30.0])
        # Sorted: 0.0(idx=2), 30.0(idx=0), 30.0(idx=4), 60.0(idx=1), 60.0(idx=3)
        sorted_by_abs_angle = ts.indices_sorted_absolute_angle()
        assert torch.equal(sorted_by_abs_angle, torch.tensor([2, 0, 4, 1, 3]))

        # Test dose sorting (dose: [75.0, 0.0, 50.0, 100.0, 25.0])
        # Sorted: 0.0(idx=1), 25.0(idx=4), 50.0(idx=2), 75.0(idx=0), 100.0(idx=3)
        sorted_by_dose = ts.indices_sorted_dose()
        assert torch.equal(sorted_by_dose, torch.tensor([1, 4, 2, 0, 3]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
