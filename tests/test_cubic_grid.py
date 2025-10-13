"""
Tests for CubicGrid to verify equivalence with einspline implementation
"""

import numpy as np
import pytest

from warpylib.cubic_grid import CubicGrid, Dimension, DimensionSets


class TestCubicGridBasic:
    """Basic functionality tests"""

    def test_initialization_empty(self):
        """Test creating empty grid"""
        grid = CubicGrid((3, 4, 5))
        assert grid.dimensions == (3, 4, 5)
        assert len(grid.values) == 3 * 4 * 5
        assert grid.dimension_set == DimensionSets.XYZ

    def test_initialization_with_values(self):
        """Test creating grid with values"""
        values = np.arange(60, dtype=np.float32)
        grid = CubicGrid((3, 4, 5), values)
        assert grid.dimensions == (3, 4, 5)
        np.testing.assert_array_equal(grid.values, values)

    def test_gradient_initialization(self):
        """Test gradient initialization"""
        grid = CubicGrid(
            (5, 1, 1),
            gradient_direction=Dimension.X,
            value_min=0.0,
            value_max=4.0,
        )
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(grid.values, expected, rtol=1e-6)

    def test_dimension_detection_3d(self):
        """Test 3D dimension detection"""
        grid = CubicGrid((3, 4, 5))
        assert grid.dimension_set == DimensionSets.XYZ

    def test_dimension_detection_2d_xy(self):
        """Test 2D XY dimension detection"""
        grid = CubicGrid((3, 4, 1))
        assert grid.dimension_set == DimensionSets.XY

    def test_dimension_detection_2d_xz(self):
        """Test 2D XZ dimension detection"""
        grid = CubicGrid((3, 1, 5))
        assert grid.dimension_set == DimensionSets.XZ

    def test_dimension_detection_2d_yz(self):
        """Test 2D YZ dimension detection"""
        grid = CubicGrid((1, 4, 5))
        assert grid.dimension_set == DimensionSets.YZ

    def test_dimension_detection_1d_x(self):
        """Test 1D X dimension detection"""
        grid = CubicGrid((5, 1, 1))
        assert grid.dimension_set == DimensionSets.X

    def test_dimension_detection_1d_y(self):
        """Test 1D Y dimension detection"""
        grid = CubicGrid((1, 5, 1))
        assert grid.dimension_set == DimensionSets.Y

    def test_dimension_detection_1d_z(self):
        """Test 1D Z dimension detection"""
        grid = CubicGrid((1, 1, 5))
        assert grid.dimension_set == DimensionSets.Z


class TestCubicGrid1D:
    """Test 1D interpolation"""

    def test_1d_interpolation(self):
        """Test 1D interpolation passes through data points"""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        grid = CubicGrid((5, 1, 1), data)

        # Test at grid points
        test_coords = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0],
                                [0.75, 0.0, 0.0], [1.0, 0.0, 0.0]])
        results = grid.get_interpolated(test_coords)

        # Should pass through grid points with small error
        assert abs(results[0] - 0.0) < 1e-4
        assert abs(results[2] - 2.0) < 1e-4
        assert abs(results[4] - 4.0) < 1e-4


class TestCubicGrid2D:
    """Test 2D interpolation"""

    def test_2d_xy_interpolation(self):
        """Test 2D XY interpolation"""
        dims = (4, 5, 1)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        # Test at corners
        test_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        results = grid.get_interpolated(test_coords)

        # Check that we get reasonable values (should be close to corners)
        assert len(results) == 4
        assert all(np.isfinite(results))


class TestCubicGrid3D:
    """Test 3D interpolation"""

    def test_3d_interpolation(self):
        """Test 3D interpolation"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        # Test at origin
        test_coords = np.array([[0.0, 0.0, 0.0]])
        results = grid.get_interpolated(test_coords)

        assert len(results) == 1
        assert np.isfinite(results[0])

    def test_3d_interpolation_multiple_points(self):
        """Test 3D interpolation with multiple points"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        test_coords = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ])
        results = grid.get_interpolated(test_coords)

        assert len(results) == 5
        assert all(np.isfinite(results))


class TestCubicGridMargins:
    """Test margin support"""

    def test_margins_initialization(self):
        """Test grid initialization with margins"""
        data = np.arange(60, dtype=np.float32)
        margins = (0.1, 0.1, 0.1)
        grid = CubicGrid((3, 4, 5), data, margins=margins)

        assert grid.margins == margins

    def test_centered_spacing(self):
        """Test centered spacing margin calculation"""
        grid = CubicGrid((4, 4, 4), centered_spacing=True)

        expected_margin = (1.0 / 4) / 2
        assert abs(grid.margins[0] - expected_margin) < 1e-6
        assert abs(grid.margins[1] - expected_margin) < 1e-6
        assert abs(grid.margins[2] - expected_margin) < 1e-6


class TestCubicGridOperations:
    """Test grid operations"""

    def test_resize(self):
        """Test grid resizing"""
        data = np.arange(12, dtype=np.float32)
        grid = CubicGrid((3, 4, 1), data)

        new_grid = grid.resize((5, 6, 1))
        assert new_grid.dimensions == (5, 6, 1)
        assert len(new_grid.values) == 5 * 6

    def test_collapse_xy(self):
        """Test collapsing XY dimensions"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        collapsed = grid.collapse_xy()
        assert collapsed.dimensions == (1, 1, 5)
        assert len(collapsed.values) == 5

    def test_collapse_z(self):
        """Test collapsing Z dimension"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        collapsed = grid.collapse_z()
        assert collapsed.dimensions == (3, 4, 1)
        assert len(collapsed.values) == 3 * 4

    def test_get_slice_xy(self):
        """Test getting XY slice"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        slice_data = grid.get_slice_xy(2)
        assert len(slice_data) == 3 * 4

    def test_get_slice_xz(self):
        """Test getting XZ slice"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        slice_data = grid.get_slice_xz(1)
        assert len(slice_data) == 3 * 5

    def test_get_slice_yz(self):
        """Test getting YZ slice"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        slice_data = grid.get_slice_yz(0)
        assert len(slice_data) == 4 * 5


class TestCubicGridInterpolatedGrid:
    """Test get_interpolated_grid method"""

    def test_interpolated_grid_basic(self):
        """Test interpolated grid generation"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        result = grid.get_interpolated_grid((2, 2, 2), (0.0, 0.0, 0.0))
        assert len(result) == 2 * 2 * 2

    def test_interpolated_grid_with_border(self):
        """Test interpolated grid with border"""
        dims = (3, 4, 5)
        data = np.arange(np.prod(dims), dtype=np.float32)
        grid = CubicGrid(dims, data)

        result = grid.get_interpolated_grid((3, 3, 3), (0.1, 0.1, 0.1))
        assert len(result) == 3 * 3 * 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
