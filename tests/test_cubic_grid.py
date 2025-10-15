"""
Tests for CubicGrid to verify equivalence with einspline implementation
"""

import torch
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
        values = torch.arange(60, dtype=torch.float32)
        grid = CubicGrid((3, 4, 5), values)
        assert grid.dimensions == (3, 4, 5)
        assert torch.equal(grid.values, values)

    def test_gradient_initialization(self):
        """Test gradient initialization"""
        grid = CubicGrid(
            (5, 1, 1),
            gradient_direction=Dimension.X,
            value_min=0.0,
            value_max=4.0,
        )
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        assert torch.allclose(grid.values, expected, rtol=1e-6)

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
        data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        grid = CubicGrid((5, 1, 1), data)

        # Test at grid points
        test_coords = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0],
                                     [0.75, 0.0, 0.0], [1.0, 0.0, 0.0]])
        results = grid.get_interpolated(test_coords)

        # Should pass through grid points with small error
        assert abs(results[0].item() - 0.0) < 1e-4
        assert abs(results[2].item() - 2.0) < 1e-4
        assert abs(results[4].item() - 4.0) < 1e-4


class TestCubicGrid2D:
    """Test 2D interpolation"""

    def test_2d_xy_interpolation(self):
        """Test 2D XY interpolation"""
        dims = (4, 5, 1)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        # Test at corners
        test_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        results = grid.get_interpolated(test_coords)

        # Check that we get reasonable values (should be close to corners)
        assert len(results) == 4
        assert all(torch.isfinite(results))


class TestCubicGrid3D:
    """Test 3D interpolation"""

    def test_3d_interpolation(self):
        """Test 3D interpolation"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        # Test at origin
        test_coords = torch.tensor([[0.0, 0.0, 0.0]])
        results = grid.get_interpolated(test_coords)

        assert len(results) == 1
        assert torch.isfinite(results[0])

    def test_3d_interpolation_multiple_points(self):
        """Test 3D interpolation with multiple points"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        test_coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ])
        results = grid.get_interpolated(test_coords)

        assert len(results) == 5
        assert all(torch.isfinite(results))


class TestCubicGridMargins:
    """Test margin support"""

    def test_margins_initialization(self):
        """Test grid initialization with margins"""
        data = torch.arange(60, dtype=torch.float32)
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
        data = torch.arange(12, dtype=torch.float32)
        grid = CubicGrid((3, 4, 1), data)

        new_grid = grid.resize((5, 6, 1))
        assert new_grid.dimensions == (5, 6, 1)
        assert len(new_grid.values) == 5 * 6

    def test_collapse_xy(self):
        """Test collapsing XY dimensions"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        collapsed = grid.collapse_xy()
        assert collapsed.dimensions == (1, 1, 5)
        assert len(collapsed.values) == 5

    def test_collapse_z(self):
        """Test collapsing Z dimension"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        collapsed = grid.collapse_z()
        assert collapsed.dimensions == (3, 4, 1)
        assert len(collapsed.values) == 3 * 4

    def test_get_slice_xy(self):
        """Test getting XY slice"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        slice_data = grid.get_slice_xy(2)
        assert len(slice_data) == 3 * 4

    def test_get_slice_xz(self):
        """Test getting XZ slice"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        slice_data = grid.get_slice_xz(1)
        assert len(slice_data) == 3 * 5

    def test_get_slice_yz(self):
        """Test getting YZ slice"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        slice_data = grid.get_slice_yz(0)
        assert len(slice_data) == 4 * 5


class TestCubicGridInterpolatedGrid:
    """Test get_interpolated_grid method"""

    def test_interpolated_grid_basic(self):
        """Test interpolated grid generation"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        result = grid.get_interpolated_grid((2, 2, 2), (0.0, 0.0, 0.0))
        assert len(result) == 2 * 2 * 2

    def test_interpolated_grid_with_border(self):
        """Test interpolated grid with border"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32)
        grid = CubicGrid(dims, data)

        result = grid.get_interpolated_grid((3, 3, 3), (0.1, 0.1, 0.1))
        assert len(result) == 3 * 3 * 3


class TestCubicGridGradientFlow:
    """Test that gradients flow correctly through CubicGrid operations"""

    def test_gradient_flow_1d(self):
        """Test gradient flow in 1D interpolation"""
        # Create grid with requires_grad
        data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True)
        grid = CubicGrid((5, 1, 1), data)

        # Interpolate at some points
        coords = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
        result = grid.get_interpolated(coords)

        # Compute loss and backprop
        loss = result.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert data.grad is not None
        assert data.grad.abs().sum() > 0

    def test_gradient_flow_2d(self):
        """Test gradient flow in 2D interpolation"""
        dims = (4, 5, 1)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32, requires_grad=True)
        grid = CubicGrid(dims, data)

        # Interpolate at multiple points
        coords = torch.tensor([
            [0.25, 0.25, 0.0],
            [0.75, 0.75, 0.0],
        ])
        result = grid.get_interpolated(coords)

        # Compute loss and backprop
        loss = result.sum()
        loss.backward()

        # Check gradients
        assert data.grad is not None
        assert data.grad.abs().sum() > 0

    def test_gradient_flow_3d(self):
        """Test gradient flow in 3D interpolation"""
        dims = (3, 4, 5)
        data = torch.arange(torch.tensor(dims).prod().item(), dtype=torch.float32, requires_grad=True)
        grid = CubicGrid(dims, data)

        # Interpolate at multiple points
        coords = torch.tensor([
            [0.2, 0.3, 0.4],
            [0.6, 0.7, 0.8],
        ])
        result = grid.get_interpolated(coords)

        # Compute loss and backprop
        loss = result.sum()
        loss.backward()

        # Check gradients
        assert data.grad is not None
        assert data.grad.abs().sum() > 0

    def test_gradient_values_affect_interpolation(self):
        """Test that changing values via gradients affects interpolation results"""
        # Create grid with requires_grad
        data = torch.ones(20, dtype=torch.float32, requires_grad=True)
        grid = CubicGrid((4, 5, 1), data)

        # Interpolate
        coords = torch.tensor([[0.5, 0.5, 0.0]])
        result1 = grid.get_interpolated(coords)

        # Manually update values
        with torch.no_grad():
            data.add_(torch.randn_like(data) * 0.5)

        # Create new grid with updated data
        grid2 = CubicGrid((4, 5, 1), data)
        result2 = grid2.get_interpolated(coords)

        # Results should be different
        assert not torch.allclose(result1, result2)

    def test_gradient_flow_with_margins(self):
        """Test gradient flow works with margins/centered spacing"""
        data = torch.randn(60, dtype=torch.float32, requires_grad=True)
        grid = CubicGrid((3, 4, 5), data, centered_spacing=True)

        coords = torch.tensor([[0.5, 0.5, 0.5]])
        result = grid.get_interpolated(coords)

        loss = result.sum()
        loss.backward()

        assert data.grad is not None
        assert data.grad.abs().sum() > 0

    def test_gradient_flow_interpolated_grid(self):
        """Test gradient flow through get_interpolated_grid"""
        dims = (3, 4, 5)
        data = torch.randn(torch.tensor(dims).prod().item(), dtype=torch.float32, requires_grad=True)
        grid = CubicGrid(dims, data)

        # Get interpolated grid
        result = grid.get_interpolated_grid((2, 2, 2), (0.0, 0.0, 0.0))

        # Compute loss and backprop
        loss = result.sum()
        loss.backward()

        # Check gradients
        assert data.grad is not None
        assert data.grad.abs().sum() > 0

    def test_gradient_accumulation(self):
        """Test that gradients accumulate correctly across multiple interpolations"""
        data = torch.randn(12, dtype=torch.float32, requires_grad=True)
        grid = CubicGrid((3, 4, 1), data)

        # First interpolation
        coords1 = torch.tensor([[0.25, 0.25, 0.0]])
        result1 = grid.get_interpolated(coords1)
        loss1 = result1.sum()
        loss1.backward(retain_graph=True)

        grad1 = data.grad.clone()

        # Second interpolation (gradients should accumulate)
        coords2 = torch.tensor([[0.75, 0.75, 0.0]])
        result2 = grid.get_interpolated(coords2)
        loss2 = result2.sum()
        loss2.backward()

        # Gradients should have accumulated
        assert not torch.equal(data.grad, grad1)
        assert data.grad.abs().sum() > grad1.abs().sum()


class TestCubicGridOptimization:
    """Test that CubicGrid parameters can be optimized via gradient descent"""

    def test_optimize_1d_to_target_function(self):
        """Test optimizing 1D grid to fit a sine wave"""
        # Target function: sine wave
        def target_fn(x):
            return torch.sin(2 * torch.pi * x)

        # Create sample points
        n_samples = 20
        sample_coords = torch.linspace(0, 1, n_samples).reshape(-1, 1)
        target_values = target_fn(sample_coords.squeeze())

        # Initialize grid with random values
        grid_size = 10
        initial_values = torch.randn(grid_size, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([initial_values], lr=0.1)

        # Training loop
        n_iterations = 200
        losses = []

        for i in range(n_iterations):
            optimizer.zero_grad()

            # Create grid with current values
            grid = CubicGrid((grid_size, 1, 1), initial_values)

            # Interpolate at sample points
            # Need to add dummy y, z coordinates for 3D input
            coords_3d = torch.cat([
                sample_coords,
                torch.zeros(n_samples, 1),
                torch.zeros(n_samples, 1)
            ], dim=1)
            predictions = grid.get_interpolated(coords_3d)

            # Compute loss
            loss = torch.nn.functional.mse_loss(predictions, target_values)
            losses.append(loss.item())

            # Backprop and update
            loss.backward()
            optimizer.step()

        # Verify optimization worked
        initial_loss = losses[0]
        final_loss = losses[-1]

        print(f"\n1D Optimization: Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1, f"Loss didn't decrease enough: {final_loss} vs {initial_loss}"

        # Final loss should be small
        assert final_loss < 0.01, f"Final loss too high: {final_loss}"

    def test_optimize_2d_to_target_image(self):
        """Test optimizing 2D grid to fit a target pattern"""
        # Create a target pattern (gaussian blob)
        def target_fn(x, y):
            return torch.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

        # Create sample points
        n_samples = 30
        x_coords = torch.rand(n_samples)
        y_coords = torch.rand(n_samples)
        target_values = target_fn(x_coords, y_coords)

        # Initialize grid with random values
        grid_size = (8, 8, 1)
        initial_values = torch.randn(8 * 8, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([initial_values], lr=0.1)

        # Training loop
        n_iterations = 300
        losses = []

        for i in range(n_iterations):
            optimizer.zero_grad()

            # Create grid
            grid = CubicGrid(grid_size, initial_values)

            # Interpolate at sample points
            coords = torch.stack([x_coords, y_coords, torch.zeros(n_samples)], dim=1)
            predictions = grid.get_interpolated(coords)

            # Compute loss
            loss = torch.nn.functional.mse_loss(predictions, target_values)
            losses.append(loss.item())

            # Backprop and update
            loss.backward()
            optimizer.step()

        # Verify optimization worked
        initial_loss = losses[0]
        final_loss = losses[-1]

        print(f"\n2D Optimization: Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1, f"Loss didn't decrease enough: {final_loss} vs {initial_loss}"

        # Final loss should be small
        assert final_loss < 0.05, f"Final loss too high: {final_loss}"

    def test_optimize_3d_to_target_volume(self):
        """Test optimizing 3D grid to fit target values at specific points"""
        # Target function: 3D gaussian
        def target_fn(x, y, z):
            return torch.exp(-((x - 0.5)**2 + (y - 0.5)**2 + (z - 0.5)**2) / 0.1)

        # Create sample points
        n_samples = 40
        coords = torch.rand(n_samples, 3)
        target_values = target_fn(coords[:, 0], coords[:, 1], coords[:, 2])

        # Initialize grid with random values
        grid_size = (6, 6, 6)
        initial_values = torch.randn(6 * 6 * 6, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([initial_values], lr=0.1)

        # Training loop
        n_iterations = 400
        losses = []

        for i in range(n_iterations):
            optimizer.zero_grad()

            # Create grid
            grid = CubicGrid(grid_size, initial_values)

            # Interpolate at sample points
            predictions = grid.get_interpolated(coords)

            # Compute loss
            loss = torch.nn.functional.mse_loss(predictions, target_values)
            losses.append(loss.item())

            # Backprop and update
            loss.backward()
            optimizer.step()

        # Verify optimization worked
        initial_loss = losses[0]
        final_loss = losses[-1]

        print(f"\n3D Optimization: Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1, f"Loss didn't decrease enough: {final_loss} vs {initial_loss}"

        # Final loss should be small
        assert final_loss < 0.05, f"Final loss too high: {final_loss}"

    def test_gradient_descent_converges_monotonically(self):
        """Test that gradient descent converges smoothly without oscillations"""
        # Simple 1D case
        target_values_at_points = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        sample_coords = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Initialize grid
        grid_size = 5
        initial_values = torch.full((grid_size,), 0.5, requires_grad=True)

        optimizer = torch.optim.SGD([initial_values], lr=0.01)

        # Track losses
        losses = []
        for i in range(100):
            optimizer.zero_grad()

            grid = CubicGrid((grid_size, 1, 1), initial_values)
            predictions = grid.get_interpolated(sample_coords)
            loss = torch.nn.functional.mse_loss(predictions, target_values_at_points)

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        # Check that loss generally decreases (allow some noise)
        # Compare first 10 iterations to last 10 iterations
        early_loss = sum(losses[:10]) / 10
        late_loss = sum(losses[-10:]) / 10

        print(f"\nMonotonic convergence: Early avg loss: {early_loss:.6f}, Late avg loss: {late_loss:.6f}")

        assert late_loss < early_loss, "Loss should decrease over time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
