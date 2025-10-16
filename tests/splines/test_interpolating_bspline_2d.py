"""
Tests for 2D interpolating B-spline implementation.
"""

import torch
import numpy as np
import pytest
from warpylib.interpolating_bspline import (
    InterpolatingBSpline2d,
    find_coefs_2d,
    interpolate_grid_2d,
)


def test_coefficient_computation_2d():
    """Test the 2D coefficient computation step."""
    # Simple 4x4 test data
    data = torch.tensor([
        [1.0, 2.0, 1.5, 2.5],
        [2.0, 3.0, 2.5, 3.5],
        [1.5, 2.5, 2.0, 3.0],
        [2.5, 3.5, 3.0, 4.0],
    ])

    # Compute coefficients
    coefs = find_coefs_2d(data)

    print("\n=== 2D Coefficient Computation Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Coefficients shape: {coefs.shape}")

    # Should have (Mx+2, My+2) coefficients for (Mx, My) data points
    assert coefs.shape[0] == data.shape[0] + 2, \
        f"Expected X dimension {data.shape[0] + 2}, got {coefs.shape[0]}"
    assert coefs.shape[1] == data.shape[1] + 2, \
        f"Expected Y dimension {data.shape[1] + 2}, got {coefs.shape[1]}"


def test_interpolation_at_data_points_2d():
    """Test that the 2D spline passes through all data points (interpolation property)."""
    # Create 4x4 test data
    data = torch.tensor([
        [1.0, 2.0, 1.5, 2.5],
        [2.0, 3.0, 2.5, 3.5],
        [1.5, 2.5, 2.0, 3.0],
        [2.5, 3.5, 3.0, 4.0],
    ])

    # Create spline using from_grid_data (API-compliant)
    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Evaluate at the data point locations
    # For 4x4 points spanning [0, 1] x [0, 1], they are at:
    # x: 0.0, 0.333, 0.666, 1.0
    # y: 0.0, 0.333, 0.666, 1.0
    Mx, My = data.shape
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)

    # Create all combinations of (x, y)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (16, 2)

    with torch.no_grad():
        values = spline(eval_points)

    # Reshape values back to grid
    values = values.squeeze(-1).reshape(Mx, My)

    print("\n=== 2D Interpolation at Data Points Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Data:\n{data.numpy()}")
    print(f"Interpolated values:\n{values.numpy()}")
    print(f"Differences:\n{(values - data).numpy()}")
    print(f"Max absolute error: {torch.max(torch.abs(values - data)).item():.6e}")

    # The spline should pass through all data points
    assert torch.allclose(values, data, atol=1e-5), \
        f"Spline does not interpolate data points. Max error: {torch.max(torch.abs(values - data)).item()}"


def test_separable_property():
    """Test that separable interpolation works correctly."""
    # Create data that is separable: f(x, y) = g(x) * h(y)
    x_vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_vals = torch.tensor([1.0, 1.5, 2.0, 2.5])

    # Create separable 2D data
    data = x_vals[:, None] * y_vals[None, :]  # (4, 4)

    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Test at data points
    Mx, My = data.shape
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        values = spline(eval_points)

    values = values.squeeze(-1).reshape(Mx, My)

    print("\n=== Separable Property Test ===")
    print(f"Data:\n{data.numpy()}")
    print(f"Interpolated:\n{values.numpy()}")
    print(f"Max error: {torch.max(torch.abs(values - data)).item():.6e}")

    assert torch.allclose(values, data, atol=1e-5), \
        "Separable interpolation failed"


def test_evaluation_between_points_2d():
    """Test evaluation at points between data points."""
    # Simple data
    data = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])

    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Evaluate at center and some intermediate points
    eval_points = torch.tensor([
        [0.5, 0.5],   # Center of first cell
        [0.25, 0.25], # Quarter way through first cell
        [0.75, 0.75], # Three-quarters through
    ])

    with torch.no_grad():
        values = spline(eval_points)

    print("\n=== 2D Evaluation Between Points Test ===")
    print(f"Eval points:\n{eval_points.numpy()}")
    print(f"Interpolated values:\n{values.numpy()}")

    # Values should be in reasonable range based on data
    assert torch.all(values >= -0.5), "Values too small"
    assert torch.all(values <= 6.5), "Values too large"


def test_differentiability_2d():
    """Test that gradients flow through the 2D spline."""
    # Create spline with learnable parameters
    spline = InterpolatingBSpline2d(resolution=(4, 4), n_channels=1)

    # Set some initial data
    with torch.no_grad():
        spline._data.copy_(torch.tensor([[
            [1.0, 2.0, 1.5, 2.5],
            [2.0, 3.0, 2.5, 3.5],
            [1.5, 2.5, 2.0, 3.0],
            [2.5, 3.5, 3.0, 4.0],
        ]]))

    # Evaluate at some points
    eval_points = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])
    values = spline(eval_points)

    # Compute loss and backprop
    loss = values.sum()
    loss.backward()

    print("\n=== 2D Differentiability Test ===")
    print(f"Data shape: {spline._data.shape}")
    print(f"Eval points:\n{eval_points.numpy()}")
    print(f"Values: {values.detach().numpy()}")
    print(f"Loss: {loss.item()}")
    print(f"Gradient norm: {torch.norm(spline._data.grad).item():.6f}")

    # Check that gradients exist and are not all zero
    assert spline._data.grad is not None, "No gradients computed"
    assert torch.any(spline._data.grad != 0), "All gradients are zero"


def test_multichannel_2d():
    """Test multi-channel 2D splines."""
    # 2 channels, 4x4 points each
    data = torch.tensor([
        [  # Channel 0
            [1.0, 2.0, 1.5, 2.5],
            [2.0, 3.0, 2.5, 3.5],
            [1.5, 2.5, 2.0, 3.0],
            [2.5, 3.5, 3.0, 4.0],
        ],
        [  # Channel 1
            [0.5, 1.0, 0.8, 1.2],
            [1.0, 1.5, 1.3, 1.7],
            [0.8, 1.3, 1.1, 1.5],
            [1.2, 1.7, 1.5, 1.9],
        ],
    ])

    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Evaluate at data points
    Mx, My = data.shape[1:]
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        values = spline(eval_points)

    print("\n=== 2D Multi-channel Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Expected shape: ({Mx * My}, 2)")

    # Each channel should interpolate independently
    assert values.shape == (Mx * My, 2), f"Expected shape ({Mx * My}, 2), got {values.shape}"

    # Reshape and check each channel
    for c in range(data.shape[0]):
        values_c = values[:, c].reshape(Mx, My)
        error = torch.max(torch.abs(values_c - data[c])).item()
        print(f"Channel {c} max error: {error:.6e}")
        assert torch.allclose(values_c, data[c], atol=1e-5), \
            f"Multi-channel interpolation failed for channel {c}"


def test_api_compliance_2d():
    """Test API compliance with torch-cubic-spline-grids for 2D."""
    # Test initialization with resolution parameter
    spline1 = InterpolatingBSpline2d(resolution=(6, 8), n_channels=2)
    assert spline1.resolution == (6, 8), f"Expected resolution (6, 8), got {spline1.resolution}"
    assert spline1.n_channels == 2, f"Expected n_channels 2, got {spline1.n_channels}"
    assert spline1.ndim == 2, f"Expected ndim 2, got {spline1.ndim}"

    # Test from_grid_data class method
    data = torch.rand(2, 4, 5)
    spline2 = InterpolatingBSpline2d.from_grid_data(data)
    assert spline2.resolution == (4, 5), f"Expected resolution (4, 5), got {spline2.resolution}"
    assert spline2.n_channels == 2, f"Expected n_channels 2, got {spline2.n_channels}"
    assert torch.allclose(spline2.data, data), "from_grid_data did not set data correctly"

    # Test data property getter/setter
    new_data = torch.rand(2, 4, 5)
    spline2.data = new_data
    assert torch.allclose(spline2.data, new_data), "data setter did not work correctly"

    print("\n=== 2D API Compliance Test ===")
    print("All 2D API compliance checks passed!")


def test_minibatch_processing_2d():
    """Test that minibatch processing works correctly for 2D."""
    data = torch.tensor([
        [1.0, 2.0, 1.5, 2.5],
        [2.0, 3.0, 2.5, 3.5],
        [1.5, 2.5, 2.0, 3.0],
        [2.5, 3.5, 3.0, 4.0],
    ])

    # Create spline with small minibatch size
    spline = InterpolatingBSpline2d.from_grid_data(data)
    spline._minibatch_size = 3  # Force minibatching with batch size 3

    # Evaluate at many points
    n_points = 10
    u = torch.rand(n_points, 2)  # Random points in [0, 1] x [0, 1]

    with torch.no_grad():
        values = spline(u)

    # Should get correct shape
    assert values.shape == (n_points, 1), f"Expected shape ({n_points}, 1), got {values.shape}"

    print("\n=== 2D Minibatch Processing Test ===")
    print(f"Input points: {n_points}")
    print(f"Minibatch size: {spline._minibatch_size}")
    print(f"Output shape: {values.shape}")
    print("2D Minibatch processing works correctly!")


def test_corner_cases_2d():
    """Test boundary evaluation for 2D splines."""
    data = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])

    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Test corners
    corners = torch.tensor([
        [0.0, 0.0],  # Bottom-left
        [1.0, 0.0],  # Bottom-right
        [0.0, 1.0],  # Top-left
        [1.0, 1.0],  # Top-right
    ])

    with torch.no_grad():
        values = spline(corners)

    expected = torch.tensor([[1.0], [3.0], [3.0], [5.0]])

    print("\n=== 2D Corner Cases Test ===")
    print(f"Corner coordinates:\n{corners.numpy()}")
    print(f"Expected values:\n{expected.numpy()}")
    print(f"Interpolated values:\n{values.numpy()}")
    print(f"Max error: {torch.max(torch.abs(values - expected)).item():.6e}")

    assert torch.allclose(values, expected, atol=1e-5), \
        "Corner interpolation failed"


if __name__ == "__main__":
    test_coefficient_computation_2d()
    test_interpolation_at_data_points_2d()
    test_separable_property()
    test_evaluation_between_points_2d()
    test_differentiability_2d()
    test_multichannel_2d()
    test_api_compliance_2d()
    test_minibatch_processing_2d()
    test_corner_cases_2d()
    print("\n=== All 2D tests passed! ===")
