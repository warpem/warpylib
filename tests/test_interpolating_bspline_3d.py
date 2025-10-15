"""
Tests for 3D interpolating B-spline implementation.
"""

import torch
import numpy as np
import pytest
from warpylib.interpolating_bspline import (
    InterpolatingBSpline3d,
    find_coefs_3d,
    interpolate_grid_3d,
)


def test_coefficient_computation_3d():
    """Test the 3D coefficient computation step."""
    # Simple 3x3x3 test data
    data = torch.rand(3, 3, 3)

    # Compute coefficients
    coefs = find_coefs_3d(data)

    print("\n=== 3D Coefficient Computation Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Coefficients shape: {coefs.shape}")

    # Should have (Mx+2, My+2, Mz+2) coefficients for (Mx, My, Mz) data points
    assert coefs.shape[0] == data.shape[0] + 2, \
        f"Expected X dimension {data.shape[0] + 2}, got {coefs.shape[0]}"
    assert coefs.shape[1] == data.shape[1] + 2, \
        f"Expected Y dimension {data.shape[1] + 2}, got {coefs.shape[1]}"
    assert coefs.shape[2] == data.shape[2] + 2, \
        f"Expected Z dimension {data.shape[2] + 2}, got {coefs.shape[2]}"


def test_interpolation_at_data_points_3d():
    """Test that the 3D spline passes through all data points (interpolation property)."""
    # Create 3x3x3 test data
    data = torch.tensor([
        [  # z=0
            [1.0, 2.0, 1.5],
            [2.0, 3.0, 2.5],
            [1.5, 2.5, 2.0],
        ],
        [  # z=1
            [2.0, 3.0, 2.5],
            [3.0, 4.0, 3.5],
            [2.5, 3.5, 3.0],
        ],
        [  # z=2
            [1.5, 2.5, 2.0],
            [2.5, 3.5, 3.0],
            [2.0, 3.0, 2.5],
        ],
    ])

    # Create spline using from_grid_data (API-compliant)
    spline = InterpolatingBSpline3d.from_grid_data(data)

    # Evaluate at the data point locations
    # For 3x3x3 points spanning [0, 1]^3
    Mx, My, Mz = data.shape
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)
    z_coords = torch.linspace(0.0, 1.0, Mz)

    # Create all combinations of (x, y, z)
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)  # (27, 3)

    with torch.no_grad():
        values = spline(eval_points)

    # Reshape values back to grid
    values = values.squeeze(-1).reshape(Mx, My, Mz)

    print("\n=== 3D Interpolation at Data Points Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Max absolute error: {torch.max(torch.abs(values - data)).item():.6e}")

    # The spline should pass through all data points
    assert torch.allclose(values, data, atol=1e-5), \
        f"Spline does not interpolate data points. Max error: {torch.max(torch.abs(values - data)).item()}"


def test_separable_property_3d():
    """Test that separable interpolation works correctly in 3D."""
    # Create data that is separable: f(x, y, z) = g(x) * h(y) * k(z)
    x_vals = torch.tensor([1.0, 2.0, 3.0])
    y_vals = torch.tensor([1.0, 1.5, 2.0])
    z_vals = torch.tensor([1.0, 2.0, 3.0])

    # Create separable 3D data
    data = x_vals[:, None, None] * y_vals[None, :, None] * z_vals[None, None, :]  # (3, 3, 3)

    spline = InterpolatingBSpline3d.from_grid_data(data)

    # Test at data points
    Mx, My, Mz = data.shape
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)
    z_coords = torch.linspace(0.0, 1.0, Mz)
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    with torch.no_grad():
        values = spline(eval_points)

    values = values.squeeze(-1).reshape(Mx, My, Mz)

    print("\n=== 3D Separable Property Test ===")
    print(f"Max error: {torch.max(torch.abs(values - data)).item():.6e}")

    assert torch.allclose(values, data, atol=1e-5), \
        "Separable 3D interpolation failed"


def test_evaluation_between_points_3d():
    """Test evaluation at points between data points in 3D."""
    # Simple linear data
    data = torch.arange(27, dtype=torch.float32).reshape(3, 3, 3)

    spline = InterpolatingBSpline3d.from_grid_data(data)

    # Evaluate at some intermediate points
    eval_points = torch.tensor([
        [0.5, 0.5, 0.5],   # Center of volume
        [0.25, 0.25, 0.25], # Quarter way through
        [0.75, 0.75, 0.75], # Three-quarters through
    ])

    with torch.no_grad():
        values = spline(eval_points)

    print("\n=== 3D Evaluation Between Points Test ===")
    print(f"Eval points:\n{eval_points.numpy()}")
    print(f"Interpolated values:\n{values.numpy()}")

    # Values should be in reasonable range based on data
    assert torch.all(values >= -1.0), "Values too small"
    assert torch.all(values <= 27.0), "Values too large"


def test_differentiability_3d():
    """Test that gradients flow through the 3D spline."""
    # Create spline with learnable parameters
    spline = InterpolatingBSpline3d(resolution=(3, 3, 3), n_channels=1)

    # Set some initial data
    with torch.no_grad():
        spline._data.copy_(torch.rand(1, 3, 3, 3))

    # Evaluate at some points
    eval_points = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ])
    values = spline(eval_points)

    # Compute loss and backprop
    loss = values.sum()
    loss.backward()

    print("\n=== 3D Differentiability Test ===")
    print(f"Data shape: {spline._data.shape}")
    print(f"Eval points:\n{eval_points.numpy()}")
    print(f"Values: {values.detach().numpy()}")
    print(f"Loss: {loss.item()}")
    print(f"Gradient norm: {torch.norm(spline._data.grad).item():.6f}")

    # Check that gradients exist and are not all zero
    assert spline._data.grad is not None, "No gradients computed"
    assert torch.any(spline._data.grad != 0), "All gradients are zero"


def test_multichannel_3d():
    """Test multi-channel 3D splines."""
    # 2 channels, 3x3x3 points each
    data = torch.stack([
        torch.arange(27, dtype=torch.float32).reshape(3, 3, 3),  # Channel 0
        torch.arange(27, dtype=torch.float32).reshape(3, 3, 3) * 0.5,  # Channel 1
    ])

    spline = InterpolatingBSpline3d.from_grid_data(data)

    # Evaluate at data points
    Mx, My, Mz = data.shape[1:]
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)
    z_coords = torch.linspace(0.0, 1.0, Mz)
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    with torch.no_grad():
        values = spline(eval_points)

    print("\n=== 3D Multi-channel Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Expected shape: ({Mx * My * Mz}, 2)")

    # Each channel should interpolate independently
    assert values.shape == (Mx * My * Mz, 2), f"Expected shape ({Mx * My * Mz}, 2), got {values.shape}"

    # Reshape and check each channel
    for c in range(data.shape[0]):
        values_c = values[:, c].reshape(Mx, My, Mz)
        error = torch.max(torch.abs(values_c - data[c])).item()
        print(f"Channel {c} max error: {error:.6e}")
        assert torch.allclose(values_c, data[c], atol=1e-5), \
            f"Multi-channel 3D interpolation failed for channel {c}"


def test_api_compliance_3d():
    """Test API compliance with torch-cubic-spline-grids for 3D."""
    # Test initialization with resolution parameter
    spline1 = InterpolatingBSpline3d(resolution=(4, 5, 6), n_channels=2)
    assert spline1.resolution == (4, 5, 6), f"Expected resolution (4, 5, 6), got {spline1.resolution}"
    assert spline1.n_channels == 2, f"Expected n_channels 2, got {spline1.n_channels}"
    assert spline1.ndim == 3, f"Expected ndim 3, got {spline1.ndim}"

    # Test from_grid_data class method
    data = torch.rand(2, 3, 4, 5)
    spline2 = InterpolatingBSpline3d.from_grid_data(data)
    assert spline2.resolution == (3, 4, 5), f"Expected resolution (3, 4, 5), got {spline2.resolution}"
    assert spline2.n_channels == 2, f"Expected n_channels 2, got {spline2.n_channels}"
    assert torch.allclose(spline2.data, data), "from_grid_data did not set data correctly"

    # Test data property getter/setter
    new_data = torch.rand(2, 3, 4, 5)
    spline2.data = new_data
    assert torch.allclose(spline2.data, new_data), "data setter did not work correctly"

    print("\n=== 3D API Compliance Test ===")
    print("All 3D API compliance checks passed!")


def test_minibatch_processing_3d():
    """Test that minibatch processing works correctly for 3D."""
    data = torch.rand(3, 3, 3)

    # Create spline with small minibatch size
    spline = InterpolatingBSpline3d.from_grid_data(data)
    spline._minibatch_size = 3  # Force minibatching with batch size 3

    # Evaluate at many points
    n_points = 10
    u = torch.rand(n_points, 3)  # Random points in [0, 1]^3

    with torch.no_grad():
        values = spline(u)

    # Should get correct shape
    assert values.shape == (n_points, 1), f"Expected shape ({n_points}, 1), got {values.shape}"

    print("\n=== 3D Minibatch Processing Test ===")
    print(f"Input points: {n_points}")
    print(f"Minibatch size: {spline._minibatch_size}")
    print(f"Output shape: {values.shape}")
    print("3D Minibatch processing works correctly!")


def test_corner_cases_3d():
    """Test boundary evaluation for 3D splines."""
    # Simple 2x2x2 data for easy corner verification
    data = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])

    spline = InterpolatingBSpline3d.from_grid_data(data)

    # Test 8 corners of the unit cube
    corners = torch.tensor([
        [0.0, 0.0, 0.0],  # Corner 0
        [1.0, 0.0, 0.0],  # Corner 1
        [0.0, 1.0, 0.0],  # Corner 2
        [1.0, 1.0, 0.0],  # Corner 3
        [0.0, 0.0, 1.0],  # Corner 4
        [1.0, 0.0, 1.0],  # Corner 5
        [0.0, 1.0, 1.0],  # Corner 6
        [1.0, 1.0, 1.0],  # Corner 7
    ])

    with torch.no_grad():
        values = spline(corners)

    expected = torch.tensor([[1.0], [5.0], [3.0], [7.0], [2.0], [6.0], [4.0], [8.0]])

    print("\n=== 3D Corner Cases Test ===")
    print(f"Corner coordinates:\n{corners.numpy()}")
    print(f"Expected values:\n{expected.numpy()}")
    print(f"Interpolated values:\n{values.numpy()}")
    print(f"Max error: {torch.max(torch.abs(values - expected)).item():.6e}")

    assert torch.allclose(values, expected, atol=1e-5), \
        "3D corner interpolation failed"


def test_larger_grid_3d():
    """Test 3D interpolation on a larger grid."""
    # Create a 5x5x5 grid
    data = torch.rand(5, 5, 5)

    spline = InterpolatingBSpline3d.from_grid_data(data)

    # Verify interpolation at data points
    Mx, My, Mz = data.shape
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)
    z_coords = torch.linspace(0.0, 1.0, Mz)
    xx, yy, zz = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    eval_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    with torch.no_grad():
        values = spline(eval_points)

    values = values.squeeze(-1).reshape(Mx, My, Mz)

    print("\n=== 3D Larger Grid Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Max absolute error: {torch.max(torch.abs(values - data)).item():.6e}")

    assert torch.allclose(values, data, atol=1e-5), \
        "Large 3D grid interpolation failed"


if __name__ == "__main__":
    test_coefficient_computation_3d()
    test_interpolation_at_data_points_3d()
    test_separable_property_3d()
    test_evaluation_between_points_3d()
    test_differentiability_3d()
    test_multichannel_3d()
    test_api_compliance_3d()
    test_minibatch_processing_3d()
    test_corner_cases_3d()
    test_larger_grid_3d()
    print("\n=== All 3D tests passed! ===")
