"""
Tests for interpolating B-spline implementation.
"""

import torch
import numpy as np
import pytest
from warpylib.interpolating_bspline import (
    InterpolatingBSpline1d,
    find_coefs_1d,
    interpolate_grid_1d,
)


def test_interpolation_at_data_points():
    """Test that the spline passes through all data points (interpolation property)."""
    # Create simple test data with 6 points
    data = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 2.0])

    # Create spline using from_grid_data (API-compliant)
    spline = InterpolatingBSpline1d.from_grid_data(data)

    # Evaluate at the data point locations
    # For 6 points spanning [0, 1], they are at 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    M = len(data)
    eval_points = torch.linspace(0.0, 1.0, M)

    with torch.no_grad():
        values = spline(eval_points)

    print("\n=== Interpolation at Data Points Test ===")
    print(f"Data points: {data.numpy()}")
    print(f"Eval locations: {eval_points.numpy()}")
    print(f"Interpolated values: {values.numpy()}")
    print(f"Differences: {(values - data).numpy()}")
    print(f"Max absolute error: {torch.max(torch.abs(values - data)).item():.6e}")

    # The spline should pass through all data points
    assert torch.allclose(values, data, atol=1e-5), \
        f"Spline does not interpolate data points. Max error: {torch.max(torch.abs(values - data)).item()}"


def test_coefficient_computation():
    """Test the coefficient computation step."""
    # Simple test data
    data = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 2.0])

    # Compute coefficients
    coefs = find_coefs_1d(data)

    print("\n=== Coefficient Computation Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Data: {data.numpy()}")
    print(f"Coefficients shape: {coefs.shape}")
    print(f"Coefficients: {coefs.numpy()}")

    # Should have M+2 coefficients for M data points
    assert coefs.shape[0] == len(data) + 2, \
        f"Expected {len(data) + 2} coefficients, got {coefs.shape[0]}"


def test_evaluation_between_points():
    """Test evaluation at points between data points."""
    # Simple linear data
    data = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    spline = InterpolatingBSpline1d.from_grid_data(data)

    # Evaluate at midpoints
    M = len(data)
    midpoints = torch.linspace(0.0, 1.0, M * 2 - 1)

    with torch.no_grad():
        values = spline(midpoints)

    print("\n=== Evaluation Between Points Test ===")
    print(f"Data: {data.numpy()}")
    print(f"Eval points: {midpoints.numpy()}")
    print(f"Interpolated values: {values.numpy()}")

    # For linear data, the spline should be close to linear interpolation
    # Check that values are in reasonable range
    assert torch.all(values >= -0.5), "Values too small"
    assert torch.all(values <= 5.5), "Values too large"


def test_differentiability():
    """Test that gradients flow through the spline."""
    # Create spline with learnable parameters
    spline = InterpolatingBSpline1d(resolution=6, n_channels=1)

    # Set some initial data
    with torch.no_grad():
        spline._data.copy_(torch.tensor([[1.0, 2.0, 1.5, 3.0, 2.5, 2.0]]))

    # Evaluate at some points
    eval_points = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    values = spline(eval_points)

    # Compute loss and backprop
    loss = values.sum()
    loss.backward()

    print("\n=== Differentiability Test ===")
    print(f"Data: {spline._data.detach().numpy()}")
    print(f"Eval points: {eval_points.numpy()}")
    print(f"Values: {values.detach().numpy()}")
    print(f"Loss: {loss.item()}")
    print(f"Data gradients: {spline._data.grad.numpy()}")

    # Check that gradients exist and are not all zero
    assert spline._data.grad is not None, "No gradients computed"
    assert torch.any(spline._data.grad != 0), "All gradients are zero"


def test_multichannel():
    """Test multi-channel splines."""
    # 3 channels, 6 points each
    data = torch.tensor([
        [1.0, 2.0, 1.5, 3.0, 2.5, 2.0],
        [0.5, 1.0, 0.8, 1.5, 1.2, 1.0],
        [2.0, 3.0, 2.5, 4.0, 3.5, 3.0],
    ])

    spline = InterpolatingBSpline1d.from_grid_data(data)

    # Evaluate at data points
    M = data.shape[1]
    eval_points = torch.linspace(0.0, 1.0, M)

    with torch.no_grad():
        values = spline(eval_points)

    print("\n=== Multi-channel Test ===")
    print(f"Data shape: {data.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Data:\n{data.numpy()}")
    print(f"Values:\n{values.numpy()}")
    print(f"Max error per channel:")
    for c in range(data.shape[0]):
        error = torch.max(torch.abs(values[:, c] - data[c])).item()
        print(f"  Channel {c}: {error:.6e}")

    # Each channel should interpolate independently
    assert values.shape == (M, 3), f"Expected shape ({M}, 3), got {values.shape}"
    assert torch.allclose(values.T, data, atol=1e-5), "Multi-channel interpolation failed"


def test_api_compliance():
    """Test API compliance with torch-cubic-spline-grids."""
    # Test initialization with resolution parameter
    spline1 = InterpolatingBSpline1d(resolution=10, n_channels=2)
    assert spline1.resolution == (10,), f"Expected resolution (10,), got {spline1.resolution}"
    assert spline1.n_channels == 2, f"Expected n_channels 2, got {spline1.n_channels}"
    assert spline1.ndim == 1, f"Expected ndim 1, got {spline1.ndim}"

    # Test initialization with tuple resolution
    spline2 = InterpolatingBSpline1d(resolution=(8,), n_channels=3)
    assert spline2.resolution == (8,), f"Expected resolution (8,), got {spline2.resolution}"
    assert spline2.n_channels == 3, f"Expected n_channels 3, got {spline2.n_channels}"

    # Test from_grid_data class method
    data = torch.rand(2, 6)
    spline3 = InterpolatingBSpline1d.from_grid_data(data)
    assert spline3.resolution == (6,), f"Expected resolution (6,), got {spline3.resolution}"
    assert spline3.n_channels == 2, f"Expected n_channels 2, got {spline3.n_channels}"
    assert torch.allclose(spline3.data, data), "from_grid_data did not set data correctly"

    # Test data property getter/setter
    new_data = torch.rand(2, 6)
    spline3.data = new_data
    assert torch.allclose(spline3.data, new_data), "data setter did not work correctly"

    print("\n=== API Compliance Test ===")
    print("All API compliance checks passed!")


def test_input_shape_handling():
    """Test handling of various input shapes (matching torch-cubic-spline-grids behavior)."""
    data = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 2.0])
    spline = InterpolatingBSpline1d.from_grid_data(data)

    # Test scalar input (torch-cubic-spline-grids returns (1, 1) for single channel scalar)
    u_scalar = 0.5
    out_scalar = spline(u_scalar)
    assert out_scalar.shape == torch.Size([1, 1]), f"Expected shape (1, 1), got {out_scalar.shape}"

    # Test 1D array input
    u_1d = torch.tensor([0.2, 0.4, 0.6])
    out_1d = spline(u_1d)
    assert out_1d.shape == torch.Size([3]), f"Expected shape (3,), got {out_1d.shape}"

    # Test 2D input with explicit dimension
    u_2d = torch.tensor([[0.2], [0.4], [0.6]])
    out_2d = spline(u_2d)
    assert out_2d.shape == torch.Size([3, 1]), f"Expected shape (3, 1), got {out_2d.shape}"

    # Test multi-channel output with scalar input
    data_multi = torch.tensor([[1.0, 2.0, 1.5], [0.5, 1.0, 0.8]])
    spline_multi = InterpolatingBSpline1d.from_grid_data(data_multi)
    out_multi_scalar = spline_multi(u_scalar)
    assert out_multi_scalar.shape == torch.Size([1, 2]), f"Expected shape (1, 2), got {out_multi_scalar.shape}"

    # Test multi-channel with 1D input
    out_multi = spline_multi(u_1d)
    assert out_multi.shape == torch.Size([3, 2]), f"Expected shape (3, 2), got {out_multi.shape}"

    print("\n=== Input Shape Handling Test ===")
    print(f"Scalar input (single channel) -> output shape: {out_scalar.shape}")
    print(f"1D input (3,) -> output shape: {out_1d.shape}")
    print(f"2D input (3, 1) -> output shape: {out_2d.shape}")
    print(f"Scalar input (multi-channel) -> output shape: {out_multi_scalar.shape}")
    print(f"1D input (multi-channel) -> output shape: {out_multi.shape}")
    print("All shape handling tests passed!")


def test_minibatch_processing():
    """Test that minibatch processing works correctly."""
    data = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 2.0])

    # Create spline with small minibatch size
    spline = InterpolatingBSpline1d.from_grid_data(data)
    spline._minibatch_size = 3  # Force minibatching with batch size 3

    # Evaluate at many points
    u = torch.linspace(0.0, 1.0, 10)

    with torch.no_grad():
        values = spline(u)

    # Should still get correct results
    assert values.shape == torch.Size([10]), f"Expected shape (10,), got {values.shape}"

    # Check interpolation at boundaries
    assert abs(values[0].item() - 1.0) < 1e-5, f"First value should be ~1.0, got {values[0].item()}"
    assert abs(values[-1].item() - 2.0) < 1e-5, f"Last value should be ~2.0, got {values[-1].item()}"

    print("\n=== Minibatch Processing Test ===")
    print(f"Input points: {len(u)}")
    print(f"Minibatch size: {spline._minibatch_size}")
    print(f"Output shape: {values.shape}")
    print("Minibatch processing works correctly!")


if __name__ == "__main__":
    test_interpolation_at_data_points()
    test_coefficient_computation()
    test_evaluation_between_points()
    test_differentiability()
    test_multichannel()
    test_api_compliance()
    test_input_shape_handling()
    test_minibatch_processing()
    print("\n=== All tests passed! ===")
