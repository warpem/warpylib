"""
Test 2D interpolating B-spline against einspline reference implementation.
"""

import torch
import numpy as np
from warpylib.interpolating_bspline import InterpolatingBSpline2d

# Precomputed einspline outputs for a 4x4 grid
# Data:  [[1, 2, 1.5, 2.5],
#         [2, 3, 2.5, 3.5],
#         [1.5, 2.5, 2, 3],
#         [2.5, 3.5, 3, 4]]

def test_2d_data_layout():
    """Test that we understand einspline's data layout correctly."""
    # Create simple 3x4 test data (3 rows X-direction, 4 columns Y-direction)
    # In PyTorch: shape (3, 4) means data[ix, iy]
    data_torch = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # ix=0, iy=0..3
        [5.0, 6.0, 7.0, 8.0],  # ix=1, iy=0..3
        [9.0, 10.0, 11.0, 12.0],  # ix=2, iy=0..3
    ])

    # Einspline expects data in Y-major order (iy is fast index)
    # So einspline's flat array should be: [data[0,0], data[0,1], data[0,2], data[0,3],
    #                                        data[1,0], data[1,1], data[1,2], data[1,3],
    #                                        data[2,0], data[2,1], data[2,2], data[2,3]]
    # Which in einspline notation means: for each x, iterate through all y values

    print("\n=== 2D Data Layout Test ===")
    print(f"PyTorch data shape: {data_torch.shape}")
    print(f"PyTorch data:\n{data_torch}")
    print(f"\nPyTorch flattened (row-major): {data_torch.flatten()}")
    print(f"\nEinspline expects for each X row: all Y values")
    print(f"ix=0: {data_torch[0,:]}")
    print(f"ix=1: {data_torch[1,:]}")
    print(f"ix=2: {data_torch[2,:]}")


def test_2d_simple_interpolation():
    """Test 2D interpolation with a simple known case."""
    # Create 3x3 grid with simple values
    data = torch.tensor([
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ], dtype=torch.float32)

    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Test at grid points (should match exactly)
    Mx, My = data.shape
    x_coords = torch.linspace(0.0, 1.0, Mx)
    y_coords = torch.linspace(0.0, 1.0, My)

    # Test corners
    test_points = torch.tensor([
        [0.0, 0.0],  # data[0, 0] = 0.0
        [1.0, 0.0],  # data[2, 0] = 2.0
        [0.0, 1.0],  # data[0, 2] = 2.0
        [1.0, 1.0],  # data[2, 2] = 4.0
        [0.5, 0.5],  # data[1, 1] = 2.0 (center point)
    ])

    with torch.no_grad():
        values = spline(test_points)

    expected = torch.tensor([[0.0], [2.0], [2.0], [4.0], [2.0]])

    print("\n=== 2D Simple Interpolation Test ===")
    print(f"Data:\n{data}")
    print(f"\nTest points:\n{test_points}")
    print(f"Expected values:\n{expected}")
    print(f"Got values:\n{values}")
    print(f"Differences:\n{(values - expected).abs()}")
    print(f"Max error: {torch.max(torch.abs(values - expected)).item():.6e}")

    assert torch.allclose(values, expected, atol=1e-5), \
        f"Simple 2D interpolation failed. Max error: {torch.max(torch.abs(values - expected)).item()}"


def test_layout_transformation():
    """Test if we need to transpose data for einspline compatibility."""
    # Create asymmetric data so we can detect transposition issues
    # 3 rows (X) x 4 columns (Y)
    data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
    ], dtype=torch.float32)

    spline = InterpolatingBSpline2d.from_grid_data(data)

    # Test at (0, 0) - should be 1.0
    # Test at (1, 0) - should be 9.0 if X is first dimension
    # Test at (0, 1) - should be 4.0 if Y is second dimension

    test_points = torch.tensor([
        [0.0, 0.0],  # Should be data[0, 0] = 1.0
        [1.0, 0.0],  # Should be data[2, 0] = 9.0 (last X, first Y)
        [0.0, 1.0],  # Should be data[0, 3] = 4.0 (first X, last Y)
        [1.0, 1.0],  # Should be data[2, 3] = 12.0 (last X, last Y)
    ])

    with torch.no_grad():
        values = spline(test_points)

    expected = torch.tensor([[1.0], [9.0], [4.0], [12.0]])

    print("\n=== Layout Transformation Test ===")
    print(f"Data shape: {data.shape} (Mx=3, My=4)")
    print(f"Data:\n{data}")
    print(f"\nTest points (x, y):\n{test_points}")
    print(f"Expected values:\n{expected}")
    print(f"Got values:\n{values}")
    print(f"Differences:\n{(values - expected).abs()}")

    if not torch.allclose(values, expected, atol=1e-5):
        print("\nMismatch detected! Checking if transposed...")
        expected_transposed = torch.tensor([[1.0], [3.0], [10.0], [12.0]])
        print(f"Expected if transposed:\n{expected_transposed}")
        if torch.allclose(values, expected_transposed, atol=1e-5):
            print("ERROR: Data appears to be transposed!")
            assert False, "Data appears to be transposed!"
        else:
            print("ERROR: Unknown layout issue")
            assert False, f"Unknown layout issue. Max error: {torch.max(torch.abs(values - expected)).item()}"

    print("Layout test PASSED!")


if __name__ == "__main__":
    test_2d_data_layout()
    test_2d_simple_interpolation()
    test_layout_transformation()
    print("\n=== All layout tests completed ===")
