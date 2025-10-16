"""
Test interpolating B-splines against einspline reference values.

The reference values were computed using einspline on a different system
and stored in .star files with two tables:
1. data_grid: The grid values (input data)
2. data_eval: Interpolated values at specific evaluation points
"""

import torch
import numpy as np
import starfile
from pathlib import Path
from warpylib.cubic_grid import CubicGrid

# Path to test data
TESTDATA_DIR = Path(__file__).parent.parent.parent / "testdata"


def test_1d_against_einspline_reference():
    """Test 1D interpolation against einspline reference values."""
    # Load reference data
    star_path = TESTDATA_DIR / "reference1d.star"
    data = starfile.read(star_path)

    # Extract grid values
    grid_df = data['grid']
    grid_values = torch.tensor(grid_df['value'].values, dtype=torch.float32)

    # Extract evaluation points and expected values
    eval_df = data['eval']
    eval_points = torch.tensor(eval_df['x'].values, dtype=torch.float32)
    expected_values = torch.tensor(eval_df['value'].values, dtype=torch.float32)

    print("\n=== 1D Einspline Reference Test ===")
    print(f"Grid size: {len(grid_values)}")
    print(f"Grid values: {grid_values[:5].numpy()}... (showing first 5)")
    print(f"Number of evaluation points: {len(eval_points)}")

    # Create CubicGrid (1D case: only X dimension is active)
    grid = CubicGrid(
        dimensions=(len(grid_values), 1, 1),
        values=grid_values,
        margins=(0.0, 0.0, 0.0)
    )

    # Evaluate at reference points
    # CubicGrid expects (N, 3) coordinates, so we need to add dummy Y and Z
    coords_3d = torch.zeros((len(eval_points), 3), dtype=torch.float32)
    coords_3d[:, 0] = eval_points  # X coordinates
    coords_3d[:, 1] = 0.0  # Dummy Y
    coords_3d[:, 2] = 0.0  # Dummy Z

    computed_values = grid.get_interpolated(coords_3d)

    # Compute errors (computed_values is already a torch tensor)
    errors = torch.abs(computed_values - expected_values)
    max_error = torch.max(errors).item()
    mean_error = torch.mean(errors).item()

    print(f"\nFirst 10 comparisons:")
    print(f"{'x':>10} {'Expected':>15} {'Computed':>15} {'Error':>15}")
    for i in range(min(10, len(eval_points))):
        x = eval_points[i].item()
        exp = expected_values[i].item()
        comp = computed_values[i].item()
        err = errors[i].item()
        print(f"{x:>10.6f} {exp:>15.8f} {comp:>15.8f} {err:>15.2e}")

    print(f"\nError statistics:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  RMS error: {torch.sqrt(torch.mean(errors**2)).item():.6e}")

    # Check if errors are acceptable
    # Einspline uses float32, so we expect small numerical differences
    tolerance = 1e-5
    assert max_error < tolerance, \
        f"1D interpolation error too large: {max_error:.6e} > {tolerance}"

    print(f"\n✓ 1D test PASSED (max error: {max_error:.6e})")


def test_2d_against_einspline_reference():
    """Test 2D interpolation against einspline reference values."""
    # Load reference data
    star_path = TESTDATA_DIR / "reference2d.star"
    data = starfile.read(star_path)

    # Extract grid values
    grid_df = data['grid']
    # Grid is stored with columns: ix, iy, value
    # We need to reshape it properly

    # Find grid dimensions
    ix_vals = grid_df['ix'].values
    iy_vals = grid_df['iy'].values
    Mx = int(ix_vals.max()) + 1
    My = int(iy_vals.max()) + 1

    # Reshape grid values to (Mx, My)
    grid_values_flat = grid_df['value'].values
    grid_values_2d = np.zeros((Mx, My))
    for idx, row in grid_df.iterrows():
        ix = int(row['ix'])
        iy = int(row['iy'])
        grid_values_2d[ix, iy] = row['value']

    grid_values = torch.tensor(grid_values_2d, dtype=torch.float32)

    # Extract evaluation points and expected values
    eval_df = data['eval']
    eval_x = torch.tensor(eval_df['x'].values, dtype=torch.float32)
    eval_y = torch.tensor(eval_df['y'].values, dtype=torch.float32)
    eval_points = torch.stack([eval_x, eval_y], dim=1)
    expected_values = torch.tensor(eval_df['value'].values, dtype=torch.float32)

    print("\n=== 2D Einspline Reference Test ===")
    print(f"Grid shape: {grid_values.shape} ({Mx} x {My})")
    print(f"Grid values sample:\n{grid_values[:3, :3].numpy()}")
    print(f"Number of evaluation points: {len(eval_points)}")

    # Create CubicGrid (2D case: X and Y dimensions are active, Z=1)
    # Need to flatten the 2D grid to einspline layout: [(y)*X+x]
    values_flat = torch.zeros(Mx * My, dtype=torch.float32)
    for iy in range(My):
        for ix in range(Mx):
            idx = iy * Mx + ix  # Einspline layout for 2D: [(y)*X+x]
            values_flat[idx] = grid_values[ix, iy]

    grid = CubicGrid(
        dimensions=(Mx, My, 1),
        values=values_flat,
        margins=(0.0, 0.0, 0.0)
    )

    # Evaluate at reference points
    # CubicGrid expects (N, 3) coordinates, add dummy Z
    coords_3d = torch.zeros((len(eval_points), 3), dtype=torch.float32)
    coords_3d[:, 0] = eval_points[:, 0]  # X coordinates
    coords_3d[:, 1] = eval_points[:, 1]  # Y coordinates
    coords_3d[:, 2] = 0.0  # Dummy Z

    computed_values = grid.get_interpolated(coords_3d)

    # Compute errors (computed_values is already a torch tensor)
    errors = torch.abs(computed_values - expected_values)
    max_error = torch.max(errors).item()
    mean_error = torch.mean(errors).item()

    print(f"\nFirst 10 comparisons:")
    print(f"{'x':>10} {'y':>10} {'Expected':>15} {'Computed':>15} {'Error':>15}")
    for i in range(min(10, len(eval_points))):
        x = eval_points[i, 0].item()
        y = eval_points[i, 1].item()
        exp = expected_values[i].item()
        comp = computed_values[i].item()
        err = errors[i].item()
        print(f"{x:>10.6f} {y:>10.6f} {exp:>15.8f} {comp:>15.8f} {err:>15.2e}")

    # Show worst cases
    worst_indices = torch.argsort(errors, descending=True)[:5]
    print(f"\n5 worst error cases:")
    print(f"{'x':>10} {'y':>10} {'Expected':>15} {'Computed':>15} {'Error':>15}")
    for idx in worst_indices:
        i = idx.item()
        x = eval_points[i, 0].item()
        y = eval_points[i, 1].item()
        exp = expected_values[i].item()
        comp = computed_values[i].item()
        err = errors[i].item()
        print(f"{x:>10.6f} {y:>10.6f} {exp:>15.8f} {comp:>15.8f} {err:>15.2e}")

    print(f"\nError statistics:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  RMS error: {torch.sqrt(torch.mean(errors**2)).item():.6e}")

    # Check if errors are acceptable
    tolerance = 1e-5
    if max_error >= tolerance:
        print(f"\n✗ 2D test FAILED: max error {max_error:.6e} >= {tolerance}")
        # Show error distribution
        print(f"\nError distribution:")
        print(f"  Errors > 1e-3: {torch.sum(errors > 1e-3).item()}")
        print(f"  Errors > 1e-4: {torch.sum(errors > 1e-4).item()}")
        print(f"  Errors > 1e-5: {torch.sum(errors > 1e-5).item()}")
        print(f"  Errors > 1e-6: {torch.sum(errors > 1e-6).item()}")

        # This will help us debug
        assert False, f"2D interpolation error too large: {max_error:.6e} > {tolerance}"

    print(f"\n✓ 2D test PASSED (max error: {max_error:.6e})")


def test_3d_against_einspline_reference():
    """Test 3D interpolation against einspline reference values."""
    # Load reference data
    star_path = TESTDATA_DIR / "reference3d.star"
    data = starfile.read(star_path)

    # Extract grid values
    grid_df = data['grid']
    # Grid is stored with columns: ix, iy, iz, value
    # We need to reshape it properly

    # Find grid dimensions
    ix_vals = grid_df['ix'].values
    iy_vals = grid_df['iy'].values
    iz_vals = grid_df['iz'].values
    Mx = int(ix_vals.max()) + 1
    My = int(iy_vals.max()) + 1
    Mz = int(iz_vals.max()) + 1

    # Reshape grid values to (Mx, My, Mz)
    grid_values_3d = np.zeros((Mx, My, Mz))
    for idx, row in grid_df.iterrows():
        ix = int(row['ix'])
        iy = int(row['iy'])
        iz = int(row['iz'])
        grid_values_3d[ix, iy, iz] = row['value']

    grid_values = torch.tensor(grid_values_3d, dtype=torch.float32)

    # Extract evaluation points and expected values
    eval_df = data['eval']
    eval_x = torch.tensor(eval_df['x'].values, dtype=torch.float32)
    eval_y = torch.tensor(eval_df['y'].values, dtype=torch.float32)
    eval_z = torch.tensor(eval_df['z'].values, dtype=torch.float32)
    eval_points = torch.stack([eval_x, eval_y, eval_z], dim=1)
    expected_values = torch.tensor(eval_df['value'].values, dtype=torch.float32)

    print("\n=== 3D Einspline Reference Test ===")
    print(f"Grid shape: {grid_values.shape} ({Mx} x {My} x {Mz})")
    print(f"Grid values sample:\n{grid_values[:2, :2, :2].numpy()}")
    print(f"Number of evaluation points: {len(eval_points)}")

    # Create CubicGrid (3D case: X, Y, and Z dimensions are all active)
    # Need to flatten the 3D grid to einspline layout: [(z)*X*Y+(y)*X+x]
    values_flat = torch.zeros(Mx * My * Mz, dtype=torch.float32)
    for iz in range(Mz):
        for iy in range(My):
            for ix in range(Mx):
                idx = iz * Mx * My + iy * Mx + ix  # Einspline layout for 3D: [(z)*X*Y+(y)*X+x]
                values_flat[idx] = grid_values[ix, iy, iz]

    grid = CubicGrid(
        dimensions=(Mx, My, Mz),
        values=values_flat,
        margins=(0.0, 0.0, 0.0)
    )

    # Evaluate at reference points
    coords_3d = eval_points

    computed_values = grid.get_interpolated(coords_3d)

    # Compute errors (computed_values is already a torch tensor)
    errors = torch.abs(computed_values - expected_values)
    max_error = torch.max(errors).item()
    mean_error = torch.mean(errors).item()

    print(f"\nFirst 10 comparisons:")
    print(f"{'x':>10} {'y':>10} {'z':>10} {'Expected':>15} {'Computed':>15} {'Error':>15}")
    for i in range(min(10, len(eval_points))):
        x = eval_points[i, 0].item()
        y = eval_points[i, 1].item()
        z = eval_points[i, 2].item()
        exp = expected_values[i].item()
        comp = computed_values[i].item()
        err = errors[i].item()
        print(f"{x:>10.6f} {y:>10.6f} {z:>10.6f} {exp:>15.8f} {comp:>15.8f} {err:>15.2e}")

    # Show worst cases
    worst_indices = torch.argsort(errors, descending=True)[:5]
    print(f"\n5 worst error cases:")
    print(f"{'x':>10} {'y':>10} {'z':>10} {'Expected':>15} {'Computed':>15} {'Error':>15}")
    for idx in worst_indices:
        i = idx.item()
        x = eval_points[i, 0].item()
        y = eval_points[i, 1].item()
        z = eval_points[i, 2].item()
        exp = expected_values[i].item()
        comp = computed_values[i].item()
        err = errors[i].item()
        print(f"{x:>10.6f} {y:>10.6f} {z:>10.6f} {exp:>15.8f} {comp:>15.8f} {err:>15.2e}")

    print(f"\nError statistics:")
    print(f"  Max error: {max_error:.6e}")
    print(f"  Mean error: {mean_error:.6e}")
    print(f"  RMS error: {torch.sqrt(torch.mean(errors**2)).item():.6e}")

    # Check if errors are acceptable
    tolerance = 1e-5
    if max_error >= tolerance:
        print(f"\n✗ 3D test FAILED: max error {max_error:.6e} >= {tolerance}")
        # Show error distribution
        print(f"\nError distribution:")
        print(f"  Errors > 1e-3: {torch.sum(errors > 1e-3).item()}")
        print(f"  Errors > 1e-4: {torch.sum(errors > 1e-4).item()}")
        print(f"  Errors > 1e-5: {torch.sum(errors > 1e-5).item()}")
        print(f"  Errors > 1e-6: {torch.sum(errors > 1e-6).item()}")

        # This will help us debug
        assert False, f"3D interpolation error too large: {max_error:.6e} > {tolerance}"

    print(f"\n✓ 3D test PASSED (max error: {max_error:.6e})")


if __name__ == "__main__":
    test_1d_against_einspline_reference()
    test_2d_against_einspline_reference()
    test_3d_against_einspline_reference()
    print("\n=== All einspline reference tests completed ===")
