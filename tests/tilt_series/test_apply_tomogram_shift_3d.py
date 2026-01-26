"""
Tests for TiltSeries.apply_tomogram_shift_3d method
"""

import math
import pytest
import torch

from warpylib.tilt_series import TiltSeries


class TestApplyTomogramShift3D:
    """Test apply_tomogram_shift_3d method"""

    def create_simple_tilt_series(self, angles: list[float]) -> TiltSeries:
        """Create a simple tilt series with given angles and no extra geometry."""
        n_tilts = len(angles)
        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = torch.tensor(angles, dtype=torch.float32)
        ts.tilt_axis_angles = torch.zeros(n_tilts)
        ts.level_angle_x = 0.0
        ts.level_angle_y = 0.0
        ts.tilt_axis_offset_x = torch.zeros(n_tilts)
        ts.tilt_axis_offset_y = torch.zeros(n_tilts)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])
        return ts

    def test_pure_x_shift(self):
        """
        Test a pure X shift in specimen space.

        X shift projects as X * cos(tilt_angle) on each tilt image.
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        ts.apply_tomogram_shift_3d(torch.tensor([100.0, 0.0, 0.0]))

        # X shift scales with cos(tilt_angle)
        expected_x = [
            100.0 * math.cos(math.radians(-30.0)),
            100.0 * math.cos(math.radians(0.0)),
            100.0 * math.cos(math.radians(30.0)),
        ]

        for i, expected in enumerate(expected_x):
            assert abs(ts.tilt_axis_offset_x[i].item() - expected) < 0.01, \
                f"Tilt {i}: expected X offset {expected:.2f}, got {ts.tilt_axis_offset_x[i].item():.2f}"

        # Y should remain 0
        assert torch.allclose(ts.tilt_axis_offset_y, torch.zeros(3), atol=0.01)

    def test_pure_y_shift(self):
        """
        Test a pure Y shift in specimen space.

        Y shift (along tilt axis) projects unchanged on all tilts.
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        ts.apply_tomogram_shift_3d(torch.tensor([0.0, 100.0, 0.0]))

        # Y shift is constant across all tilts (parallel to tilt axis)
        for i in range(3):
            assert abs(ts.tilt_axis_offset_y[i].item() - 100.0) < 0.01, \
                f"Tilt {i}: expected Y offset 100.0, got {ts.tilt_axis_offset_y[i].item():.2f}"

        # X should remain 0
        assert torch.allclose(ts.tilt_axis_offset_x, torch.zeros(3), atol=0.01)

    def test_pure_z_shift(self):
        """
        Test a pure Z shift in specimen space.

        Z shift (along beam at 0° tilt) projects as -Z * sin(tilt_angle) in image X.
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        ts.apply_tomogram_shift_3d(torch.tensor([0.0, 0.0, 100.0]))

        # Z shift projects as -Z * sin(tilt_angle) in X
        expected_x = [
            -100.0 * math.sin(math.radians(-30.0)),  # +50 at -30°
            -100.0 * math.sin(math.radians(0.0)),    # 0 at 0°
            -100.0 * math.sin(math.radians(30.0)),   # -50 at 30°
        ]

        for i, expected in enumerate(expected_x):
            assert abs(ts.tilt_axis_offset_x[i].item() - expected) < 0.01, \
                f"Tilt {i}: expected X offset {expected:.2f}, got {ts.tilt_axis_offset_x[i].item():.2f}"

        # Y should remain 0
        assert torch.allclose(ts.tilt_axis_offset_y, torch.zeros(3), atol=0.01)

    def test_combined_xyz_shift(self):
        """
        Test a combined 3D shift.
        """
        ts = self.create_simple_tilt_series([0.0, 45.0])

        ts.apply_tomogram_shift_3d(torch.tensor([100.0, 50.0, 100.0]))

        # At 0° tilt:
        # - X projects as 100 * cos(0) = 100
        # - Z projects as -100 * sin(0) = 0
        # Total X: 100
        assert abs(ts.tilt_axis_offset_x[0].item() - 100.0) < 0.01
        assert abs(ts.tilt_axis_offset_y[0].item() - 50.0) < 0.01

        # At 45° tilt:
        # - X projects as 100 * cos(45) = 70.7
        # - Z projects as -100 * sin(45) = -70.7
        # Total X: 70.7 - 70.7 = 0
        assert abs(ts.tilt_axis_offset_x[1].item()) < 0.01
        assert abs(ts.tilt_axis_offset_y[1].item() - 50.0) < 0.01

    def test_additive_to_existing_offsets(self):
        """
        Test that 3D shifts are additive to existing offsets.
        """
        ts = self.create_simple_tilt_series([0.0, 30.0])

        # Set initial offsets
        ts.tilt_axis_offset_x = torch.tensor([10.0, 20.0])
        ts.tilt_axis_offset_y = torch.tensor([5.0, 15.0])

        ts.apply_tomogram_shift_3d(torch.tensor([100.0, 50.0, 0.0]))

        # At 0°: X = 10 + 100, Y = 5 + 50
        assert abs(ts.tilt_axis_offset_x[0].item() - 110.0) < 0.01
        assert abs(ts.tilt_axis_offset_y[0].item() - 55.0) < 0.01

        # At 30°: X = 20 + 100*cos(30), Y = 15 + 50
        expected_x_30 = 20.0 + 100.0 * math.cos(math.radians(30.0))
        assert abs(ts.tilt_axis_offset_x[1].item() - expected_x_30) < 0.01
        assert abs(ts.tilt_axis_offset_y[1].item() - 65.0) < 0.01

    def test_rotated_tilt_axis_x_shift(self):
        """
        Test X shift with a 90° rotated tilt axis.

        With tilt_axis_angle = 90°, specimen X becomes image +Y direction
        (due to the ZYZ Euler convention with psi = -tilt_axis_angle).
        """
        n_tilts = 3
        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0], dtype=torch.float32)
        ts.tilt_axis_angles = torch.full((n_tilts,), 90.0)
        ts.level_angle_x = 0.0
        ts.level_angle_y = 0.0
        ts.tilt_axis_offset_x = torch.zeros(n_tilts)
        ts.tilt_axis_offset_y = torch.zeros(n_tilts)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        ts.apply_tomogram_shift_3d(torch.tensor([100.0, 0.0, 0.0]))

        # With 90° tilt axis rotation, specimen X projects to image +Y
        # and attenuates with cos(tilt_angle)
        expected_y = [
            100.0 * math.cos(math.radians(-30.0)),
            100.0 * math.cos(math.radians(0.0)),
            100.0 * math.cos(math.radians(30.0)),
        ]

        for i, expected in enumerate(expected_y):
            assert abs(ts.tilt_axis_offset_y[i].item() - expected) < 0.01, \
                f"Tilt {i}: expected Y offset {expected:.2f}, got {ts.tilt_axis_offset_y[i].item():.2f}"

    def test_equivalence_with_propagate_at_zero_tilt(self):
        """
        Test that apply_tomogram_shift_3d with Z=0 is equivalent to
        apply_tilt_shift_and_propagate from the 0° tilt.
        """
        # Create two identical tilt series
        ts1 = self.create_simple_tilt_series([-30.0, 0.0, 30.0])
        ts2 = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Apply 3D shift with Z=0
        ts1.apply_tomogram_shift_3d(torch.tensor([100.0, 50.0, 0.0]))

        # Apply 2D shift at 0° tilt and propagate
        ts2.apply_tilt_shift_and_propagate(
            source_tilt_id=1,  # 0° tilt
            shift=torch.tensor([100.0, 50.0]),
            propagate_to="both"
        )

        # Results should be identical
        assert torch.allclose(ts1.tilt_axis_offset_x, ts2.tilt_axis_offset_x, atol=0.01)
        assert torch.allclose(ts1.tilt_axis_offset_y, ts2.tilt_axis_offset_y, atol=0.01)

    def test_gradient_flow(self):
        """
        Test that gradients flow back through the shift tensor.
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Create shift tensor with requires_grad
        shift = torch.tensor([100.0, 50.0, 25.0], requires_grad=True)

        ts.apply_tomogram_shift_3d(shift)

        # Compute some loss from the offsets
        loss = ts.tilt_axis_offset_x.sum() + ts.tilt_axis_offset_y.sum()

        # Backpropagate
        loss.backward()

        # Verify gradients exist and are non-zero
        assert shift.grad is not None, "Gradients should flow back to shift tensor"
        assert shift.grad[0].item() != 0.0, "X gradient should be non-zero"
        assert shift.grad[1].item() != 0.0, "Y gradient should be non-zero"
        assert shift.grad[2].item() != 0.0, "Z gradient should be non-zero (projects to X)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])