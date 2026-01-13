"""
Tests for TiltSeries.apply_tilt_shift_and_propagate method
"""

import math
import pytest
import torch

from warpylib.tilt_series import TiltSeries


class TestApplyTiltShiftAndPropagate:
    """Test apply_tilt_shift_and_propagate method"""

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

    def test_shift_central_tilt_propagate_both(self):
        """
        Test shifting the central tilt (0 degrees) and propagating to both neighbors.

        Geometry: For a Y-axis tilt series with no tilt axis rotation,
        an X shift at angle theta_src propagates to angle theta_tgt as:
            X_tgt = X_src * cos(theta_tgt - theta_src)
            Y_tgt = Y_src  (unchanged)
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Apply 100 A shift in X at central tilt (index 1, angle 0)
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="both"
        )

        # Expected X shifts:
        # - At 0° (source): 100.0
        # - At -30°: 100 * cos(-30° - 0°) = 100 * cos(-30°) = 86.6
        # - At +30°: 100 * cos(30° - 0°) = 100 * cos(30°) = 86.6
        expected_x = [
            100.0 * math.cos(math.radians(-30.0)),  # tilt 0 (-30°)
            100.0,                                   # tilt 1 (0°, source)
            100.0 * math.cos(math.radians(30.0)),   # tilt 2 (+30°)
        ]

        for i, expected in enumerate(expected_x):
            assert abs(ts.tilt_axis_offset_x[i].item() - expected) < 0.01, \
                f"Tilt {i}: expected X offset {expected:.2f}, got {ts.tilt_axis_offset_x[i].item():.2f}"

        # Y offsets should all be ~0 (no Y shift applied)
        assert torch.allclose(ts.tilt_axis_offset_y, torch.zeros(3), atol=0.01)

    def test_shift_highest_tilt_propagate_both(self):
        """
        Test shifting the highest tilt (+30 degrees) and propagating to both neighbors.
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Apply 100 A shift in X at highest tilt (index 2, angle +30°)
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=2,
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="both"
        )

        # Expected X shifts:
        # - At -30°: 100 * cos(-30° - 30°) = 100 * cos(-60°) = 50.0
        # - At 0°: 100 * cos(0° - 30°) = 100 * cos(-30°) = 86.6
        # - At +30° (source): 100.0
        expected_x = [
            100.0 * math.cos(math.radians(-60.0)),  # tilt 0 (-30°)
            100.0 * math.cos(math.radians(-30.0)),  # tilt 1 (0°)
            100.0,                                   # tilt 2 (+30°, source)
        ]

        for i, expected in enumerate(expected_x):
            assert abs(ts.tilt_axis_offset_x[i].item() - expected) < 0.01, \
                f"Tilt {i}: expected X offset {expected:.2f}, got {ts.tilt_axis_offset_x[i].item():.2f}"

    def test_shift_with_y_component(self):
        """
        Test that Y shifts propagate unchanged (Y is parallel to tilt axis).
        """
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Apply shift with both X and Y components at central tilt
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=100.0,
            shift_y=50.0,
            propagate_to="both"
        )

        # Y shifts should be the same across all tilts
        expected_y = 50.0
        for i in range(3):
            assert abs(ts.tilt_axis_offset_y[i].item() - expected_y) < 0.01, \
                f"Tilt {i}: expected Y offset {expected_y:.2f}, got {ts.tilt_axis_offset_y[i].item():.2f}"

    def test_propagate_lower_only(self):
        """Test propagating only to lower tilt indices."""
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Apply shift at central tilt, propagate only to lower indices
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="lower"
        )

        # Tilt 0 should be propagated
        expected_x_tilt0 = 100.0 * math.cos(math.radians(-30.0))
        assert abs(ts.tilt_axis_offset_x[0].item() - expected_x_tilt0) < 0.01

        # Tilt 1 (source) should have the shift
        assert abs(ts.tilt_axis_offset_x[1].item() - 100.0) < 0.01

        # Tilt 2 should NOT be propagated (still 0)
        assert abs(ts.tilt_axis_offset_x[2].item()) < 0.01

    def test_propagate_higher_only(self):
        """Test propagating only to higher tilt indices."""
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Apply shift at central tilt, propagate only to higher indices
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="higher"
        )

        # Tilt 0 should NOT be propagated (still 0)
        assert abs(ts.tilt_axis_offset_x[0].item()) < 0.01

        # Tilt 1 (source) should have the shift
        assert abs(ts.tilt_axis_offset_x[1].item() - 100.0) < 0.01

        # Tilt 2 should be propagated
        expected_x_tilt2 = 100.0 * math.cos(math.radians(30.0))
        assert abs(ts.tilt_axis_offset_x[2].item() - expected_x_tilt2) < 0.01

    def test_additive_shifts(self):
        """Test that shifts are additive to existing offsets."""
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        # Set initial offsets
        ts.tilt_axis_offset_x = torch.tensor([10.0, 20.0, 30.0])
        ts.tilt_axis_offset_y = torch.tensor([5.0, 10.0, 15.0])

        # Apply shift at central tilt
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=100.0,
            shift_y=50.0,
            propagate_to="both"
        )

        # Source tilt should have original + new shift
        assert abs(ts.tilt_axis_offset_x[1].item() - (20.0 + 100.0)) < 0.01
        assert abs(ts.tilt_axis_offset_y[1].item() - (10.0 + 50.0)) < 0.01

        # Other tilts should have original + propagated shift
        expected_x_tilt0 = 10.0 + 100.0 * math.cos(math.radians(-30.0))
        assert abs(ts.tilt_axis_offset_x[0].item() - expected_x_tilt0) < 0.01
        assert abs(ts.tilt_axis_offset_y[0].item() - (5.0 + 50.0)) < 0.01

    def test_invalid_source_tilt_id(self):
        """Test error handling for invalid source_tilt_id."""
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        with pytest.raises(ValueError, match="source_tilt_id must be between"):
            ts.apply_tilt_shift_and_propagate(source_tilt_id=-1, shift_x=100.0, shift_y=0.0)

        with pytest.raises(ValueError, match="source_tilt_id must be between"):
            ts.apply_tilt_shift_and_propagate(source_tilt_id=3, shift_x=100.0, shift_y=0.0)

    def test_invalid_propagate_to(self):
        """Test error handling for invalid propagate_to value."""
        ts = self.create_simple_tilt_series([-30.0, 0.0, 30.0])

        with pytest.raises(ValueError, match="propagate_to must be"):
            ts.apply_tilt_shift_and_propagate(
                source_tilt_id=1, shift_x=100.0, shift_y=0.0, propagate_to="invalid"
            )

    def test_high_tilt_angle_attenuation(self):
        """
        Test that shifts are attenuated correctly at high tilt angles.

        At 60° tilt, an X shift from 0° should be attenuated to 50%.
        """
        ts = self.create_simple_tilt_series([0.0, 60.0])

        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=0,  # source at 0°
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="both"
        )

        # At 0° (source): 100.0
        assert abs(ts.tilt_axis_offset_x[0].item() - 100.0) < 0.01

        # At 60°: 100 * cos(60°) = 50.0
        assert abs(ts.tilt_axis_offset_x[1].item() - 50.0) < 0.01

    def test_rotated_tilt_axis(self):
        """
        Test with a 90° rotated tilt axis.

        With tilt_axis_angle = 90°, the tilt axis is along the image X direction.
        This means:
        - X shifts in the image are parallel to the tilt axis → unchanged across tilts
        - Y shifts in the image are perpendicular to the tilt axis → attenuate with cos(tilt)
        """
        n_tilts = 3
        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0], dtype=torch.float32)
        ts.tilt_axis_angles = torch.full((n_tilts,), 90.0)  # 90° rotation
        ts.level_angle_x = 0.0
        ts.level_angle_y = 0.0
        ts.tilt_axis_offset_x = torch.zeros(n_tilts)
        ts.tilt_axis_offset_y = torch.zeros(n_tilts)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Apply a pure X shift at central tilt (0°)
        # With 90° tilt axis, X is parallel to tilt axis → should stay constant
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="both"
        )

        # X shift should be the same across all tilts (parallel to tilt axis)
        for i in range(3):
            assert abs(ts.tilt_axis_offset_x[i].item() - 100.0) < 0.01, \
                f"Tilt {i}: X offset should be 100.0, got {ts.tilt_axis_offset_x[i].item():.2f}"

        # Y should remain 0
        assert torch.allclose(ts.tilt_axis_offset_y, torch.zeros(3), atol=0.01)

    def test_rotated_tilt_axis_y_shift(self):
        """
        Test Y shift with a 90° rotated tilt axis.

        With tilt_axis_angle = 90°:
        - Y shifts in the image are perpendicular to the tilt axis → attenuate with tilt
        """
        n_tilts = 3
        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = torch.tensor([-30.0, 0.0, 30.0], dtype=torch.float32)
        ts.tilt_axis_angles = torch.full((n_tilts,), 90.0)  # 90° rotation
        ts.level_angle_x = 0.0
        ts.level_angle_y = 0.0
        ts.tilt_axis_offset_x = torch.zeros(n_tilts)
        ts.tilt_axis_offset_y = torch.zeros(n_tilts)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Apply a pure Y shift at central tilt (0°)
        # With 90° tilt axis, Y is perpendicular to tilt axis → should attenuate
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=1,
            shift_x=0.0,
            shift_y=100.0,
            propagate_to="both"
        )

        # Y shift should attenuate with cos(tilt_angle)
        expected_y = [
            100.0 * math.cos(math.radians(-30.0)),  # tilt 0 (-30°)
            100.0,                                   # tilt 1 (0°, source)
            100.0 * math.cos(math.radians(30.0)),   # tilt 2 (+30°)
        ]

        for i, expected in enumerate(expected_y):
            assert abs(ts.tilt_axis_offset_y[i].item() - expected) < 0.01, \
                f"Tilt {i}: expected Y offset {expected:.2f}, got {ts.tilt_axis_offset_y[i].item():.2f}"

        # X should remain 0
        assert torch.allclose(ts.tilt_axis_offset_x, torch.zeros(3), atol=0.01)

    def test_45_degree_tilt_axis(self):
        """
        Test with a 45° tilt axis angle.

        With tilt_axis_angle = 45°, both X and Y shifts have components
        parallel and perpendicular to the tilt axis, so both should
        partially attenuate.
        """
        n_tilts = 2
        ts = TiltSeries(n_tilts=n_tilts)
        ts.angles = torch.tensor([0.0, 60.0], dtype=torch.float32)
        ts.tilt_axis_angles = torch.full((n_tilts,), 45.0)  # 45° rotation
        ts.level_angle_x = 0.0
        ts.level_angle_y = 0.0
        ts.tilt_axis_offset_x = torch.zeros(n_tilts)
        ts.tilt_axis_offset_y = torch.zeros(n_tilts)
        ts.volume_dimensions_physical = torch.tensor([100.0, 100.0, 50.0])
        ts.image_dimensions_physical = torch.tensor([100.0, 100.0])

        # Apply an X shift at tilt 0°
        ts.apply_tilt_shift_and_propagate(
            source_tilt_id=0,
            shift_x=100.0,
            shift_y=0.0,
            propagate_to="both"
        )

        # At source (0°): X = 100, Y = 0
        assert abs(ts.tilt_axis_offset_x[0].item() - 100.0) < 0.01
        assert abs(ts.tilt_axis_offset_y[0].item()) < 0.01

        # At 60° tilt: the shift should have both X and Y components
        # due to the 45° tilt axis mixing them during projection
        # Just verify the result is reasonable (not 0, not 100 for both)
        x_at_60 = ts.tilt_axis_offset_x[1].item()
        y_at_60 = ts.tilt_axis_offset_y[1].item()

        # The total magnitude should be less than 100 (some attenuation)
        magnitude = math.sqrt(x_at_60**2 + y_at_60**2)
        assert magnitude < 100.0, f"Expected attenuation, but magnitude is {magnitude:.2f}"
        assert magnitude > 0.0, "Expected non-zero propagated shift"

        # Both components should be non-zero due to 45° mixing
        assert abs(x_at_60) > 1.0, f"Expected non-zero X, got {x_at_60:.2f}"
        assert abs(y_at_60) > 1.0, f"Expected non-zero Y, got {y_at_60:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])