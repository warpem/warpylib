"""
Tests for aligned stack generation (get_aligned_stack and friends)
"""

from pathlib import Path

import mrcfile
import pytest
import torch
import torch.nn.functional as F

from warpylib.cubic_grid import CubicGrid
from warpylib.tilt_series import TiltSeries

TESTDATA = Path(__file__).parent.parent.parent / "testdata"


def _load_ts1() -> TiltSeries:
    """TS_1 has tilt axis angles, offsets, level angles and 6x4x41 movement grids."""
    ts = TiltSeries()
    ts.load_meta(str(TESTDATA / "TS_1.xml"))
    return ts


def _random_volume_coords(ts: TiltSeries, n: int, margin: float = 0.2) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    frac = margin + (1 - 2 * margin) * torch.rand(n, 3, generator=g)
    return frac * ts.volume_dimensions_physical


def _assert_round_trip(ts: TiltSeries, coords: torch.Tensor):
    per_tilt = coords.unsqueeze(-2).expand(-1, ts.n_tilts, 3)
    expected = ts.get_position_in_all_tilts(per_tilt)[..., :2]
    aligned = ts.get_positions_in_aligned_stack(per_tilt)
    tilt_ids = torch.arange(ts.n_tilts).expand(coords.shape[0], -1)
    raw = ts.get_raw_positions_from_aligned(aligned, tilt_ids)
    assert torch.allclose(raw, expected, atol=1e-2), (raw - expected).abs().max()


def test_round_trip_matches_forward_model():
    """aligned -> raw composed with 3D -> aligned must equal the full forward model."""
    ts = _load_ts1()
    assert ts.grid_movement_x.values.abs().max() > 1  # movement grids are exercised
    ts.size_rounding_factors = torch.tensor([1.003, 0.997, 1.0])
    _assert_round_trip(ts, _random_volume_coords(ts, 50))

    ts.are_angles_inverted = True  # only affects Z, must not change XY
    _assert_round_trip(ts, _random_volume_coords(ts, 50))


def test_round_trip_with_volume_warp():
    """Volume warp moves points in the aligned frame too, so the round trip still holds."""
    ts = _load_ts1()
    g = torch.Generator().manual_seed(1)
    for name in ("grid_volume_warp_x", "grid_volume_warp_y", "grid_volume_warp_z"):
        setattr(ts, name, CubicGrid((3, 3, 2, 4), torch.randn(4, 2, 3, 3, generator=g) * 20))
    _assert_round_trip(ts, _random_volume_coords(ts, 50))


def test_aligned_frame_has_tilt_axis_along_y():
    """With axis/offsets/warp removed, a point on the tilt axis projects to x = center."""
    ts = _load_ts1()
    ts.level_angle_x = 0.0
    ts.level_angle_y = 0.0
    center = ts.volume_dimensions_physical / 2
    coords = torch.stack([center.clone() for _ in range(3)])
    coords[:, 1] += torch.tensor([-1000.0, 0.0, 1000.0])  # move along Y (tilt axis)
    aligned = ts.get_positions_in_aligned_stack(coords.unsqueeze(-2).expand(-1, ts.n_tilts, 3))
    image_center = ts.image_dimensions_physical / 2
    assert torch.allclose(aligned[..., 0], image_center[0].expand(3, ts.n_tilts), atol=1e-2)
    assert torch.allclose(aligned[:, :, 1], (image_center[1] + coords[:, 1:2] - center[1]).expand(-1, ts.n_tilts), atol=1e-2)


def test_identity_alignment_returns_input():
    ts = TiltSeries(n_tilts=3)
    ts.angles = torch.tensor([-30.0, 0.0, 30.0])
    ts.image_dimensions_physical = torch.tensor([64.0 * 5, 48.0 * 5])
    ts.volume_dimensions_physical = torch.tensor([320.0, 240.0, 100.0])
    tilt_data = torch.randn(3, 48, 64)
    stack = ts.get_aligned_stack(tilt_data, pixel_size=5.0, interpolation="bilinear")
    assert torch.allclose(stack, tilt_data, atol=1e-4)


def test_blobs_land_at_aligned_positions():
    """Render blobs at raw positions of 3D points, align, find them at aligned positions."""
    ts = _load_ts1()
    pixel_size = 40.0
    w = int(round(float(ts.image_dimensions_physical[0]) / pixel_size / 2)) * 2
    h = int(round(float(ts.image_dimensions_physical[1]) / pixel_size / 2)) * 2
    ts.size_rounding_factors = torch.tensor([
        w / (float(ts.image_dimensions_physical[0]) / pixel_size),
        h / (float(ts.image_dimensions_physical[1]) / pixel_size),
        1.0,
    ])
    tilt_ids = torch.tensor([0, 10, 20, 30, 40])

    # Well-separated points in the volume's mid-plane so blobs don't overlap and
    # stay in view at high tilt
    fx, fy = torch.meshgrid(torch.tensor([0.25, 0.75]), torch.tensor([0.25, 0.5, 0.75]), indexing="ij")
    frac = torch.stack([fx.flatten(), fy.flatten(), torch.full((6,), 0.5)], dim=-1)
    coords = frac * ts.volume_dimensions_physical
    per_tilt = coords.unsqueeze(-2).expand(-1, ts.n_tilts, 3)
    raw_px = ts.get_position_in_all_tilts(per_tilt)[..., :2] / pixel_size  # (P, T, 2)
    aligned_px = ts.get_positions_in_aligned_stack(per_tilt) / pixel_size

    sigma = 1.5
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    tilt_data = torch.zeros(ts.n_tilts, h, w)
    for p in range(coords.shape[0]):
        d2 = (xx - raw_px[p, :, 0, None, None]) ** 2 + (yy - raw_px[p, :, 1, None, None]) ** 2
        tilt_data += torch.exp(-d2 / (2 * sigma**2))

    stack = ts.get_aligned_stack(tilt_data, pixel_size, tilt_ids=tilt_ids, fill_value=0.0)
    for i, t in enumerate(tilt_ids.tolist()):
        for p in range(coords.shape[0]):
            cx, cy = aligned_px[p, t].tolist()
            x0, y0 = int(round(cx)), int(round(cy))
            patch = stack[i, y0 - 4:y0 + 5, x0 - 4:x0 + 5]
            wy, wx = torch.meshgrid(
                torch.arange(y0 - 4, y0 + 5).float(), torch.arange(x0 - 4, x0 + 5).float(),
                indexing="ij",
            )
            centroid = torch.tensor([(patch * wx).sum(), (patch * wy).sum()]) / patch.sum()
            assert torch.allclose(centroid, torch.tensor([cx, cy]), atol=0.1), (t, p, centroid, cx, cy)


def test_matches_imod_xf_transform():
    """
    After importing an IMOD .xf, the aligned stack must equal applying the .xf
    directly (IMOD's newstack convention: aligned = A @ (raw - c) + d + c).
    """
    stack_path = TESTDATA / "00254.stack.mrc"
    xf_dir = TESTDATA / "xf_import"
    with mrcfile.open(stack_path, permissive=True) as mrc:
        raw = torch.from_numpy(mrc.data.copy()).float()
    n_tilts, h, w = raw.shape
    pixel_size = 10.0

    ts = TiltSeries(n_tilts=n_tilts)
    ts.image_dimensions_physical = torch.tensor([w * pixel_size, h * pixel_size])
    ts.import_alignments(str(xf_dir / "00254.xf"), pixel_size, tlt_path=str(xf_dir / "00254.tlt"))

    tilt_ids = torch.tensor([0, 20, 40])
    stack = ts.get_aligned_stack(
        raw, pixel_size, tilt_ids=tilt_ids, interpolation="bilinear", fill_value=0.0
    )

    xf = torch.tensor([[float(v) for v in line.split()]
                       for line in (xf_dir / "00254.xf").read_text().splitlines() if line.strip()])
    center = torch.tensor([w / 2, h / 2])
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    q = torch.stack([xx, yy], dim=-1)
    for i, t in enumerate(tilt_ids.tolist()):
        a = xf[t, :4].reshape(2, 2)  # IMOD order: A11 A12 A21 A22
        d = xf[t, 4:]
        src = torch.einsum("ij,...j->...i", torch.linalg.inv(a), q - center - d) + center
        grid = torch.stack([src[..., 0] / (w - 1), src[..., 1] / (h - 1)], dim=-1) * 2 - 1
        expected = F.grid_sample(raw[t][None, None], grid[None], mode="bilinear",
                                 padding_mode="zeros", align_corners=True)[0, 0]
        inside = (src[..., 0] >= 0) & (src[..., 0] <= w - 1) & (src[..., 1] >= 0) & (src[..., 1] <= h - 1)
        scale = raw[t].std()
        assert ((stack[i] - expected)[inside].abs().max() / scale) < 0.05


def test_used_tilts_only_input_and_write(tmp_path):
    ts = _load_ts1()
    ts.use_tilt[[3, 7]] = False
    pixel_size = 80.0
    w = int(round(float(ts.image_dimensions_physical[0]) / pixel_size / 2)) * 2
    h = int(round(float(ts.image_dimensions_physical[1]) / pixel_size / 2)) * 2
    full = torch.randn(ts.n_tilts, h, w)
    used = full[ts.use_tilt]

    from_full = ts.get_aligned_stack(full, pixel_size, interpolation="bilinear")
    from_used = ts.get_aligned_stack(used, pixel_size, interpolation="bilinear")
    assert from_full.shape == (ts.n_tilts - 2, h, w)
    assert torch.allclose(from_full, from_used)

    with pytest.raises(ValueError):
        ts.get_aligned_stack(used, pixel_size, tilt_ids=torch.tensor([3]))

    out = tmp_path / "aligned" / "TS_1.mrc"
    written = ts.write_aligned_stack(used, pixel_size, str(out), interpolation="bilinear",
                                     output_shape=(h + 10, w - 6))
    assert written.shape == (ts.n_tilts - 2, h + 10, w - 6)
    with mrcfile.open(out) as mrc:
        assert mrc.data.shape == written.shape
        assert mrc.voxel_size.x == pytest.approx(pixel_size)
    angles = [float(a) for a in out.with_suffix(".tlt").read_text().split()]
    expected = (ts.angles[ts.use_tilt] + ts.level_angle_y).tolist()
    assert angles == pytest.approx(expected, abs=0.01)


def test_volume_warp_warns():
    ts = TiltSeries(n_tilts=2)
    ts.angles = torch.tensor([0.0, 10.0])
    ts.image_dimensions_physical = torch.tensor([32.0, 32.0])
    ts.grid_volume_warp_x = CubicGrid((2, 2, 2, 2), torch.ones(2, 2, 2, 2))
    with pytest.warns(UserWarning, match="volume warp"):
        ts.get_aligned_stack(torch.randn(2, 32, 32), pixel_size=1.0)


def test_coarse_mapping_matches_exact():
    """map_step lattice + bilinear upsampling must match per-pixel evaluation."""
    ts = _load_ts1()
    # Coarse enough that the warp knot spacing (~300 px) allows a lattice step > 1
    ts.image_dimensions_physical = ts.image_dimensions_physical * 6
    pixel_size = 20.0
    w = int(round(float(ts.image_dimensions_physical[0]) / pixel_size / 2)) * 2
    h = int(round(float(ts.image_dimensions_physical[1]) / pixel_size / 2)) * 2
    tilt_data = torch.randn(ts.n_tilts, h, w)
    tilt_ids = torch.tensor([0, 40])
    exact = ts.get_aligned_stack(tilt_data, pixel_size, tilt_ids=tilt_ids, map_step=1,
                                 interpolation="bilinear")
    coarse = ts.get_aligned_stack(tilt_data, pixel_size, tilt_ids=tilt_ids, map_step=16,
                                  interpolation="bilinear")
    assert (coarse - exact).abs().max() < 0.02
