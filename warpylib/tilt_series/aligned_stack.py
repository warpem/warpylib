"""
Aligned tilt stack generation.

The forward model in positions.py maps a 3D volume coordinate to a raw image
position via::

    raw = R2(-tilt_axis_angle) @ q + tilt_axis_offset + image_center
    raw = raw - movement(raw / image_dimensions, tilt)

where ``q`` is the xy part of ``Ry(angle + level_angle_y) @ Rx(level_angle_x) @
(x - volume_center + volume_warp)``. The "aligned frame" is ``q + image_center``:
tilt axis along Y, no per-tilt offsets and no 2D image warp. Since the mapping
from aligned frame to raw image is explicit, an aligned stack is produced by
pulling raw pixels through it - no inversion of the warp grids is needed.

The 4D volume warp grids act on 3D coordinates before projection and therefore
cannot be removed from a 2D stack; they are ignored here (with a warning).
"""

import copy
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import mrcfile
import torch
import torch.nn.functional as F

from ..cubic_grid import CubicGrid
from ..euler import euler_to_matrix
from .positions import get_position_in_all_tilts


def get_raw_positions_from_aligned(
    ts: "TiltSeries",
    coords: torch.Tensor,
    tilt_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Map positions in the aligned frame to positions in the raw tilt images.

    Applies the 2D part of the forward model: tilt axis rotation, tilt axis
    offsets and the image warp (movement) grids. Volume warp grids are not
    involved.

    Args:
        ts: TiltSeries instance
        coords: Positions in the aligned frame, shape (..., 2) as (X, Y). Units are
               the same as the XY output of get_position_in_all_tilts (Angstroms,
               including size_rounding_factors), i.e. divide by pixel_size to get
               pixels. The image center lies at image_dimensions_physical / 2.
        tilt_ids: Tilt index for each position, integer tensor broadcastable to
                 coords.shape[:-1]

    Returns:
        Raw image positions, shape (..., 2), in the same units as coords
    """
    device = ts.angles.device
    coords = coords.to(device=device, dtype=torch.float32)
    tilt_ids = torch.as_tensor(tilt_ids, device=device).long()
    tilt_ids = tilt_ids.expand(coords.shape[:-1])

    srf = ts.size_rounding_factors[:2]
    image_center = ts.image_dimensions_physical / 2
    grid_step = 1.0 / (ts.n_tilts - 1) if ts.n_tilts > 1 else 0.0

    # The in-plane rotation is the psi part of the Euler matrix built in
    # get_position_in_all_tilts; since Rz only mixes X and Y, its top-left 2x2
    # block maps aligned XY to raw XY.
    euler_angles = torch.stack([
        torch.zeros(ts.n_tilts, dtype=torch.float32, device=device),
        torch.zeros(ts.n_tilts, dtype=torch.float32, device=device),
        -ts.tilt_axis_angles * (torch.pi / 180.0),
    ], dim=-1)
    rotations = euler_to_matrix(euler_angles)[:, :2, :2]  # (n_tilts, 2, 2)
    offsets = torch.stack([ts.tilt_axis_offset_x, ts.tilt_axis_offset_y], dim=-1)

    # Undo size rounding so the geometry is evaluated in physical Angstroms
    centered = coords / srf - image_center
    raw = torch.einsum('...ji,...i->...j', rotations[tilt_ids], centered)
    raw = raw + offsets[tilt_ids] + image_center

    # Image warp, evaluated at the pre-warp position as in the forward model
    grid_coords = torch.stack([
        raw[..., 0] / ts.image_dimensions_physical[0],
        raw[..., 1] / ts.image_dimensions_physical[1],
        tilt_ids.float() * grid_step,
    ], dim=-1).reshape(-1, 3)
    movement = torch.stack([
        ts.grid_movement_x.get_interpolated(grid_coords),
        ts.grid_movement_y.get_interpolated(grid_coords),
    ], dim=-1).reshape(raw.shape)
    raw = raw - movement

    return raw * srf


def get_positions_in_aligned_stack(ts: "TiltSeries", coords: torch.Tensor) -> torch.Tensor:
    """
    Transform 3D volume coordinates to 2D positions in the aligned stack.

    Same as get_position_in_all_tilts but without tilt axis rotation, tilt axis
    offsets and image warp, so that
    ``get_raw_positions_from_aligned(ts, get_positions_in_aligned_stack(ts, c), t)``
    equals ``get_position_in_all_tilts(ts, c)[..., :2]``. Volume warp is still
    applied, as it is a property of the 3D coordinate.

    Args:
        ts: TiltSeries instance
        coords: Coordinates in volume space (Angstroms), shape (..., n_tilts, 3)

    Returns:
        Positions in the aligned frame, shape (..., n_tilts, 2), in the same units
        as get_position_in_all_tilts
    """
    device = ts.angles.device
    ts_aligned = copy.copy(ts)
    ts_aligned.tilt_axis_angles = torch.zeros_like(ts.tilt_axis_angles)
    ts_aligned.tilt_axis_offset_x = torch.zeros_like(ts.tilt_axis_offset_x)
    ts_aligned.tilt_axis_offset_y = torch.zeros_like(ts.tilt_axis_offset_y)
    ts_aligned.grid_movement_x = CubicGrid((1, 1, 1)).to(device)
    ts_aligned.grid_movement_y = CubicGrid((1, 1, 1)).to(device)
    return get_position_in_all_tilts(ts_aligned, coords)[..., :2]


def get_aligned_stack(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    pixel_size: float,
    tilt_ids: Optional[torch.Tensor] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "bicubic",
    fill_value: Union[float, str] = "mean",
    map_step: int = 16,
) -> torch.Tensor:
    """
    Resample tilt images into an aligned stack.

    Corrects for tilt axis angle, tilt axis offsets and image warp (movement)
    grids. The result has the tilt axis along Y and the tilt axis through the
    image center; tilt angles for it are ``ts.angles + ts.level_angle_y``. 4D
    volume warp grids cannot be represented in a 2D stack and are ignored.

    Args:
        ts: TiltSeries instance
        tilt_data: Tilt images, shape (n_tilts, H, W) or (n_used_tilts, H, W)
                  containing only the tilts where ts.use_tilt is True
        pixel_size: Pixel size of tilt_data in Angstroms (output has the same)
        tilt_ids: Indices of tilts to include, in output order. Defaults to the
                 tilts where ts.use_tilt is True.
        output_shape: (H, W) of the output images. Defaults to tilt_data's (H, W).
        interpolation: 'bicubic', 'bilinear' or 'nearest'. On devices without
                      bicubic grid_sample (MPS) bicubic sampling runs on CPU.
        fill_value: Value for output pixels that fall outside the raw image, or
                   'mean' to use each raw image's mean
        map_step: Maximum spacing in output pixels at which the aligned -> raw
                 mapping is evaluated exactly before bilinear upsampling. The
                 spacing is further limited to 1/50 of the image warp grid's knot
                 spacing. 1 evaluates the mapping at every pixel.

    Returns:
        Aligned stack, shape (len(tilt_ids), H_out, W_out)
    """
    for grid in (ts.grid_volume_warp_x, ts.grid_volume_warp_y, ts.grid_volume_warp_z):
        if torch.any(grid.values != 0):
            warnings.warn(
                "TiltSeries has non-zero volume warp grids; these cannot be applied "
                "to a 2D stack and are ignored in the aligned stack.",
                stacklevel=2,
            )
            break

    if tilt_ids is None:
        tilt_ids = torch.nonzero(ts.use_tilt, as_tuple=True)[0]
    tilt_ids = torch.as_tensor(tilt_ids).long().cpu()

    # Map tilt indices to rows of tilt_data
    if tilt_data.shape[0] == ts.n_tilts:
        rows = tilt_ids
    else:
        used_idx = torch.nonzero(ts.use_tilt, as_tuple=True)[0].cpu()
        if tilt_data.shape[0] != len(used_idx):
            raise ValueError(
                f"tilt_data has {tilt_data.shape[0]} frames but TiltSeries has "
                f"{ts.n_tilts} tilts ({len(used_idx)} used). "
                "tilt_data must have either all tilts or only the used tilts."
            )
        row_of_tilt = torch.full((ts.n_tilts,), -1, dtype=torch.long)
        row_of_tilt[used_idx] = torch.arange(len(used_idx))
        rows = row_of_tilt[tilt_ids]
        if torch.any(rows < 0):
            raise ValueError("tilt_ids contains unused tilts, which are absent from tilt_data")

    in_h, in_w = tilt_data.shape[-2:]
    out_h, out_w = output_shape if output_shape is not None else (in_h, in_w)
    device = tilt_data.device

    # The aligned -> raw mapping is affine plus the smooth image warp. Evaluating
    # the warp spline at every pixel dominates the runtime, so the mapping is
    # evaluated exactly on a lattice spanning the output (first and last pixel
    # included) and bilinearly upsampled: exact for the affine part, and far
    # well below a pixel when the lattice is much finer than the warp grid's knot
    # spacing (error scales with (step / knot spacing)^2).
    step = map_step
    for grid in (ts.grid_movement_x, ts.grid_movement_y):
        for n_knots, n_pixels in zip(grid.dimensions[:2], (in_w, in_h)):
            if n_knots > 1:
                step = min(step, int(n_pixels / (n_knots - 1) / 50))
    step = max(step, 1)
    n_y = min(out_h, -(-(out_h - 1) // step) + 1)
    n_x = min(out_w, -(-(out_w - 1) // step) + 1)
    # Output center (W/2, H/2) maps to the image center, as raw pixel N/2 does
    # in particle extraction
    y, x = torch.meshgrid(
        torch.linspace(0, out_h - 1, n_y, device=device) - out_h // 2 + in_h / 2,
        torch.linspace(0, out_w - 1, n_x, device=device) - out_w // 2 + in_w / 2,
        indexing="ij",
    )
    aligned = torch.stack([x, y], dim=-1) * pixel_size  # (n_y, n_x, 2)

    sample_device = device
    if interpolation == "bicubic" and device.type == "mps":
        sample_device = torch.device("cpu")

    output = torch.empty(len(tilt_ids), out_h, out_w, dtype=torch.float32, device=device)
    for i, (t, row) in enumerate(zip(tilt_ids.tolist(), rows.tolist())):
        raw = get_raw_positions_from_aligned(ts, aligned, torch.tensor(t)).to(device)
        raw = raw / pixel_size  # pixels, (X, Y)
        if (n_y, n_x) != (out_h, out_w):
            raw = F.interpolate(
                raw.permute(2, 0, 1)[None], size=(out_h, out_w),
                mode="bilinear", align_corners=True,
            )[0].permute(1, 2, 0)

        image = tilt_data[row].float()
        # align_corners=True: -1 and 1 are the centers of the first and last pixel
        grid = torch.stack([
            raw[..., 0] / (in_w - 1) * 2 - 1,
            raw[..., 1] / (in_h - 1) * 2 - 1,
        ], dim=-1)
        sampled = F.grid_sample(
            image[None, None].to(sample_device),
            grid[None].to(sample_device),
            mode=interpolation,
            padding_mode="reflection",
            align_corners=True,
        )[0, 0].to(device)

        # Small tolerance so pixels mapping exactly onto the border survive
        # float rounding
        eps = 1e-3
        outside = (
            (raw[..., 0] < -eps) | (raw[..., 0] > in_w - 1 + eps)
            | (raw[..., 1] < -eps) | (raw[..., 1] > in_h - 1 + eps)
        )
        fill = image.mean() if fill_value == "mean" else fill_value
        output[i] = torch.where(outside, fill, sampled)

    return output


def write_aligned_stack(
    ts: "TiltSeries",
    tilt_data: torch.Tensor,
    pixel_size: float,
    path: str,
    tilt_ids: Optional[torch.Tensor] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: str = "bicubic",
    fill_value: Union[float, str] = "mean",
    map_step: int = 16,
) -> torch.Tensor:
    """
    Write an aligned stack as MRC, plus a .tlt file with its tilt angles.

    See get_aligned_stack for what is corrected. The .tlt file (same path with
    a .tlt suffix) contains ``ts.angles + ts.level_angle_y`` for the written
    tilts. ts.level_angle_x is a constant rotation of the specimen about X that a
    tilt-angle file cannot express; a reconstruction from the aligned stack is
    rotated by it relative to Warp's volume frame.

    Args:
        ts: TiltSeries instance
        tilt_data: Tilt images, shape (n_tilts, H, W) or (n_used_tilts, H, W)
        pixel_size: Pixel size of tilt_data in Angstroms
        path: Output MRC path
        tilt_ids, output_shape, interpolation, fill_value, map_step: see
            get_aligned_stack

    Returns:
        The aligned stack that was written, shape (len(tilt_ids), H_out, W_out)
    """
    if tilt_ids is None:
        tilt_ids = torch.nonzero(ts.use_tilt, as_tuple=True)[0]
    tilt_ids = torch.as_tensor(tilt_ids).long().cpu()

    stack = get_aligned_stack(
        ts, tilt_data, pixel_size,
        tilt_ids=tilt_ids,
        output_shape=output_shape,
        interpolation=interpolation,
        fill_value=fill_value,
        map_step=map_step,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(stack.cpu().numpy().astype("float32"))
        mrc.voxel_size = (pixel_size, pixel_size, pixel_size)

    angles = (ts.angles.cpu()[tilt_ids] + ts.level_angle_y).tolist()
    path.with_suffix(".tlt").write_text("\n".join(f"{a:.2f}" for a in angles) + "\n")

    return stack
