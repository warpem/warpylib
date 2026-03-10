# warpylib

A Python library replicating core functionality from [WarpLib](https://github.com/warpem/warp), a C# framework for cryo-electron tomography (cryo-ET) data processing. Built on PyTorch for GPU acceleration and automatic differentiation.

## Overview

warpylib provides:

- **Cubic B-spline grid interpolation** — spatially varying parameter fields with smooth interpolation
- **CTF modelling** — contrast transfer function calculation with full WarpLib parameter compatibility
- **Tilt series geometry** — 3D↔2D coordinate transforms, angle handling, metadata I/O
- **Subtomogram reconstruction** — weighted backprojection with CTF correction
- **Image operations** — Fourier-space shifting, resizing, normalization, bandpass filtering

The library preserves WarpLib's coordinate conventions and unit system (Ångströms, degrees, µm) for compatibility with Warp-processed datasets. If you're looking for a library that reads Warp metadata and reproduces Warp's exact tilt series model behavior, this is it!

## Installation

```bash
pip install warpylib
```

For development:

```bash
git clone <repo>
cd warpylib
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.8

| Package | Version | Purpose |
|---|---|---|
| [`torch`](https://github.com/pytorch/pytorch) | ≥ 2.0.0 | Core computation, GPU support, autograd |
| [`torch-cubic-spline-grids`](https://github.com/teamtomo/torch-cubic-spline-grids) | ≥ 0.0.10 | B-spline interpolation |
| [`torch-projectors`](https://github.com/warpem/torch-projectors) | — | Fourier-space backprojection |
| [`torch-subpixel-crop`](https://github.com/teamtomo/torch-subpixel-crop) | ≥ 0.1.1 | Image extraction at fractional pixel coordinates |
| [`numpy`](https://github.com/numpy/numpy) | ≥ 1.20.0 | Array utilities |
| [`lxml`](https://github.com/lxml/lxml) | ≥ 4.9.0 | XML metadata I/O |
| [`starfile`](https://github.com/teamtomo/starfile) | ≥ 0.5.0 | RELION STAR file I/O |
| [`mrcfile`](https://github.com/ccpem/mrcfile) | — | MRC file I/O |
| [`pillow`](https://github.com/python-pillow/Pillow) | — | Image loading/saving |

With [torch-projectors](https://github.com/warpem/torch-projectors), you're responsible for sourcing the correct package version for your CUDA environment if you want to use the GPU. Unfortunately, pip can't do this automatically.

## Quick Start

```python
import torch
from warpylib import TiltSeries, CTF, CubicGrid, euler_to_matrix

# Load a tilt series from WarpLib XML metadata
ts = TiltSeries(path="path/to/tiltseries.xml")
ts.load_meta("path/to/tiltseries.xml")

# Load tilt images resampled to a target pixel size
tilt_data = ts.load_images(original_pixel_size=10.0, target_pixel_size=5.0)

# Get per-tilt 2D positions for a set of particle coordinates
coords_3d = torch.tensor([[512.0, 512.0, 256.0]])  # (N, 3) in Ångströms
coords_2d = ts.get_position_in_all_tilts(coords_3d)  # (N, n_tilts, 3)

# Reconstruct subtomograms
subtomos = ts.reconstruct_subvolumes(
    tilt_data=tilt_data,
    coords=coords_2d,
    pixel_size=5.0,
    size=64,
)
# subtomos: (N, 64, 64, 64)
```

## Modules

### `TiltSeries`

The central class for cryo-ET tilt series. Stores geometry, CTF parameters, and motion corrections as spatially varying cubic B-spline grids.

#### Loading metadata

```python
from warpylib import TiltSeries

# From WarpLib XML file
ts = TiltSeries(path="tiltseries.xml")
ts.load_meta("tiltseries.xml")

# From a RELION tomo STAR file
ts = TiltSeries(n_tilts=41)
ts.initialize_from_tomo_star("tomo.star")

# Save back to XML
ts.save_meta("tiltseries_modified.xml")
```

#### Key attributes

```python
ts.angles             # (n_tilts,) tilt angles in degrees
ts.dose               # (n_tilts,) accumulated electron dose
ts.use_tilt           # (n_tilts,) bool mask — which tilts to include
ts.tilt_axis_angles   # (n_tilts,) in-plane rotation of the tilt axis
ts.tilt_movie_paths   # list of paths to per-tilt image files

# Spatially varying grids (CubicGrid objects)
ts.grid_ctf_defocus        # defocus as a function of XY position and tilt
ts.grid_movement_x         # motion correction X shifts
ts.grid_movement_y         # motion correction Y shifts
ts.grid_volume_warp_x      # 3D volume deformation field (X component)
```

#### Coordinate transforms

All coordinates are in Ångströms. Volume coordinates use the WarpLib convention (X, Y, Z).

```python
import torch

# 3D particle positions: (N, 3)
particle_positions = torch.tensor([
    [1024.0, 1024.0, 512.0],
    [2048.0, 768.0,  300.0],
])

# Project to all tilts: (N, n_tilts, 3) — X, Y in Å, Z is defocus in µm
coords_2d = ts.get_position_in_all_tilts(particle_positions)

# Project to a single tilt: (N, 3)
coords_tilt5 = ts.get_positions_in_one_tilt(particle_positions, tilt_id=5)
```

#### Euler angle transforms

Particle orientations (ZYZ Euler angles) can be transformed into the reference frame of each tilt:

```python
# Particle orientations in the volume frame: (N, 3) — rot, tilt, psi in radians
orientations = torch.zeros(len(particle_positions), 3)

# Transform to each tilt's frame: (N, n_tilts, 3)
tilt_orientations = ts.get_angle_in_all_tilts(orientations)

# For a single tilt: (N, 3)
tilt5_orientations = ts.get_angles_in_one_tilt(orientations, tilt_id=5)
```

#### CTF generation

```python
# Generate CTFs for all particles at their per-tilt positions
# coords has shape (N, n_tilts, 3) from get_position_in_all_tilts
ctfs = ts.get_ctfs_for_particles(
    coords=coords_2d,
    pixel_size=5.0,
)
# ctfs is a CTF object with batched parameters of shape (N, n_tilts)
```

#### Image loading

```python
# Load all tilt images, optionally resampling
tilt_data = ts.load_images(
    original_pixel_size=10.0,   # pixel size of raw images in Å/px
    target_pixel_size=5.0,      # desired output pixel size
)
# tilt_data: (n_tilts, H, W) float tensor

# Just load image dimensions without reading data
ts.load_image_dimensions(original_pixel_size=10.0)
```

#### Subtomogram reconstruction

```python
subtomos = ts.reconstruct_subvolumes(
    tilt_data=tilt_data,        # (n_tilts, H, W)
    coords=coords_2d,           # (N, n_tilts, 3) from get_position_in_all_tilts
    pixel_size=5.0,             # Å/px
    size=64,                    # output box size (isotropic, must be even)
    oversampling=2.0,           # internal oversampling for aliasing control
    apply_ctf=True,             # multiply by CTF in Fourier space
    ctf_weighted=True,          # weight by CTF² for Wiener-filter-like correction
    correct_attenuation=True,   # apply sinc² correction for linear interpolation
    tilt_ids=None,              # optionally restrict to a subset of tilts
)
# subtomos: (N, 64, 64, 64)
```

#### Full tomogram reconstruction

```python
tomogram = ts.reconstruct_full(
    tilt_data=tilt_data,
    output_size=(1000, 1000, 200),   # (X, Y, Z) voxels
    pixel_size=5.0,
    oversampling=1.0,
)
# tomogram: (Z, Y, X) float tensor
```

#### CTF volume reconstruction

Reconstruct the CTF modulation as a 3D volume (useful for CTF refinement):

```python
ctf_vols = ts.reconstruct_subvolume_ctfs(
    coords=coords_2d,
    pixel_size=5.0,
    size=64,
)
# ctf_vols: (N, 64, 64, 64)
```

#### Importing alignments

```python
# Import per-tilt shifts and angles from external alignment files
ts.import_alignments(
    shifts=torch.zeros(ts.n_tilts, 2),     # (n_tilts, 2) XY shifts in Å
    angles=torch.zeros(ts.n_tilts),         # (n_tilts,) in-plane rotation in degrees
)
```

#### Coordinate shifts

```python
# Shift a tilt's grid fields and propagate changes to motion/deformation grids
ts.apply_tilt_shift_and_propagate(tilt_id=3, shift_x=10.0, shift_y=-5.0)

# Apply a 3D shift to the volume origin and all per-tilt positions
ts.apply_tomogram_shift_3d(shift=torch.tensor([10.0, 0.0, -5.0]))
```

---

### `CTF`

Models the contrast transfer function of a cryo-EM experiment. All parameters match WarpLib's conventions and units.

```python
from warpylib import CTF
import torch

ctf = CTF()

# Microscope parameters
ctf.voltage = 300.0          # kV
ctf.cs = 2.7                 # spherical aberration (mm)
ctf.pixel_size = 1.5         # Å/px

# Defocus (underfocus positive, in µm)
ctf.defocus = 2.0
ctf.defocus_delta = 0.1      # astigmatism magnitude (µm)
ctf.defocus_angle = 45.0     # astigmatism angle (degrees)

# Amplitude contrast and phase shift
ctf.amplitude = 0.07
ctf.phase_shift = 0.0        # in units of π

# B-factor (dose weighting)
ctf.bfactor = -50.0          # Å²

# Compute 2D CTF in real-frequency half (rfft) format
ctf_2d = ctf.get_2d(size=(256, 256))
# ctf_2d: (256, 129) — use torch.fft.irfftn to go back to real space

# Compute radial 1D profile
ctf_1d = ctf.get_1d(width=128)
# ctf_1d: (128,)
```

**Batched parameters:** Any scalar parameter can be replaced with a tensor to compute CTFs for multiple particles simultaneously:

```python
ctf = CTF()
ctf.voltage = 300.0
ctf.cs = 2.7
ctf.pixel_size = 1.5

# Per-particle defocus values
ctf.defocus = torch.tensor([1.5, 2.0, 2.5, 3.0])  # (4,)

ctf_2d = ctf.get_2d(size=(256, 256))
# ctf_2d: (4, 256, 129)
```

**Options for `get_2d` / `get_1d`:**

```python
ctf.get_2d(
    size=(256, 256),
    amp_squared=False,         # if True, return |CTF|² (power spectrum)
    ignore_bfactor=False,      # skip dose-weighting B-factor
    ignore_scale=False,        # skip amplitude scale factor
    ignore_below_res=0.0,      # taper CTF to zero below this resolution (Å)
    ignore_transition_res=0.0, # transition width for the above taper (Å)
)
```

---

### `CubicGrid`

Cubic B-spline interpolation grids for spatially varying parameters. Matches WarpLib's `CubicGrid` implementation, supporting gradients for optimization.

```python
from warpylib import CubicGrid
import torch

# Create a 10×10×10 grid with manually specified values
values = torch.zeros(10 * 10 * 10)  # flat, in einspline layout
grid = CubicGrid(
    dimensions=(10, 10, 10),
    values=values,
    margins=(0.0, 0.0, 0.0),
    centered_spacing=True,
)

# Interpolate at normalized coordinates in [0, 1]³
# Input shape: (N, 3), coords in (x, y, z) order
coords = torch.rand(100, 3)
interpolated = grid.get_interpolated(coords)   # (100,)

# Evaluate on a regular grid
grid_values = grid.get_interpolated_grid(
    value_grid=(20, 20, 20),   # output resolution
    border=(0.0, 0.0, 0.0),
)
# grid_values: (8000,) — flattened 20×20×20 in einspline layout
```

**Convenience constructors:**

```python
# Uniform-value grid
grid = CubicGrid.from_scalar(dimensions=(5, 5, 5), value=1.0)

# Linear gradient along one axis
grid = CubicGrid(
    dimensions=(10, 10, 10),
    gradient_direction=0,   # 0=X, 1=Y, 2=Z
    value_min=0.0,
    value_max=1.0,
)
```

**Resampling and slicing:**

```python
# Resize to new resolution (bilinear resampling of control points)
grid_fine = grid.resize(new_size=(20, 20, 20))

# Extract 2D slices
slice_xy = grid.get_slice_xy(index=5)   # (X*Y,)
slice_xz = grid.get_slice_xz(index=5)
slice_yz = grid.get_slice_yz(index=5)

# Reduce dimensionality
grid_2d = grid.collapse_z()    # average over Z
grid_1d = grid.collapse_xy()   # average over X and Y
```

**Device placement:**

```python
grid_gpu = grid.to(torch.device("cuda"))
```

**XML serialization** (WarpLib-compatible format):

```python
from lxml import etree

root = etree.Element("TiltSeries")
grid.save_to_xml(root)

# Load back
grid_loaded = CubicGrid.load_from_xml(root)
```

---

### Euler angle utilities

Functions for converting between rotation matrices and Euler angles, supporting batched inputs.

```python
from warpylib import euler_to_matrix, matrix_to_euler
from warpylib import euler_xyz_extrinsic_to_matrix, matrix_to_euler_xyz_extrinsic
from warpylib import rotate_x, rotate_y, rotate_z
import torch
import math

# ZYZ Euler convention (cryo-EM standard: rot, tilt, psi — all in radians)
angles = torch.tensor([[0.0, math.radians(30.0), math.radians(45.0)])  # (1, 3)
R = euler_to_matrix(angles)         # (1, 3, 3)
angles_back = matrix_to_euler(R)    # (1, 3)

# XYZ extrinsic convention
angles_xyz = torch.tensor([[0.1, 0.2, 0.3]])
R_xyz = euler_xyz_extrinsic_to_matrix(angles_xyz)   # (1, 3, 3)

# Batched: convert 1000 rotation matrices at once
R_batch = torch.rand(1000, 3, 3)   # not valid rotations, just for shape demo
# In practice these would be orthonormal matrices
angles_batch = matrix_to_euler(R_batch)   # (1000, 3)

# Elementary rotations (angle in radians, batched)
angle = torch.tensor([math.radians(45.0)])
Rx = rotate_x(angle)   # (1, 3, 3)
Ry = rotate_y(angle)
Rz = rotate_z(angle)
```

---

### `InterpolatingBSpline1d` / `InterpolatingBSpline2d`

Callable torch modules that solve for cubic B-spline coefficients and evaluate them. Useful as differentiable interpolators within neural networks or optimization loops.

```python
from warpylib import InterpolatingBSpline1d, InterpolatingBSpline2d
import torch

# 1D: fit a spline through 20 data points and evaluate at 100 query points
spline = InterpolatingBSpline1d()

data = torch.sin(torch.linspace(0, 2 * torch.pi, 20))  # (20,)
query = torch.linspace(0, 1, 100)                       # normalized coords in [0, 1]

values = spline(data, query)   # (100,)

# 2D: fit through a (32, 32) grid, evaluate at arbitrary (x, y) coords
spline2d = InterpolatingBSpline2d()

data2d = torch.rand(32, 32)
coords = torch.rand(500, 2)    # (N, 2), normalized to [0, 1]²

values2d = spline2d(data2d, coords)   # (500,)
```

Both modules preserve gradients, so they can be used inside `torch.autograd` computations.

---

### `warpylib.ops` — Image operations

Low-level image processing operations in real and Fourier space.

```python
import torch
import warpylib.ops as ops
```

#### Resizing

```python
image = torch.rand(256, 256)

# Real-space resizing (Fourier-domain crop/pad then IFFT)
small = ops.resize(image, new_size=(128, 128))
large = ops.resize(image, new_size=(512, 512))

# Already in Fourier space (full FFT)
image_ft = torch.fft.fftn(image)
small_ft = ops.resize_ft(image_ft, new_size=(128, 128))

# In half-space (rFFT)
image_rft = torch.fft.rfftn(image)
small_rft = ops.resize_rft(image_rft, new_size=(128, 128))

# Non-integer scaling
image_2x = ops.rescale(image, scale_factor=2.0)
```

#### Normalization

```python
# Zero-mean, unit-variance — optionally restricted to a circular region
image_norm = ops.norm(image, dimensionality=2)
image_norm = ops.norm(image, dimensionality=2, diameter=200, mode="inner")

# In Fourier space
image_rft = torch.fft.rfftn(image)
image_rft_norm = ops.norm_rft(image_rft, dimensionality=2)
```

#### Sinc² correction

Linear interpolation suppresses high-frequency signal by a sinc² envelope. This corrects for that:

```python
# 2D correction map
correction = ops.get_sinc2_correction(size=(256, 256), oversampling=2.0)
image_corrected = torch.fft.irfftn(
    torch.fft.rfftn(image) * ops.get_sinc2_correction_rft((256, 256), oversampling=2.0)
)

# 3D
correction_3d = ops.get_sinc2_correction(size=(64, 64, 64))
```

#### Masking and plane subtraction

```python
# Rectangular mask (zeros outside a centred box)
masked = ops.mask_rectangular(volume, size=(50, 50, 50))

# Fit and remove a plane from a tomogram slice
plane_normal, plane_offset = ops.fit_plane(slice_2d)
flat_slice = ops.subtract_plane(slice_2d)
```

#### Tilt data preprocessing

Convenience function applying normalization, masking, and optional bandpass to a tilt stack:

```python
tilt_data = torch.rand(41, 3712, 3712)   # (n_tilts, H, W)

processed = ops.preprocess_tilt_data(
    tilt_data,
    normalize=True,
    mask_radius=0.9,           # fraction of half-width to use as circular mask
    bandpass_highpass=500.0,   # high-pass resolution cutoff in Å (None to skip)
    bandpass_lowpass=10.0,     # low-pass resolution cutoff in Å (None to skip)
    pixel_size=5.0,            # needed when bandpass is specified
)
```

---

## Design notes

**Method binding pattern:** `TiltSeries` methods are defined in focused submodules (`positions.py`, `ctf.py`, etc.) and attached to the class in `__init__.py`. This keeps related code together while exposing a unified API.

**Coordinate conventions:** Volume coordinates follow WarpLib's (X, Y, Z) convention in Ångströms. PyTorch tensors internally use (Z, Y, X) indexing; the library handles conversion transparently.

**Grid layout:** `CubicGrid` values are stored in einspline layout `[(z*Y+y)*X+x]` to match WarpLib's serialization format.

**Batching:** CTF parameters, Euler angles, and coordinate transforms all support arbitrary leading batch dimensions via PyTorch broadcasting.

**Gradients:** `CubicGrid`, `InterpolatingBSpline1d/2d`, CTF calculations, and Fourier-space operations all preserve gradients for use in differentiable optimization.

## Running tests

```bash
pytest tests/
```

## License

MIT
