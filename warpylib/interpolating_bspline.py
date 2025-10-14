"""
Interpolating B-spline implementation in PyTorch.

This module implements cubic interpolating B-splines that match einspline's behavior.
Unlike standard B-splines which use data values as control points, interpolating B-splines
solve for coefficients that make the spline pass exactly through the data points.

This implementation supports:
- Natural boundary conditions (second derivative = 0 at boundaries)
- Uniform grids only
- Full API compliance with torch-cubic-spline-grids
"""

from typing import Callable, Optional, Tuple, Union
import einops
import torch
import torch.nn as nn


# Einspline's basis matrix (transposed format)
# This is multiplied by [t^3, t^2, t, 1] to get basis functions
EINSPLINE_BASIS_MATRIX = torch.tensor([
    [-1.0/6.0,  3.0/6.0, -3.0/6.0,  1.0/6.0],
    [ 3.0/6.0, -6.0/6.0,  0.0/6.0,  4.0/6.0],
    [-3.0/6.0,  3.0/6.0,  3.0/6.0,  1.0/6.0],
    [ 1.0/6.0,  0.0/6.0,  0.0/6.0,  0.0/6.0]
], dtype=torch.float32)


def solve_band_matrix_1d(
    bands: torch.Tensor,
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Solve a band matrix system for B-spline coefficients.

    Uses torch.linalg.solve for full differentiability.

    Args:
        bands: (M+2, 4) band matrix with boundary conditions.
               Each row contains [left, diag, right, rhs] for the system.
        data: (M,) data values to interpolate (can require gradients)

    Returns:
        coefs: (M+2,) B-spline coefficients

    The band matrix format is:
    - Row 0: boundary condition at left
    - Rows 1 to M: interpolation equations
    - Row M+1: boundary condition at right
    """
    M = data.shape[0]

    # Build the full matrix (M+2 x M+2)
    # The band storage format varies by row:
    # - Row 0: bands[0, :] = [coef_0, coef_1, coef_2, rhs]
    # - Rows 1-M: bands[i, :] = [coef_{i-1}, coef_i, coef_{i+1}, rhs]
    # - Row M+1: bands[M+1, :] = [coef_{M-1}, coef_M, coef_{M+1}, rhs]

    # Create dense matrix from band storage
    A = torch.zeros(M + 2, M + 2, dtype=data.dtype, device=data.device)

    # Row 0: three coefficients [0, 1, 2]
    A[0, 0] = bands[0, 0]
    A[0, 1] = bands[0, 1]
    A[0, 2] = bands[0, 2]

    # Rows 1 to M: tridiagonal [i-1, i, i+1]
    for i in range(1, M + 1):
        A[i, i - 1] = bands[i, 0]  # left
        A[i, i] = bands[i, 1]      # diag
        A[i, i + 1] = bands[i, 2]  # right

    # Row M+1: three coefficients [M-1, M, M+1]
    A[M + 1, M - 1] = bands[M + 1, 0]
    A[M + 1, M] = bands[M + 1, 1]
    A[M + 1, M + 1] = bands[M + 1, 2]

    # Build RHS vector
    b = bands[:, 3].clone()
    b[1:M+1] = data

    # Solve using PyTorch's differentiable linear solver
    coefs = torch.linalg.solve(A, b)

    return coefs


def setup_natural_boundary_1d(
    M: int,
    delta_inv: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    Setup band matrix with natural boundary conditions for 1D interpolation.
    Natural boundary condition: second derivative = 0 at boundaries.

    Args:
        M: Number of data points
        delta_inv: Inverse of grid spacing (1 / delta)
        dtype: Data type for tensors
        device: Device for tensors

    Returns:
        bands: (M+2, 4) band matrix with boundary conditions
    """
    bands = torch.zeros(M + 2, 4, dtype=dtype, device=device)

    # Left boundary: second derivative = 0
    bands[0, 0] = 1.0 * delta_inv * delta_inv
    bands[0, 1] = -2.0 * delta_inv * delta_inv
    bands[0, 2] = 1.0 * delta_inv * delta_inv
    bands[0, 3] = 0.0

    # Right boundary: second derivative = 0
    bands[M + 1, 0] = 1.0 * delta_inv * delta_inv
    bands[M + 1, 1] = -2.0 * delta_inv * delta_inv
    bands[M + 1, 2] = 1.0 * delta_inv * delta_inv
    bands[M + 1, 3] = 0.0

    # Interior interpolation equations
    # B-spline basis: [1/6, 2/3, 1/6]
    basis = torch.tensor([1.0/6.0, 2.0/3.0, 1.0/6.0], dtype=dtype, device=device)
    for i in range(M):
        bands[i + 1, 0] = basis[0]
        bands[i + 1, 1] = basis[1]
        bands[i + 1, 2] = basis[2]
        # RHS (data) will be filled in by solver
        bands[i + 1, 3] = 0.0

    return bands


def find_coefs_1d(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Find B-spline coefficients for 1D data with natural boundary conditions.

    Args:
        data: (M,) or (C, M) data values to interpolate.
              If 2D, solves independently for each channel.

    Returns:
        coefs: (M+2,) or (C, M+2) B-spline coefficients
    """
    if data.ndim == 1:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, M = data.shape

    # Compute grid spacing (assuming grid covers [0, 1])
    delta = 1.0 / (M - 1)
    delta_inv = 1.0 / delta

    # Setup band matrix with natural boundary conditions
    bands = setup_natural_boundary_1d(
        M, delta_inv,
        dtype=data.dtype, device=data.device
    )

    # Solve for coefficients in each channel
    coefs = torch.zeros(C, M + 2, dtype=data.dtype, device=data.device)
    for c in range(C):
        coefs[c] = solve_band_matrix_1d(bands, data[c])

    if squeeze_output:
        coefs = coefs.squeeze(0)

    return coefs


def interpolate_grid_1d(
    data: torch.Tensor,
    u: torch.Tensor,
    matrix: Optional[torch.Tensor] = None,
    monotonicity: Optional[str] = None,
) -> torch.Tensor:
    """
    Interpolate 1D grid data using interpolating B-splines.

    This function signature matches torch-cubic-spline-grids API for compatibility.

    Args:
        data: (C, M+2) padded grid data (B-spline coefficients)
        u: (B, 1) batch of coordinates in [0, 1]
        matrix: Ignored (for API compatibility - einspline uses fixed basis matrix)
        monotonicity: Optional monotonicity constraint (not yet implemented)

    Returns:
        values: (B, C) interpolated values
    """
    if monotonicity is not None:
        raise NotImplementedError("Monotonicity constraints not yet implemented for interpolating B-splines")

    # u is (B, 1), squeeze to (B,)
    u = u.squeeze(-1)

    C, M_plus_2 = data.shape
    M = M_plus_2 - 2

    # Get basis matrix on correct device
    basis_matrix = EINSPLINE_BASIS_MATRIX.to(device=data.device, dtype=data.dtype)

    # Transform coordinates to grid space
    # u is in [0, 1], map to [0, M-1] in grid indices
    u_norm = u * (M - 1)

    # Find grid cell and local coordinate
    # Handle boundary cases like einspline's split_fraction
    i = torch.floor(u_norm).long()
    t = u_norm - i.float()

    # Handle out-of-bounds cases
    # if u < 0: i=0, t=u
    mask_low = u_norm < 0
    i = torch.where(mask_low, torch.zeros_like(i), i)
    t = torch.where(mask_low, u_norm, t)

    # if u >= M-2: i=M-2, t=u-(M-2)
    mask_high = u_norm >= M - 2
    i = torch.where(mask_high, torch.full_like(i, M - 2), i)
    t = torch.where(mask_high, u_norm - (M - 2), t)

    # Build power vector [t^3, t^2, t, 1]
    t_powers = torch.stack([t**3, t**2, t, torch.ones_like(t)], dim=1)  # (B, 4)

    # Evaluate for each channel
    B = u.shape[0]
    values = torch.zeros(B, C, dtype=data.dtype, device=data.device)

    for c in range(C):
        # Extract 4 control points for each query point
        control_points = torch.stack([
            data[c, i],
            data[c, i + 1],
            data[c, i + 2],
            data[c, i + 3]
        ], dim=1)  # (B, 4)

        # Einspline evaluation formula:
        # result = sum_j coef[i+j] * (matrix_row_j @ [t^3, t^2, t, 1])

        # Compute basis functions: matrix @ t_powers.T gives (4, B)
        # Each row is one basis function evaluated at all B points
        basis_vals = torch.matmul(basis_matrix, t_powers.T)  # (4, B)

        # Now compute weighted sum: control_points^T @ basis_vals
        # control_points: (B, 4) -> transpose to (4, B)
        # basis_vals: (4, B)
        # Result: (B,)
        values[:, c] = torch.sum(control_points.T * basis_vals, dim=0)

    return values


def coerce_to_multichannel_grid(grid: torch.Tensor, grid_ndim: int) -> torch.Tensor:
    """If missing, add a channel dimension to a multidimensional grid.

    e.g. for a 1D (M,) grid
          `M -> 1 M`
        `C M -> C M`
    """
    grid_is_multichannel = grid.ndim == grid_ndim + 1
    grid_is_single_channel = grid.ndim == grid_ndim
    if grid_is_single_channel is False and grid_is_multichannel is False:
        raise ValueError(f'expected a {grid_ndim}D grid, got {grid.ndim}')
    if grid_is_single_channel:
        grid = einops.rearrange(grid, '... -> 1 ...')
    return grid


def batch_iterator(iterable, n: int = 1):
    """Split an iterable into batches of constant length."""
    max_len = len(iterable)
    for idx in range(0, max_len, n):
        yield iterable[idx : min(idx + n, max_len)]


class InterpolatingBSpline1d(nn.Module):
    """
    1D Interpolating B-spline grid with API compliance to torch-cubic-spline-grids.

    This implements cubic B-splines that interpolate (pass through) the data points,
    matching einspline's behavior with natural boundary conditions.
    The coefficients are computed by solving a linear system.

    Args:
        resolution: Number of data points (int or tuple)
        n_channels: Number of channels
        minibatch_size: Maximum batch size for evaluation
    """

    ndim: int = 1
    _interpolation_function: Callable = staticmethod(interpolate_grid_1d)
    _interpolation_matrix: torch.Tensor = EINSPLINE_BASIS_MATRIX
    _data: nn.Parameter
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Union[int, Tuple[int]]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[str] = None,
    ):
        super().__init__()

        if resolution is None:
            resolution = 2
        if isinstance(resolution, tuple):
            resolution = resolution[0]

        # Initialize data as learnable parameters
        grid_shape = (n_channels, resolution)
        self.data = torch.zeros(size=grid_shape)

        self._minibatch_size = minibatch_size
        self._monotonicity = monotonicity

        # Register interpolation matrix as buffer
        self.register_buffer(
            name='interpolation_matrix',
            tensor=self._interpolation_matrix,
            persistent=False,
        )

        # Cache for coefficients
        self._coefs_cache = None
        self._data_version = None

    @property
    def data(self) -> torch.Tensor:
        """Get data values."""
        return self._data.detach()

    @data.setter
    def data(self, grid_data: torch.Tensor) -> None:
        """Set data values."""
        grid_data = coerce_to_multichannel_grid(grid_data, grid_ndim=self.ndim)
        self._data = nn.Parameter(grid_data)
        self._coefs_cache = None  # Invalidate cache

    @property
    def n_channels(self) -> int:
        """Number of channels in the grid."""
        return int(self._data.size(0))

    @property
    def resolution(self) -> Tuple[int, ...]:
        """Grid resolution (number of data points per dimension)."""
        return tuple(self._data.shape[1:])

    def _compute_coefs(self) -> torch.Tensor:
        """Compute B-spline coefficients from data."""
        return find_coefs_1d(self._data)

    def _get_coefs_cached(self) -> torch.Tensor:
        """Get coefficients with caching for inference."""
        current_version = self._data._version
        if self._coefs_cache is None or self._data_version != current_version:
            with torch.no_grad():
                self._coefs_cache = self._compute_coefs()
                self._data_version = current_version
        return self._coefs_cache

    def _interpolate(self, u: torch.Tensor) -> torch.Tensor:
        """Interpolate at given coordinates (internal method)."""
        # Compute coefficients (this will be part of computational graph if training)
        if self.training or torch.is_grad_enabled():
            coefs = self._compute_coefs()
        else:
            coefs = self._get_coefs_cached()

        return self._interpolation_function(
            coefs,
            u,
            matrix=self.interpolation_matrix,
            monotonicity=self._monotonicity,
        )

    def _coerce_to_batched_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        """Convert input coordinates to batched format (B, 1)."""
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim

        if self._input_is_coordinate_like is False and self.ndim == 1:
            u = einops.rearrange(u, '... -> ... 1')  # add singleton coord dimension
        else:
            u = torch.atleast_2d(u)  # add batch dimension if missing

        u, self._packed_shapes = einops.pack([u], pattern='* coords')

        if u.shape[-1] != self.ndim:
            ndim = u.shape[-1]
            raise ValueError(
                f'Cannot interpolate on a {self.ndim}D grid with {ndim}D coordinates'
            )
        return u

    def _unpack_interpolated_output(self, interpolated: torch.Tensor) -> torch.Tensor:
        """Convert batched output back to input format."""
        [interpolated] = einops.unpack(
            interpolated, packed_shapes=self._packed_shapes, pattern='* coords'
        )
        # Only squeeze the coordinate dimension if input was not coordinate-like
        # and we have a single channel (otherwise we want to keep the channel dimension)
        if self._input_is_coordinate_like is False and self.ndim == 1 and self.n_channels == 1:
            interpolated = einops.rearrange(interpolated, '... 1 -> ...')
        return interpolated

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline at given coordinates.

        Args:
            u: Coordinates in [0, 1] to evaluate. Can be:
               - (N,) for N points
               - (N, 1) for N points with explicit dimension
               - (..., 1) for arbitrary batch shapes

        Returns:
            values: Interpolated values with shape matching input:
                    - (N,) if single channel and input is (N,)
                    - (N, C) if multiple channels
                    - (..., C) for arbitrary batch shapes
        """
        u = self._coerce_to_batched_coordinates(u)  # (B, 1)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch_iterator(u, n=self._minibatch_size)
        ]  # List[Tensor[(B, C)]]
        interpolated = torch.cat(interpolated, dim=0)  # (B, C)

        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> "InterpolatingBSpline1d":
        """
        Create spline from existing data (API-compliant factory method).

        Args:
            data: (M,) or (C, M) data values

        Returns:
            Initialized InterpolatingBSpline1d instance
        """
        grid = cls()
        grid.data = data
        return grid

    # Alias for compatibility
    @classmethod
    def from_data(cls, data: torch.Tensor) -> "InterpolatingBSpline1d":
        """Alias for from_grid_data (backward compatibility)."""
        return cls.from_grid_data(data)
