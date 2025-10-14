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


def _build_1d_system_matrix(M: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Build the (M+2)x(M+2) system matrix for 1D interpolating B-spline.

    This matrix is the same for all channels and data values, depending only on M.
    Can be cached and reused.

    Args:
        M: Number of data points
        dtype: Data type for the matrix
        device: Device for the matrix

    Returns:
        A: (M+2, M+2) system matrix
    """
    # Compute grid spacing (assuming grid covers [0, 1])
    delta = 1.0 / (M - 1)
    delta_inv_sq = (1.0 / delta) ** 2

    # Build system matrix A: (M+2) x (M+2)
    A = torch.zeros(M + 2, M + 2, dtype=dtype, device=device)

    # Row 0: Left boundary condition (second derivative = 0)
    # c_0 - 2*c_1 + c_2 = 0
    A[0, 0] = delta_inv_sq
    A[0, 1] = -2.0 * delta_inv_sq
    A[0, 2] = delta_inv_sq

    # Rows 1 to M: Interpolation equations
    # (1/6)*c_{i-1} + (2/3)*c_i + (1/6)*c_{i+1} = data[i-1]
    for i in range(1, M + 1):
        A[i, i - 1] = 1.0 / 6.0
        A[i, i] = 2.0 / 3.0
        A[i, i + 1] = 1.0 / 6.0

    # Row M+1: Right boundary condition (second derivative = 0)
    # c_{M-1} - 2*c_M + c_{M+1} = 0
    A[M + 1, M - 1] = delta_inv_sq
    A[M + 1, M] = -2.0 * delta_inv_sq
    A[M + 1, M + 1] = delta_inv_sq

    return A


def find_coefs_1d(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Find B-spline coefficients for 1D data with natural boundary conditions.

    Solves the linear system Ac = d where:
    - A is the (M+2)x(M+2) system matrix
    - c are the B-spline coefficients (unknowns)
    - d is the RHS vector (data values with boundary conditions)

    Natural boundary condition: second derivative = 0 at boundaries.

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

    # Build system matrix (same for all channels)
    A = _build_1d_system_matrix(M, dtype=data.dtype, device=data.device)

    # Solve for each channel
    coefs = torch.zeros(C, M + 2, dtype=data.dtype, device=data.device)
    for c in range(C):
        # Build RHS vector
        b = torch.zeros(M + 2, dtype=data.dtype, device=data.device)
        b[0] = 0.0  # Left boundary condition RHS
        b[1:M+1] = data[c]  # Interpolation equations RHS
        b[M+1] = 0.0  # Right boundary condition RHS

        # Solve Ac = b
        coefs[c] = torch.linalg.solve(A, b)

    if squeeze_output:
        coefs = coefs.squeeze(0)

    return coefs


def find_coefs_2d(
    data: torch.Tensor,
) -> torch.Tensor:
    """
    Find B-spline coefficients for 2D data with natural boundary conditions.

    Uses separable approach:
    1. Solve 1D problems along X-direction for each Y row
    2. Solve 1D problems along Y-direction for each X column

    Args:
        data: (C, Mx, My) data values to interpolate

    Returns:
        coefs: (C, Mx+2, My+2) B-spline coefficients
    """
    if data.ndim == 2:
        data = data.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    C, Mx, My = data.shape

    # Build system matrices (reused for all channels and rows/columns)
    Ax = _build_1d_system_matrix(Mx, dtype=data.dtype, device=data.device)
    Ay = _build_1d_system_matrix(My, dtype=data.dtype, device=data.device)

    # Step 1: Solve along X-direction for each Y row
    # This expands Mx -> Mx+2 while keeping My unchanged
    coefs_x = torch.zeros(C, Mx + 2, My, dtype=data.dtype, device=data.device)

    for c in range(C):
        for j in range(My):
            # Extract row: data[c, :, j] is (Mx,)
            b = torch.zeros(Mx + 2, dtype=data.dtype, device=data.device)
            b[0] = 0.0
            b[1:Mx+1] = data[c, :, j]
            b[Mx+1] = 0.0

            # Solve and store
            coefs_x[c, :, j] = torch.linalg.solve(Ax, b)

    # Step 2: Solve along Y-direction for each X column
    # This expands My -> My+2, final shape: (C, Mx+2, My+2)
    coefs = torch.zeros(C, Mx + 2, My + 2, dtype=data.dtype, device=data.device)

    for c in range(C):
        for i in range(Mx + 2):
            # Extract column: coefs_x[c, i, :] is (My,)
            b = torch.zeros(My + 2, dtype=data.dtype, device=data.device)
            b[0] = 0.0
            b[1:My+1] = coefs_x[c, i, :]
            b[My+1] = 0.0

            # Solve and store
            coefs[c, i, :] = torch.linalg.solve(Ay, b)

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
    _ = matrix  # Unused, for API compatibility
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


def interpolate_grid_2d(
    data: torch.Tensor,
    u: torch.Tensor,
    matrix: Optional[torch.Tensor] = None,
    monotonicity: Optional[str] = None,
) -> torch.Tensor:
    """
    Interpolate 2D grid data using interpolating B-splines.

    This function signature matches torch-cubic-spline-grids API for compatibility.

    Args:
        data: (C, Mx+2, My+2) padded grid data (B-spline coefficients)
        u: (B, 2) batch of coordinates in [0, 1] x [0, 1]
        matrix: Ignored (for API compatibility - einspline uses fixed basis matrix)
        monotonicity: Optional monotonicity constraint (not yet implemented)

    Returns:
        values: (B, C) interpolated values
    """
    _ = matrix  # Unused, for API compatibility
    if monotonicity is not None:
        raise NotImplementedError("Monotonicity constraints not yet implemented for interpolating B-splines")

    C, Mx_plus_2, My_plus_2 = data.shape
    Mx = Mx_plus_2 - 2
    My = My_plus_2 - 2

    # Extract x and y coordinates
    ux = u[:, 0]  # (B,)
    uy = u[:, 1]  # (B,)

    # Get basis matrix on correct device
    basis_matrix = EINSPLINE_BASIS_MATRIX.to(device=data.device, dtype=data.dtype)

    # Transform coordinates to grid space
    ux_norm = ux * (Mx - 1)
    uy_norm = uy * (My - 1)

    # Find grid cells and local coordinates for X dimension
    ix = torch.floor(ux_norm).long()
    tx = ux_norm - ix.float()

    # Handle X boundary cases
    mask_low_x = ux_norm < 0
    ix = torch.where(mask_low_x, torch.zeros_like(ix), ix)
    tx = torch.where(mask_low_x, ux_norm, tx)

    mask_high_x = ux_norm >= Mx - 2
    ix = torch.where(mask_high_x, torch.full_like(ix, Mx - 2), ix)
    tx = torch.where(mask_high_x, ux_norm - (Mx - 2), tx)

    # Find grid cells and local coordinates for Y dimension
    iy = torch.floor(uy_norm).long()
    ty = uy_norm - iy.float()

    # Handle Y boundary cases
    mask_low_y = uy_norm < 0
    iy = torch.where(mask_low_y, torch.zeros_like(iy), iy)
    ty = torch.where(mask_low_y, uy_norm, ty)

    mask_high_y = uy_norm >= My - 2
    iy = torch.where(mask_high_y, torch.full_like(iy, My - 2), iy)
    ty = torch.where(mask_high_y, uy_norm - (My - 2), ty)

    # Build power vectors
    tx_powers = torch.stack([tx**3, tx**2, tx, torch.ones_like(tx)], dim=1)  # (B, 4)
    ty_powers = torch.stack([ty**3, ty**2, ty, torch.ones_like(ty)], dim=1)  # (B, 4)

    # Compute basis functions
    basis_x = torch.matmul(basis_matrix, tx_powers.T)  # (4, B)
    basis_y = torch.matmul(basis_matrix, ty_powers.T)  # (4, B)

    # Evaluate for each channel
    B = u.shape[0]
    values = torch.zeros(B, C, dtype=data.dtype, device=data.device)

    for c in range(C):
        # Extract 4x4 control point grid for each query point
        # We need to stack control points at positions:
        # [ix+i, iy+j] for i,j in 0..3

        # For each batch element, we need a (4, 4) grid of control points
        # Result will be (B, 4, 4)
        control_grid = torch.zeros(B, 4, 4, dtype=data.dtype, device=data.device)

        for i in range(4):
            for j in range(4):
                control_grid[:, i, j] = data[c, ix + i, iy + j]

        # Apply separable 2D cubic interpolation
        # First apply basis in X: control_grid (B, 4, 4) @ basis_x (4, B)
        # We want: for each B, multiply (4, 4) by (4,) to get (4,)
        # Then multiply result by basis_y

        # Compute intermediate: apply basis_x along first dimension of control_grid
        # control_grid: (B, 4, 4)
        # basis_x: (4, B) -> we need (B, 4) for batch matmul
        # Result after X interpolation: (B, 4)
        intermediate = torch.zeros(B, 4, dtype=data.dtype, device=data.device)
        for i in range(4):
            # control_grid[:, :, i]: (B, 4) - controls in X direction at y offset i
            # basis_x: (4, B)
            # We want: for each b, sum over 4 control points weighted by basis_x[:, b]
            intermediate[:, i] = torch.sum(control_grid[:, :, i] * basis_x.T, dim=1)

        # Apply basis in Y: intermediate (B, 4) @ basis_y
        # For each b, sum over 4 intermediate values weighted by basis_y[:, b]
        values[:, c] = torch.sum(intermediate * basis_y.T, dim=1)

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


class InterpolatingBSpline2d(nn.Module):
    """
    2D Interpolating B-spline grid with API compliance to torch-cubic-spline-grids.

    This implements cubic B-splines that interpolate (pass through) the data points,
    matching einspline's behavior with natural boundary conditions.
    The coefficients are computed by solving separable linear systems.

    Args:
        resolution: Number of data points per dimension (tuple of 2 ints)
        n_channels: Number of channels
        minibatch_size: Maximum batch size for evaluation
    """

    ndim: int = 2
    _interpolation_function: Callable = staticmethod(interpolate_grid_2d)
    _interpolation_matrix: torch.Tensor = EINSPLINE_BASIS_MATRIX
    _data: nn.Parameter
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[str] = None,
    ):
        super().__init__()

        if resolution is None:
            resolution = (2, 2)

        # Initialize data as learnable parameters
        grid_shape = (n_channels, *resolution)
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
        return find_coefs_2d(self._data)

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
        """Convert input coordinates to batched format (B, 2)."""
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim

        if self._input_is_coordinate_like is False:
            raise ValueError(
                f'For 2D grids, coordinates must have shape (..., 2), got {u.shape}'
            )

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
        return interpolated

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline at given coordinates.

        Args:
            u: Coordinates in [0, 1] x [0, 1] to evaluate. Must have shape:
               - (N, 2) for N points
               - (..., 2) for arbitrary batch shapes

        Returns:
            values: Interpolated values with shape:
                    - (N, C) for C channels
                    - (..., C) for arbitrary batch shapes
        """
        u = self._coerce_to_batched_coordinates(u)  # (B, 2)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch_iterator(u, n=self._minibatch_size)
        ]  # List[Tensor[(B, C)]]
        interpolated = torch.cat(interpolated, dim=0)  # (B, C)

        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> "InterpolatingBSpline2d":
        """
        Create spline from existing data (API-compliant factory method).

        Args:
            data: (Mx, My) or (C, Mx, My) data values

        Returns:
            Initialized InterpolatingBSpline2d instance
        """
        grid = cls()
        grid.data = data
        return grid
