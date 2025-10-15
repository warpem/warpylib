"""
CubicGrid - Cubic B-spline grid interpolation

Replicates the functionality of WarpLib's CubicGrid.cs using torch-cubic-spline-grids.
"""

from enum import IntFlag
from typing import Optional, Tuple, Union
import torch
from torch_cubic_spline_grids import (
    CubicCatmullRomGrid4d
)
from lxml import etree
from warpylib.interpolating_bspline import (
    find_coefs_1d, find_coefs_2d, find_coefs_3d,
    interpolate_grid_1d, interpolate_grid_2d, interpolate_grid_3d,
    EINSPLINE_BASIS_MATRIX
)


class InterpolatingBSplineOperator1D:
    """Stateless 1D interpolating B-spline operator that preserves gradients."""

    def __call__(self, data: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate data at given coordinates.

        Args:
            data: (C, M) data tensor (gradients flow through this)
            coords: (B, 1) coordinates in [0, 1]

        Returns:
            values: (B, C) interpolated values
        """
        # Ensure data has channel dimension
        if data.ndim == 1:
            data = data.unsqueeze(0)

        # Compute coefficients (part of computational graph)
        coefs = find_coefs_1d(data)

        # Interpolate
        return interpolate_grid_1d(coefs, coords, matrix=EINSPLINE_BASIS_MATRIX)


class InterpolatingBSplineOperator2D:
    """Stateless 2D interpolating B-spline operator that preserves gradients."""

    def __call__(self, data: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate data at given coordinates.

        Args:
            data: (C, Mx, My) data tensor (gradients flow through this)
            coords: (B, 2) coordinates in [0, 1] x [0, 1]

        Returns:
            values: (B, C) interpolated values
        """
        # Ensure data has channel dimension
        if data.ndim == 2:
            data = data.unsqueeze(0)

        # Compute coefficients (part of computational graph)
        coefs = find_coefs_2d(data)

        # Interpolate
        return interpolate_grid_2d(coefs, coords, matrix=EINSPLINE_BASIS_MATRIX)


class InterpolatingBSplineOperator3D:
    """Stateless 3D interpolating B-spline operator that preserves gradients."""

    def __call__(self, data: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Interpolate data at given coordinates.

        Args:
            data: (C, Mx, My, Mz) data tensor (gradients flow through this)
            coords: (B, 3) coordinates in [0, 1]^3

        Returns:
            values: (B, C) interpolated values
        """
        # Ensure data has channel dimension
        if data.ndim == 3:
            data = data.unsqueeze(0)

        # Compute coefficients (part of computational graph)
        coefs = find_coefs_3d(data)

        # Interpolate
        return interpolate_grid_3d(coefs, coords, matrix=EINSPLINE_BASIS_MATRIX)


class Dimension(IntFlag):
    """Dimension enum matching C# implementation"""
    X = 0
    Y = 1
    Z = 2
    W = 3  # 4th dimension


class DimensionSets(IntFlag):
    """Dimension sets enum matching C# implementation"""
    NONE = 0
    X = 1 << 0
    Y = 1 << 1
    Z = 1 << 2
    W = 1 << 3
    XY = 1 << 4
    XZ = 1 << 5
    YZ = 1 << 6
    XYZ = 1 << 7
    XYZW = 1 << 8  # 4D case


class CubicGrid:
    """
    Cubic B-spline grid interpolation matching WarpLib's CubicGrid.cs API.

    This implementation uses torch-cubic-spline-grids internally but handles all
    necessary coordinate and data transformations to match the einspline behavior.

    Key coordinate conventions:
    - C#/Einspline 3D: dims=(X,Y,Z), coords=(x,y,z), data layout [(z*Y+y)*X+x]
    - Torch 3D: dims=(Z,Y,X), coords=(z,y,x), data layout [z][y][x]
    - C#/Einspline 4D: dims=(X,Y,Z,W), coords=(x,y,z,w), data layout [((w*Z+z)*Y+y)*X+x]
    - Torch 4D: dims=(W,Z,Y,X), coords=(w,z,y,x), data layout [w][z][y][x]

    This class handles the transformations automatically.
    """

    def __init__(
        self,
        dimensions: Union[Tuple[int, int, int], Tuple[int, int, int, int]],
        values: Optional[torch.Tensor] = None,
        margins: Optional[Union[Tuple[float, float, float], Tuple[float, float, float, float]]] = None,
        centered_spacing: bool = False,
        gradient_direction: Optional[Dimension] = None,
        value_min: float = 0.0,
        value_max: float = 1.0,
    ):
        """
        Initialize CubicGrid.

        Args:
            dimensions: (X, Y, Z) or (X, Y, Z, W) grid dimensions
            values: Flat torch tensor of values in einspline layout [(z*Y+y)*X+x] or [((w*Z+z)*Y+y)*X+x]
                   If tensor has requires_grad=True, gradients will flow through interpolation.
            margins: (X, Y, Z) or (X, Y, Z, W) margins for grid boundaries
            centered_spacing: If True, automatically compute margins for centered spacing
            gradient_direction: Direction for gradient initialization (requires value_min/max)
            value_min: Minimum value for gradient initialization
            value_max: Maximum value for gradient initialization
        """
        self.dimensions = tuple(dimensions)
        self.dimension_set = self._get_dimensions(self.dimensions)

        # Set default margins if not provided
        if margins is None:
            margins = tuple(0.0 for _ in dimensions)

        # Handle centered spacing margins
        if centered_spacing:
            margins_list = list(margins)
            for i, dim in enumerate(self.dimensions):
                if dim > 1:
                    margins_list[i] = (1.0 / dim) / 2.0
                else:
                    margins_list[i] = 0.0
            margins = tuple(margins_list)

        self.margins = margins

        # Initialize values
        if values is not None:
            # Keep gradients - use .to() which preserves the computation graph
            self.values = values.to(dtype=torch.float32)
        elif gradient_direction is not None:
            self.values = self._create_gradient_values(
                gradient_direction, value_min, value_max
            )
        else:
            total_size = int(torch.tensor(self.dimensions).prod().item())
            self.values = torch.zeros(total_size, dtype=torch.float32)

        # Create torch spline grid operator and data view
        self._grid_operator, self._grid_data = self._create_torch_grid()

    @property
    def flat_values(self) -> torch.Tensor:
        """Get flat values tensor (einspline layout)"""
        return self.values

    def _get_dimensions(self, dims: Union[Tuple[int, int, int], Tuple[int, int, int, int]]) -> DimensionSets:
        """Determine which dimensions are active (>1)"""
        if len(dims) == 4:
            x, y, z, w = dims
            if x > 1 and y > 1 and z > 1 and w > 1:
                return DimensionSets.XYZW
            # For now, we only support full 4D or treat as 3D
            # Fall through to 3D handling
            dims = dims[:3]

        x, y, z = dims[:3]

        if x > 1 and y > 1 and z > 1:
            return DimensionSets.XYZ
        if x > 1 and y > 1:
            return DimensionSets.XY
        if x > 1 and z > 1:
            return DimensionSets.XZ
        if y > 1 and z > 1:
            return DimensionSets.YZ
        if x > 1:
            return DimensionSets.X
        if y > 1:
            return DimensionSets.Y
        if z > 1:
            return DimensionSets.Z

        return DimensionSets.NONE

    def _create_gradient_values(
        self, gradient_direction: Dimension, value_min: float, value_max: float
    ) -> torch.Tensor:
        """Create values with a gradient in specified direction"""
        total_size = int(torch.tensor(self.dimensions).prod().item())
        values = torch.zeros(total_size, dtype=torch.float32)

        if len(self.dimensions) == 4:
            x_dim, y_dim, z_dim, w_dim = self.dimensions
            step = value_max - value_min
            if gradient_direction == Dimension.X:
                step /= max(1, x_dim - 1)
            elif gradient_direction == Dimension.Y:
                step /= max(1, y_dim - 1)
            elif gradient_direction == Dimension.Z:
                step /= max(1, z_dim - 1)
            elif gradient_direction == Dimension.W:
                step /= max(1, w_dim - 1)

            for w in range(w_dim):
                for z in range(z_dim):
                    for y in range(y_dim):
                        for x in range(x_dim):
                            value = value_min
                            if gradient_direction == Dimension.X:
                                value += x * step
                            elif gradient_direction == Dimension.Y:
                                value += y * step
                            elif gradient_direction == Dimension.Z:
                                value += z * step
                            elif gradient_direction == Dimension.W:
                                value += w * step

                            # Einspline layout: [((w*Z+z)*Y+y)*X+x]
                            idx = ((w * z_dim + z) * y_dim + y) * x_dim + x
                            values[idx] = value
        else:
            x_dim, y_dim, z_dim = self.dimensions
            step = value_max - value_min
            if gradient_direction == Dimension.X:
                step /= max(1, x_dim - 1)
            elif gradient_direction == Dimension.Y:
                step /= max(1, y_dim - 1)
            elif gradient_direction == Dimension.Z:
                step /= max(1, z_dim - 1)

            for z in range(z_dim):
                for y in range(y_dim):
                    for x in range(x_dim):
                        value = value_min
                        if gradient_direction == Dimension.X:
                            value += x * step
                        elif gradient_direction == Dimension.Y:
                            value += y * step
                        elif gradient_direction == Dimension.Z:
                            value += z * step

                        # Einspline layout: [(z*Y+y)*X+x]
                        idx = (z * y_dim + y) * x_dim + x
                        values[idx] = value

        return values

    def _create_torch_grid(self) -> tuple:
        """Create torch spline grid operator and reshaped data view.

        Returns a tuple of (operator, data_view) where:
        - operator: stateless interpolator that takes (data, coords) -> values
        - data_view: reshaped view of self.values for the operator (preserves gradients)
        """
        if self.dimension_set == DimensionSets.NONE:
            return (None, None)

        # 4D case - use CubicCatmullRomGrid4d (TODO: make this an operator too)
        if self.dimension_set == DimensionSets.XYZW:
            x_dim, y_dim, z_dim, w_dim = self.dimensions
            data_torch = self.values.reshape((w_dim, z_dim, y_dim, x_dim))
            return (CubicCatmullRomGrid4d.from_grid_data(data_torch), None)

        x_dim, y_dim, z_dim = self.dimensions[:3]

        # 3D case
        if self.dimension_set == DimensionSets.XYZ:
            data_torch = self.values.reshape((z_dim, y_dim, x_dim))
            return (InterpolatingBSplineOperator3D(), data_torch)

        # 2D cases
        elif self.dimension_set == DimensionSets.XY:
            data_torch = self.values.reshape((y_dim, x_dim))
            return (InterpolatingBSplineOperator2D(), data_torch)

        elif self.dimension_set == DimensionSets.XZ:
            data_torch = self.values.reshape((z_dim, x_dim))
            return (InterpolatingBSplineOperator2D(), data_torch)

        elif self.dimension_set == DimensionSets.YZ:
            data_torch = self.values.reshape((z_dim, y_dim))
            return (InterpolatingBSplineOperator2D(), data_torch)

        # 1D cases
        elif self.dimension_set in (DimensionSets.X, DimensionSets.Y, DimensionSets.Z):
            if self.dimension_set == DimensionSets.X:
                data_torch = self.values.reshape(x_dim)
            elif self.dimension_set == DimensionSets.Y:
                data_torch = self.values.reshape(y_dim)
            else:  # Z
                data_torch = self.values.reshape(z_dim)
            return (InterpolatingBSplineOperator1D(), data_torch)

        return (None, None)

    def _transform_coords_to_torch(
        self, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform einspline coordinates to torch coordinates.

        Args:
            coords: Nx3 or Nx4 tensor of (x, y, z) or (x, y, z, w) coordinates in [0, 1] range

        Returns:
            torch_coords: transformed coordinates for torch spline grids
        """
        coords_transformed = coords.clone()

        # Apply margin transformation if needed
        if any(m > 0 for m in self.margins):
            # Transform from [0,1] to [margin, 1-margin] for active dimensions
            for i, (dim, margin) in enumerate(zip(self.dimensions, self.margins)):
                if dim > 1 and margin > 0:
                    coords_transformed[:, i] = (
                        coords_transformed[:, i] * (1.0 - 2 * margin) + margin
                    )

        # Extract and swap coordinates based on dimension set
        if self.dimension_set == DimensionSets.XYZW:
            # Swap (x,y,z,w) -> (w,z,y,x)
            torch_coords = coords_transformed[:, [3, 2, 1, 0]]

        elif self.dimension_set == DimensionSets.XYZ:
            # Swap (x,y,z) -> (z,y,x)
            torch_coords = coords_transformed[:, [2, 1, 0]]

        elif self.dimension_set == DimensionSets.XY:
            # Use (y, x) - swap order
            torch_coords = coords_transformed[:, [1, 0]]

        elif self.dimension_set == DimensionSets.XZ:
            # Use (z, x) - swap order
            torch_coords = coords_transformed[:, [2, 0]]

        elif self.dimension_set == DimensionSets.YZ:
            # Use (z, y) - swap order
            torch_coords = coords_transformed[:, [2, 1]]

        elif self.dimension_set == DimensionSets.X:
            torch_coords = coords_transformed[:, [0]]

        elif self.dimension_set == DimensionSets.Y:
            torch_coords = coords_transformed[:, [1]]

        elif self.dimension_set == DimensionSets.Z:
            torch_coords = coords_transformed[:, [2]]

        else:  # NONE
            torch_coords = coords_transformed

        return torch_coords.to(dtype=torch.float32)

    def get_interpolated(
        self, coords: Union[torch.Tensor, Tuple[float, ...]]
    ) -> torch.Tensor:
        """
        Get interpolated values at specified coordinates.

        Args:
            coords: Either a single (x, y, z) or (x, y, z, w) tuple, or Nx3/Nx4 tensor of coordinates
                   Coordinates should be in [0, 1] range

        Returns:
            Tensor of interpolated values. Gradients will flow if input has requires_grad=True.
        """
        # Handle single coordinate
        if isinstance(coords, (tuple, list)):
            coords = torch.tensor([coords], dtype=torch.float32)
        elif not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        else:
            coords = coords.to(dtype=torch.float32)

        # Ensure 2D tensor
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)

        # Handle degenerate case
        if self._grid_operator is None or self.dimension_set == DimensionSets.NONE:
            return torch.full((len(coords),), self.values[0].item(), dtype=torch.float32)

        # Transform coordinates
        torch_coords = self._transform_coords_to_torch(coords)

        # Evaluate spline (gradients will flow through data -> operator -> result)
        if self._grid_data is not None:
            # Use our operator-based approach (preserves gradients)
            result = self._grid_operator(self._grid_data, torch_coords).squeeze(-1)
        else:
            # 4D case still using old module (TODO: fix this)
            result = self._grid_operator(torch_coords).squeeze(-1)

        return result

    def get_interpolated_grid(
        self, value_grid: Tuple[int, int, int], border: Tuple[float, float, float]
    ) -> torch.Tensor:
        """
        Get interpolated values on a regular grid.

        Args:
            value_grid: (X, Y, Z) dimensions of output grid
            border: (X, Y, Z) border offsets

        Returns:
            Flat tensor of interpolated values in einspline layout
        """
        vx, vy, vz = value_grid
        bx, by, bz = border

        # Compute grid coordinates
        step_x = (1.0 - bx * 2) / max(1, vx - 1)
        offset_x = bx

        step_y = (1.0 - by * 2) / max(1, vy - 1)
        offset_y = by

        step_z = (1.0 - bz * 2) / max(vz - 1, 1)
        offset_z = 0.5 if vz == 1 else bz

        # Generate coordinates in einspline order
        total_size = int(torch.tensor(value_grid).prod().item())
        coords = torch.zeros((total_size, 3), dtype=torch.float32)
        idx = 0
        for z in range(vz):
            for y in range(vy):
                for x in range(vx):
                    coords[idx] = torch.tensor([
                        x * step_x + offset_x,
                        y * step_y + offset_y,
                        z * step_z + offset_z,
                    ])
                    idx += 1

        return self.get_interpolated(coords)

    def resize(self, new_size: Tuple[int, int, int]) -> "CubicGrid":
        """
        Resize grid by resampling at new resolution.

        Args:
            new_size: (X, Y, Z) new dimensions

        Returns:
            New CubicGrid with resampled values
        """
        nx, ny, nz = new_size

        step_x = 1.0 / max(1, nx - 1)
        step_y = 1.0 / max(1, ny - 1)
        step_z = 1.0 / max(1, nz - 1)

        # Generate coordinates
        total_size = int(torch.tensor(new_size).prod().item())
        result = torch.zeros(total_size, dtype=torch.float32)
        idx = 0
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    coord = torch.tensor([[x * step_x, y * step_y, z * step_z]])
                    result[idx] = self.get_interpolated(coord)[0]
                    idx += 1

        # Preserve margins if any were set
        has_margins = any(m > 0 for m in self.margins)
        return CubicGrid(new_size, result, centered_spacing=has_margins)

    def collapse_xy(self) -> "CubicGrid":
        """
        Collapse X and Y dimensions by averaging, keeping only Z.

        Returns:
            New 1D CubicGrid along Z axis
        """
        x_dim, y_dim, z_dim = self.dimensions
        collapsed = torch.zeros(z_dim, dtype=torch.float32)

        for z in range(z_dim):
            mean = 0.0
            for y in range(y_dim):
                for x in range(x_dim):
                    idx = (z * y_dim + y) * x_dim + x
                    mean += self.values[idx].item()
            mean /= x_dim * y_dim
            collapsed[z] = mean

        has_margins = any(m > 0 for m in self.margins)
        return CubicGrid((1, 1, z_dim), collapsed, centered_spacing=has_margins)

    def collapse_z(self) -> "CubicGrid":
        """
        Collapse Z dimension by averaging, keeping X and Y.

        Returns:
            New 2D CubicGrid in XY plane
        """
        x_dim, y_dim, z_dim = self.dimensions
        collapsed = torch.zeros(x_dim * y_dim, dtype=torch.float32)

        for y in range(y_dim):
            for x in range(x_dim):
                mean = 0.0
                for z in range(z_dim):
                    idx = (z * y_dim + y) * x_dim + x
                    mean += self.values[idx].item()
                mean /= z_dim
                collapsed[y * x_dim + x] = mean

        has_margins = any(m > 0 for m in self.margins)
        return CubicGrid((x_dim, y_dim, 1), collapsed, centered_spacing=has_margins)

    def get_slice_xy(self, z: int) -> torch.Tensor:
        """Get XY slice at given Z index"""
        x_dim, y_dim, _ = self.dimensions
        result = torch.zeros(x_dim * y_dim, dtype=torch.float32)

        for y in range(y_dim):
            for x in range(x_dim):
                idx = (z * y_dim + y) * x_dim + x
                result[y * x_dim + x] = self.values[idx]

        return result

    def get_slice_xz(self, y: int) -> torch.Tensor:
        """Get XZ slice at given Y index"""
        x_dim, y_dim, z_dim = self.dimensions
        result = torch.zeros(x_dim * z_dim, dtype=torch.float32)

        for z in range(z_dim):
            for x in range(x_dim):
                idx = (z * y_dim + y) * x_dim + x
                result[z * x_dim + x] = self.values[idx]

        return result

    def get_slice_yz(self, x: int) -> torch.Tensor:
        """Get YZ slice at given X index"""
        x_dim, y_dim, z_dim = self.dimensions
        result = torch.zeros(y_dim * z_dim, dtype=torch.float32)

        for z in range(z_dim):
            for y in range(y_dim):
                idx = (z * y_dim + y) * x_dim + x
                result[z * y_dim + y] = self.values[idx]

        return result

    def save_to_xml(self, parent_element: etree._Element) -> None:
        """
        Save grid to XML element.

        Args:
            parent_element: XML element to save grid data to
        """
        if len(self.dimensions) == 4:
            # 4D grid
            x_dim, y_dim, z_dim, w_dim = self.dimensions
            parent_element.set("Width", str(x_dim))
            parent_element.set("Height", str(y_dim))
            parent_element.set("Depth", str(z_dim))
            parent_element.set("Duration", str(w_dim))

            # NOTE: 4D grids do NOT save margin attributes in C# implementation

            # Save all nodes
            for w in range(w_dim):
                for z in range(z_dim):
                    for y in range(y_dim):
                        for x in range(x_dim):
                            node = etree.SubElement(parent_element, "Node")
                            node.set("X", str(x))
                            node.set("Y", str(y))
                            node.set("Z", str(z))
                            node.set("W", str(w))
                            idx = ((w * z_dim + z) * y_dim + y) * x_dim + x
                            node.set("Value", f"{self.values[idx].item():.9g}")
        else:
            # 3D grid (or lower)
            x_dim, y_dim, z_dim = self.dimensions[:3]
            parent_element.set("Width", str(x_dim))
            parent_element.set("Height", str(y_dim))
            parent_element.set("Depth", str(z_dim))

            # Save margins
            parent_element.set("MarginX", f"{self.margins[0]:.9g}")
            parent_element.set("MarginY", f"{self.margins[1]:.9g}")
            parent_element.set("MarginZ", f"{self.margins[2]:.9g}")

            # Save all nodes
            for z in range(z_dim):
                for y in range(y_dim):
                    for x in range(x_dim):
                        node = etree.SubElement(parent_element, "Node")
                        node.set("X", str(x))
                        node.set("Y", str(y))
                        node.set("Z", str(z))
                        idx = (z * y_dim + y) * x_dim + x
                        node.set("Value", f"{self.values[idx].item():.9g}")

    @staticmethod
    def load_from_xml(element: etree._Element) -> "CubicGrid":
        """
        Load grid from XML element.

        Args:
            element: XML element containing grid data

        Returns:
            Loaded CubicGrid instance
        """
        # Load dimensions
        width = int(element.get("Width", "1"))
        height = int(element.get("Height", "1"))
        depth = int(element.get("Depth", "1"))
        duration = element.get("Duration")

        # Check if 4D
        if duration is not None:
            w_dim = int(duration)
            dimensions = (width, height, depth, w_dim)

            # Load margins
            margin_x = float(element.get("MarginX", "0"))
            margin_y = float(element.get("MarginY", "0"))
            margin_z = float(element.get("MarginZ", "0"))
            margin_w = float(element.get("MarginW", "0"))
            margins = (margin_x, margin_y, margin_z, margin_w)

            # Initialize values tensor
            total_size = int(torch.tensor(dimensions).prod().item())
            values = torch.zeros(total_size, dtype=torch.float32)

            # Load node values
            for node in element.findall("Node"):
                x = int(node.get("X", "0"))
                y = int(node.get("Y", "0"))
                z = int(node.get("Z", "0"))
                w = int(node.get("W", "0"))
                value = float(node.get("Value", "0"))

                idx = ((w * depth + z) * height + y) * width + x
                values[idx] = value

            return CubicGrid(dimensions, values, margins)
        else:
            # 3D grid
            dimensions = (width, height, depth)

            # Load margins
            margin_x = float(element.get("MarginX", "0"))
            margin_y = float(element.get("MarginY", "0"))
            margin_z = float(element.get("MarginZ", "0"))
            margins = (margin_x, margin_y, margin_z)

            # Initialize values tensor
            total_size = int(torch.tensor(dimensions).prod().item())
            values = torch.zeros(total_size, dtype=torch.float32)

            # Load node values
            for node in element.findall("Node"):
                x = int(node.get("X", "0"))
                y = int(node.get("Y", "0"))
                z = int(node.get("Z", "0"))
                value = float(node.get("Value", "0"))

                idx = (z * height + y) * width + x
                values[idx] = value

            return CubicGrid(dimensions, values, margins)
