"""
CTF (Contrast Transfer Function) data structure.

This module contains the CTF class that replicates the basic data structure
from WarpLib's CTF.cs, including properties for microscope parameters,
defocus, aberrations, and XML serialization.
"""

from typing import Optional
import torch
from lxml import etree


class CTF:
    """
    Contrast Transfer Function parameters.

    This class stores microscope and optical parameters used for CTF correction
    in cryo-EM/cryo-ET data processing. All properties use the same units as
    the C# version for compatibility.
    """

    def __init__(self):
        """Initialize CTF with default values matching CTF.cs defaults."""

        # Basic microscope parameters
        self.pixel_size: float = 1.0  # Pixel size in Angstrom
        self.pixel_size_delta_percent: float = 0.0  # Pixel size anisotropy delta in percent
        self.pixel_size_angle: float = 0.0  # Pixel size anisotropy angle in degrees

        # Distortion (simplified - would need Matrix2 implementation for full compatibility)
        # For now, store as 4 values: m00, m01, m10, m11
        self.distortion: torch.Tensor = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)

        # Aberrations
        self.cs: float = 2.7  # Spherical aberration in mm
        self.cc: float = 2.7  # Chromatic aberration in mm
        self.voltage: float = 300.0  # Voltage in kV

        # Defocus parameters
        self.defocus: float = 1.0  # Defocus in um, underfocus (first peak positive) is positive
        self.defocus_delta: float = 0.0  # Astigmatism delta defocus in um
        self.defocus_angle: float = 0.0  # Astigmatism angle in degrees

        # Amplitude and phase
        self.amplitude: float = 0.07  # Amplitude contrast
        self.phase_shift: float = 0.0  # Phase shift in Pi

        # B-factor parameters
        self.bfactor: float = 0.0  # B factor in Angstrom^2
        self.bfactor_delta: float = 0.0  # B factor anisotropy in Angstrom^2
        self.bfactor_angle: float = 0.0  # B factor anisotropy X axis angle in degrees

        # Scale
        self.scale: float = 1.0  # Scale, i.e. CTF oscillates within [-Scale; +Scale]

        # Beam tilt (simplified - would need float2/float3 types for full compatibility)
        self.beam_tilt: torch.Tensor = torch.zeros(2, dtype=torch.float32)  # Beam tilt in millirad
        self.beam_tilt2: torch.Tensor = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)  # Higher-order beam tilt

        # Zernike coefficients for aberrations
        self.zernike_coeffs_odd: torch.Tensor = torch.zeros(0, dtype=torch.float32)
        self.zernike_coeffs_even: torch.Tensor = torch.zeros(0, dtype=torch.float32)

        # Thickness
        self.thickness: float = 0.0  # Thickness in nm

    def get_copy(self) -> "CTF":
        """
        Create a deep copy of this CTF object.

        Returns:
            A new CTF instance with copied values
        """
        copy = CTF()

        # Copy all scalar values
        copy.pixel_size = self.pixel_size
        copy.pixel_size_delta_percent = self.pixel_size_delta_percent
        copy.pixel_size_angle = self.pixel_size_angle
        copy.cs = self.cs
        copy.cc = self.cc
        copy.voltage = self.voltage
        copy.defocus = self.defocus
        copy.defocus_delta = self.defocus_delta
        copy.defocus_angle = self.defocus_angle
        copy.amplitude = self.amplitude
        copy.phase_shift = self.phase_shift
        copy.bfactor = self.bfactor
        copy.bfactor_delta = self.bfactor_delta
        copy.bfactor_angle = self.bfactor_angle
        copy.scale = self.scale
        copy.thickness = self.thickness

        # Copy tensors
        copy.distortion = self.distortion.clone()
        copy.beam_tilt = self.beam_tilt.clone()
        copy.beam_tilt2 = self.beam_tilt2.clone()
        copy.zernike_coeffs_odd = self.zernike_coeffs_odd.clone()
        copy.zernike_coeffs_even = self.zernike_coeffs_even.clone()

        return copy

    @classmethod
    def load_from_xml(cls, element: etree._Element) -> "CTF":
        """
        Load CTF parameters from an XML element.

        Args:
            element: XML element containing CTF data (typically <CTF> node)

        Returns:
            A new CTF instance with loaded values
        """
        ctf = cls()

        # Helper function to load parameter from <Param> child elements
        def get_param(name: str, default: float) -> float:
            param = element.find(f"Param[@Name='{name}']")
            if param is not None:
                value = param.get("Value")
                if value is not None:
                    return float(value)
            return default

        def get_param_str(name: str, default: str = "") -> str:
            param = element.find(f"Param[@Name='{name}']")
            if param is not None:
                value = param.get("Value")
                if value is not None:
                    return value
            return default

        # Load all properties from Param child elements
        ctf.pixel_size = get_param("PixelSize", 1.0)
        ctf.pixel_size_delta_percent = get_param("PixelSizeDeltaPercent", 0.0)
        ctf.pixel_size_angle = get_param("PixelSizeAngle", 0.0)

        # Distortion matrix
        distortion_str = get_param_str("Distortion")
        if distortion_str:
            values = distortion_str.strip("()").split(",")
            if len(values) == 4:
                ctf.distortion = torch.tensor([float(v.strip()) for v in values], dtype=torch.float32)

        ctf.cs = get_param("Cs", 2.7)
        ctf.cc = get_param("Cc", 2.7)
        ctf.voltage = get_param("Voltage", 300.0)
        ctf.defocus = get_param("Defocus", 1.0)
        ctf.defocus_delta = get_param("DefocusDelta", 0.0)
        ctf.defocus_angle = get_param("DefocusAngle", 0.0)
        ctf.amplitude = get_param("Amplitude", 0.07)
        ctf.phase_shift = get_param("PhaseShift", 0.0)
        ctf.bfactor = get_param("Bfactor", 0.0)
        ctf.bfactor_delta = get_param("BfactorDelta", 0.0)
        ctf.bfactor_angle = get_param("BfactorAngle", 0.0)
        ctf.scale = get_param("Scale", 1.0)
        ctf.thickness = get_param("Thickness", 0.0)

        # Beam tilt
        beam_tilt_str = get_param_str("BeamTilt")
        if beam_tilt_str:
            values = beam_tilt_str.strip("()").split(",")
            if len(values) == 2:
                ctf.beam_tilt = torch.tensor([float(v.strip()) for v in values], dtype=torch.float32)

        beam_tilt2_str = get_param_str("BeamTilt2")
        if beam_tilt2_str:
            values = beam_tilt2_str.strip("()").split(",")
            if len(values) == 3:
                ctf.beam_tilt2 = torch.tensor([float(v.strip()) for v in values], dtype=torch.float32)

        # Zernike coefficients (stored as Param elements with semicolon-separated values)
        zernike_odd_str = get_param_str("ZernikeCoeffsOdd")
        if zernike_odd_str:
            values = [float(x.strip()) for x in zernike_odd_str.split(";") if x.strip()]
            if values:
                ctf.zernike_coeffs_odd = torch.tensor(values, dtype=torch.float32)

        zernike_even_str = get_param_str("ZernikeCoeffsEven")
        if zernike_even_str:
            values = [float(x.strip()) for x in zernike_even_str.split(";") if x.strip()]
            if values:
                ctf.zernike_coeffs_even = torch.tensor(values, dtype=torch.float32)

        return ctf

    def save_to_xml(self, element: etree._Element) -> None:
        """
        Save CTF parameters to an XML element using Param child elements.

        Args:
            element: XML element to save into (typically <CTF> node)
        """
        # Helper function to add a Param child element
        def add_param(name: str, value: str):
            param = etree.SubElement(element, "Param")
            param.set("Name", name)
            param.set("Value", value)

        # Save all properties as Param child elements
        add_param("PixelSize", f"{self.pixel_size:.9g}")
        add_param("PixelSizeDeltaPercent", f"{self.pixel_size_delta_percent:.9g}")
        add_param("PixelSizeAngle", f"{self.pixel_size_angle:.9g}")

        # Distortion matrix - save as "m00, m01, m10, m11" (no parentheses)
        distortion_str = f"{self.distortion[0]:.9g}, {self.distortion[1]:.9g}, {self.distortion[2]:.9g}, {self.distortion[3]:.9g}"
        add_param("Distortion", distortion_str)

        add_param("Cs", f"{self.cs:.9g}")
        add_param("Cc", f"{self.cc:.9g}")
        add_param("Voltage", f"{self.voltage:.9g}")
        add_param("Defocus", f"{self.defocus:.9g}")
        add_param("DefocusDelta", f"{self.defocus_delta:.9g}")
        add_param("DefocusAngle", f"{self.defocus_angle:.9g}")
        add_param("Amplitude", f"{self.amplitude:.9g}")
        add_param("PhaseShift", f"{self.phase_shift:.9g}")
        add_param("Bfactor", f"{self.bfactor:.9g}")
        add_param("BfactorDelta", f"{self.bfactor_delta:.9g}")
        add_param("BfactorAngle", f"{self.bfactor_angle:.9g}")
        add_param("Scale", f"{self.scale:.9g}")
        add_param("Thickness", f"{self.thickness:.9g}")

        # Beam tilt - save as "x,y" (no spaces, no parentheses)
        beam_tilt_str = f"{self.beam_tilt[0]:.9g},{self.beam_tilt[1]:.9g}"
        add_param("BeamTilt", beam_tilt_str)

        # Beam tilt2 - save as "x,y,z"
        beam_tilt2_str = f"{self.beam_tilt2[0]:.9g},{self.beam_tilt2[1]:.9g},{self.beam_tilt2[2]:.9g}"
        add_param("BeamTilt2", beam_tilt2_str)

        # Zernike coefficients - save as Param elements with semicolon-separated values
        # Always write them, even if empty (to match C# behavior)
        if len(self.zernike_coeffs_odd) > 0:
            zernike_odd_str = ";".join([f"{x:.9g}" for x in self.zernike_coeffs_odd.tolist()])
        else:
            zernike_odd_str = ""
        add_param("ZernikeCoeffsOdd", zernike_odd_str)

        if len(self.zernike_coeffs_even) > 0:
            zernike_even_str = ";".join([f"{x:.9g}" for x in self.zernike_coeffs_even.tolist()])
        else:
            zernike_even_str = ""
        add_param("ZernikeCoeffsEven", zernike_even_str)

    def _tensorize_params(self, device: Optional[torch.device] = None) -> tuple[dict[str, torch.Tensor], int]:
        """
        Convert all parameters to tensors and find maximum dimensionality.

        Args:
            device: Device to put tensors on

        Returns:
            Tuple of (parameter dict, max_ndim)
        """
        params = {}

        # Convert all relevant parameters to tensors
        def to_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.to(device) if device is not None else value
            else:
                return torch.tensor(value, dtype=torch.float32, device=device)

        params['pixel_size'] = to_tensor(self.pixel_size)
        params['pixel_size_delta_percent'] = to_tensor(self.pixel_size_delta_percent)
        params['pixel_size_angle'] = to_tensor(self.pixel_size_angle)
        params['cs'] = to_tensor(self.cs)
        params['voltage'] = to_tensor(self.voltage)
        params['defocus'] = to_tensor(self.defocus)
        params['defocus_delta'] = to_tensor(self.defocus_delta)
        params['defocus_angle'] = to_tensor(self.defocus_angle)
        params['amplitude'] = to_tensor(self.amplitude)
        params['phase_shift'] = to_tensor(self.phase_shift)
        params['bfactor'] = to_tensor(self.bfactor)
        params['bfactor_delta'] = to_tensor(self.bfactor_delta)
        params['bfactor_angle'] = to_tensor(self.bfactor_angle)
        params['scale'] = to_tensor(self.scale)

        # Find maximum number of dimensions
        max_ndim = max(p.ndim for p in params.values())

        return params, max_ndim

    def _get_ks(self, params: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate CTF constants K1, K2, K3, K4.

        Args:
            params: Tensorized parameters dict

        Returns:
            Tuple of (K1, K2, K3, K4) constants for CTF calculation
        """
        voltage = params['voltage'] * 1e3  # kV to V
        lambda_val = 12.2643247 / torch.sqrt(voltage * (1.0 + voltage * 0.978466e-6))
        cs = params['cs'] * 1e7  # mm to Angstrom
        amplitude = params['amplitude']

        K1 = torch.pi * lambda_val
        K2 = torch.pi * 0.5 * cs * lambda_val ** 3
        K3 = torch.sqrt(1.0 - amplitude ** 2)
        K4 = params['bfactor'] * 0.25

        return (K1, K2, K3, K4)

    def get_1d(
        self,
        width: int,
        amp_squared: bool = False,
        ignore_bfactor: bool = False,
        ignore_scale: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calculate 1D CTF values with support for batched parameters.

        Parameters can be scalars (floats) or tensors of any dimensionality.
        All parameters will be broadcast to the same shape, and the output
        will have that batch shape plus one frequency dimension.

        Args:
            width: Number of frequency samples
            amp_squared: If True, return absolute value (amplitude spectrum)
            ignore_bfactor: If True, don't apply B-factor
            ignore_scale: If True, don't apply scale factor
            device: Device to put the result tensor on

        Returns:
            Tensor of CTF values. If all parameters are scalars, shape is (width,).
            If any parameter is batched, shape is (*batch_shape, width).
        """
        params, max_ndim = self._tensorize_params(device)

        # Broadcast all parameters to common shape
        ny = 0.5 / params['pixel_size'] / width  # Nyquist frequency step
        defocus = -params['defocus'] * 1e4  # um to Angstrom
        amplitude = params['amplitude']
        scale = torch.ones_like(params['scale']) if ignore_scale else params['scale']
        phaseshift = params['phase_shift'] * torch.pi

        K1, K2, K3, K4 = self._get_ks(params)

        # Broadcast all tensors to the same shape
        ny, defocus, amplitude, scale, phaseshift, K1, K2, K3, K4 = torch.broadcast_tensors(
            ny, defocus, amplitude, scale, phaseshift, K1, K2, K3, K4
        )

        # Create frequency array - shape (..., width) where ... matches broadcast shape
        freq = torch.arange(width, dtype=torch.float32, device=device)
        freq = freq * ny[..., None]  # Broadcasting: ny shape (*,) -> freq shape (*, width)

        r2 = freq ** 2
        r4 = r2 ** 2

        # Calculate CTF - broadcasting happens automatically
        argument = K1[..., None] * defocus[..., None] * r2 + K2[..., None] * r4 - phaseshift[..., None]
        retval = amplitude[..., None] * torch.cos(argument) - K3[..., None] * torch.sin(argument)

        if not ignore_bfactor and torch.any(K4 != 0):
            retval = retval * torch.exp(K4[..., None] * r2)

        if amp_squared:
            retval = torch.abs(retval)

        return retval * scale[..., None]

    def get_2d(
        self,
        size: int,
        original_size: Optional[int] = None,
        amp_squared: bool = False,
        ignore_bfactor: bool = False,
        ignore_scale: bool = False,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Calculate 2D CTF in rfft format (half Hermitian) with support for batched parameters.

        Parameters can be scalars (floats) or tensors of any dimensionality.
        All parameters will be broadcast to the same shape, and the output
        will have that batch shape plus two spatial frequency dimensions.

        Args:
            size: Size of the 2D image
            original_size: Original image size (for scaling coordinates), defaults to size
            amp_squared: If True, return absolute value (amplitude spectrum)
            ignore_bfactor: If True, don't apply B-factor
            ignore_scale: If True, don't apply scale factor
            device: Device to put the result tensor on

        Returns:
            Tensor of CTF values in rfft format. If all parameters are scalars,
            shape is (size, size//2 + 1). If any parameter is batched, shape is
            (*batch_shape, size, size//2 + 1).
        """
        if original_size is None:
            original_size = size

        params, max_ndim = self._tensorize_params(device)

        # Get CTF coordinates
        coords_r, coords_angle = self.get_ctf_coords(size, original_size, device=device)

        # Unit conversions matching C# code
        pixelsize = params['pixel_size']
        pixeldelta = params['pixel_size'] * params['pixel_size_delta_percent']
        pixelangle = torch.deg2rad(params['pixel_size_angle'])

        voltage = params['voltage'] * 1e3  # kV to V
        lambda_val = 12.2643247 / torch.sqrt(voltage * (1.0 + voltage * 0.978466e-6))

        defocus = -params['defocus'] * 1e4  # um to Angstrom
        defocusdelta = -params['defocus_delta'] * 1e4 * 0.5  # um to Angstrom, half for formula
        astigmatismangle = torch.deg2rad(params['defocus_angle'])

        cs = params['cs'] * 1e7  # mm to Angstrom
        amplitude = params['amplitude']
        scale = torch.ones_like(params['scale']) if ignore_scale else params['scale']
        phaseshift = params['phase_shift'] * torch.pi

        K1 = torch.pi * lambda_val
        K2 = torch.pi * 0.5 * cs * lambda_val ** 3
        K3 = torch.atan(amplitude / torch.sqrt(1.0 - amplitude ** 2))

        # Broadcast all tensors to the same shape (for B-factor we'll handle separately)
        pixelsize, pixeldelta, pixelangle, defocus, defocusdelta, astigmatismangle, K1, K2, K3, phaseshift, scale = torch.broadcast_tensors(
            pixelsize, pixeldelta, pixelangle, defocus, defocusdelta, astigmatismangle, K1, K2, K3, phaseshift, scale
        )

        # Apply pixel anisotropy to radius
        # Use [..., None, None] to add spatial dimensions after batch dimensions
        r = coords_r / (pixelsize[..., None, None] + pixeldelta[..., None, None] * torch.cos(2.0 * (coords_angle - pixelangle[..., None, None])))
        r2 = r ** 2
        r4 = r2 ** 2

        # Apply astigmatism to defocus
        deltaf = defocus[..., None, None] + defocusdelta[..., None, None] * torch.cos(2.0 * (coords_angle - astigmatismangle[..., None, None]))

        # Calculate CTF - note the different formula compared to 1D!
        # This matches line 186 in CTF.cs: argument = K1 * deltaf * r2 + K2 * r4 - phaseshift - K3;
        # retval = -(float)Math.Sin(argument);
        argument = K1[..., None, None] * deltaf * r2 + K2[..., None, None] * r4 - phaseshift[..., None, None] - K3[..., None, None]
        retval = -torch.sin(argument)

        # Apply B-factor if needed
        if not ignore_bfactor and torch.any(params['bfactor'] != 0):
            bfactor_aniso = params['bfactor'] * 0.25
            if torch.any(params['bfactor_delta'] != 0):
                bfactor_angle_rad = torch.deg2rad(params['bfactor_angle'])
                # Broadcast bfactor parameters
                bfactor_aniso, bfactor_angle_rad = torch.broadcast_tensors(bfactor_aniso, bfactor_angle_rad)
                bfactor_aniso = bfactor_aniso + (params['bfactor_delta'] * 0.25) * torch.cos(2.0 * (coords_angle - bfactor_angle_rad[..., None, None]))
            retval = retval * torch.exp(bfactor_aniso[..., None, None] * r2)

        if amp_squared:
            retval = torch.abs(retval)

        if not ignore_scale:
            retval = scale[..., None, None] * retval

        return retval

    @staticmethod
    def get_ctf_coords(
        size: int,
        original_size: Optional[int] = None,
        pixel_size: float = 1.0,
        pixel_size_delta: float = 0.0,
        pixel_size_angle: float = 0.0,
        device: Optional[torch.device] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate CTF coordinate grids in rfft format.

        This matches the C# GetCTFCoords static method, generating polar coordinates
        (r, angle) for each point in the rfft half-Hermitian format.

        Args:
            size: Size of the 2D image
            original_size: Original image size for scaling, defaults to size
            pixel_size: Pixel size scaling factor
            pixel_size_delta: Pixel size anisotropy delta
            pixel_size_angle: Pixel size anisotropy angle in degrees
            device: Device to put tensors on

        Returns:
            Tuple of (r, angle) tensors, both of shape (size, size//2 + 1)
            r: radial frequency in 1/Angstrom
            angle: angle in radians
        """
        if original_size is None:
            original_size = size

        # Create coordinate grids for rfft format
        # y coordinates: [0, 1, ..., size//2, -(size//2-1), ..., -1]
        # x coordinates: [0, 1, ..., size//2]
        y_coords = torch.fft.fftfreq(size, d=1.0, device=device) * size
        x_coords = torch.fft.rfftfreq(size, d=1.0, device=device) * size

        # Create meshgrid
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Scale to frequency units
        xs = xx / original_size
        ys = yy / original_size

        # Calculate radius and angle
        r = torch.sqrt(xs ** 2 + ys ** 2)
        angle = torch.atan2(yy, xx)

        # Apply pixel anisotropy if needed
        if pixel_size != 1.0 or pixel_size_delta != 0.0:
            pixel_angle_rad = torch.deg2rad(torch.tensor(pixel_size_angle, device=device))
            r = r / (pixel_size + pixel_size_delta * torch.cos(2.0 * (angle - pixel_angle_rad)))

        return r, angle

    def __repr__(self) -> str:
        """String representation of CTF."""
        return (
            f"CTF(pixel_size={self.pixel_size:.2f}Å, "
            f"voltage={self.voltage:.0f}kV, "
            f"cs={self.cs:.2f}mm, "
            f"defocus={self.defocus:.2f}µm, "
            f"amplitude={self.amplitude:.3f})"
        )
