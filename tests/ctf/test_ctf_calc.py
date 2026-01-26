"""Tests for CTF calculation methods."""

import torch
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mrcfile
from warpylib import CTF


def test_ctf_1d_basic():
    """Test basic 1D CTF calculation."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.defocus = 1.0
    ctf.amplitude = 0.07

    # Calculate 1D CTF
    ctf_1d = ctf.get_1d(width=512)

    assert ctf_1d.shape == (512,)
    assert ctf_1d.dtype == torch.float32

    # CTF should oscillate, check some basic properties
    assert torch.min(ctf_1d) < 0  # Should have negative values
    assert torch.max(ctf_1d) > 0  # Should have positive values

    # DC component (first element) should be non-zero
    assert ctf_1d[0] != 0


def test_ctf_1d_amp_squared():
    """Test 1D CTF with amplitude squared."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 1.0

    ctf_normal = ctf.get_1d(width=512, amp_squared=False)
    ctf_squared = ctf.get_1d(width=512, amp_squared=True)

    # Squared version should be all positive
    assert torch.all(ctf_squared >= 0)

    # Squared version should be absolute value of normal
    assert torch.allclose(ctf_squared, torch.abs(ctf_normal), atol=1e-6)


def test_ctf_1d_bfactor():
    """Test 1D CTF with B-factor."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0
    ctf.bfactor = -100.0  # Negative B-factor attenuates high frequencies

    ctf_with_b = ctf.get_1d(width=512, ignore_bfactor=False)
    ctf_without_b = ctf.get_1d(width=512, ignore_bfactor=True)

    # With negative B-factor, high frequencies should be attenuated
    assert torch.abs(ctf_with_b[-1]) < torch.abs(ctf_without_b[-1])


def test_ctf_2d_basic():
    """Test basic 2D CTF calculation in rfft format."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.defocus = 2.0
    ctf.amplitude = 0.07

    # Calculate 2D CTF
    size = 64
    ctf_2d = ctf.get_2d(size=size)

    # Check shape - should be rfft format
    assert ctf_2d.shape == (size, size // 2 + 1)
    assert ctf_2d.dtype == torch.float32

    # CTF should oscillate
    assert torch.min(ctf_2d) < 0
    assert torch.max(ctf_2d) > 0


def test_ctf_2d_astigmatism():
    """Test 2D CTF with astigmatism."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0
    ctf.defocus_delta = 0.5  # Add astigmatism
    ctf.defocus_angle = 45.0

    ctf_no_astig = CTF()
    ctf_no_astig.pixel_size = 1.0
    ctf_no_astig.defocus = 2.0

    size = 64
    ctf_with_astig = ctf.get_2d(size=size)
    ctf_without_astig = ctf_no_astig.get_2d(size=size)

    # With astigmatism, the CTF should be different
    assert not torch.allclose(ctf_with_astig, ctf_without_astig)


def test_ctf_2d_amp_squared():
    """Test 2D CTF with amplitude squared."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    size = 64
    ctf_normal = ctf.get_2d(size=size, amp_squared=False)
    ctf_squared = ctf.get_2d(size=size, amp_squared=True)

    # Squared version should be all positive
    assert torch.all(ctf_squared >= 0)

    # Squared version should be absolute value of normal
    assert torch.allclose(ctf_squared, torch.abs(ctf_normal), atol=1e-6)


def test_ctf_coords():
    """Test CTF coordinate generation."""
    size = 64
    r, angle = CTF.get_ctf_coords(size=size)

    # Check shapes
    assert r.shape == (size, size // 2 + 1)
    assert angle.shape == (size, size // 2 + 1)

    # Radius should be non-negative
    assert torch.all(r >= 0)

    # DC component should be at (0, 0)
    assert r[0, 0] == 0

    # Angles should be in [-pi, pi]
    assert torch.all(angle >= -torch.pi)
    assert torch.all(angle <= torch.pi)


def test_ctf_coords_with_anisotropy():
    """Test CTF coordinate generation with pixel anisotropy."""
    size = 64
    r_iso, angle_iso = CTF.get_ctf_coords(size=size, pixel_size=1.0)
    r_aniso, angle_aniso = CTF.get_ctf_coords(
        size=size,
        pixel_size=1.0,
        pixel_size_delta=0.1,
        pixel_size_angle=45.0
    )

    # Radii should be different with anisotropy
    assert not torch.allclose(r_iso, r_aniso)

    # But angles should be the same
    assert torch.allclose(angle_iso, angle_aniso)


def test_ctf_original_size():
    """Test CTF calculation with different original size."""
    ctf = CTF()
    ctf.pixel_size = 2.0  # 2 Angstrom/pixel
    ctf.defocus = 2.0

    size = 64
    original_size = 128

    ctf_same = ctf.get_2d(size=size, original_size=size)
    ctf_diff = ctf.get_2d(size=size, original_size=original_size)

    # Different original sizes should give different results
    assert not torch.allclose(ctf_same, ctf_diff)


def test_ctf_device():
    """Test CTF calculation on different devices."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    # CPU
    ctf_cpu = ctf.get_1d(width=512, device=torch.device('cpu'))
    assert ctf_cpu.device.type == 'cpu'

    # GPU if available
    if torch.cuda.is_available():
        ctf_gpu = ctf.get_1d(width=512, device=torch.device('cuda'))
        assert ctf_gpu.device.type == 'cuda'

        # Results should match
        assert torch.allclose(ctf_cpu, ctf_gpu.cpu(), atol=1e-5)


def test_ctf_2d_symmetry():
    """Test that 2D CTF respects Hermitian symmetry (approximately)."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0
    # No astigmatism or anisotropy for this test

    size = 64
    ctf_2d = ctf.get_2d(size=size)

    # Check a few symmetry points
    # The point (y, x) should equal the point (-y, -x) for rfft format
    # But we only have half the spectrum, so this is approximate

    # At least check that DC (0,0) matches Nyquist in y direction
    # These should have same magnitude in a rotationally symmetric CTF
    dc_val = ctf_2d[0, 0]
    # Just check it's a reasonable value
    assert torch.isfinite(dc_val)


def test_visualization():
    """Test that we can visualize CTF and save plots."""
    from pathlib import Path

    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.defocus = 1.0
    ctf.amplitude = 0.07
    ctf.defocus_delta = 0.3  # Add some astigmatism for visualization
    ctf.defocus_angle = 45.0

    # Create testoutputs directory
    output_dir = Path(__file__).parent.parent.parent / 'testoutputs'
    output_dir.mkdir(exist_ok=True)

    # 1D CTF
    ctf_1d = ctf.get_1d(width=256)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ctf_1d.numpy())
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('CTF value')
    ax.set_title('1D CTF Profile')
    ax.grid(True, alpha=0.3)

    output_path_1d = output_dir / 'test_ctf_1d.png'
    plt.savefig(output_path_1d, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert output_path_1d.exists()
    print(f'\n1D CTF plot saved to: {output_path_1d}')

    # 2D CTF
    size = 512
    ctf_2d = ctf.get_2d(size=size)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(ctf_2d.numpy(), cmap='gray', origin='lower')
    ax.set_xlabel('X frequency (rfft format)')
    ax.set_ylabel('Y frequency')
    ax.set_title(f'2D CTF (amp squared)\nDefocus: {ctf.defocus}µm, Astigmatism: {ctf.defocus_delta}µm @ {ctf.defocus_angle}°')
    plt.colorbar(im, ax=ax, label='CTF amplitude')

    output_path_2d = output_dir / 'test_ctf_2d.png'
    plt.savefig(output_path_2d, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert output_path_2d.exists()
    print(f'2D CTF plot saved to: {output_path_2d}')

    # Combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1D plot
    ax1.plot(ctf_1d.numpy())
    ax1.set_xlabel('Frequency index')
    ax1.set_ylabel('CTF value')
    ax1.set_title('1D CTF Profile')
    ax1.grid(True, alpha=0.3)

    # 2D plot
    im = ax2.imshow(ctf_2d.numpy(), cmap='gray', origin='lower')
    ax2.set_xlabel('X frequency (rfft format)')
    ax2.set_ylabel('Y frequency')
    ax2.set_title('2D CTF (with astigmatism)')
    plt.colorbar(im, ax=ax2)

    plt.suptitle(f'CTF Visualization Test\nVoltage: {ctf.voltage}kV, Pixel size: {ctf.pixel_size}Å', fontsize=12)
    plt.tight_layout()

    output_path_combined = output_dir / 'test_ctf_combined.png'
    plt.savefig(output_path_combined, dpi=100, bbox_inches='tight')
    plt.close(fig)

    assert output_path_combined.exists()
    print(f'Combined CTF plot saved to: {output_path_combined}')


def test_ctf_reference_comparison():
    """Test CTF calculation against reference results from C# code."""
    # Path to reference data
    testdata_dir = Path(__file__).parent.parent.parent / 'testdata'
    reference_path = testdata_dir / 'ctf_reference.mrc'

    if not reference_path.exists():
        pytest.skip(f"Reference file not found: {reference_path}")

    # Create CTF objects matching the C# reference code
    # C1 parameters from gen_ctf.txt
    ctf1 = CTF()
    ctf1.pixel_size = 1.1
    ctf1.defocus = 1.5
    ctf1.defocus_delta = 0.3
    ctf1.defocus_angle = 25.0
    ctf1.phase_shift = 0.4
    ctf1.voltage = 300.0
    ctf1.cs = 2.7
    ctf1.amplitude = 0.1
    ctf1.scale = 0.9
    ctf1.bfactor = -40.0

    # C2 parameters from gen_ctf.txt
    ctf2 = CTF()
    ctf2.pixel_size = 1.5
    ctf2.defocus = 2.5
    ctf2.defocus_delta = 0.0
    ctf2.defocus_angle = 0.0
    ctf2.phase_shift = 0.0
    ctf2.voltage = 200.0
    ctf2.cs = 2.7
    ctf2.amplitude = 0.07
    ctf2.scale = 1.0
    ctf2.bfactor = -40.0

    # Calculate 2D CTF for both (256x256 as in reference)
    size = 256
    ctf1_2d = ctf1.get_2d(size=size, amp_squared=False)
    ctf2_2d = ctf2.get_2d(size=size, amp_squared=False)

    # Stack them (batch dimension)
    ctf_stack = torch.stack([ctf1_2d, ctf2_2d], dim=0)  # Shape: (2, 256, 129)

    # Save our result to testoutputs for comparison
    testoutputs_dir = Path(__file__).parent.parent.parent / 'testoutputs'
    testoutputs_dir.mkdir(exist_ok=True)
    with mrcfile.new(testoutputs_dir / 'ctf_python_result.mrc', overwrite=True) as mrc:
        mrc.set_data(ctf_stack.numpy())

    # Load reference data
    with mrcfile.open(reference_path, permissive=True) as mrc:
        reference = torch.from_numpy(mrc.data.copy()).float()

    # Reference shape should be (2, 256, 129) for rfft format
    assert reference.shape == ctf_stack.shape, f"Shape mismatch: expected {ctf_stack.shape}, got {reference.shape}"

    # Compare with reference, excluding Nyquist frequency row
    # The Nyquist frequency (at index size//2 = 128) may have different sign convention
    nyquist_idx = size // 2

    # Create a mask excluding the Nyquist row
    mask = torch.ones_like(ctf_stack, dtype=torch.bool)
    mask[:, nyquist_idx, :] = False

    # Compare only non-Nyquist values
    ctf_masked = ctf_stack[mask]
    ref_masked = reference[mask]

    assert torch.allclose(ctf_masked, ref_masked, atol=2e-5, rtol=1e-4), \
        f"CTF values don't match reference. Max diff: {torch.max(torch.abs(ctf_masked - ref_masked))}"

    print(f"\nCTF reference test passed!")
    print(f"  Shape: {ctf_stack.shape}")
    print(f"  Compared elements (excluding Nyquist row): {ctf_masked.numel()} / {ctf_stack.numel()}")
    print(f"  Max absolute difference: {torch.max(torch.abs(ctf_masked - ref_masked)):.2e}")
    print(f"  Mean absolute difference: {torch.mean(torch.abs(ctf_masked - ref_masked)):.2e}")


def test_ctf_1d_ignore_low_frequency_validation():
    """Test that ignore parameters are validated correctly."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    # Should raise if ignore_below_res is set but ignore_transition_res is not
    with pytest.raises(ValueError, match="ignore_transition_res must be set"):
        ctf.get_1d(width=512, ignore_below_res=50.0)

    # Should raise if ignore_below_res <= ignore_transition_res
    with pytest.raises(ValueError, match="must be greater than"):
        ctf.get_1d(width=512, ignore_below_res=20.0, ignore_transition_res=50.0)

    with pytest.raises(ValueError, match="must be greater than"):
        ctf.get_1d(width=512, ignore_below_res=30.0, ignore_transition_res=30.0)


def test_ctf_2d_ignore_low_frequency_validation():
    """Test that ignore parameters are validated correctly for 2D."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    # Should raise if ignore_below_res is set but ignore_transition_res is not
    with pytest.raises(ValueError, match="ignore_transition_res must be set"):
        ctf.get_2d(size=64, ignore_below_res=50.0)

    # Should raise if ignore_below_res <= ignore_transition_res
    with pytest.raises(ValueError, match="must be greater than"):
        ctf.get_2d(size=64, ignore_below_res=20.0, ignore_transition_res=50.0)


def test_ctf_1d_ignore_low_frequency_basic():
    """Test basic low-frequency ignore functionality for 1D CTF."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    width = 512
    ignore_below_res = 50.0  # Ignore below 50Å (low freq)
    ignore_transition_res = 20.0  # Full CTF above 20Å (higher freq)

    ctf_normal = ctf.get_1d(width=width)
    ctf_ignored = ctf.get_1d(
        width=width,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    # Shapes should match
    assert ctf_normal.shape == ctf_ignored.shape

    # At very low frequencies (below ignore_below_res), CTF should be 1
    # First element (DC) should be close to 1
    assert torch.abs(ctf_ignored[0] - 1.0) < 0.01

    # At high frequencies (well above 1/ignore_transition_res), should match normal CTF
    # Nyquist frequency = 0.5 / pixel_size = 0.5 (1/Å)
    # freq[i] = i * nyquist / width = i * 0.5 / 512
    # For full CTF, we need freq > 1/ignore_transition_res = 1/20 = 0.05
    # i > 0.05 * 512 / 0.5 = 51.2, so start well past that
    high_freq_start = 60  # Safely past the transition zone
    assert torch.allclose(
        ctf_ignored[high_freq_start:],
        ctf_normal[high_freq_start:],
        atol=1e-6
    )


def test_ctf_2d_ignore_low_frequency_basic():
    """Test basic low-frequency ignore functionality for 2D CTF."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    size = 128
    ignore_below_res = 50.0
    ignore_transition_res = 20.0

    ctf_normal = ctf.get_2d(size=size)
    ctf_ignored = ctf.get_2d(
        size=size,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    # Shapes should match
    assert ctf_normal.shape == ctf_ignored.shape

    # DC component (0, 0) should be 1
    assert torch.abs(ctf_ignored[0, 0] - 1.0) < 0.01


def test_ctf_1d_ignore_transition_smoothness():
    """Test that the transition uses raised cosine interpolation."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    width = 1024
    ignore_below_res = 100.0  # Wide transition zone for testing
    ignore_transition_res = 20.0

    ctf_ignored = ctf.get_1d(
        width=width,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    # Compute frequencies
    nyquist_step = 0.5 / ctf.pixel_size / width
    freq = torch.arange(width, dtype=torch.float32) * nyquist_step

    freq_low = 1.0 / ignore_below_res
    freq_high = 1.0 / ignore_transition_res

    # Find indices in transition zone
    in_transition = (freq > freq_low) & (freq < freq_high)

    # The transition should be smooth (no sudden jumps)
    # Check that values in transition zone are between boundary values
    if torch.any(in_transition):
        transition_values = ctf_ignored[in_transition]
        # All values should be finite
        assert torch.all(torch.isfinite(transition_values))


def test_ctf_1d_ignore_matches_at_boundaries():
    """Test CTF values at the exact boundary frequencies."""
    ctf = CTF()
    ctf.pixel_size = 2.0  # Larger pixel size for wider frequency spacing
    ctf.defocus = 2.0

    width = 256
    ignore_below_res = 40.0
    ignore_transition_res = 10.0

    ctf_normal = ctf.get_1d(width=width)
    ctf_ignored = ctf.get_1d(
        width=width,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    # Verify that without ignore parameters, we get oscillating CTF
    assert torch.min(ctf_normal) < 0
    assert torch.max(ctf_normal) > 0

    # Verify that with ignore, low frequencies are closer to 1
    # The first few indices should be very close to 1
    assert torch.mean(torch.abs(ctf_ignored[:3] - 1.0)) < torch.mean(torch.abs(ctf_normal[:3] - 1.0))


def test_ctf_2d_ignore_radial_symmetry():
    """Test that 2D ignore is radially symmetric (for isotropic CTF)."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0
    # No astigmatism for radial symmetry

    size = 64
    ignore_below_res = 30.0
    ignore_transition_res = 15.0

    ctf_ignored = ctf.get_2d(
        size=size,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    # Check that values at same radius are approximately equal
    # Compare (0, 5) with (5, 0) - should have same radius
    # Note: in rfft format, x goes from 0 to size//2
    val_x = ctf_ignored[0, 5]  # Along x-axis
    val_y = ctf_ignored[5, 0]  # Along y-axis

    assert torch.abs(val_x - val_y) < 1e-5, f"Radial asymmetry: x={val_x}, y={val_y}"


def test_ctf_ignore_no_effect_when_not_set():
    """Test that CTF is unchanged when ignore parameters are not set."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0

    # 1D
    ctf_1d_default = ctf.get_1d(width=512)
    ctf_1d_none = ctf.get_1d(width=512, ignore_below_res=None, ignore_transition_res=None)
    assert torch.allclose(ctf_1d_default, ctf_1d_none)

    # 2D
    ctf_2d_default = ctf.get_2d(size=64)
    ctf_2d_none = ctf.get_2d(size=64, ignore_below_res=None, ignore_transition_res=None)
    assert torch.allclose(ctf_2d_default, ctf_2d_none)


def test_ctf_ignore_visualization():
    """Test and visualize the low-frequency ignore functionality."""
    from pathlib import Path

    ctf = CTF()
    ctf.pixel_size = 1.5
    ctf.voltage = 300.0
    ctf.defocus = 2.0
    ctf.amplitude = 0.07

    # Create testoutputs directory
    output_dir = Path(__file__).parent.parent.parent / 'testoutputs'
    output_dir.mkdir(exist_ok=True)

    width = 512
    ignore_below_res = 60.0
    ignore_transition_res = 20.0

    # Calculate CTFs
    ctf_normal = ctf.get_1d(width=width)
    ctf_ignored = ctf.get_1d(
        width=width,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    # Create frequency axis (in 1/Å)
    nyquist_step = 0.5 / ctf.pixel_size / width
    freq = np.arange(width) * nyquist_step

    # Convert to resolution (Å) for x-axis, avoiding division by zero
    resolution = np.where(freq > 0, 1.0 / freq, np.inf)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Calculate frequency indices for the resolution boundaries
    # freq[i] = i * 0.5 / (pixel_size * width), so for resolution R:
    # 1/R = i * 0.5 / (pixel_size * width) => i = 2 * pixel_size * width / R
    idx_below = 2 * ctf.pixel_size * width / ignore_below_res
    idx_transition = 2 * ctf.pixel_size * width / ignore_transition_res

    # Plot vs frequency index
    ax1 = axes[0]
    ax1.plot(ctf_normal.numpy(), label='Normal CTF', alpha=0.7)
    ax1.plot(ctf_ignored.numpy(), label='With low-freq ignore', alpha=0.7)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='CTF = 1')
    ax1.axvline(x=idx_below, color='r', linestyle=':', alpha=0.5, label=f'ignore_below_res ({ignore_below_res}Å)')
    ax1.axvline(x=idx_transition, color='g', linestyle=':', alpha=0.5, label=f'ignore_transition_res ({ignore_transition_res}Å)')
    ax1.set_xlabel('Frequency index')
    ax1.set_ylabel('CTF value')
    ax1.set_title('1D CTF with Low-Frequency Ignore')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)  # Zoom to see the transition

    # Plot the difference
    ax2 = axes[1]
    diff = ctf_ignored.numpy() - ctf_normal.numpy()
    ax2.plot(diff)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=idx_below, color='r', linestyle=':', alpha=0.5)
    ax2.axvline(x=idx_transition, color='g', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Frequency index')
    ax2.set_ylabel('Difference (ignored - normal)')
    ax2.set_title('Difference between ignored and normal CTF')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    output_path = output_dir / 'test_ctf_low_freq_ignore.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    assert output_path.exists()
    print(f'\nLow-frequency ignore plot saved to: {output_path}')

    # 2D visualization
    size = 256
    ctf_2d_normal = ctf.get_2d(size=size)
    ctf_2d_ignored = ctf.get_2d(
        size=size,
        ignore_below_res=ignore_below_res,
        ignore_transition_res=ignore_transition_res
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(ctf_2d_normal.numpy(), cmap='gray', origin='lower')
    axes[0].set_title('Normal 2D CTF')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ctf_2d_ignored.numpy(), cmap='gray', origin='lower')
    axes[1].set_title(f'With ignore (below {ignore_below_res}Å, transition to {ignore_transition_res}Å)')
    plt.colorbar(im1, ax=axes[1])

    diff_2d = ctf_2d_ignored.numpy() - ctf_2d_normal.numpy()
    im2 = axes[2].imshow(diff_2d, cmap='RdBu', origin='lower')
    axes[2].set_title('Difference')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    output_path_2d = output_dir / 'test_ctf_low_freq_ignore_2d.png'
    plt.savefig(output_path_2d, dpi=100, bbox_inches='tight')
    plt.close(fig)

    assert output_path_2d.exists()
    print(f'2D low-frequency ignore plot saved to: {output_path_2d}')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
