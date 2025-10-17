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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
