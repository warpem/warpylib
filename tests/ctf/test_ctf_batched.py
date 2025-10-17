"""Tests for batched CTF calculations."""

import torch
import pytest
from warpylib import CTF


def test_batched_1d_single_param():
    """Test batched 1D CTF with a single batched parameter."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.amplitude = 0.07

    # Make defocus a 1D tensor with 3 values
    ctf.defocus = torch.tensor([1.0, 2.0, 3.0])

    # Calculate 1D CTF
    ctf_1d = ctf.get_1d(width=128)

    # Should have shape (3, 128) - 3 defocus values, 128 frequencies
    assert ctf_1d.shape == (3, 128)
    assert ctf_1d.dtype == torch.float32

    # Each defocus should produce different CTF
    assert not torch.allclose(ctf_1d[0], ctf_1d[1])
    assert not torch.allclose(ctf_1d[1], ctf_1d[2])


def test_batched_1d_multiple_params():
    """Test batched 1D CTF with multiple batched parameters of same shape."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = torch.tensor([200.0, 300.0])  # 2 values
    ctf.defocus = torch.tensor([1.0, 2.0])  # 2 values (same shape)
    ctf.amplitude = 0.07

    # Calculate 1D CTF - should work with same shape
    ctf_1d = ctf.get_1d(width=128)

    # Should have shape (2, 128)
    assert ctf_1d.shape == (2, 128)


def test_batched_1d_2d_param():
    """Test batched 1D CTF with 2D parameter."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.amplitude = 0.07

    # Make defocus a 2D tensor (2x3)
    ctf.defocus = torch.tensor([[1.0, 1.5, 2.0],
                                 [2.5, 3.0, 3.5]])

    # Calculate 1D CTF
    ctf_1d = ctf.get_1d(width=128)

    # Should have shape (2, 3, 128)
    assert ctf_1d.shape == (2, 3, 128)


def test_batched_2d_single_param():
    """Test batched 2D CTF with a single batched parameter."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.amplitude = 0.07

    # Make defocus a 1D tensor with 3 values
    ctf.defocus = torch.tensor([1.0, 2.0, 3.0])

    # Calculate 2D CTF
    size = 64
    ctf_2d = ctf.get_2d(size=size)

    # Should have shape (3, 64, 33) - 3 defocus values, rfft format
    assert ctf_2d.shape == (3, size, size // 2 + 1)
    assert ctf_2d.dtype == torch.float32

    # Each defocus should produce different CTF
    assert not torch.allclose(ctf_2d[0], ctf_2d[1])
    assert not torch.allclose(ctf_2d[1], ctf_2d[2])


def test_batched_2d_multiple_params():
    """Test batched 2D CTF with multiple batched parameters of same shape."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = torch.tensor([200.0, 300.0])  # 2 values
    ctf.defocus = torch.tensor([1.0, 2.0])  # 2 values (same shape)
    ctf.amplitude = 0.07

    # Calculate 2D CTF
    size = 64
    ctf_2d = ctf.get_2d(size=size)

    # Should have shape (2, 64, 33)
    assert ctf_2d.shape == (2, size, size // 2 + 1)


def test_batched_2d_with_astigmatism():
    """Test batched 2D CTF with astigmatism."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = torch.tensor([1.0, 2.0])
    ctf.defocus_delta = torch.tensor([0.0, 0.5])  # No astig, then astig
    ctf.defocus_angle = 45.0

    size = 64
    ctf_2d = ctf.get_2d(size=size)

    # Should have shape (2, 64, 33)
    assert ctf_2d.shape == (2, size, size // 2 + 1)

    # The two CTFs should be different due to astigmatism
    assert not torch.allclose(ctf_2d[0], ctf_2d[1])


def test_batched_scalar_mode_compatibility():
    """Test that scalar mode still works correctly."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.defocus = 2.0
    ctf.amplitude = 0.07

    # All scalars - should return unbatched result
    ctf_1d = ctf.get_1d(width=128)
    assert ctf_1d.shape == (128,)

    ctf_2d = ctf.get_2d(size=64)
    assert ctf_2d.shape == (64, 33)


def test_batched_bfactor():
    """Test batched CTF with batched B-factor."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = 2.0
    ctf.bfactor = torch.tensor([-100.0, 0.0, 100.0])  # Attenuate, none, sharpen

    ctf_1d = ctf.get_1d(width=512, ignore_bfactor=False)

    # Should have shape (3, 512)
    assert ctf_1d.shape == (3, 512)

    # High frequencies should be affected differently
    # Negative B-factor attenuates, positive sharpens
    assert torch.abs(ctf_1d[0, -1]) < torch.abs(ctf_1d[1, -1])  # -100 < 0
    assert torch.abs(ctf_1d[2, -1]) > torch.abs(ctf_1d[1, -1])  # 100 > 0


def test_batched_amp_squared():
    """Test batched CTF with amp_squared option."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = torch.tensor([1.0, 2.0])

    ctf_normal = ctf.get_1d(width=256, amp_squared=False)
    ctf_squared = ctf.get_1d(width=256, amp_squared=True)

    # Both should be batched
    assert ctf_normal.shape == (2, 256)
    assert ctf_squared.shape == (2, 256)

    # Squared should be all positive
    assert torch.all(ctf_squared >= 0)

    # Squared should be absolute value of normal
    assert torch.allclose(ctf_squared, torch.abs(ctf_normal), atol=1e-6)


def test_batched_device():
    """Test batched CTF on different devices."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.defocus = torch.tensor([1.0, 2.0])

    # CPU
    ctf_cpu = ctf.get_1d(width=128, device=torch.device('cpu'))
    assert ctf_cpu.device.type == 'cpu'
    assert ctf_cpu.shape == (2, 128)

    # GPU if available
    if torch.cuda.is_available():
        ctf_gpu = ctf.get_1d(width=128, device=torch.device('cuda'))
        assert ctf_gpu.device.type == 'cuda'
        assert ctf_gpu.shape == (2, 128)

        # Results should match
        assert torch.allclose(ctf_cpu, ctf_gpu.cpu(), atol=1e-5)


def test_batched_mixed_dimensions():
    """Test CTF with parameters of broadcastable dimensions."""
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = torch.tensor([[200.0], [300.0]])  # (2, 1)
    ctf.defocus = torch.tensor([1.0, 2.0])  # (2,)
    ctf.amplitude = 0.07

    # Should broadcast (2,1) and (2,) to (2,2)
    ctf_1d = ctf.get_1d(width=64)

    # Expected shape: (2, 1) broadcasts with (2,) -> (2, 2)
    # Final: (2, 2, 64)
    assert ctf_1d.shape == (2, 2, 64)


def test_batched_visualization():
    """Test visualization of batched CTF calculations."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Create CTF with batched defocus values
    ctf = CTF()
    ctf.pixel_size = 1.0
    ctf.voltage = 300.0
    ctf.amplitude = 0.07
    ctf.defocus = torch.tensor([0.5, 1.0, 2.0, 3.0])  # 4 different defocus values
    ctf.defocus_delta = 0.3  # Add some astigmatism
    ctf.defocus_angle = 45.0

    # Create testoutputs directory
    output_dir = Path(__file__).parent.parent.parent / 'testoutputs'
    output_dir.mkdir(exist_ok=True)

    # 1D CTF comparison
    ctf_1d = ctf.get_1d(width=256)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(ctf.defocus)):
        ax.plot(ctf_1d[i].numpy(), label=f'Defocus {ctf.defocus[i]:.1f}µm', alpha=0.8)
    ax.set_xlabel('Frequency index')
    ax.set_ylabel('CTF value')
    ax.set_title('Batched 1D CTF Profiles (Different Defocus Values)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path_1d = output_dir / 'test_ctf_batched_1d.png'
    plt.savefig(output_path_1d, dpi=100, bbox_inches='tight')
    plt.close(fig)

    assert output_path_1d.exists()
    print(f'\nBatched 1D CTF plot saved to: {output_path_1d}')

    # 2D CTF comparison (2x2 grid)
    size = 256
    ctf_2d = ctf.get_2d(size=size)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(len(ctf.defocus)):
        im = axes[i].imshow(ctf_2d[i].numpy(), cmap='gray', origin='lower')
        axes[i].set_xlabel('X frequency (rfft format)')
        axes[i].set_ylabel('Y frequency')
        axes[i].set_title(f'Defocus: {ctf.defocus[i]:.1f}µm')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(f'Batched 2D CTF (with astigmatism)\nVoltage: {ctf.voltage:.0f}kV, Pixel size: {ctf.pixel_size}Å', fontsize=14)
    plt.tight_layout()

    output_path_2d = output_dir / 'test_ctf_batched_2d.png'
    plt.savefig(output_path_2d, dpi=100, bbox_inches='tight')
    plt.close(fig)

    assert output_path_2d.exists()
    print(f'Batched 2D CTF plot saved to: {output_path_2d}')

    # 2D batched parameters (voltage and defocus)
    ctf2 = CTF()
    ctf2.pixel_size = 1.0
    ctf2.voltage = torch.tensor([[200.0], [300.0]])  # (2, 1) - two voltages
    ctf2.defocus = torch.tensor([1.0, 2.0])  # (2,) - two defocus values
    ctf2.amplitude = 0.07

    # This will broadcast to (2, 2) - 4 combinations
    ctf_2d_batched = ctf2.get_2d(size=256)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for i in range(2):
        for j in range(2):
            im = axes[i, j].imshow(ctf_2d_batched[i, j].numpy(), cmap='gray', origin='lower')
            axes[i, j].set_xlabel('X frequency')
            axes[i, j].set_ylabel('Y frequency')
            axes[i, j].set_title(f'V={ctf2.voltage[i, 0]:.0f}kV, D={ctf2.defocus[j]:.1f}µm')
            plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

    plt.suptitle('Batched 2D CTF (Voltage × Defocus Grid)', fontsize=14)
    plt.tight_layout()

    output_path_grid = output_dir / 'test_ctf_batched_grid.png'
    plt.savefig(output_path_grid, dpi=100, bbox_inches='tight')
    plt.close(fig)

    assert output_path_grid.exists()
    print(f'Batched grid CTF plot saved to: {output_path_grid}')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
