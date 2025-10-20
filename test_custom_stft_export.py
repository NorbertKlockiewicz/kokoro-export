"""
Test CustomSTFT export to ExecuTorch
Check if NaNs appear in STFT operations
"""

import torch
from torch.export import export
from kokoro.custom_stft import CustomSTFT
import os

print("Testing CustomSTFT export")
print("=" * 80)

# Create CustomSTFT module
stft = CustomSTFT(
    filter_length=800, hop_length=200, win_length=800, window="hann"
).eval()

# Test with simple sine wave input
duration = 1.0  # seconds
sample_rate = 24000
t = torch.linspace(0, duration, int(sample_rate * duration))
# 440 Hz sine wave
test_audio = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)

print(f"Input audio shape: {test_audio.shape}")
print(f"Input range: [{test_audio.min():.3f}, {test_audio.max():.3f}]")

# Test PyTorch version
with torch.no_grad():
    mag_torch, phase_torch = stft.transform(test_audio)
    print(f"\nPyTorch transform:")
    print(
        f"  Magnitude shape: {mag_torch.shape}, range: [{mag_torch.min():.3f}, {mag_torch.max():.3f}]"
    )
    print(
        f"  Phase shape: {phase_torch.shape}, range: [{phase_torch.min():.3f}, {phase_torch.max():.3f}]"
    )
    print(
        f"  Contains NaN: mag={mag_torch.isnan().any()}, phase={phase_torch.isnan().any()}"
    )

    audio_torch = stft.inverse(mag_torch, phase_torch)
    print(f"\nPyTorch inverse:")
    print(f"  Audio shape: {audio_torch.shape}")
    print(f"  Audio range: [{audio_torch.min():.3f}, {audio_torch.max():.3f}]")
    print(f"  Contains NaN: {audio_torch.isnan().any()}")

# Export to ExecuTorch
print("\n" + "=" * 80)
print("Exporting to ExecuTorch...")


class STFTWrapper(torch.nn.Module):
    def __init__(self, stft):
        super().__init__()
        self.stft = stft

    def forward(self, audio):
        mag, phase = self.stft.transform(audio)
        return self.stft.inverse(mag, phase)


wrapper = STFTWrapper(stft).eval()

with torch.no_grad():
    # Export
    exported = export(wrapper, (test_audio,), strict=False)
    print("✓ torch.export successful")

    # Test exported program
    output_exported = exported.module()(test_audio)
    print(f"\nExported program output:")
    print(f"  Shape: {output_exported.shape}")
    print(f"  Range: [{output_exported.min():.3f}, {output_exported.max():.3f}]")
    print(f"  Contains NaN: {output_exported.isnan().any()}")

    # Lower to ExecuTorch
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    print("✓ Lowered to ExecuTorch")

    # Save
    os.makedirs("exported_pte/debug", exist_ok=True)
    pte_path = "exported_pte/debug/custom_stft.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Saved to {pte_path}")

    # Load and test
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    output_et = method.execute((test_audio,))[0]
    print(f"\nExecuTorch output:")
    print(f"  Shape: {output_et.shape}")
    print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
    print(f"  Contains NaN: {output_et.isnan().any()}")

    # Compare
    if output_et.isnan().any():
        print("\n⚠ NaNs detected in ExecuTorch output!")
        nan_count = output_et.isnan().sum().item()
        total = output_et.numel()
        print(f"  NaN count: {nan_count}/{total} ({100 * nan_count / total:.2f}%)")
    else:
        diff = torch.abs(audio_torch - output_et).max().item()
        print(f"\n✓ No NaNs! Max difference from PyTorch: {diff:.6f}")

print("\n" + "=" * 80)
