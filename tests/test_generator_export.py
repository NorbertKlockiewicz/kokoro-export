"""
Test Generator (without full Decoder) export to ExecuTorch
Check if NaNs appear in Generator operations
"""

import torch
from torch.export import export
from kokoro import KModel
import os

print("Testing Generator export")
print("=" * 80)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

# Extract Generator
generator = model.decoder.generator

# Create test inputs matching Generator.forward(x, s, f0)
batch = 1
channels = 512  # From decoder output before generator
frames = 78
x = torch.randn(batch, channels, frames)
s = torch.randn(batch, 128)  # Style vector (prosody)
f0 = torch.randn(batch, frames * 2)  # F0 curve

print(f"\nInput shapes:")
print(f"  x: {x.shape}")
print(f"  s: {s.shape}")
print(f"  f0: {f0.shape}")

# Test PyTorch version
with torch.no_grad():
    output_torch = generator(x, s, f0)
    print(f"\nPyTorch Generator output:")
    print(f"  Shape: {output_torch.shape}")
    print(f"  Range: [{output_torch.min():.3f}, {output_torch.max():.3f}]")
    print(f"  Contains NaN: {output_torch.isnan().any()}")
    print(f"  Contains Inf: {output_torch.isinf().any()}")

# Export
print("\n" + "=" * 80)
print("Exporting to ExecuTorch...")

class GeneratorWrapper(torch.nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.generator = gen

    def forward(self, x, s, f0):
        return self.generator(x, s, f0)

wrapper = GeneratorWrapper(generator).eval()

with torch.no_grad():
    # Export
    exported = export(wrapper, (x, s, f0), strict=False)
    print("✓ torch.export successful")

    # Test exported program
    output_exported = exported.module()(x, s, f0)
    print(f"\nExported program output:")
    print(f"  Shape: {output_exported.shape}")
    print(f"  Range: [{output_exported.min():.3f}, {output_exported.max():.3f}]")
    print(f"  Contains NaN: {output_exported.isnan().any()}")
    print(f"  Contains Inf: {output_exported.isinf().any()}")

    # Lower to ExecuTorch
    from executorch.exir import to_edge

    edge_program = to_edge(exported).to_executorch()
    print("✓ Lowered to ExecuTorch")

    # Save
    os.makedirs("exported_pte/debug", exist_ok=True)
    pte_path = "exported_pte/debug/generator.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Saved to {pte_path}")

    # Load and test
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    output_et = method.execute((x, s, f0))[0]
    print(f"\nExecuTorch output:")
    print(f"  Shape: {output_et.shape}")
    print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
    print(f"  Contains NaN: {output_et.isnan().any()}")
    print(f"  Contains Inf: {output_et.isinf().any()}")

    # Compare
    if output_et.isnan().any():
        print("\n⚠ NaNs detected in ExecuTorch output!")
        nan_count = output_et.isnan().sum().item()
        total = output_et.numel()
        print(f"  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")

        # Check where NaNs appear
        nan_mask = output_et.isnan()
        print(f"  First NaN at position: {nan_mask.nonzero()[0] if nan_mask.any() else 'N/A'}")
    else:
        diff = torch.abs(output_torch - output_et).max().item()
        print(f"\n✓ No NaNs! Max difference from PyTorch: {diff:.6f}")

print("\n" + "=" * 80)
