"""
Export Decoder to ExecuTorch WITHOUT XNNPACK
Test if XNNPACK partitioner is causing NaN issues
"""

import torch
from torch import nn
from torch.export import export
import os

print("Exporting Decoder WITHOUT XNNPACK")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte", exist_ok=True)

# Load model with disable_complex=True
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded with disable_complex=True")

class DecoderWrapper(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(
        self,
        asr: torch.FloatTensor,
        F0_pred: torch.FloatTensor,
        N_pred: torch.FloatTensor,
        ref_s: torch.FloatTensor,
    ):
        return self.decoder(asr, F0_pred, N_pred, ref_s).squeeze()

decoder = DecoderWrapper(model)

# Create example inputs - 78 frames (matching actual usage)
frames = 78
asr = torch.randn(1, 512, frames)
F0_pred = torch.randn(1, frames * 2)
N_pred = torch.randn(1, frames * 2)
ref_s_prosody = torch.randn(1, 128)

print(f"Input shapes: asr={asr.shape}, F0={F0_pred.shape}, N={N_pred.shape}, ref_s={ref_s_prosody.shape}")

# Test PyTorch version
with torch.no_grad():
    output_pytorch = decoder(asr, F0_pred, N_pred, ref_s_prosody)
    print(f"\nPyTorch output:")
    print(f"  Shape: {output_pytorch.shape}")
    print(f"  Range: [{output_pytorch.min():.3f}, {output_pytorch.max():.3f}]")
    print(f"  Contains NaN: {output_pytorch.isnan().any()}")

    # Export
    exported = export(decoder, (asr, F0_pred, N_pred, ref_s_prosody), strict=False)
    output_exported = exported.module()(asr, F0_pred, N_pred, ref_s_prosody)
    print(f"\nExported output:")
    print(f"  Shape: {output_exported.shape}")
    print(f"  Range: [{output_exported.min():.3f}, {output_exported.max():.3f}]")
    print(f"  Contains NaN: {output_exported.isnan().any()}")
    print("✓ Torch export successful")

    # Lower to ExecuTorch WITHOUT XNNPACK
    from executorch.exir import to_edge

    print("\nLowering to ExecuTorch (NO XNNPACK)...")
    edge_program = to_edge(exported).to_executorch()
    print("✓ Converted to ExecuTorch (portable kernels only)")

    pte_path = "exported_pte/decoder_no_xnnpack.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

    # Test ExecuTorch runtime
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")

    print("\nTesting ExecuTorch runtime...")
    outputs = method.execute((asr, F0_pred, N_pred, ref_s_prosody))
    output_et = outputs[0]

    print(f"ExecuTorch output:")
    print(f"  Shape: {output_et.shape}")
    print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
    print(f"  Contains NaN: {output_et.isnan().any()}")

    if output_et.isnan().any():
        nan_count = output_et.isnan().sum().item()
        total = output_et.numel()
        print(f"\n⚠ NaNs detected: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        print("  Problem is NOT caused by XNNPACK")
    else:
        diff = torch.abs(output_pytorch - output_et).max().item()
        print(f"\n✓ No NaNs! XNNPACK was the culprit!")
        print(f"  Max difference from PyTorch: {diff:.6f}")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
