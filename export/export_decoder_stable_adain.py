"""
Export Decoder to ExecuTorch with stable AdaIN
Replaces InstanceNorm1d with numerically stable manual implementation
"""

import torch
from torch import nn
from torch.export import export
import os

print("Exporting Decoder with Stable AdaIN")
print("=" * 80)

from kokoro import KModel
from kokoro.stable_adain import StableAdaIN1d

os.makedirs("exported_pte", exist_ok=True)

# Load model with disable_complex=True
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded with disable_complex=True")

# Replace all AdaIN1d instances with StableAdaIN1d
def replace_adain_recursive(module):
    """Recursively replace AdaIN1d with StableAdaIN1d"""
    from kokoro.istftnet import AdaIN1d
    count = 0
    for name, child in module.named_children():
        if isinstance(child, AdaIN1d):
            # Create stable replacement
            stable_adain = StableAdaIN1d(
                style_dim=child.fc.in_features,
                num_features=child.fc.out_features // 2,
                eps=1e-5
            )
            # Copy weights
            stable_adain.fc.load_state_dict(child.fc.state_dict())
            stable_adain.weight.data = child.norm.weight.data.view(1, -1, 1)
            stable_adain.bias.data = child.norm.bias.data.view(1, -1, 1)

            setattr(module, name, stable_adain)
            count += 1
        else:
            count += replace_adain_recursive(child)
    return count

replaced = replace_adain_recursive(model.decoder)
print(f"✓ Replaced {replaced} AdaIN1d instances with StableAdaIN1d")

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

# Create example inputs - 78 frames
frames = 78
asr = torch.randn(1, 512, frames)
F0_pred = torch.randn(1, frames * 2)
N_pred = torch.randn(1, frames * 2)
ref_s_prosody = torch.randn(1, 128)

print(f"Input shapes: asr={asr.shape}, F0={F0_pred.shape}, N={N_pred.shape}, ref_s={ref_s_prosody.shape}")

# Test PyTorch version first
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

    # Lower to ExecuTorch
    from executorch.exir import to_edge

    edge_program = to_edge(exported).to_executorch()
    print("✓ Converted to ExecuTorch")

    pte_path = "exported_pte/decoder_stable.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

    # Test ExecuTorch runtime
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((asr, F0_pred, N_pred, ref_s_prosody))
    output_et = outputs[0]

    print(f"\nExecuTorch output:")
    print(f"  Shape: {output_et.shape}")
    print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
    print(f"  Contains NaN: {output_et.isnan().any()}")

    if output_et.isnan().any():
        print("\n⚠ Still has NaNs with stable AdaIN")
    else:
        diff = torch.abs(output_pytorch - output_et).max().item()
        print(f"\n✓ No NaNs! Max difference from PyTorch: {diff:.6f}")

print("\n" + "=" * 80)
print("Decoder exported with Stable AdaIN")
print("=" * 80)
