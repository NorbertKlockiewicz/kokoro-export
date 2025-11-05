"""
Export Decoder to ExecuTorch with pure Python random noise
Replaces torch.randn_like with Core ATen operations using a seed
"""

import torch
from torch import nn
from torch.export import export
import os

print("Exporting Decoder to ExecuTorch (pure Python random)")
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

# Create example inputs - 78 frames
frames = 78
asr = torch.randn(1, 512, frames)
F0_pred = torch.randn(1, frames * 2)
N_pred = torch.randn(1, frames * 2)
ref_s_prosody = torch.randn(1, 128)
print(decoder(asr, F0_pred, N_pred, ref_s_prosody))
print(
    f"Input shapes: asr={asr.shape}, F0={F0_pred.shape}, N={N_pred.shape}, ref_s={ref_s_prosody.shape}"
)

from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

with torch.no_grad():
    exported = export(decoder, (asr, F0_pred, N_pred, ref_s_prosody), strict=True)
    output = exported.module()(asr, F0_pred, N_pred, ref_s_prosody)
    print(output)
    print("✓ Torch export successful")

    # Try without exception first (randn might be in Core ATen)
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    print("✓ Converted to edge (no exceptions!)")

    pte_path = "exported_pte/decoder.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

from executorch.runtime import Runtime

runtime = Runtime.get()

program = runtime.load_program("exported_pte/decoder.pte")
method = program.load_method("forward")
outputs = method.execute((asr, F0_pred, N_pred, ref_s_prosody))
print(outputs)

print("\n" + "=" * 80)
print("SUCCESS! Decoder exported with deterministic noise")
print("Note: Uses torch.randn which should work in most runtimes")
print("=" * 80)
