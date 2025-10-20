"""
Export only F0N Predictor to ExecuTorch with static shapes
"""

from kokoro import KModel
import torch
from torch import nn
from torch.export import export, Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import os

print("Exporting F0N Predictor to ExecuTorch")
print("=" * 80)

# Create output directory
os.makedirs("exported_pte", exist_ok=True)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M").eval()


# Define F0N Predictor
class F0NPredictor(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.shared = model.predictor.shared
        self.F0_blocks = model.predictor.F0
        self.F0_proj = model.predictor.F0_proj
        self.N_blocks = model.predictor.N
        self.N_proj = model.predictor.N_proj

    def forward(self, en: torch.FloatTensor, s: torch.FloatTensor):
        x = en.transpose(-1, -2)
        # torch._check_is_size(x.shape[1])  # Commented out - causes dynamic shape to specialize
        x, _ = self.shared(x)

        F0 = x.transpose(-1, -2)
        for block in self.F0_blocks:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N_blocks:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1)


f0n_predictor = F0NPredictor(model)

# Create example inputs - you can modify these shapes
# en shape: [batch, hidden_dim, time_frames]
# s shape: [batch, style_dim]

# Example with specific time frames (modify as needed)
en = torch.randn(1, 640, 100)  # 100 time frames
s = torch.randn(1, 128)

print(f"Input shapes: en={en.shape}, s={s.shape}")

# Export to ExecuTorch

with torch.no_grad():
    # Export with torch.export
    from torch.export import Dim

    exported = export(f0n_predictor, (en, s), strict=False)
    print(f"✓ Torch export successful")

    # Convert to edge
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    print(f"✓ Converted to edge")

    # Save to .pte
    pte_path = "exported_pte/f0n_predictor.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

print("\n" + "=" * 80)
print(f"SUCCESS! F0N Predictor exported with shape: en={en.shape}")
print("Note: Input to ExecuTorch must match this exact shape")
print("=" * 80)
