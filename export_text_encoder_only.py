"""
Export only Text Encoder to ExecuTorch with static shapes
"""

from kokoro import KModel
import torch
from torch import nn
from torch.export import export, Dim
import os

print("Exporting Text Encoder to ExecuTorch")
print("=" * 80)

# Create output directory
os.makedirs("exported_pte", exist_ok=True)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M").eval()


# Define Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids: torch.LongTensor):
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
        return self.text_encoder(input_ids, input_lengths, ~text_mask)


text_encoder = TextEncoder(model)

# Create example inputs
# input_ids shape: [batch, seq_len]
# You can modify seq_len (16, 32, 64, etc.)

seq_len = 16  # Modify this for different input lengths
input_ids = torch.randint(1, 100, (1, seq_len))

print(f"Input shape: input_ids={input_ids.shape}")

# Export to ExecuTorch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

with torch.no_grad():
    # input_ids shape: [batch, seq_len] - dimension 1 is seq_len
    exported = export(text_encoder, (input_ids,), strict=False)
    print(f"✓ Torch export successful")

    # Convert to edge and lower with XNNPACK backend
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    print(f"✓ Converted to edge and lowered with XNNPACK")

    # Save to .pte
    pte_path = "exported_pte/text_encoder.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

print("\n" + "=" * 80)
print(f"SUCCESS! Text Encoder exported with shape: input_ids={input_ids.shape}")
print("Note: Input to ExecuTorch must match this exact shape")
print("=" * 80)
