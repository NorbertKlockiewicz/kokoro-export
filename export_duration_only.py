"""
Export only Duration Predictor to ExecuTorch with static shapes
"""

from kokoro import KModel
import torch
from torch import nn
from torch.export import export
import os

print("Exporting Duration Predictor to ExecuTorch")
print("=" * 80)

# Create output directory
os.makedirs("exported_pte", exist_ok=True)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M").eval()


# Define Duration Predictor
class DurationPredictor(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.text_encoder_module = model.predictor.text_encoder
        self.lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj

    def forward(
        self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.Tensor
    ):
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)

        bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.text_encoder_module(d_en, s, input_lengths, ~text_mask)
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        return pred_dur, d, s


duration_predictor = DurationPredictor(model)

# Create example inputs
# input_ids shape: [batch, seq_len]
# ref_s (voice) shape: [batch, 256]
# speed shape: [1] (1D tensor, NOT scalar!)

seq_len = 64  # Modify this for different input lengths (16, 32, 64, 128, 256)
input_ids = torch.randint(1, 100, (1, seq_len))
ref_s = torch.randn(1, 256)
speed = torch.tensor([1.0])  # Must be 1D tensor for ExecuTorch

print(f"Input shapes:")
print(f"  input_ids: {input_ids.shape}")
print(f"  ref_s (voice): {ref_s.shape}")
print(f"  speed: {speed.shape}")

# Export to ExecuTorch
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

with torch.no_grad():
    # Export with STATIC shape (bidirectional LSTM can't handle dynamic)
    exported = export(duration_predictor, (input_ids, ref_s, speed), strict=False)
    print(f"✓ Torch export successful")

    # Convert to edge and lower with XNNPACK backend
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    print(f"✓ Converted to edge and lowered with XNNPACK")

    # Save to .pte
    pte_path = "exported_pte/duration_predictor.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

print("\n" + "=" * 80)
print(f"SUCCESS! Duration Predictor exported")
print(f"Input shapes required:")
print(f"  - input_ids: {input_ids.shape}")
print(f"  - ref_s: {ref_s.shape}")
print(f"  - speed: {speed.shape}")
print("=" * 80)
