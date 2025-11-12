"""
Export Kokoro model parts to ExecuTorch .pte format
"""
from kokoro import KModel
import torch
from torch import nn
from torch.export import export
import os

print("=" * 80)
print("Exporting Kokoro Model Parts to ExecuTorch")
print("=" * 80)

# Create output directory
os.makedirs("exported_pte", exist_ok=True)

# Load model
repo_id = "hexgrad/Kokoro-82M"
model = KModel(repo_id=repo_id).eval()

# =============================================================================
# Define wrapper classes
# =============================================================================

class DurationPredictor(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.text_encoder_module = model.predictor.text_encoder
        self.lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.Tensor):
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
        torch._check_is_size(x.shape[1])
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


class TextEncoderWrapper(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids: torch.LongTensor):
        # WARNING: Duplicated with DurationPredictor
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
        return self.text_encoder(input_ids, input_lengths, ~text_mask)


class DecoderWrapper(nn.Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr: torch.FloatTensor, F0_pred: torch.FloatTensor,
                N_pred: torch.FloatTensor, ref_s: torch.FloatTensor):
        return self.decoder(asr, F0_pred, N_pred, ref_s).squeeze()


# =============================================================================
# Export Part 1: Duration Predictor
# =============================================================================
print("\n[1/4] Exporting Duration Predictor...")

try:
    duration_predictor = DurationPredictor(model)

    # Create example inputs
    input_ids = torch.randint(1, 100, (14,))
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.tensor([1.0])  # Must be a 1D tensor for ExecuTorch

    # Export to ExecuTorch
    from executorch.exir import to_edge, EdgeCompileConfig

    with torch.no_grad():
        # First export using torch.export
        exported = export(
            duration_predictor,
            (input_ids, style, speed),
            strict=False
        )
        print(f"  ✓ Torch export successful")

        # Convert to edge
        edge_program = to_edge(exported)
        print(f"  ✓ Converted to edge")

        # Save to .pte
        pte_path = "exported_pte/duration_predictor.pte"
        with open(pte_path, "wb") as f:
            edge_program.to_executorch().write_to_file(f)

        print(f"  ✓ Saved to {pte_path}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Export Part 3: F0/N Predictor
# =============================================================================
print("\n[2/4] Exporting F0/N Predictor...")

try:
    f0n_predictor = F0NPredictor(model)

    # Create example inputs (from duration predictor output)
    with torch.no_grad():
        pred_dur, d, s = duration_predictor(input_ids, style, speed)

        # Create alignment
        device = input_ids.device
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=device),
            pred_dur
        )
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg

        print(f"  Export shapes: en={en.shape}, s={s.shape}")

    # Export - ExecuTorch doesn't support dynamic shapes well, so use static
    # Note: This means input must match the exported shape exactly

    with torch.no_grad():
        exported = export(
            f0n_predictor,
            (en, s),
            strict=False
        )
        print(f"  ✓ Torch export successful (static shape: en={en.shape})")

        edge_program = to_edge(exported)
        print(f"  ✓ Converted to edge")

        pte_path = "exported_pte/f0n_predictor.pte"
        with open(pte_path, "wb") as f:
            edge_program.to_executorch().write_to_file(f)

        print(f"  ✓ Saved to {pte_path}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Export Part 4: Text Encoder
# =============================================================================
print("\n[3/4] Exporting Text Encoder...")

try:
    text_encoder = TextEncoderWrapper(model)

    with torch.no_grad():
        exported = export(
            text_encoder,
            (input_ids,),
            strict=False
        )
        print(f"  ✓ Torch export successful")

        edge_program = to_edge(exported)
        print(f"  ✓ Converted to edge")

        pte_path = "exported_pte/text_encoder.pte"
        with open(pte_path, "wb") as f:
            edge_program.to_executorch().write_to_file(f)

        print(f"  ✓ Saved to {pte_path}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Export Part 5: Decoder
# =============================================================================
print("\n[4/4] Exporting Decoder...")

try:
    decoder = DecoderWrapper(model)

    with torch.no_grad():
        # Get example inputs from previous parts
        F0_pred, N_pred = f0n_predictor(en, s)
        t_en = text_encoder(input_ids)
        asr = t_en @ pred_aln_trg
        ref_s_prosody = style[:, :128]

    with torch.no_grad():
        exported = export(
            decoder,
            (asr, F0_pred, N_pred, ref_s_prosody),
            strict=False
        )
        print(f"  ✓ Torch export successful")

        # Allow operators not in core ATen (decoder uses ISTFT with complex numbers)
        compile_config = EdgeCompileConfig(
            _core_aten_ops_exception_list=[
                torch.ops.aten.randn_like.default,
                torch.ops.aten.unfold.default,
                torch.ops.aten.angle.default,
                torch.ops.aten.conj_physical.default,
                torch.ops.aten.view_as_real.default,
                torch.ops.aten.view_as_complex.default,
            ]
        )
        edge_program = to_edge(exported, compile_config=compile_config)
        print(f"  ✓ Converted to edge")

        pte_path = "exported_pte/decoder.pte"
        with open(pte_path, "wb") as f:
            edge_program.to_executorch().write_to_file(f)

        print(f"  ✓ Saved to {pte_path}")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Export Summary")
print("=" * 80)

import glob
pte_files = glob.glob("exported_pte/*.pte")
if pte_files:
    print(f"\nSuccessfully exported {len(pte_files)} parts:")
    for pte_file in sorted(pte_files):
        size = os.path.getsize(pte_file) / (1024 * 1024)  # MB
        print(f"  - {pte_file} ({size:.2f} MB)")

    print("\nNote: Part 2 (Alignment Builder) stays in Python (data-dependent)")
    print("\nNext step: Create ExecuTorch runtime pipeline to use these .pte files")
else:
    print("\n✗ No .pte files were created. Check errors above.")
