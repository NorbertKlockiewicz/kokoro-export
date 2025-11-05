"""
Fix torch.rsqrt(torch.tensor(2)) issue and re-export decoder
Replace dynamic tensor creation with constant
"""

import torch
from torch import nn
from torch.export import export
import os
from huggingface_hub import hf_hub_download

print("Fixing rsqrt issue and exporting decoder")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte", exist_ok=True)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

# Fix all AdainResBlk1d blocks - replace rsqrt(2) with constant
def fix_rsqrt_recursive(module):
    """Replace torch.rsqrt(torch.tensor(2)) in forward with constant"""
    from kokoro.istftnet import AdainResBlk1d
    count = 0

    for name, child in module.named_children():
        if isinstance(child, AdainResBlk1d):
            # Monkey-patch the forward method
            original_forward = child.forward

            def make_fixed_forward(orig_fwd):
                def fixed_forward(x, s):
                    out = orig_fwd.__self__._residual(x, s)
                    # Use constant instead of torch.rsqrt(torch.tensor(2))
                    out = (out + orig_fwd.__self__._shortcut(x)) * 0.7071067811865476  # 1/sqrt(2)
                    return out
                return fixed_forward

            child.forward = make_fixed_forward(original_forward)
            count += 1
        else:
            count += fix_rsqrt_recursive(child)
    return count

fixed = fix_rsqrt_recursive(model.decoder)
print(f"✓ Fixed {fixed} AdainResBlk1d blocks")

# Generate real inputs
voice_file = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/af_bella.pt")
voice_style = torch.load(voice_file, weights_only=True)
if voice_style.dim() == 3:
    voice_style = voice_style.mean(dim=0)
if voice_style.dim() == 2 and voice_style.shape[0] != 1:
    if voice_style.shape[1] == 256:
        voice_style = voice_style[0:1]

phonemes = "həlˈoʊ wˈɝld"
input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
TARGET_TOKENS = 16
while len(input_ids) < (TARGET_TOKENS - 2):
    input_ids.append(0)
input_ids = input_ids[:(TARGET_TOKENS - 2)]
input_ids = torch.LongTensor([[0, *input_ids, 0]])

print("Generating real decoder inputs...")
with torch.no_grad():
    input_lengths = torch.tensor(input_ids.shape[-1])
    text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
    bert_dur = model.bert(input_ids, attention_mask=text_mask.int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = voice_style[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, ~text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1)
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    device = input_ids.device
    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=device), pred_dur)
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln_trg
    x = en.transpose(-1, -2)
    x, _ = model.predictor.shared(x)
    F0 = x.transpose(-1, -2)
    for block in model.predictor.F0:
        F0 = block(F0, s)
    F0_pred = model.predictor.F0_proj(F0).squeeze(1)
    N = x.transpose(-1, -2)
    for block in model.predictor.N:
        N = block(N, s)
    N_pred = model.predictor.N_proj(N).squeeze(1)
    t_en = model.text_encoder(input_ids, input_lengths, ~text_mask)
    asr = t_en @ pred_aln_trg
    ref_s = voice_style[:, :128]

print(f"✓ Generated real inputs")

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

# Test PyTorch
with torch.no_grad():
    output_pytorch = decoder(asr, F0_pred, N_pred, ref_s)
    print(f"\nPyTorch output:")
    print(f"  Shape: {output_pytorch.shape}")
    print(f"  Range: [{output_pytorch.min():.3f}, {output_pytorch.max():.3f}]")
    print(f"  Contains NaN: {output_pytorch.isnan().any()}")

    # Export
    print("\nExporting to ExecuTorch...")
    exported = export(decoder, (asr, F0_pred, N_pred, ref_s), strict=False)
    output_exported = exported.module()(asr, F0_pred, N_pred, ref_s)
    print(f"Exported: range=[{output_exported.min():.3f}, {output_exported.max():.3f}] NaN={output_exported.isnan().any()}")
    print("✓ Torch export successful")

    from executorch.exir import to_edge
    edge_program = to_edge(exported).to_executorch()
    print("✓ Converted to ExecuTorch")

    pte_path = "exported_pte/decoder_fixed.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Saved to {pte_path} ({size_mb:.2f} MB)")

    # Test ExecuTorch
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((asr, F0_pred, N_pred, ref_s))
        output_et = outputs[0]

        print(f"\nExecuTorch output:")
        print(f"  Shape: {output_et.shape}")
        print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
        print(f"  Contains NaN: {output_et.isnan().any()}")

        if output_et.isnan().any():
            nan_pct = 100 * output_et.isnan().sum().item() / output_et.numel()
            print(f"\n⚠ Still has NaNs ({nan_pct:.2f}%) - rsqrt was not the only issue")
        else:
            diff = torch.abs(output_pytorch - output_et).max().item()
            print(f"\n✓ FIXED! No NaNs!")
            print(f"  Max diff from PyTorch: {diff:.6f}")
    except RuntimeError as e:
        print(f"\n⚠ Runtime error: {e}")

print("\n" + "=" * 80)
