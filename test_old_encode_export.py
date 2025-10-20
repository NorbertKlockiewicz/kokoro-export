"""
Test the old encode export from step2_encode.pte
See if that one produces NaNs or works
"""

import torch
from huggingface_hub import hf_hub_download
from executorch.runtime import Runtime
from pathlib import Path

print("Testing old encode export")
print("=" * 80)

from kokoro import KModel

model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

# Generate inputs
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

    # Run encode in PyTorch
    F0_conv = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N_conv = model.decoder.N_conv(N_pred.unsqueeze(1))
    x_cat = torch.cat([asr, F0_conv, N_conv], axis=1)
    x_enc_pt = model.decoder.encode(x_cat, ref_s)

print(f"PyTorch encode: range=[{x_enc_pt.min():.3f}, {x_enc_pt.max():.3f}], NaN={x_enc_pt.isnan().any()}")

# Test old export
runtime = Runtime.get()

print("\nTesting step2_encode.pte:")
try:
    program = runtime.load_program(Path("exported_pte/debug/step2_encode.pte"))
    method = program.load_method("forward")
    outputs = method.execute((asr, F0_conv, N_conv, ref_s))
    x_enc_et = outputs[0]

    print(f"ExecuTorch: range=[{x_enc_et.min():.3f}, {x_enc_et.max():.3f}], NaN={x_enc_et.isnan().any()}")

    if x_enc_et.isnan().any():
        print("  ⚠ Old export also produces NaNs")
    else:
        print("  ✓ Old export WORKS!")
        diff = torch.abs(x_enc_pt - x_enc_et).max().item()
        print(f"  Max diff from PyTorch: {diff:.6f}")

except Exception as e:
    print(f"  ⚠ Error: {e}")

print("\nTesting new decoder_encode_single.pte:")
try:
    program = runtime.load_program(Path("exported_pte/decoder_encode_single.pte"))
    method = program.load_method("forward")
    outputs = method.execute((asr, F0_pred, N_pred, ref_s))
    x_enc_et2 = outputs[0]

    print(f"ExecuTorch: range=[{x_enc_et2.min():.3f}, {x_enc_et2.max():.3f}], NaN={x_enc_et2.isnan().any()}")

    if x_enc_et2.isnan().any():
        print("  ⚠ New export produces NaNs")
    else:
        print("  ✓ New export WORKS!")

except Exception as e:
    print(f"  ⚠ Error: {e}")

print("\n" + "=" * 80)
print("Compare the two exports to see what's different")
print("=" * 80)
