"""
Test the split decoder (4 parts) to verify no NaNs
"""

import torch
from huggingface_hub import hf_hub_download
from executorch.runtime import Runtime
from pathlib import Path

print("Testing Split Decoder (4 parts)")
print("=" * 80)

from kokoro import KModel

# Load PyTorch model for comparison
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

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

print(f"✓ Generated inputs\n")
print(f"  asr: {asr.shape}")
print(f"  F0_pred: {F0_pred.shape}")
print(f"  N_pred: {N_pred.shape}")
print(f"  ref_s: {ref_s.shape}\n")

# PyTorch reference
with torch.no_grad():
    audio_pytorch = model.decoder(asr, F0_pred, N_pred, ref_s).squeeze()
    print(f"PyTorch decoder output:")
    print(f"  Shape: {audio_pytorch.shape}")
    print(f"  Range: [{audio_pytorch.min():.3f}, {audio_pytorch.max():.3f}]")
    print(f"  NaN: {audio_pytorch.isnan().any()}\n")

# Load ExecuTorch models
runtime = Runtime.get()

print("=" * 80)
print("Running Split Decoder Pipeline")
print("=" * 80)

# Part 1: Encode
print("\nPart 1: Encode...")
part1 = runtime.load_program(Path("exported_pte/decoder_part1_encode.pte"))
method1 = part1.load_method("forward")
outputs1 = method1.execute((asr, F0_pred, N_pred, ref_s))
x_enc, F0, N, asr_res = outputs1[0], outputs1[1], outputs1[2], outputs1[3]
print(f"  x_enc: {x_enc.shape}, range=[{x_enc.min():.3f}, {x_enc.max():.3f}], NaN={x_enc.isnan().any()}")
print(f"  F0: {F0.shape}, range=[{F0.min():.3f}, {F0.max():.3f}], NaN={F0.isnan().any()}")
print(f"  N: {N.shape}, range=[{N.min():.3f}, {N.max():.3f}], NaN={N.isnan().any()}")
print(f"  asr_res: {asr_res.shape}, range=[{asr_res.min():.3f}, {asr_res.max():.3f}], NaN={asr_res.isnan().any()}")

if x_enc.isnan().any():
    print("  ⚠ Part 1 produces NaNs!")
else:
    print("  ✓ Part 1 OK")

# Part 2: Decode blocks
print("\nPart 2: Decode blocks...")
part2 = runtime.load_program(Path("exported_pte/decoder_part2_decode.pte"))
method2 = part2.load_method("forward")
outputs2 = method2.execute((x_enc, asr_res, F0, N, ref_s))
x_dec = outputs2[0]
print(f"  x_dec: {x_dec.shape}, range=[{x_dec.min():.3f}, {x_dec.max():.3f}], NaN={x_dec.isnan().any()}")

if x_dec.isnan().any():
    print("  ⚠ Part 2 produces NaNs!")
else:
    print("  ✓ Part 2 OK")

# Part 3a: Generator upsampling
print("\nPart 3a: Generator upsampling...")
part3a = runtime.load_program(Path("exported_pte/decoder_part3a_gen_upsample.pte"))
method3a = part3a.load_method("forward")
outputs3a = method3a.execute((x_dec, ref_s, F0_pred))
x_upsampled = outputs3a[0]
print(f"  x_upsampled: {x_upsampled.shape}, range=[{x_upsampled.min():.3f}, {x_upsampled.max():.3f}], NaN={x_upsampled.isnan().any()}")

if x_upsampled.isnan().any():
    print("  ⚠ Part 3a produces NaNs!")
else:
    print("  ✓ Part 3a OK")

# Part 3b: Generator post + ISTFT
print("\nPart 3b: Generator post + ISTFT...")
part3b = runtime.load_program(Path("exported_pte/decoder_part3b_gen_post.pte"))
method3b = part3b.load_method("forward")
outputs3b = method3b.execute((x_upsampled,))
audio_split = outputs3b[0]
print(f"  audio: {audio_split.shape}, range=[{audio_split.min():.3f}, {audio_split.max():.3f}], NaN={audio_split.isnan().any()}")

if audio_split.isnan().any():
    print("  ⚠ Part 3b produces NaNs!")
else:
    print("  ✓ Part 3b OK")

# Final comparison
print("\n" + "=" * 80)
print("RESULTS:")
print("=" * 80)

if audio_split.isnan().any():
    nan_pct = 100 * audio_split.isnan().sum().item() / audio_split.numel()
    print(f"⚠ Split decoder still produces NaNs ({nan_pct:.2f}%)")
else:
    # Compare with PyTorch
    min_len = min(len(audio_split), len(audio_pytorch))
    split_trimmed = audio_split[:min_len]
    pytorch_trimmed = audio_pytorch[:min_len]

    mse = torch.mean((split_trimmed - pytorch_trimmed) ** 2).item()
    mae = torch.mean(torch.abs(split_trimmed - pytorch_trimmed)).item()
    max_diff = torch.max(torch.abs(split_trimmed - pytorch_trimmed)).item()

    print(f"✓ SUCCESS! Split decoder produces valid audio")
    print(f"\nComparison with PyTorch:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max diff: {max_diff:.6f}")

    # Save audio
    import torchaudio
    torchaudio.save("hello_split_decoder.wav", audio_split.unsqueeze(0), 24000)
    print(f"\n✓ Saved to hello_split_decoder.wav")

print("\n" + "=" * 80)
