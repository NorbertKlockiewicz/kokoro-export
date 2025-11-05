"""
Test split decoder parts 2, 3a, 3b
Skip Part 1 - use PyTorch encode outputs as inputs
"""

import torch
from huggingface_hub import hf_hub_download
from executorch.runtime import Runtime
from pathlib import Path

print("Testing Split Decoder (skipping Part 1)")
print("=" * 80)

from kokoro import KModel

# Load PyTorch model
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

    # Run Part 1 with PyTorch to get intermediate values
    F0_conv = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N_conv = model.decoder.N_conv(N_pred.unsqueeze(1))
    x_cat = torch.cat([asr, F0_conv, N_conv], axis=1)
    x_enc = model.decoder.encode(x_cat, ref_s)
    asr_res = model.decoder.asr_res(asr)

print(f"✓ Generated PyTorch outputs from Part 1:\n")
print(f"  x_enc: {x_enc.shape}, range=[{x_enc.min():.3f}, {x_enc.max():.3f}], NaN={x_enc.isnan().any()}")
print(f"  F0_conv: {F0_conv.shape}, range=[{F0_conv.min():.3f}, {F0_conv.max():.3f}], NaN={F0_conv.isnan().any()}")
print(f"  N_conv: {N_conv.shape}, range=[{N_conv.min():.3f}, {N_conv.max():.3f}], NaN={N_conv.isnan().any()}")
print(f"  asr_res: {asr_res.shape}, range=[{asr_res.min():.3f}, {asr_res.max():.3f}], NaN={asr_res.isnan().any()}\n")

# PyTorch reference (full decoder)
with torch.no_grad():
    audio_pytorch = model.decoder(asr, F0_pred, N_pred, ref_s).squeeze()
    print(f"PyTorch decoder full output:")
    print(f"  Shape: {audio_pytorch.shape}")
    print(f"  Range: [{audio_pytorch.min():.3f}, {audio_pytorch.max():.3f}]")
    print(f"  NaN: {audio_pytorch.isnan().any()}\n")

# Load ExecuTorch models
runtime = Runtime.get()

print("=" * 80)
print("Running Split Decoder Pipeline (using PyTorch Part 1 outputs)")
print("=" * 80)

# Part 2: Decode blocks (using PyTorch Part 1 outputs)
print("\nPart 2: Decode blocks...")
try:
    part2 = runtime.load_program(Path("exported_pte/decoder_part2_decode.pte"))
    method2 = part2.load_method("forward")
    outputs2 = method2.execute((x_enc, asr_res, F0_conv, N_conv, ref_s))
    x_dec = outputs2[0]
    print(f"  x_dec: {x_dec.shape}, range=[{x_dec.min():.3f}, {x_dec.max():.3f}], NaN={x_dec.isnan().any()}")

    if x_dec.isnan().any():
        print("  ⚠ Part 2 produces NaNs even with PyTorch inputs!")
    else:
        print("  ✓ Part 2 OK")
except Exception as e:
    print(f"  ⚠ Part 2 error: {e}")
    x_dec = None

# Part 3a: Generator upsampling
if x_dec is not None and not x_dec.isnan().any():
    print("\nPart 3a: Generator upsampling...")
    try:
        part3a = runtime.load_program(Path("exported_pte/decoder_part3a_gen_upsample.pte"))
        method3a = part3a.load_method("forward")
        outputs3a = method3a.execute((x_dec, ref_s, F0_pred))
        x_upsampled = outputs3a[0]
        print(f"  x_upsampled: {x_upsampled.shape}, range=[{x_upsampled.min():.3f}, {x_upsampled.max():.3f}], NaN={x_upsampled.isnan().any()}")

        if x_upsampled.isnan().any():
            print("  ⚠ Part 3a produces NaNs!")
        else:
            print("  ✓ Part 3a OK")
    except Exception as e:
        print(f"  ⚠ Part 3a error: {e}")
        x_upsampled = None
else:
    print("\nSkipping Part 3a (Part 2 failed)")
    # Use PyTorch output
    with torch.no_grad():
        x = x_enc
        res = True
        for block in model.decoder.decode:
            if res:
                x = torch.cat([x, asr_res, F0_conv, N_conv], axis=1)
            x = block(x, ref_s)
            if block.upsample_type != "none":
                res = False
        x_dec_pt = x

        gen = model.decoder.generator
        f0_up = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        har_source, noi_source, uv = gen.m_source(f0_up)
        har_source_t = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source_t)
        har = torch.cat([har_spec, har_phase], dim=1)

        x = x_dec_pt
        for i in range(gen.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = gen.noise_convs[i](har)
            x_source = gen.noise_res[i](x_source, ref_s)
            x = gen.ups[i](x)
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(gen.num_kernels):
                if xs is None:
                    xs = gen.resblocks[i * gen.num_kernels + j](x, ref_s)
                else:
                    xs += gen.resblocks[i * gen.num_kernels + j](x, ref_s)
            x = xs / gen.num_kernels
        x_upsampled = x

        print("\nUsing PyTorch Part 3a output:")
        print(f"  x_upsampled: {x_upsampled.shape}, range=[{x_upsampled.min():.3f}, {x_upsampled.max():.3f}], NaN={x_upsampled.isnan().any()}")

# Part 3b: Generator post + ISTFT
if x_upsampled is not None and not x_upsampled.isnan().any():
    print("\nPart 3b: Generator post + ISTFT...")
    try:
        part3b = runtime.load_program(Path("exported_pte/decoder_part3b_gen_post.pte"))
        method3b = part3b.load_method("forward")
        outputs3b = method3b.execute((x_upsampled,))
        audio_split = outputs3b[0]
        print(f"  audio: {audio_split.shape}, range=[{audio_split.min():.3f}, {audio_split.max():.3f}], NaN={audio_split.isnan().any()}")

        if audio_split.isnan().any():
            print("  ⚠ Part 3b produces NaNs!")
        else:
            print("  ✓ Part 3b OK")

            # Compare with PyTorch
            min_len = min(len(audio_split), len(audio_pytorch))
            split_trimmed = audio_split[:min_len]
            pytorch_trimmed = audio_pytorch[:min_len]

            mae = torch.mean(torch.abs(split_trimmed - pytorch_trimmed)).item()
            max_diff = torch.max(torch.abs(split_trimmed - pytorch_trimmed)).item()

            print(f"\nComparison with PyTorch:")
            print(f"  MAE: {mae:.6f}")
            print(f"  Max diff: {max_diff:.6f}")

    except Exception as e:
        print(f"  ⚠ Part 3b error: {e}")
else:
    print("\nSkipping Part 3b (previous part failed)")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If Part 2/3a/3b work with PyTorch Part 1 output,")
print("then the problem is ONLY in Part 1 (Encode)")
print("=" * 80)
