"""
Split Decoder into parts and test each separately
Find which part causes NaNs in ExecuTorch
"""

import torch
from torch import nn
from torch.export import export
import os
from huggingface_hub import hf_hub_download

print("Testing Decoder Parts Separately")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte/debug", exist_ok=True)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

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

print(f"\nDecoder inputs:")
print(f"  asr: {asr.shape}")
print(f"  F0_pred: {F0_pred.shape}")
print(f"  N_pred: {N_pred.shape}")
print(f"  ref_s: {ref_s.shape}")

# ============================================================================
# Part 1: Encode Block (first part of decoder)
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: Encode Block")
print("=" * 80)

class EncodePart(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode
        self.asr_res = decoder.asr_res

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        return x, F0, N, asr_res

encode_part = EncodePart(model.decoder).eval()

with torch.no_grad():
    x_enc, F0, N, asr_res = encode_part(asr, F0_pred, N_pred, ref_s)
    print(f"PyTorch: x_enc={x_enc.shape}, range=[{x_enc.min():.3f}, {x_enc.max():.3f}], NaN={x_enc.isnan().any()}")

    exported = export(encode_part, (asr, F0_pred, N_pred, ref_s), strict=False)
    from executorch.exir import to_edge
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/part1_encode.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((asr, F0_pred, N_pred, ref_s))
        x_enc_et = outputs[0]
        print(f"ExecuTorch: range=[{x_enc_et.min():.3f}, {x_enc_et.max():.3f}], NaN={x_enc_et.isnan().any()}")
        if x_enc_et.isnan().any():
            print("  ⚠ Part 1 (Encode) produces NaNs!")
        else:
            print("  ✓ Part 1 (Encode) OK")
    except Exception as e:
        print(f"  ⚠ Part 1 failed: {e}")

# ============================================================================
# Part 2: Decode Blocks (middle processing)
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: Decode Blocks")
print("=" * 80)

class DecodePart(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decode = decoder.decode

    def forward(self, x, asr_res, F0, N, s):
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return x

decode_part = DecodePart(model.decoder).eval()

with torch.no_grad():
    # Use outputs from encode part
    x_dec = decode_part(x_enc, asr_res, F0, N, ref_s)
    print(f"PyTorch: x_dec={x_dec.shape}, range=[{x_dec.min():.3f}, {x_dec.max():.3f}], NaN={x_dec.isnan().any()}")

    exported = export(decode_part, (x_enc, asr_res, F0, N, ref_s), strict=False)
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/part2_decode.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x_enc, asr_res, F0, N, ref_s))
        x_dec_et = outputs[0]
        print(f"ExecuTorch: range=[{x_dec_et.min():.3f}, {x_dec_et.max():.3f}], NaN={x_dec_et.isnan().any()}")
        if x_dec_et.isnan().any():
            print("  ⚠ Part 2 (Decode blocks) produces NaNs!")
        else:
            print("  ✓ Part 2 (Decode blocks) OK")
    except Exception as e:
        print(f"  ⚠ Part 2 failed: {e}")

# ============================================================================
# Part 3: Generator (final audio generation)
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: Generator")
print("=" * 80)

class GeneratorPart(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.generator = decoder.generator

    def forward(self, x, s, F0_curve):
        return self.generator(x, s, F0_curve).squeeze()

generator_part = GeneratorPart(model.decoder).eval()

with torch.no_grad():
    audio_gen = generator_part(x_dec, ref_s, F0_pred)
    print(f"PyTorch: audio={audio_gen.shape}, range=[{audio_gen.min():.3f}, {audio_gen.max():.3f}], NaN={audio_gen.isnan().any()}")

    exported = export(generator_part, (x_dec, ref_s, F0_pred), strict=False)
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/part3_generator.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x_dec, ref_s, F0_pred))
        audio_et = outputs[0]
        print(f"ExecuTorch: range=[{audio_et.min():.3f}, {audio_et.max():.3f}], NaN={audio_et.isnan().any()}")
        if audio_et.isnan().any():
            print("  ⚠ Part 3 (Generator) produces NaNs!")
        else:
            print("  ✓ Part 3 (Generator) OK")
    except Exception as e:
        print(f"  ⚠ Part 3 failed: {e}")

print("\n" + "=" * 80)
print("Test complete - check which part failed above")
print("=" * 80)
