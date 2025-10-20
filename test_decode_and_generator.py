"""
Test decode blocks and generator separately
Now that we know encode works, isolate which part after encode causes NaNs
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download
import os

print("Testing Decode Blocks and Generator Separately")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte/debug", exist_ok=True)

model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

# Generate REAL inputs
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

    # Generate intermediate values by running through encode
    F0_conv = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N_conv = model.decoder.N_conv(N_pred.unsqueeze(1))
    x_cat = torch.cat([asr, F0_conv, N_conv], axis=1)
    x_enc = model.decoder.encode(x_cat, ref_s)
    asr_res = model.decoder.asr_res(asr)

print("✓ Generated inputs and ran through encode\n")

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

runtime = Runtime.get()

# ============================================================================
# TEST 1: Just the decode blocks (no generator)
# ============================================================================
print("=" * 80)
print("TEST 1: Decode blocks only (no generator)")
print("=" * 80)

class DecodeBlocksOnly(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decode = decoder.decode

    def forward(self, x_enc, asr_res, F0, N, s):
        x = x_enc
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return x

decode_only = DecodeBlocksOnly(model.decoder).eval()

with torch.no_grad():
    x_dec = decode_only(x_enc, asr_res, F0_conv, N_conv, ref_s)
    print(f"PyTorch: shape={x_dec.shape}, range=[{x_dec.min():.3f}, {x_dec.max():.3f}], NaN={x_dec.isnan().any()}")

    exported = export(decode_only, (x_enc, asr_res, F0_conv, N_conv, ref_s), strict=False)
    output_exp = exported.module()(x_enc, asr_res, F0_conv, N_conv, ref_s)
    print(f"Exported: range=[{output_exp.min():.3f}, {output_exp.max():.3f}], NaN={output_exp.isnan().any()}")

    try:
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        pte_path = "exported_pte/debug/decode_blocks_only.pte"
        with open(pte_path, "wb") as f:
            f.write(edge_program.buffer)
        print(f"✓ Exported to {pte_path}")

        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x_enc, asr_res, F0_conv, N_conv, ref_s))
        output_et = outputs[0]

        print(f"ExecuTorch: range=[{output_et.min():.3f}, {output_et.max():.3f}], NaN={output_et.isnan().any()}")

        if output_et.isnan().any():
            nan_pct = 100 * output_et.isnan().sum().item() / output_et.numel()
            print(f"⚠ Decode blocks produce NaNs ({nan_pct:.2f}%)")
        else:
            print(f"✓ Decode blocks work fine")

    except Exception as e:
        print(f"⚠ Error: {e}")

# ============================================================================
# TEST 2: Just the generator (using decode output)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Generator only (using decode output)")
print("=" * 80)

class GeneratorOnly(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.generator = decoder.generator

    def forward(self, x, s, F0_curve):
        return self.generator(x, s, F0_curve).squeeze()

generator_only = GeneratorOnly(model.decoder).eval()

with torch.no_grad():
    audio_gen = generator_only(x_dec, ref_s, F0_pred)
    print(f"PyTorch: shape={audio_gen.shape}, range=[{audio_gen.min():.3f}, {audio_gen.max():.3f}], NaN={audio_gen.isnan().any()}")

    exported = export(generator_only, (x_dec, ref_s, F0_pred), strict=False)
    output_exp = exported.module()(x_dec, ref_s, F0_pred)
    print(f"Exported: range=[{output_exp.min():.3f}, {output_exp.max():.3f}], NaN={output_exp.isnan().any()}")

    try:
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        pte_path = "exported_pte/debug/generator_only.pte"
        with open(pte_path, "wb") as f:
            f.write(edge_program.buffer)
        print(f"✓ Exported to {pte_path}")

        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x_dec, ref_s, F0_pred))
        output_et = outputs[0]

        print(f"ExecuTorch: range=[{output_et.min():.3f}, {output_et.max():.3f}], NaN={output_et.isnan().any()}")

        if output_et.isnan().any():
            nan_pct = 100 * output_et.isnan().sum().item() / output_et.numel()
            print(f"⚠ Generator produces NaNs ({nan_pct:.2f}%)")
        else:
            print(f"✓ Generator works fine")

    except Exception as e:
        print(f"⚠ Error: {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If both decode blocks AND generator work separately,")
print("then NaNs appear only when chaining ALL parts together.")
print("=" * 80)
