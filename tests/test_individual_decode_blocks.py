"""
Test each of the 4 decode blocks individually
Find which specific block produces NaNs
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download
import os

print("Testing Individual Decode Blocks")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte/debug", exist_ok=True)

model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

# Generate REAL inputs (abbreviated)
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

    # Run through encode to get starting point
    F0_conv = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N_conv = model.decoder.N_conv(N_pred.unsqueeze(1))
    x_cat = torch.cat([asr, F0_conv, N_conv], axis=1)
    x_enc = model.decoder.encode(x_cat, ref_s)
    asr_res = model.decoder.asr_res(asr)

print("✓ Generated inputs\n")

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

runtime = Runtime.get()

# Test each decode block individually
print("Decode block structure:")
print("  Block 0: AdainResBlk1d(1024+2+64, 1024, no upsample)")
print("  Block 1: AdainResBlk1d(1024+2+64, 1024, no upsample)")
print("  Block 2: AdainResBlk1d(1024+2+64, 1024, no upsample)")
print("  Block 3: AdainResBlk1d(1024+2+64, 512, WITH upsample)\n")

# Simulate the decode loop to get correct input shapes for each block
x = x_enc
res = True
block_inputs = []

for i, block in enumerate(model.decoder.decode):
    if res:
        x_input = torch.cat([x, asr_res, F0_conv, N_conv], axis=1)
        block_inputs.append((x_input.clone(), ref_s.clone()))
    else:
        block_inputs.append((x.clone(), ref_s.clone()))

    x = block(x_input if res else x, ref_s)
    if block.upsample_type != "none":
        res = False

# Now test each block individually
for i, block in enumerate(model.decoder.decode):
    print("=" * 80)
    print(f"TEST: Decode Block {i}")
    print("=" * 80)

    x_in, s_in = block_inputs[i]

    class BlockWrapper(nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.block = blk

        def forward(self, x, s):
            return self.block(x, s)

    wrapper = BlockWrapper(block).eval()

    with torch.no_grad():
        output_pt = wrapper(x_in, s_in)
        print(f"PyTorch: shape={output_pt.shape}, range=[{output_pt.min():.3f}, {output_pt.max():.3f}], NaN={output_pt.isnan().any()}")

        exported = export(wrapper, (x_in, s_in), strict=False)
        output_exp = exported.module()(x_in, s_in)
        print(f"Exported: range=[{output_exp.min():.3f}, {output_exp.max():.3f}], NaN={output_exp.isnan().any()}")

        try:
            edge_program = to_edge_transform_and_lower(
                exported,
                partitioner=[XnnpackPartitioner()],
            ).to_executorch()

            pte_path = f"exported_pte/debug/decode_block_{i}.pte"
            with open(pte_path, "wb") as f:
                f.write(edge_program.buffer)

            program = runtime.load_program(pte_path)
            method = program.load_method("forward")
            outputs = method.execute((x_in, s_in))
            output_et = outputs[0]

            print(f"ExecuTorch: range=[{output_et.min():.3f}, {output_et.max():.3f}], NaN={output_et.isnan().any()}")

            if output_et.isnan().any():
                nan_pct = 100 * output_et.isnan().sum().item() / output_et.numel()
                print(f"⚠ Block {i} FAILS with {nan_pct:.2f}% NaNs")
            else:
                print(f"✓ Block {i} OK")

        except Exception as e:
            print(f"⚠ Error: {str(e)[:100]}")

print("\n" + "=" * 80)
print("Check which decode block(s) produce NaNs above")
print("=" * 80)
