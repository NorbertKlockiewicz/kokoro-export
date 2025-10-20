"""
Test if conv_post is the culprit
Since stage 0+1 works but stage 0+1+conv_post fails
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download
import os

print("Testing conv_post operation")
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

    # Run through full pipeline up to conv_post
    F0_conv = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N_conv = model.decoder.N_conv(N_pred.unsqueeze(1))
    x_cat = torch.cat([asr, F0_conv, N_conv], axis=1)
    x_enc = model.decoder.encode(x_cat, ref_s)
    asr_res = model.decoder.asr_res(asr)
    x = x_enc
    res = True
    for block in model.decoder.decode:
        if res:
            x = torch.cat([x, asr_res, F0_conv, N_conv], axis=1)
        x = block(x, ref_s)
        if block.upsample_type != "none":
            res = False
    x_dec = x

    # Generate harmonic source
    gen = model.decoder.generator
    f0_up = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
    har_source, noi_source, uv = gen.m_source(f0_up)
    har_source_t = har_source.transpose(1, 2).squeeze(1)
    har_spec, har_phase = gen.stft.transform(har_source_t)
    har = torch.cat([har_spec, har_phase], dim=1)

    # Run through all upsampling stages
    x = x_dec
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

    # x is now the output after all upsampling
    x_before_post = x.clone()

print(f"✓ Generated input before conv_post: shape={x_before_post.shape}\n")

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

runtime = Runtime.get()

# ============================================================================
# TEST 1: Just LeakyReLU + conv_post
# ============================================================================
print("=" * 80)
print("TEST 1: LeakyReLU + conv_post only")
print("=" * 80)

class ConvPostOnly(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.conv_post = gen.conv_post

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(x)
        x = self.conv_post(x)
        return x

conv_post_test = ConvPostOnly(gen).eval()

with torch.no_grad():
    output_pt = conv_post_test(x_before_post)
    print(f"PyTorch: shape={output_pt.shape}, range=[{output_pt.min():.3f}, {output_pt.max():.3f}], NaN={output_pt.isnan().any()}")

    exported = export(conv_post_test, (x_before_post,), strict=False)
    output_exp = exported.module()(x_before_post)
    print(f"Exported: range=[{output_exp.min():.3f}, {output_exp.max():.3f}], NaN={output_exp.isnan().any()}")

    try:
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        pte_path = "exported_pte/debug/conv_post_only.pte"
        with open(pte_path, "wb") as f:
            f.write(edge_program.buffer)

        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x_before_post,))
        output_et = outputs[0]

        print(f"ExecuTorch: range=[{output_et.min():.3f}, {output_et.max():.3f}], NaN={output_et.isnan().any()}")

        if output_et.isnan().any():
            print(f"⚠ conv_post itself produces NaNs!")
        else:
            print(f"✓ conv_post alone works fine")
            print(f"\n  → NaNs must come from the COMBINATION of all stages + conv_post")
            print(f"  → This suggests numerical accumulation or precision loss in XNNPACK")

    except Exception as e:
        print(f"⚠ Error: {str(e)[:100]}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If conv_post alone works, then XNNPACK has numerical issues")
print("when processing the full generator pipeline sequentially.")
print("=" * 80)
