"""
Export decoder parts with SINGLE outputs only
Hypothesis: XNNPACK + AdainResBlk1d + multiple outputs = NaNs
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download
import os

print("Exporting Decoder Parts (Single Outputs Only)")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte", exist_ok=True)

model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

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

print("✓ Generated example inputs\n")

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# ============================================================================
# Export each component separately with SINGLE output
# ============================================================================

# Part 1a: Encode ONLY
print("=" * 80)
print("PART 1a: Encode (single output)")
print("=" * 80)

class EncodeOnly(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode

    def forward(self, asr, F0_pred, N_pred, s):
        F0 = self.F0_conv(F0_pred.unsqueeze(1))
        N = self.N_conv(N_pred.unsqueeze(1))
        x = torch.cat([asr, F0, N], axis=1)
        x_enc = self.encode(x, s)
        return x_enc  # SINGLE output

encode_only = EncodeOnly(model.decoder).eval()

with torch.no_grad():
    x_enc = encode_only(asr, F0_pred, N_pred, ref_s)
    print(f"Test output: x_enc={x_enc.shape}, range=[{x_enc.min():.3f}, {x_enc.max():.3f}]")

    exported = export(encode_only, (asr, F0_pred, N_pred, ref_s), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    pte_path = "exported_pte/decoder_encode_single.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Exported to {pte_path} ({size_mb:.2f} MB)")

    # Test it
    from executorch.runtime import Runtime
    runtime = Runtime.get()
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((asr, F0_pred, N_pred, ref_s))
    x_enc_et = outputs[0]
    print(f"ExecuTorch: range=[{x_enc_et.min():.3f}, {x_enc_et.max():.3f}], NaN={x_enc_et.isnan().any()}")

    if x_enc_et.isnan().any():
        print("  ⚠ Still produces NaNs with single output!")
    else:
        print("  ✓ WORKS with single output!")

# Part 1b: F0/N conv (separate, for auxiliary outputs)
print("\n" + "=" * 80)
print("PART 1b: F0/N Conv (auxiliary)")
print("=" * 80)

class ConvsOnly(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, F0_pred, N_pred):
        F0 = self.F0_conv(F0_pred.unsqueeze(1))
        N = self.N_conv(N_pred.unsqueeze(1))
        return F0, N  # Two outputs but no AdainResBlk1d

convs_only = ConvsOnly(model.decoder).eval()

with torch.no_grad():
    F0, N = convs_only(F0_pred, N_pred)
    print(f"Test output: F0={F0.shape}, N={N.shape}")

    exported = export(convs_only, (F0_pred, N_pred), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    pte_path = "exported_pte/decoder_convs.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

# Part 1c: ASR residual
print("\n" + "=" * 80)
print("PART 1c: ASR residual (auxiliary)")
print("=" * 80)

class AsrResOnly(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.asr_res = decoder.asr_res

    def forward(self, asr):
        return self.asr_res(asr)

asr_res_only = AsrResOnly(model.decoder).eval()

with torch.no_grad():
    asr_res = asr_res_only(asr)
    print(f"Test output: asr_res={asr_res.shape}")

    exported = export(asr_res_only, (asr,), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    pte_path = "exported_pte/decoder_asr_res.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

# Part 2: Decode blocks (single output)
print("\n" + "=" * 80)
print("PART 2: Decode Blocks (single output)")
print("=" * 80)

F0_conv = model.decoder.F0_conv(F0_pred.unsqueeze(1))
N_conv = model.decoder.N_conv(N_pred.unsqueeze(1))
asr_res = model.decoder.asr_res(asr)

class DecodeBlocksSingle(nn.Module):
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
        return x  # SINGLE output

decode_single = DecodeBlocksSingle(model.decoder).eval()

with torch.no_grad():
    x_dec = decode_single(x_enc, asr_res, F0_conv, N_conv, ref_s)
    print(f"Test output: x_dec={x_dec.shape}, range=[{x_dec.min():.3f}, {x_dec.max():.3f}]")

    exported = export(decode_single, (x_enc, asr_res, F0_conv, N_conv, ref_s), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    pte_path = "exported_pte/decoder_decode_single.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Exported to {pte_path} ({size_mb:.2f} MB)")

    # Test it
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((x_enc, asr_res, F0_conv, N_conv, ref_s))
    x_dec_et = outputs[0]
    print(f"ExecuTorch: range=[{x_dec_et.min():.3f}, {x_dec_et.max():.3f}], NaN={x_dec_et.isnan().any()}")

    if x_dec_et.isnan().any():
        print("  ⚠ Still produces NaNs with single output!")
    else:
        print("  ✓ WORKS with single output!")

# Part 3a: Generator upsampling (single output)
print("\n" + "=" * 80)
print("PART 3a: Generator Upsampling (single output)")
print("=" * 80)

class GeneratorUpsampleSingle(nn.Module):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, x, s, f0):
        # Harmonic source
        f0_up = self.gen.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, noi_source, uv = self.gen.m_source(f0_up)
        har_source_t = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.gen.stft.transform(har_source_t)
        har = torch.cat([har_spec, har_phase], dim=1)

        # Upsampling stages
        for i in range(self.gen.num_upsamples):
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
            x_source = self.gen.noise_convs[i](har)
            x_source = self.gen.noise_res[i](x_source, s)
            x = self.gen.ups[i](x)
            if i == self.gen.num_upsamples - 1:
                x = self.gen.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.gen.num_kernels):
                if xs is None:
                    xs = self.gen.resblocks[i * self.gen.num_kernels + j](x, s)
                else:
                    xs += self.gen.resblocks[i * self.gen.num_kernels + j](x, s)
            x = xs / self.gen.num_kernels

        return x  # SINGLE output

gen_upsample_single = GeneratorUpsampleSingle(model.decoder.generator).eval()

with torch.no_grad():
    x_upsampled = gen_upsample_single(x_dec, ref_s, F0_pred)
    print(f"Test output: x_upsampled={x_upsampled.shape}, range=[{x_upsampled.min():.3f}, {x_upsampled.max():.3f}]")

    exported = export(gen_upsample_single, (x_dec, ref_s, F0_pred), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    pte_path = "exported_pte/decoder_gen_upsample_single.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    size_mb = os.path.getsize(pte_path) / (1024 * 1024)
    print(f"✓ Exported to {pte_path} ({size_mb:.2f} MB)")

    # Test it
    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((x_dec, ref_s, F0_pred))
    x_upsampled_et = outputs[0]
    print(f"ExecuTorch: range=[{x_upsampled_et.min():.3f}, {x_upsampled_et.max():.3f}], NaN={x_upsampled_et.isnan().any()}")

    if x_upsampled_et.isnan().any():
        print("  ⚠ Part 3a produces NaNs!")
    else:
        print("  ✓ Part 3a WORKS!")

print("\n" + "=" * 80)
print("HYPOTHESIS TEST:")
print("If single-output exports work, then XNNPACK has a bug with")
print("multi-output functions containing AdainResBlk1d")
print("=" * 80)
