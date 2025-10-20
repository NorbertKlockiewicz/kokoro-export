"""
Test: Export cat operations separately from encode block
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download
import os

from kokoro import KModel
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime
from kokoro.istftnet import AdainResBlk1d
from torch.nn.utils.parametrizations import weight_norm
from executorch.devtools import BundledProgram, Inspector
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite

os.makedirs("exported_pte/debug", exist_ok=True)

model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

voice_file = hf_hub_download(
    repo_id="hexgrad/Kokoro-82M", filename="voices/af_bella.pt"
)
voice_style = torch.load(voice_file, weights_only=True)
if voice_style.dim() == 3:
    voice_style = voice_style.mean(dim=0)
if voice_style.dim() == 2 and voice_style.shape[0] != 1:
    if voice_style.shape[1] == 256:
        voice_style = voice_style[0:1]

phonemes = "həlˈoʊ wˈɝld"
input_ids = list(
    filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes))
)
TARGET_TOKENS = 16
while len(input_ids) < (TARGET_TOKENS - 2):
    input_ids.append(0)
input_ids = input_ids[: (TARGET_TOKENS - 2)]
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
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=device), pred_dur
    )
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

runtime = Runtime.get()


# Part 1: Conv + Cat (first 3 steps combined)
class ConvCat(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, asr, F0_curve, N):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        return x, F0, N_out


conv_cat = ConvCat(model.decoder).eval()

with torch.no_grad():
    exported = export(conv_cat, (asr, F0_pred, N_pred), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    pte_path = "exported_pte/debug/conv_cat.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((asr, F0_pred, N_pred))
    x_cat_et = outputs
    print(f"Part 1 (Conv+Cat) ExecuTorch")


# Part 3: Encode
class EncodeAsrRes(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.encode = decoder.encode

    def forward(self, x, s):
        x = self.encode(x, s)
        return x


encode_asr = EncodeAsrRes(model.decoder).eval()

with torch.no_grad():
    exported = export(encode_asr, (x_cat_et[0], ref_s), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    pte_path = "exported_pte/debug/encode_asr.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((x_cat_et[0], ref_s))
    x_enc_et2 = outputs[0]

    print(
        f"Part 2 (Encode+AsrRes) ExecuTorch: range=[{x_enc_et2.min():.3f}, {x_enc_et2.max():.3f}], NaN={x_enc_et2.isnan().any()}"
    )

print("\nSummary: Testing encode with asr_res")


class DecoderPart3(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decode = decoder.decode
        self.asr_res = decoder.asr_res

    def forward(self, asr):
        asr_res = self.asr_res(asr)
        return asr_res


decode_asr_res = DecoderPart3(model.decoder).eval()

with torch.no_grad():
    exported = export(decode_asr_res, (asr,), strict=False)
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    pte_path = "exported_pte/debug/decode_asr_res.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((asr,))
    asr_res_et = outputs[0]

    print(
        f"Part 3 (Decode+AsrRes) ExecuTorch: range=[{asr_res_et.min():.3f}, {asr_res_et.max():.3f}], NaN={asr_res_et.isnan().any()}"
    )


class DecoderPart4(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decode = decoder.decode

    def forward(self, x, asr, F0, N, s):
        res = True

        if res:
            x = torch.cat([x, asr, F0, N], axis=1)
        x = self.decode[0](x, s)
        if self.decode[0].upsample_type != "none":
            res = False

        # if res:
        #     x = torch.cat([x, asr, F0, N], axis=1)
        # x = self.decode[1](x, s)
        # if self.decode[1].upsample_type != "none":
        #     res = False

        # if res:
        #     x = torch.cat([x, asr, F0, N], axis=1)
        # x = self.decode[2](x, s)
        # if self.decode[2].upsample_type != "none":
        #     res = False

        # if res:
        #     x = torch.cat([x, asr, F0, N], axis=1)
        # x = self.decode[3](x, s)
        # if self.decode[3].upsample_type != "none":
        #     res = False

        return x


import copy
from executorch.devtools import generate_etrecord
from executorch.devtools import BundledProgram

decode = DecoderPart4(model.decoder).eval()
with torch.no_grad():
    # Generate F0 and N_out from PyTorch for export (not from ExecuTorch outputs)
    F0_pt = model.decoder.F0_conv(F0_pred.unsqueeze(1))
    N_out_pt = model.decoder.N_conv(N_pred.unsqueeze(1))

    pt_output = decode(
        x_enc_et2, asr_res_et, F0_pt, N_out_pt, ref_s
    )  # dry run
    print(
        f"PT output: range=[{pt_output.min():.3f}, {pt_output.max():.3f}], NaN={pt_output.isnan().any()}"
    )
    exported = export(
        decode, (x_enc_et2, asr_res_et, F0_pt, N_out_pt, ref_s), strict=False
    )
    edge_program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    )
    edge_program_et = edge_program.to_executorch()
    pte_path = "exported_pte/debug/decode.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program_et.buffer)

    etrecord_path = "etrecord.bin"
    edge_program_manager_copy = copy.deepcopy(edge_program)
    generate_etrecord(etrecord_path, edge_program_manager_copy, edge_program_et)

    # Create bundled program with test case for debugging
    method_test_suites = [
        MethodTestSuite(
            method_name="forward",
            test_cases=[
                MethodTestCase(
                    inputs=(x_enc_et2, asr_res_et, F0_pt, N_out_pt, ref_s),
                    expected_outputs=pt_output,
                )
            ],
        )
    ]

    bundled_program = BundledProgram(edge_program_et, method_test_suites)
    bundled_path = "exported_pte/debug/decode_bundled.pte"
    with open(bundled_path, "wb") as f:
        f.write(bundled_program.get_program())

    bundled_etrecord_path = "exported_pte/debug/decode.etrecord"
    with open(bundled_etrecord_path, "wb") as f:
        f.write(bundled_program.get_etrecord())

    program = runtime.load_program(pte_path)
    method = program.load_method("forward")
    outputs = method.execute((x_enc_et2, asr_res_et, F0_pt, N_out_pt, ref_s))
    audio_et = outputs[0]
    print(
        f"Part 4 (Decode) ExecuTorch: range=[{audio_et.min():.3f}, {audio_et.max():.3f}], NaN={audio_et.isnan().any()}"
    )

    # Debug NaN detection - print intermediate values
    if audio_et.isnan().any():
        print("\n⚠️  NaN detected in output! Analyzing intermediate operations...")
        print(f"Bundled program saved to: {bundled_path}")
        print(f"ETRecord saved to: {bundled_etrecord_path}")
        print("\nTo debug with Inspector after running with ETDump:")
        print(
            "  inspector = Inspector(etdump_path='decode.etdp', etrecord='exported_pte/debug/decode.etrecord')"
        )
        print("  for event_block in inspector.event_blocks:")
        print("    for event in event_block.events:")
        print("      # Check event outputs for NaNs")
