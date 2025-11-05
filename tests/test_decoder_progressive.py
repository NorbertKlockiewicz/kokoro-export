"""
Progressive decoder export - add operations one by one
Find exactly where NaNs start appearing
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


def test_module(name, module_class, inputs):
    wrapper = module_class(model.decoder).eval()
    with torch.no_grad():
        output_pt = wrapper(*inputs)
        exported = export(wrapper, inputs, strict=False)
        edge_program = to_edge_transform_and_lower(
            exported, partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        pte_path = f"exported_pte/debug/prog_{name}.pte"
        with open(pte_path, "wb") as f:
            f.write(edge_program.buffer)

        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute(inputs)
        output_et = outputs[0]
        print(output_et)
        has_nan = output_et.isnan().any()
        print(f"{name}: {'NaN' if has_nan else 'OK'}")
        if has_nan:
            return False
        return True


# Test 1: F0_conv only
class Test1(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv

    def forward(self, F0_curve):
        return self.F0_conv(F0_curve.unsqueeze(1))


if not test_module("1_F0_conv", Test1, (F0_pred,)):
    exit()


# Test 2: F0_conv + N_conv
class Test2(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, F0_curve, N):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        return N_out


if not test_module("2_F0_N_conv", Test2, (F0_pred, N_pred)):
    exit()


# Test 3: F0_conv + N_conv + cat
class Test3(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, asr, F0_curve, N):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1).contiguous()
        return x


if not test_module("3_cat", Test3, (asr, F0_pred, N_pred)):
    exit()


# Test 4: + encode
class Test4(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1).contiguous()
        x = self.encode(x, s)
        return x


if not test_module("4_encode", Test4, (asr, F0_pred, N_pred, ref_s)):
    exit()


# Test 5: + asr_res
class Test5(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode
        self.asr_res = decoder.asr_res

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        return x


if not test_module("5_asr_res", Test5, (asr, F0_pred, N_pred, ref_s)):
    exit()


# Test 6: + decode blocks
class Test6(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode
        self.asr_res = decoder.asr_res
        self.decode = decoder.decode

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_out], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        return x


if not test_module("6_decode", Test6, (asr, F0_pred, N_pred, ref_s)):
    exit()

print("All tests passed!")
