"""
Progressive testing: Build up the decoder piece by piece
Find the exact point where NaNs start appearing
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download
import os

print("Progressive Decoder Testing")
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

print("✓ Generated REAL inputs\n")

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.runtime import Runtime

runtime = Runtime.get()

def test_export(name, module_class, inputs, description):
    """Helper to test export and execution"""
    print("=" * 80)
    print(f"TEST: {name}")
    print(f"      {description}")
    print("=" * 80)

    wrapper = module_class(model.decoder).eval()

    with torch.no_grad():
        # PyTorch
        output_pt = wrapper(*inputs)
        print(f"PyTorch: shape={output_pt.shape}, range=[{output_pt.min():.3f}, {output_pt.max():.3f}], NaN={output_pt.isnan().any()}")

        # Export
        exported = export(wrapper, inputs, strict=False)
        output_exp = exported.module()(*inputs)
        print(f"Exported: range=[{output_exp.min():.3f}, {output_exp.max():.3f}], NaN={output_exp.isnan().any()}")

        # Lower with XNNPACK
        try:
            edge_program = to_edge_transform_and_lower(
                exported,
                partitioner=[XnnpackPartitioner()],
            ).to_executorch()

            pte_path = f"exported_pte/debug/test_{name}.pte"
            with open(pte_path, "wb") as f:
                f.write(edge_program.buffer)

            # Test runtime
            program = runtime.load_program(pte_path)
            method = program.load_method("forward")
            outputs = method.execute(inputs)
            output_et = outputs[0]

            print(f"ExecuTorch: range=[{output_et.min():.3f}, {output_et.max():.3f}], NaN={output_et.isnan().any()}")

            if output_et.isnan().any():
                nan_pct = 100 * output_et.isnan().sum().item() / output_et.numel()
                print(f"⚠ FAILS HERE! NaNs: {nan_pct:.2f}%")
                return False
            else:
                print(f"✓ PASS")
                return True

        except Exception as e:
            print(f"⚠ Error: {str(e)[:100]}")
            return False

# ============================================================================
# Progressive tests
# ============================================================================

# Test 1: Just convolutions
class Test1_Convs(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, F0_curve, N):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        return F0

test_export("1_convs", Test1_Convs, (F0_pred, N_pred), "F0_conv + N_conv only")

# Test 2: Convs + Cat
class Test2_ConvsCat(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, asr, F0_curve, N):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        return x

test_export("2_convs_cat", Test2_ConvsCat, (asr, F0_pred, N_pred), "Convs + Cat")

# Test 3: Convs + Cat + Encode
class Test3_Encode(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        return self.encode(x, s)

test_export("3_encode", Test3_Encode, (asr, F0_pred, N_pred, ref_s), "Convs + Cat + Encode")

# Test 4: Add asr_res
class Test4_WithAsrRes(nn.Module):
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
        return x, asr_res

test_export("4_with_asr_res", Test4_WithAsrRes, (asr, F0_pred, N_pred, ref_s), "Encode + ASR residual")

# Test 5: Encode + First decode block
class Test5_FirstDecode(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.encode = decoder.encode
        self.asr_res = decoder.asr_res
        self.decode_0 = decoder.decode[0]

    def forward(self, asr, F0_curve, N, s):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N_out = self.N_conv(N.unsqueeze(1))
        x = torch.cat([asr, F0, N_out], axis=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        # First decode block
        x = torch.cat([x, asr_res, F0, N_out], axis=1)
        x = self.decode_0(x, s)
        return x

test_export("5_first_decode", Test5_FirstDecode, (asr, F0_pred, N_pred, ref_s), "Encode + First decode block")

# Test 6: All decode blocks (no generator)
class Test6_AllDecode(nn.Module):
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

test_export("6_all_decode", Test6_AllDecode, (asr, F0_pred, N_pred, ref_s), "All encode/decode blocks (no generator)")

print("\n" + "=" * 80)
print("SUMMARY: Check which test failed first to isolate the problem")
print("=" * 80)
