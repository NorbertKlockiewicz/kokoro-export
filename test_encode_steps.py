"""
Test Encode block step-by-step to find where NaNs appear
"""

import torch
from torch import nn
from torch.export import export
from huggingface_hub import hf_hub_download

print("Testing Encode Block Step-by-Step")
print("=" * 80)

from kokoro import KModel
import os

os.makedirs("exported_pte/debug", exist_ok=True)

# Load model and generate real inputs
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

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

print("✓ Generated real inputs")

# ============================================================================
# Step 1: F0_conv and N_conv
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: F0_conv + N_conv")
print("=" * 80)

class ConvStep(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv

    def forward(self, F0_curve, N):
        F0 = self.F0_conv(F0_curve.unsqueeze(1))
        N = self.N_conv(N.unsqueeze(1))
        return F0, N

conv_step = ConvStep(model.decoder).eval()

with torch.no_grad():
    F0, N = conv_step(F0_pred, N_pred)
    print(f"PyTorch: F0={F0.shape} range=[{F0.min():.3f}, {F0.max():.3f}], N={N.shape} range=[{N.min():.3f}, {N.max():.3f}]")

    exported = export(conv_step, (F0_pred, N_pred), strict=False)
    from executorch.exir import to_edge
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/step1_conv.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((F0_pred, N_pred))
        F0_et, N_et = outputs[0], outputs[1]
        print(f"ExecuTorch: F0 range=[{F0_et.min():.3f}, {F0_et.max():.3f}] NaN={F0_et.isnan().any()}, N range=[{N_et.min():.3f}, {N_et.max():.3f}] NaN={N_et.isnan().any()}")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

# ============================================================================
# Step 2: Cat + Encode AdaINResBlk1d
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Cat + Encode AdaINResBlk1d")
print("=" * 80)

class EncodeStep(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.encode = decoder.encode

    def forward(self, asr, F0, N, s):
        x = torch.cat([asr, F0, N], axis=1)
        return self.encode(x, s)

encode_step = EncodeStep(model.decoder).eval()

with torch.no_grad():
    x_enc = encode_step(asr, F0, N, ref_s)
    print(f"PyTorch: x_enc={x_enc.shape} range=[{x_enc.min():.3f}, {x_enc.max():.3f}] NaN={x_enc.isnan().any()}")

    exported = export(encode_step, (asr, F0, N, ref_s), strict=False)
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/step2_encode.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((asr, F0, N, ref_s))
        x_enc_et = outputs[0]
        print(f"ExecuTorch: range=[{x_enc_et.min():.3f}, {x_enc_et.max():.3f}] NaN={x_enc_et.isnan().any()}")
        if x_enc_et.isnan().any():
            print("  ⚠ Encode block (after cat) produces NaNs!")
        else:
            print("  ✓ Encode block OK")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

# ============================================================================
# Step 3: ASR residual connection
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: ASR residual")
print("=" * 80)

class AsrResStep(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.asr_res = decoder.asr_res

    def forward(self, asr):
        return self.asr_res(asr)

asr_res_step = AsrResStep(model.decoder).eval()

with torch.no_grad():
    asr_res = asr_res_step(asr)
    print(f"PyTorch: asr_res={asr_res.shape} range=[{asr_res.min():.3f}, {asr_res.max():.3f}] NaN={asr_res.isnan().any()}")

    exported = export(asr_res_step, (asr,), strict=False)
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/step3_asr_res.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)

    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((asr,))
        asr_res_et = outputs[0]
        print(f"ExecuTorch: range=[{asr_res_et.min():.3f}, {asr_res_et.max():.3f}] NaN={asr_res_et.isnan().any()}")
        if asr_res_et.isnan().any():
            print("  ⚠ ASR residual produces NaNs!")
        else:
            print("  ✓ ASR residual OK")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

print("\n" + "=" * 80)
print("Check which step produces NaNs above")
print("=" * 80)
