"""
Test decoder.pte directly with real inputs from the full pipeline
to isolate whether NaNs come from decoder or from input shapes
"""

import torch
from kokoro import KModel
from executorch.runtime import Runtime
from pathlib import Path

print("Testing decoder.pte with real pipeline inputs")
print("=" * 80)

# Load PyTorch model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ PyTorch model loaded")

# Load voice
from huggingface_hub import hf_hub_download
voice_file = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/af_bella.pt")
voice_style = torch.load(voice_file, weights_only=True)
if voice_style.dim() == 3:
    voice_style = voice_style.mean(dim=0)
if voice_style.dim() == 2 and voice_style.shape[0] != 1:
    if voice_style.shape[1] == 256:
        voice_style = voice_style[0:1]

# Create inputs using the actual pipeline
phonemes = "həlˈoʊ wˈɝld"
input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))

# Pad to 16 tokens
TARGET_TOKENS = 16
while len(input_ids) < (TARGET_TOKENS - 2):
    input_ids.append(0)
input_ids = input_ids[:(TARGET_TOKENS - 2)]
input_ids = torch.LongTensor([[0, *input_ids, 0]])

print(f"\nGenerating intermediate values...")
with torch.no_grad():
    # Run through parts 1-4 to get decoder inputs
    input_lengths = torch.tensor(input_ids.shape[-1])
    text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)

    # Part 1: Duration prediction
    bert_dur = model.bert(input_ids, attention_mask=text_mask.int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = voice_style[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, ~text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1)
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    # Part 2: Alignment
    device = input_ids.device
    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=device), pred_dur)
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln_trg

    # Part 3: F0/N prediction
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

    # Part 4: Text encoder
    t_en = model.text_encoder(input_ids, input_lengths, ~text_mask)
    asr = t_en @ pred_aln_trg

    print(f"Decoder inputs:")
    print(f"  asr: {asr.shape}, range: [{asr.min():.3f}, {asr.max():.3f}]")
    print(f"  F0_pred: {F0_pred.shape}, range: [{F0_pred.min():.3f}, {F0_pred.max():.3f}]")
    print(f"  N_pred: {N_pred.shape}, range: [{N_pred.min():.3f}, {N_pred.max():.3f}]")
    print(f"  ref_s: {voice_style[:, :128].shape}, range: [{voice_style[:, :128].min():.3f}, {voice_style[:, :128].max():.3f}]")

    # Test PyTorch decoder
    output_pytorch = model.decoder(asr, F0_pred, N_pred, voice_style[:, :128]).squeeze()
    print(f"\nPyTorch decoder output:")
    print(f"  Shape: {output_pytorch.shape}")
    print(f"  Range: [{output_pytorch.min():.3f}, {output_pytorch.max():.3f}]")
    print(f"  Contains NaN: {output_pytorch.isnan().any()}")

    # Test ExecuTorch decoder
    runtime = Runtime.get()
    try:
        program = runtime.load_program(Path("exported_pte/decoder.pte"))
        method = program.load_method("forward")

        print(f"\nTesting ExecuTorch decoder.pte...")
        output_et = method.execute((asr, F0_pred, N_pred, voice_style[:, :128]))[0]

        print(f"ExecuTorch decoder output:")
        print(f"  Shape: {output_et.shape}")
        print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
        print(f"  Contains NaN: {output_et.isnan().any()}")

        if output_et.isnan().any():
            # Find where NaNs start
            nan_positions = output_et.isnan().nonzero()
            if len(nan_positions) > 0:
                first_nan_idx = nan_positions[0].item()
                print(f"  First NaN at index: {first_nan_idx}/{output_et.numel()}")
                print(f"  NaN percentage: {100 * output_et.isnan().sum().item() / output_et.numel():.2f}%")
        else:
            diff = torch.abs(output_pytorch - output_et).max().item()
            print(f"  ✓ No NaNs! Max diff from PyTorch: {diff:.6f}")

    except FileNotFoundError:
        print("\n⚠ exported_pte/decoder.pte not found. Run export script first.")
    except RuntimeError as e:
        print(f"\n⚠ ExecuTorch runtime error: {e}")

print("\n" + "=" * 80)
