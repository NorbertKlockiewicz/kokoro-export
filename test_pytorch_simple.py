"""
PyTorch Split Model Pipeline - 5 Parts
Generates proper "Hello World" audio
"""

from pathlib import Path
import torch
from torch import nn
import time
from executorch.runtime import Verification, Runtime, Program, Method

et_runtime: Runtime = Runtime.get()

print("=" * 80)
print("PyTorch Split Model - Hello World")
print("=" * 80)

# =============================================================================
# Load model
# =============================================================================
print("\n[1/3] Loading model...")

from kokoro import KModel

model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
print("  ✓ Model loaded")

# =============================================================================
# Create split model parts
# =============================================================================
print("\n[2/3] Creating split model parts...")


# Part 1: Duration Predictor
class DurationPredictor(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.bert = m.bert
        self.bert_encoder = m.bert_encoder
        self.text_encoder_module = m.predictor.text_encoder
        self.lstm = m.predictor.lstm
        self.duration_proj = m.predictor.duration_proj

    def forward(self, input_ids, ref_s, speed=1.0):
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
        bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.text_encoder_module(d_en, s, input_lengths, ~text_mask)
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        return pred_dur, d, s


# Part 3: F0/N Predictor
class F0NPredictor(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.shared = m.predictor.shared
        self.F0_blocks = m.predictor.F0
        self.F0_proj = m.predictor.F0_proj
        self.N_blocks = m.predictor.N
        self.N_proj = m.predictor.N_proj

    def forward(self, en, s):
        x = en.transpose(-1, -2)
        torch._check_is_size(x.shape[1])
        x, _ = self.shared(x)
        F0 = x.transpose(-1, -2)
        for block in self.F0_blocks:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N_blocks:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


# Part 4: Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.text_encoder = m.text_encoder

    def forward(self, input_ids):
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
        return self.text_encoder(input_ids, input_lengths, ~text_mask)


# Part 5: Decoder
class Decoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.decoder = m.decoder

    def forward(self, asr, F0_pred, N_pred, ref_s):
        return self.decoder(asr, F0_pred, N_pred, ref_s).squeeze()


duration_pred = DurationPredictor(model).eval()
duration_pred_executorch = et_runtime.load_program(
    Path("exported_pte/duration_predictor.pte"), verification=Verification.Minimal
)
duration_pred_method = duration_pred_executorch.load_method("forward")
f0n_pred = F0NPredictor(model).eval()
f0n_pred_executorch = et_runtime.load_program(
    Path("exported_pte/f0n_predictor.pte"), verification=Verification.Minimal
)
f0n_pred_method = f0n_pred_executorch.load_method("forward")
text_enc = TextEncoder(model).eval()
text_enc_executorch = et_runtime.load_program(
    Path("exported_pte/text_encoder.pte"), verification=Verification.Minimal
)
text_enc_method = text_enc_executorch.load_method("forward")
decoder = Decoder(model).eval()
decoder_executorch = et_runtime.load_program(
    Path("exported_pte/decoder.pte"), verification=Verification.Minimal
)
decoder_method = decoder_executorch.load_method("forward")

print("  ✓ Created all 5 parts")

# =============================================================================
# Run inference through split model
# =============================================================================
print("\n[3/3] Running split model inference...")

# Pre-computed "Hello World" phonemes: həlˈoʊ wˈɝld
# Using the model's vocab mapping
phonemes = "həlˈoʊ wˈɝld"
text = "Hello World"

# Load voice from HuggingFace
from huggingface_hub import hf_hub_download

voice_file = hf_hub_download(
    repo_id="hexgrad/Kokoro-82M", filename="voices/af_bella.pt"
)
voice_style = torch.load(voice_file, weights_only=True)

# Fix voice shape
if voice_style.dim() == 3:
    voice_style = voice_style.mean(dim=0)
if voice_style.dim() == 2 and voice_style.shape[0] != 1:
    if voice_style.shape[1] == 256:
        voice_style = voice_style[0:1]

# Tokenize using model's vocab
input_ids = list(
    filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes))
)

# Pad to match BOTH duration predictor AND text encoder export sizes
# Duration predictor expects 16 total tokens (14 phonemes + BOS + EOS)
# Text encoder should be exported with same size (16)
# If you exported text encoder with different size, change this:
TARGET_TOKENS = 16  # Must match both duration_predictor.pte and text_encoder.pte

# Pad phonemes to reach target (subtract 2 for BOS/EOS)
while len(input_ids) < (TARGET_TOKENS - 2):
    input_ids.append(0)  # Pad with EOS token
# Truncate if too long
input_ids = input_ids[: (TARGET_TOKENS - 2)]
input_ids = torch.LongTensor([[0, *input_ids, 0]])

print(f"  Text: '{text}'")
print(f"  Phonemes: {phonemes}")
print(f"  Input IDs: {input_ids.shape}")
print(f"  Voice: af_bella {voice_style.shape}")

speed = torch.tensor([1.0])  # Must be 1D tensor for ExecuTorch

with torch.no_grad():
    start = time.time()

    # Part 1: Duration Prediction
    duration_pred_input = (input_ids, voice_style, speed)
    pred_dur, d, s = duration_pred_method.execute(duration_pred_input)
    print(f"\n  Part 1 (Duration): {pred_dur.shape}")

    # Part 2: Alignment (Python, data-dependent)
    device = input_ids.device
    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=device), pred_dur
    )
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln_trg

    print(f"  Part 2 (Alignment): {en.shape}, pred_aln_trg: {pred_aln_trg.shape}")

    # Part 3: F0/N Prediction
    # Use PyTorch fallback since ExecuTorch has static shapes
    F0_pred, N_pred = f0n_pred(en, s)
    print(f"  Part 3 (F0/N - PyTorch): {F0_pred.shape}, {N_pred.shape}")

    # Part 4: Text Encoder
    # Note: Text encoder might expect different input size than what we have
    # If you get shape mismatch, export text encoder with matching size
    t_en_output = text_enc_method.execute((input_ids,))
    t_en = t_en_output[0]  # ExecuTorch returns tuple, get first element
    asr = t_en @ pred_aln_trg
    print(f"  Part 4 (Text Encoder): {t_en.shape}")

    # Part 5: Decoder
    # Test PyTorch decoder first
    decoder_pytorch_out = decoder(asr, F0_pred, N_pred, voice_style[:, :128])
    print(f"  Part 5 (Decoder PyTorch): {decoder_pytorch_out.shape}")
    print(f"    Range: [{decoder_pytorch_out.min():.3f}, {decoder_pytorch_out.max():.3f}]")
    print(f"    Contains NaN: {decoder_pytorch_out.isnan().any()}")

    decoder_output = decoder_method.execute(
        (asr, F0_pred, N_pred, voice_style[:, :128])
    )
    audio_split = decoder_output[0]  # ExecuTorch returns tuple, get first element
    print(f"  Part 5 (Decoder ExecuTorch): {audio_split.shape}")
    print(f"    Range: [{audio_split.min():.3f}, {audio_split.max():.3f}]")
    print(f"    Contains NaN: {audio_split.isnan().any()}")
    print(f"    Contains Inf: {audio_split.isinf().any()}")

    # Also generate with original model for comparison
    audio_original = model(
        phonemes, voice_style, speed.item()
    )  # Original model expects float

    total_time = time.time() - start

print(f"\n  Inference time: {total_time * 1000:.2f}ms")

# =============================================================================
# Compare outputs
# =============================================================================
print("\n[4/4] Comparing ExecuTorch vs PyTorch outputs...")

import torchaudio

# Calculate differences
audio_split_cpu = audio_split.cpu()
audio_original_cpu = audio_original.cpu()

# Ensure same length for comparison
min_len = min(len(audio_split_cpu), len(audio_original_cpu))
split_trimmed = audio_split_cpu[:min_len]
original_trimmed = audio_original_cpu[:min_len]

# Calculate metrics
mse = torch.mean((split_trimmed - original_trimmed) ** 2).item()
mae = torch.mean(torch.abs(split_trimmed - original_trimmed)).item()
max_diff = torch.max(torch.abs(split_trimmed - original_trimmed)).item()

# Calculate correlation
correlation = torch.corrcoef(torch.stack([split_trimmed, original_trimmed]))[
    0, 1
].item()

print(
    f"  Audio length - ExecuTorch: {audio_split_cpu.shape[0] / 24000:.2f}s, PyTorch: {audio_original_cpu.shape[0] / 24000:.2f}s"
)
print(f"  Mean Squared Error (MSE): {mse:.6f}")
print(f"  Mean Absolute Error (MAE): {mae:.6f}")
print(f"  Max Absolute Difference: {max_diff:.6f}")
print(f"  Correlation: {correlation:.6f}")

if correlation > 0.95:
    print(f"  ✓ Outputs are very similar! (correlation > 0.95)")
elif correlation > 0.8:
    print(f"  ✓ Outputs are similar (correlation > 0.8)")
else:
    print(f"  ⚠ Outputs differ significantly (correlation < 0.8)")

# Save both outputs
torchaudio.save("hello_executorch.wav", audio_split_cpu.unsqueeze(0), 24000)
print(f"\n  ✓ Saved ExecuTorch: hello_executorch.wav")

torchaudio.save("hello_pytorch.wav", audio_original_cpu.unsqueeze(0), 24000)
print(f"  ✓ Saved PyTorch: hello_pytorch.wav")

print("\n" + "=" * 80)
print("SUCCESS! Compare hello_executorch.wav and hello_pytorch.wav")
print("=" * 80)
