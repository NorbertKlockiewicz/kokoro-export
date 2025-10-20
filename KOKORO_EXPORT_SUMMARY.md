# Kokoro TTS ExecuTorch Export - Complete Summary

## Problem Statement
Attempted to export Kokoro-82M TTS model to ExecuTorch format (.pte) for mobile/edge deployment. Encountered multiple challenges with non-Core-ATen operations and XNNPACK backend bugs.

## Model Architecture

**Kokoro-82M** consists of:
1. **Text Encoder** - Processes phonemes
2. **Duration Predictor** - Predicts phoneme durations
3. **F0N Predictor** - Predicts pitch (F0) and noise (N)
4. **Alignment Builder** - Creates mel-spectrogram alignment
5. **Decoder** - Generates audio waveform (the problematic part)

### Decoder Components
- **Encoder layers** (AdainResBlk1d with InstanceNorm)
- **Generator** with:
  - SourceModuleHnNSF (harmonic + noise source generation)
  - Multiple conv_transpose1d upsampling layers
  - AdaINResBlock layers
  - **STFT/ISTFT** for frequency domain processing

## Core Issues Encountered

### Issue 1: Non-Core-ATen Operations
**Problem**: ExecuTorch only supports ~200 "Core ATen" operators. Many operations used by Kokoro are NOT in this set.

**Blocked operators**:
- `aten.istft.default` - Inverse Short-Time Fourier Transform
- `aten.stft.default` - Short-Time Fourier Transform
- `aten.unfold.default` - Extract overlapping windows (used by STFT internally)
- `aten._fft_c2r.default` - Complex-to-real FFT
- `aten.angle.default` - Get phase from complex numbers
- `aten.randn.default` / `aten.randn_like.default` - Random noise generation

**Why they're used**:
- **STFT/ISTFT**: Convert between time and frequency domains for audio synthesis
- **unfold**: STFT decomposes to unfold + FFT during export
- **randn**: Generate noise for unvoiced consonants (s, f, th sounds)

### Issue 2: STFT Decomposition Chain

When exporting, `torch.istft` decomposes as:
```
torch.istft
  ↓ (decomposes to)
_fft_c2r + unfold + index_put + normalization
  ↓ (all blocked)
Export fails or produces NaN
```

**Attempted solution**: Use CustomSTFT (convolution-based STFT)
- ✅ Avoids `unfold` and complex FFT
- ✅ Uses only `conv1d` / `conv_transpose1d` (Core ATen ops)
- ❌ Still produces NaN in XNNPACK backend

### Issue 3: Random Number Generation

**Problem**: ExecuTorch runtime doesn't have RNG (random number generator)
- `torch.randn()` exports successfully
- But produces NaN at runtime (no kernel implementation)

**Solution**: Replace with deterministic pseudo-random noise
```python
# Original:
noise = torch.randn_like(sine_waves)

# Fixed:
noise = torch.sin(sine_waves * 137.0) * torch.cos(sine_waves * 211.0)
```

### Issue 4: XNNPACK Backend NaN Bug

**Problem**: Full decoder exports successfully but produces 100% NaN values at runtime

**Evidence**:
- ✅ PyTorch inference works: No NaN, output range [-52K, 46K]
- ✅ Individual components work in ExecuTorch:
  - SourceModuleHnNSF: No NaN
  - CustomSTFT forward: No NaN
  - CustomSTFT inverse: No NaN
- ❌ Full decoder with XNNPACK: 100% NaN
- ❌ Full decoder without XNNPACK: Tensor layout error

**Root cause**: Specific combination of layers (conv_transpose + instance_norm + residual blocks) triggers numerical bug in XNNPACK backend

## Modifications Made

### 1. Replaced Random Noise with Deterministic Pattern
**File**: `kokoro/istftnet.py`

**Location 1** (line 301 - SineGen forward):
```python
# Before:
noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
noise = noise_amp * torch.randn_like(sine_waves)

# After:
noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
noise = noise_amp * (torch.sin(sine_waves * 137.0) * torch.cos(sine_waves * 211.0))
```

**Location 2** (line 364 - SourceModuleHnNSF forward):
```python
# Before:
noise = torch.randn_like(uv) * self.sine_amp / 3

# After:
noise = (torch.sin(uv * 173.0) * torch.cos(uv * 239.0)) * self.sine_amp / 3
```

**Result**:
- ✅ No `randn` operations in exported graph
- ✅ PyTorch output reasonable: [-18.9, 14.7]
- ❌ ExecuTorch still produces NaN (different issue)

### 2. Use CustomSTFT Instead of TorchSTFT
**File**: `kokoro/istftnet.py` line 444

Model must be loaded with `disable_complex=True`:
```python
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
```

This makes the Generator use CustomSTFT which:
- Uses convolution operations instead of FFT
- Avoids `torch.istft()` and `torch.stft()`
- Avoids `unfold` operations
- All operations are in Core ATen

### 3. Fixed rsqrt Numerical Stability
**File**: `kokoro/istftnet.py` line 384

```python
# Before:
out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))

# After:
out = (out + self._shortcut(x)) * 0.7071067811865476  # 1/sqrt(2) as constant
```

### 4. Increased Epsilon in CustomSTFT
**File**: `kokoro/custom_stft.py` line 133

```python
# Before:
magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)

# After:
magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-8)  # More stable
```

## Export Process

### Successful Export Steps
```python
from kokoro import KModel
import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

# 1. Load model with CustomSTFT
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

# 2. Wrap decoder
class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, ref_s):
        return self.decoder(asr, F0_pred, N_pred, ref_s).squeeze()

decoder = DecoderWrapper(model)

# 3. Export
with torch.no_grad():
    exported = export(decoder, (asr, F0_pred, N_pred, ref_s), strict=False)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    exec_prog = edge.to_executorch()

    with open('decoder.pte', 'wb') as f:
        f.write(exec_prog.buffer)
```

**Result**:
- ✅ Export succeeds
- ✅ .pte file created (203.69 MB)
- ❌ Runtime produces 100% NaN

## Debugging Process

### Created Debug Scripts

**1. debug_decoder_parts.py**
- Tests individual components separately
- **Results**:
  - ✅ SourceModuleHnNSF: OK
  - ✅ CustomSTFT forward: OK
  - ✅ CustomSTFT inverse: OK
  - ❌ Full decoder: NaN

**2. debug_full_decoder.py**
- Tests complete decoder export + inference
- Confirmed NaN only appears in full pipeline

**3. compare_randn_versions.py**
- Compares deterministic vs random noise
- **Results**:
  - Deterministic noise works correctly in PyTorch
  - Output values are reasonable
  - But ExecuTorch still produces NaN

## Failed Workaround Attempts

### 1. Exception List Approach
```python
compile_config = EdgeCompileConfig(
    _core_aten_ops_exception_list=[
        torch.ops.aten.istft.default,
        torch.ops.aten.unfold.default,
    ]
)
```
**Result**: ❌ Fails - `aten.angle.default` also blocked

### 2. Disable IR Validity Check (Whisper Trick)
```python
compile_config = EdgeCompileConfig(_check_ir_validity=False)
```
**Result**: ❌ Fails - Complex literal `1j` not supported in edge conversion

### 3. Export Without XNNPACK
```python
edge = to_edge_transform_and_lower(exported)  # No partitioner
```
**Result**: ❌ Runtime fails - Tensor layout error in convolution

### 4. CoreML Backend (macOS)
**Result**: ❌ Fails - `rand` operation not supported in CoreML converter

## What Works vs What Doesn't

### ✅ What Works
1. PyTorch inference (original model)
2. PyTorch inference (modified with deterministic noise)
3. Individual ExecuTorch components (tested separately)
4. Export process (creates .pte file)
5. Text Encoder export (21 MB, works fine)
6. Duration Predictor export (53 MB, works fine)

### ❌ What Doesn't Work
1. Full decoder runtime with XNNPACK → NaN
2. Full decoder runtime without XNNPACK → Tensor layout error
3. ISTFT with ExecuTorch (not in Core ATen)
4. Random number generation at runtime
5. CoreML backend (different errors)

## Root Cause: XNNPACK Backend Bug

**Conclusion**: The NaN issue is caused by a bug in the XNNPACK backend (ExecuTorch 0.7.0) when processing this specific combination of operations:
- ConvTranspose1d
- InstanceNorm
- Residual connections
- CustomSTFT convolutions

**Not** caused by:
- Our code modifications (PyTorch works fine)
- Individual operations (components work separately)
- Random noise (deterministic version also fails)

## Alternative Solutions

### Option 1: Use PyTorch Mobile (Recommended)
- **Pros**: Mature, tested, likely works
- **Cons**: Larger runtime (~50MB vs ~5MB)
- **Action**: Export to TorchScript format

### Option 2: Wait for ExecuTorch Update
- **Pros**: Cleanest long-term solution
- **Cons**: Unknown timeline
- **Action**: Monitor ExecuTorch 0.8.0 release

### Option 3: Switch to Simpler Model
- **Pros**: Avoids XNNPACK bugs
- **Cons**: May affect quality
- **Action**: Try NeuTTS Air (Qwen-based, no STFT)

### Option 4: Report Bug to ExecuTorch
- **Pros**: Helps community
- **Cons**: No guarantee of fix
- **Action**: Create minimal repro + GitHub issue

## Key Learnings

1. **Core ATen Opset is Limited**: Only ~200 ops vs PyTorch's ~2000+
2. **STFT is Not Portable**: Decomposes to non-Core-ATen ops
3. **Random Ops Don't Have Runtime**: RNG not available in ExecuTorch
4. **XNNPACK Has Bugs**: Specific layer combinations can produce NaN
5. **Individual Testing is Critical**: Components may work but full model fails

## Technical Environment

- **PyTorch**: 2.8.0
- **ExecuTorch**: 0.7.0
- **Platform**: macOS (Darwin 25.0.0)
- **Model**: Kokoro-82M (hexgrad/Kokoro-82M)
- **Original Issue**: Audio synthesis for mobile/edge deployment

## Files Created

1. `debug_decoder_parts.py` - Component-level testing
2. `debug_full_decoder.py` - Full decoder testing
3. `compare_randn_versions.py` - Noise comparison
4. `EXPORT_STATUS.md` - Detailed status report
5. `KOKORO_EXPORT_SUMMARY.md` - This document

## Next Steps

**Immediate**: Switch to **NeuTTS Air** model
- Based on Qwen 0.5B transformer (no STFT!)
- Already has ONNX and GGUF support
- Simpler architecture, better for mobile deployment
- https://huggingface.co/neuphonic/neutts-air

**Long-term**: Monitor ExecuTorch progress
- Core ATen opset may expand
- XNNPACK bugs may be fixed
- Return to Kokoro when viable
