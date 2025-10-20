# Kokoro Decoder ExecuTorch Export Status

## Summary
Successfully modified the Kokoro decoder to remove non-Core-ATen operations (`randn`, `unfold`, `istft`), but encountering NaN output in ExecuTorch runtime due to XNNPACK backend bug.

## What Works ✅

1. **PyTorch inference**: Works perfectly with deterministic noise
   - Output range: [-52K, 46K] (reasonable for raw audio before normalization)
   - No NaN, No Inf

2. **Individual components in ExecuTorch**:
   - ✅ SourceModuleHnNSF (F0 processing + noise generation)
   - ✅ CustomSTFT forward (magnitude/phase extraction)
   - ✅ CustomSTFT inverse (ISTFT reconstruction)

3. **Export process**:
   - ✅ PyTorch export successful
   - ✅ No `randn` operations (replaced with deterministic sin/cos)
   - ✅ No `unfold` operations (CustomSTFT avoids them)
   - ✅ Edge conversion successful
   - ✅ ExecuTorch conversion successful
   - ✅ .pte file created (203.69 MB)

## What Doesn't Work ❌

1. **Full decoder in ExecuTorch with XNNPACK**:
   - ❌ Produces 100% NaN values
   - Problem is NOT in individual STFT components
   - Problem is in Generator conv/resblock layers when combined with XNNPACK

2. **Full decoder in ExecuTorch without XNNPACK**:
   - ❌ Fails with tensor layout error in convolution operation
   - Error: "Expected tensor to have default or channels last dim order"

3. **CoreML backend**:
   - ❌ Fails on `rand` operation conversion

## Root Cause Analysis

### The NaN Issue
- **Location**: Generator's neural network layers (conv_transpose, instance_norm, resblocks)
- **Cause**: XNNPACK backend bug in ExecuTorch 0.7.0
- **Evidence**:
  - Individual parts work fine
  - PyTorch works fine
  - Only full model + XNNPACK produces NaN

### Why XNNPACK is Needed
- ExecuTorch without XNNPACK can't handle the tensor layouts
- XNNPACK provides optimized kernels for ARM/mobile
- But XNNPACK has numerical bugs with certain layer combinations

## Modifications Made

### 1. Replaced `randn_like` / `randn` with Deterministic Noise
**File**: `kokoro/istftnet.py`

**Location 1** (line 301 - SineGen):
```python
# Old: noise = noise_amp * torch.randn_like(sine_waves)
# New:
noise = noise_amp * (torch.sin(sine_waves * 137.0) * torch.cos(sine_waves * 211.0))
```

**Location 2** (line 364 - SourceModuleHnNSF):
```python
# Old: noise = torch.randn_like(uv) * self.sine_amp / 3
# New:
noise = (torch.sin(uv * 173.0) * torch.cos(uv * 239.0)) * self.sine_amp / 3
```

### 2. Used CustomSTFT (avoids unfold/FFT ops)
**File**: `kokoro/istftnet.py` line 444

Model must be loaded with `disable_complex=True` to use CustomSTFT instead of TorchSTFT.

### 3. Fixed rsqrt for numerical stability
**File**: `kokoro/istftnet.py` line 384
```python
# Old: out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
# New:
out = (out + self._shortcut(x)) * 0.7071067811865476  # 1/sqrt(2)
```

### 4. Increased epsilon in CustomSTFT sqrt
**File**: `kokoro/custom_stft.py` line 133
```python
# Old: magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-14)
# New:
magnitude = torch.sqrt(real_out**2 + imag_out**2 + 1e-8)
```

## Options Going Forward

### Option 1: Wait for ExecuTorch Fix
- **Pros**: Cleanest solution, no workarounds needed
- **Cons**: Unknown timeline, might be months
- **Action**: Report bug to ExecuTorch team

### Option 2: Use PyTorch Mobile Instead
- **Pros**: More mature, better tested, likely works
- **Cons**: Larger runtime, not as optimized for edge devices
- **Action**: Export to TorchScript instead of ExecuTorch

### Option 3: Simplify Model Architecture
- **Pros**: Might avoid XNNPACK bug
- **Cons**: Could affect audio quality, requires retraining
- **Action**: Replace instance_norm or other problematic layers

### Option 4: Use Different Backend
- **Pros**: Might avoid XNNPACK-specific bugs
- **Cons**: Platform-specific (CoreML=Apple only, QNN=Qualcomm only)
- **Action**: Try CoreML on macOS (needs fixing rand issue first)

### Option 5: Report & Hope for Workaround
- **Pros**: Community might find solution
- **Cons**: No guarantee of fix
- **Action**: Create minimal repro and file GitHub issue

## Test Files Created

1. `debug_decoder_parts.py` - Tests individual decoder components
2. `debug_full_decoder.py` - Tests full decoder export/inference
3. `compare_randn_versions.py` - Compares deterministic vs random noise
4. `export_decoder_pure_python.py` - Export script (user modified for CoreML)

## Recommended Next Steps

1. **Short term**: Use PyTorch Mobile for deployment
   - More reliable, proven to work
   - Slightly larger but functional

2. **Medium term**: Report bug to ExecuTorch
   - Create minimal reproduction case
   - File issue at https://github.com/pytorch/executorch/issues

3. **Long term**: Wait for ExecuTorch 0.8.0+
   - XNNPACK bugs might be fixed
   - Core ATen opset might expand

## Version Info
- PyTorch: 2.8.0
- ExecuTorch: 0.7.0
- Platform: macOS (Darwin 25.0.0)
