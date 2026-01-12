# Kokoro Model ExecuTorch Export Status

## Summary
The Kokoro model is fully exported using static input shapes. This involved replacing several model parts with custom implementations and resolving various unsupported code segments. The system is divided into four primary modules: `DurationPredictor`, `F0NPredictor`, `Encoder`, and `Decoder`. Currently, only the `Decoder` has been successfully exported with dynamic input shapes.

## What Works ✅

1. **Full Static Export**: All components (`DurationPredictor`, `F0NPredictor`, `Encoder`, `Decoder`) are functional with static shapes.
2. **Dynamic Decoder**: The `Decoder` module supports dynamic input shapes in ExecuTorch.
3. **Custom Implementations**: 
   - Replaced `randn` with deterministic noise logic.
   - Replaced `unfold` and `istft` operations via `CustomSTFT`.

## What Doesn't Work ❌

1. **Dynamic Predictors**: `DurationPredictor` and `F0NPredictor` fail to export with dynamic shapes.
2. **LSTM Support**: The primary blocker is the `LSTM` module, which does not support dynamic shape tracing in the current ExecuTorch export flow.
3. **Encoder Dynamics**: Exporting the `Encoder` with dynamic shapes is not possible without rewriting the LSTM module.

## Root Cause Analysis

### The LSTM Issue
- **Location**: `DurationPredictor` and `F0NPredictor`.
- **Cause**: Core `torch.nn.LSTM` lacks metadata support for dynamic sequence lengths during the conversion to Edge/ExecuTorch IR.
- **Impact**: Prevents end-to-end dynamic shape support for text-to-speech inference, including the `Encoder` module.

## Recommended Next Steps

1. **Rewrite LSTM Module**: Implement a custom `Lstm` using basic ATen operations (gates and loops) to enable dynamic shape compatibility.
2. **Test Encoder**: After LSTM is rewritten, verify if the `Encoder` module can be exported with dynamic shapes.
