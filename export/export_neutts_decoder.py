#!/usr/bin/env python3
"""
Export NeuTTS Air decoder (NeuCodec) to ExecuTorch.

This script:
1. Loads the NeuCodec decoder
2. Tests it with PyTorch inference
3. Attempts to export to ExecuTorch format (.pte)
4. Tests the exported model
"""

import torch
import numpy as np
from neucodec import NeuCodec

# ============================================================================
# Step 1: Load NeuCodec
# ============================================================================

print("=" * 80)
print("Step 1: Loading NeuCodec from neuphonic/neucodec")
print("=" * 80)

codec = NeuCodec.from_pretrained("neuphonic/neucodec")
codec.eval()

print(f"✓ NeuCodec loaded successfully")
print(f"  Device: {codec.device}")
print()

# ============================================================================
# Step 2: Prepare test input
# ============================================================================

print("=" * 80)
print("Step 2: Preparing test input")
print("=" * 80)

# NeuCodec decode_code expects (batch, 1, time) tensor of integer codes
# Codes are in range [0, 1024) for the finite scalar quantization
test_codes = torch.randint(0, 1024, (1, 1, 100), dtype=torch.long)

print(f"Test codes shape: {test_codes.shape}")
print(f"Test codes dtype: {test_codes.dtype}")
print(f"Test codes range: [{test_codes.min()}, {test_codes.max()}]")
print()

# ============================================================================
# Step 3: Test PyTorch inference
# ============================================================================

print("=" * 80)
print("Step 3: Testing PyTorch inference")
print("=" * 80)

with torch.no_grad():
    try:
        pytorch_output = codec.decode_code(test_codes)
        print(f"✓ PyTorch inference successful!")
        print(f"  Output shape: {pytorch_output.shape}")
        print(f"  Output dtype: {pytorch_output.dtype}")
        print(
            f"  Output range: [{pytorch_output.min():.2f}, {pytorch_output.max():.2f}]"
        )
        print(f"  Has NaN: {torch.isnan(pytorch_output).any()}")
        print(f"  Has Inf: {torch.isinf(pytorch_output).any()}")
        print()
    except Exception as e:
        print(f"✗ PyTorch inference failed!")
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

# ============================================================================
# Step 4: Export to ExecuTorch
# ============================================================================

print("=" * 80)
print("Step 4: Exporting to ExecuTorch")
print("=" * 80)


# Create wrapper module for export
class NeuCodecDecoderWrapper(torch.nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, codes):
        """
        Args:
            codes: (batch, 1, time) tensor of integer codes in [0, 1024)
        Returns:
            audio: (batch, 1, samples) tensor of audio waveform
        """
        return self.codec.decode_code(codes)


wrapper = NeuCodecDecoderWrapper(codec).eval()

# Test wrapper
print("Testing wrapper module...")
with torch.no_grad():
    wrapper_output = wrapper(test_codes)
    print(f"✓ Wrapper works! Output shape: {wrapper_output.shape}")
print()

# Try torch.export first
print("Attempting torch.export.export()...")
from torch.export import export

# Try strict=False since we know there might be dynamic shapes
exported_program = export(wrapper, (test_codes,), strict=False)
print("✓ torch.export.export() successful!")
print()
