"""
Capture intermediate values from Decoder to test Generator with real shapes
"""

import torch
from kokoro import KModel

print("Capturing Decoder intermediate values")
print("=" * 80)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

# Create inputs matching actual usage
frames = 78
asr = torch.randn(1, 512, frames)
F0_pred = torch.randn(1, frames * 2)
N_pred = torch.randn(1, frames * 2)
ref_s = torch.randn(1, 128)

print(f"\nDecoder inputs:")
print(f"  asr: {asr.shape}")
print(f"  F0_pred: {F0_pred.shape}")
print(f"  N_pred: {N_pred.shape}")
print(f"  ref_s: {ref_s.shape}")

# Hook to capture generator inputs
generator_inputs = {}

def hook_fn(module, input, output):
    generator_inputs['x'] = input[0].clone()
    generator_inputs['s'] = input[1].clone()
    generator_inputs['f0'] = input[2].clone()

handle = model.decoder.generator.register_forward_hook(hook_fn)

# Run decoder
with torch.no_grad():
    output = model.decoder(asr, F0_pred, N_pred, ref_s)
    print(f"\nDecoder output:")
    print(f"  Shape: {output.shape}")
    print(f"  Range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Contains NaN: {output.isnan().any()}")

handle.remove()

print(f"\nGenerator inputs (captured from decoder):")
print(f"  x: {generator_inputs['x'].shape}")
print(f"  s: {generator_inputs['s'].shape}")
print(f"  f0: {generator_inputs['f0'].shape}")

# Save for testing
torch.save({
    'x': generator_inputs['x'],
    's': generator_inputs['s'],
    'f0': generator_inputs['f0'],
    'output': output
}, 'generator_test_inputs.pt')

print(f"\n✓ Saved to generator_test_inputs.pt")

# Test generator standalone with captured inputs
with torch.no_grad():
    gen_output = model.decoder.generator(
        generator_inputs['x'],
        generator_inputs['s'],
        generator_inputs['f0']
    )
    print(f"\nGenerator standalone output:")
    print(f"  Shape: {gen_output.shape}")
    print(f"  Range: [{gen_output.min():.3f}, {gen_output.max():.3f}]")
    print(f"  Contains NaN: {gen_output.isnan().any()}")

    # Check if matches decoder output
    match = torch.allclose(gen_output, output, rtol=1e-4, atol=1e-5)
    print(f"  Matches decoder output: {match}")

print("\n" + "=" * 80)
