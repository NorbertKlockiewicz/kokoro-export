"""
Test individual operations in AdaIN block
Find which operation causes NaNs
"""

import torch
from torch import nn
from torch.export import export
import os

print("Testing AdaIN Block Operations")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte/debug", exist_ok=True)

# Load model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
print("✓ Model loaded")

# Get first AdaIN block from encode
adain_block = model.decoder.encode

# Create test inputs
x = torch.randn(1, 514, 78)  # asr (512) + F0 (1) + N (1)
s = torch.randn(1, 128)

print(f"Input: x={x.shape}, s={s.shape}")

# ============================================================================
# Test 1: InstanceNorm1d alone
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: InstanceNorm1d")
print("=" * 80)

class InstanceNormTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm1d(514, affine=True)

    def forward(self, x):
        return self.norm(x)

norm_test = InstanceNormTest().eval()
# Copy weights from actual model
norm_test.norm.load_state_dict(adain_block.norm1.norm.state_dict())

with torch.no_grad():
    norm_out = norm_test(x)
    print(f"PyTorch: range=[{norm_out.min():.3f}, {norm_out.max():.3f}], NaN={norm_out.isnan().any()}")

    exported = export(norm_test, (x,), strict=False)
    from executorch.exir import to_edge
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/test_instancenorm.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

    from executorch.runtime import Runtime
    runtime = Runtime.get()
    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x,))
        norm_out_et = outputs[0]
        print(f"ExecuTorch: range=[{norm_out_et.min():.3f}, {norm_out_et.max():.3f}], NaN={norm_out_et.isnan().any()}")
        if norm_out_et.isnan().any():
            print("  ⚠ InstanceNorm1d is the culprit!")
        else:
            print("  ✓ InstanceNorm1d OK")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

# ============================================================================
# Test 2: Full AdaIN block
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Full AdaIN block")
print("=" * 80)

class AdaINTest(nn.Module):
    def __init__(self, adain):
        super().__init__()
        self.adain = adain

    def forward(self, x, s):
        return self.adain(x, s)

adain_test = AdaINTest(adain_block.norm1).eval()

with torch.no_grad():
    adain_out = adain_test(x, s)
    print(f"PyTorch: range=[{adain_out.min():.3f}, {adain_out.max():.3f}], NaN={adain_out.isnan().any()}")

    exported = export(adain_test, (x, s), strict=False)
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/test_adain.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x, s))
        adain_out_et = outputs[0]
        print(f"ExecuTorch: range=[{adain_out_et.min():.3f}, {adain_out_et.max():.3f}], NaN={adain_out_et.isnan().any()}")
        if adain_out_et.isnan().any():
            print("  ⚠ Full AdaIN block produces NaNs!")
        else:
            print("  ✓ Full AdaIN block OK")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

# ============================================================================
# Test 3: Manual instance norm (stable version)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Manual InstanceNorm (stable)")
print("=" * 80)

class ManualInstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        # x shape: (B, C, T)
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, unbiased=False, keepdim=True)
        # Clamp variance to prevent division by zero
        var = torch.clamp(var, min=self.eps)
        std = torch.sqrt(var)
        x_norm = (x - mean) / std
        return x_norm * self.weight + self.bias

manual_norm = ManualInstanceNorm(514).eval()
# Copy weights
manual_norm.weight.data = adain_block.norm1.norm.weight.data.view(1, -1, 1)
manual_norm.bias.data = adain_block.norm1.norm.bias.data.view(1, -1, 1)

with torch.no_grad():
    manual_out = manual_norm(x)
    print(f"PyTorch: range=[{manual_out.min():.3f}, {manual_out.max():.3f}], NaN={manual_out.isnan().any()}")

    exported = export(manual_norm, (x,), strict=False)
    edge_program = to_edge(exported).to_executorch()

    pte_path = "exported_pte/debug/test_manual_norm.pte"
    with open(pte_path, "wb") as f:
        f.write(edge_program.buffer)
    print(f"✓ Exported to {pte_path}")

    try:
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")
        outputs = method.execute((x,))
        manual_out_et = outputs[0]
        print(f"ExecuTorch: range=[{manual_out_et.min():.3f}, {manual_out_et.max():.3f}], NaN={manual_out_et.isnan().any()}")
        if manual_out_et.isnan().any():
            print("  ⚠ Even manual norm produces NaNs - ExecuTorch has bugs!")
        else:
            print("  ✓ Manual norm works - use this instead of InstanceNorm1d!")
            diff = torch.abs(manual_out - manual_out_et).max().item()
            print(f"    Max diff from PyTorch: {diff:.6f}")
    except Exception as e:
        print(f"  ⚠ Failed: {e}")

print("\n" + "=" * 80)
print("Summary: If manual norm works, replace all InstanceNorm1d with manual implementation")
print("=" * 80)
