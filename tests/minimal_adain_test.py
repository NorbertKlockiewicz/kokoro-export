"""
Minimal test: Export just the problematic AdainResBlk1d with real-shaped inputs
to isolate the exact operation causing NaNs
"""

import torch
from torch import nn
from torch.export import export
import os

print("Minimal AdainResBlk1d Test")
print("=" * 80)

from kokoro import KModel

os.makedirs("exported_pte/debug", exist_ok=True)

model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()

# Get the exact encode block that causes NaNs
encode_block = model.decoder.encode

# Create inputs matching the problematic concatenation
# asr (512) + F0 (1) + N (1) = 514 channels
x = torch.randn(1, 514, 78)  # Batch=1, Channels=514, Time=78
s = torch.randn(1, 128)  # Style vector

print(f"Input shapes: x={x.shape}, s={s.shape}")

# Test PyTorch
with torch.no_grad():
    output_pt = encode_block(x, s)
    print(f"\nPyTorch output:")
    print(f"  Shape: {output_pt.shape}")
    print(f"  Range: [{output_pt.min():.3f}, {output_pt.max():.3f}]")
    print(f"  NaN: {output_pt.isnan().any()}")

    # Export
    class Wrapper(nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.block = blk

        def forward(self, x, s):
            return self.block(x, s)

    wrapper = Wrapper(encode_block).eval()

    print("\nExporting...")
    exported = export(wrapper, (x, s), strict=False)
    output_exp = exported.module()(x, s)
    print(f"Exported: range=[{output_exp.min():.3f}, {output_exp.max():.3f}] NaN={output_exp.isnan().any()}")

    # Lower with XNNPACK
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print("\nLowering with XNNPACK...")
    try:
        edge_program = to_edge_transform_and_lower(
            exported,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()
        print("✓ Lowered successfully")

        pte_path = "exported_pte/debug/minimal_adain_xnnpack.pte"
        with open(pte_path, "wb") as f:
            f.write(edge_program.buffer)
        print(f"✓ Saved to {pte_path}")

        # Test runtime
        from executorch.runtime import Runtime
        runtime = Runtime.get()
        program = runtime.load_program(pte_path)
        method = program.load_method("forward")

        print("\nTesting ExecuTorch runtime...")
        outputs = method.execute((x, s))
        output_et = outputs[0]

        print(f"ExecuTorch output:")
        print(f"  Range: [{output_et.min():.3f}, {output_et.max():.3f}]")
        print(f"  NaN: {output_et.isnan().any()}")

        if output_et.isnan().any():
            nan_count = output_et.isnan().sum().item()
            total = output_et.numel()
            print(f"\n⚠ CONFIRMED: AdainResBlk1d produces {nan_count}/{total} NaNs in ExecuTorch")
            print(f"  This is a bug in XNNPACK's handling of this operation pattern")
        else:
            print(f"\n✓ No NaNs")

    except Exception as e:
        print(f"\n⚠ Failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
