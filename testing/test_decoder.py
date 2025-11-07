from kokoro import KModel
from executorch.runtime import Runtime
import torch

runtime = Runtime.get()


# --------------------------
# Decoder - PyTorch instance
# --------------------------

# Create model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

decoder = model.decoder
decoder.eval()


# -----------------------------
# Decoder - ExecuTorch instance
# -----------------------------

# -------------------------------------------------------------
pte_path_xnn = "exported_models/text_decoder_16.pte"
# -------------------------------------------------------------

program_xnn = runtime.load_program(pte_path_xnn)
method_xnn = program_xnn.load_method("forward")


# ----------------------------
# Decoder - input data loading
# ----------------------------

import re
# Parse all input shapes at once from the string representation
meta_str = str(method_xnn.metadata)
input_shapes = [
    tuple(int(x) for x in sizes.split(','))
    for sizes in re.findall(r'sizes=\[([^\]]+)\]', meta_str)
]
input_shapes = input_shapes[:4]     # We expect 4 shapes (4 inputs)

# Define all possible input sets
input_sets = {
    "test": lambda: (
        torch.load("original_models/data/text_decoder_input.pt")["asr"],
        torch.load("original_models/data/text_decoder_input.pt")["F0_pred"],
        torch.load("original_models/data/text_decoder_input.pt")["N_pred"],
        torch.load("original_models/data/text_decoder_input.pt")["ref_s"],
    ),
    "random-small": lambda: (
        torch.randn(size=(1, 512, 64)),
        torch.randn(size=(1, 128)),
        torch.randn(size=(1, 128)),
        torch.randn(size=(1, 128)),
    ),
    "random-medium": lambda: (
        torch.randn(size=(1, 512, 256)),
        torch.randn(size=(1, 512)),
        torch.randn(size=(1, 512)),
        torch.randn(size=(1, 128)),
    ),
    "random-big": lambda: (
        torch.randn(size=(1, 512, 1024)),
        torch.randn(size=(1, 4096)),
        torch.randn(size=(1, 4096)),
        torch.randn(size=(1, 128)),
    ),
}

# Map input set names to their shapes
input_set_shapes = {
    "test": [
        torch.load("original_models/data/text_decoder_input.pt")["asr"].shape,
        torch.load("original_models/data/text_decoder_input.pt")["F0_pred"].shape,
        torch.load("original_models/data/text_decoder_input.pt")["N_pred"].shape,
        torch.load("original_models/data/text_decoder_input.pt")["ref_s"].shape,
    ],
    "random-small": [(1, 512, 64), (1, 128), (1, 128), (1, 128)],
    "random-medium": [(1, 512, 256), (1, 512), (1, 512), (1, 128)],
    "random-big": [(1, 512, 1024), (1, 4096), (1, 4096), (1, 128)],
}

# Find matching input set
selected_mode = None
for mode, shapes in input_set_shapes.items():
    if all(tuple(s) == tuple(ms) for s, ms in zip(shapes, input_shapes)):
        selected_mode = mode
        break

if selected_mode is None:
    raise RuntimeError(f"No matching input set for ExecuTorch input shapes: {input_shapes}")

asr, F0_pred, N_pred, ref_s = input_sets[selected_mode]()

print(f"Selected input mode: {selected_mode}")
print(f"Input shapes: {[x.shape for x in [asr, F0_pred, N_pred, ref_s]]}")

inputs = (asr, F0_pred, N_pred, ref_s)


# -----------------------
# Decoder - perform tests
# -----------------------

output_pytorch = decoder(*inputs)

print("\nTesting ExecuTorch runtime (WITH XNNPACK)...")
outputs_xnn = method_xnn.execute((asr, F0_pred, N_pred, ref_s))
output_et_xnn = outputs_xnn[0]

print(f"ExecuTorch output:")
print(f"  Range: [{output_et_xnn.min():.3f}, {output_et_xnn.max():.3f}]")
print(f"  Contains NaN: {output_et_xnn.isnan().any()}")

if output_et_xnn.isnan().any():
    nan_pct = 100 * output_et_xnn.isnan().sum().item() / output_et_xnn.numel()
    print(f"\n⚠ XNNPACK produces NaNs ({nan_pct:.2f}%)")
else:
    diff = torch.abs(output_pytorch - output_et_xnn).max().item()
    print(f"\n✓ XNNPACK works! No NaNs!")
    print(f"  Max diff from PyTorch: {diff:.6f}")

    mse = torch.nn.functional.mse_loss(output_et_xnn, output_pytorch)
    print("  MSE loss:", mse.item())