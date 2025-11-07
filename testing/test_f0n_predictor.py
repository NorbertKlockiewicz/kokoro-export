from kokoro import KModel
from executorch.runtime import Runtime
from export.export_f0n_predictor import F0NPredictorWrapper
import torch

runtime = Runtime.get()


# --------------------------------
# F0N Predictor - PyTorch instance
# --------------------------------

# Create model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

f0n_predictor = F0NPredictorWrapper(model)
f0n_predictor.eval()


# -----------------------------
# Decoder - ExecuTorch instance
# -----------------------------

# -------------------------------------------------------------
pte_path_xnn = "exported_models/tmp/f0n_predictor_test.pte"
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
input_shapes = input_shapes[:2]     # We expect 4 shapes (4 inputs)

# Define all possible input sets
input_sets = {
    "test": lambda: (
        torch.load("original_models/data/f0n_predictor_input.pt")["en"],
        torch.load("original_models/data/f0n_predictor_input.pt")["s"],
    ),
    "random-small": lambda: (
        torch.randn(size=(1, 640, 64)),
        torch.randn(size=(1, 128)),
    ),
    "random-medium": lambda: (
        torch.randn(size=(1, 640, 256)),
        torch.randn(size=(1, 128)),
    ),
    "random-big": lambda: (
        torch.randn(size=(1, 640, 1024)),
        torch.randn(size=(1, 128)),
    ),
}

# Map input set names to their shapes
input_set_shapes = {
    "test": [
        torch.load("original_models/data/f0n_predictor_input.pt")["en"].shape,
        torch.load("original_models/data/f0n_predictor_input.pt")["s"].shape,
    ],
    "random-small": [(1, 640, 64), (1, 128),],
    "random-medium": [(1, 640, 256), (1, 128),],
    "random-big": [(1, 640, 1024), (1, 128),],
}

# Find matching input set
selected_mode = None
for mode, shapes in input_set_shapes.items():
    if all(tuple(s) == tuple(ms) for s, ms in zip(shapes, input_shapes)):
        selected_mode = mode
        break

if selected_mode is None:
    raise RuntimeError(f"No matching input set for ExecuTorch input shapes: {input_shapes}")

en, s = input_sets[selected_mode]()

print(f"Selected input mode: {selected_mode}")
print(f"Input shapes: {[x.shape for x in [en, s]]}")

inputs = (en, s)


# -----------------------
# Decoder - perform tests
# -----------------------

output_pytorch_F0, output_pytorch_N = f0n_predictor(*inputs)

print("\nTesting ExecuTorch runtime (WITH XNNPACK)...")
outputs_xnn = method_xnn.execute((en, s))
output_F0_xnn, output_N_xnn = outputs_xnn

outputs = [
    ("F0", output_pytorch_F0, output_F0_xnn),
    ("N", output_pytorch_N, output_N_xnn),
]

for name, output_pytorch, output_et_xnn in outputs:
    print(f"\nExecuTorch output [{name}]:")
    print(f"  Range: [{output_et_xnn.min():.3f}, {output_et_xnn.max():.3f}]")
    print(f"  Contains NaN: {output_et_xnn.isnan().any()}")

    if output_et_xnn.isnan().any():
        nan_pct = 100 * output_et_xnn.isnan().sum().item() / output_et_xnn.numel()
        print(f"⚠ XNNPACK produces NaNs in {name} ({nan_pct:.2f}%)")
    else:
        diff = torch.abs(output_pytorch - output_et_xnn).max().item()
        print(f"✓ XNNPACK works for {name}! No NaNs!")
        print(f"  Max diff from PyTorch: {diff:.6f}")

        l1 = torch.nn.functional.l1_loss(
            output_et_xnn.float(), output_pytorch.float()
        )
        print(f"  L1 loss: {l1.item()}")