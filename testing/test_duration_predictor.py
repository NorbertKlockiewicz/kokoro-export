from kokoro import KModel
from executorch.runtime import Runtime
from export.export_duration_predictor import DurationPredictorWrapper
import torch

runtime = Runtime.get()


# -------------------------------------
# Duration predictor - PyTorch instance
# -------------------------------------

# Create model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

duration_predictor = DurationPredictorWrapper(model)
duration_predictor.eval()


# -----------------------------
# Decoder - ExecuTorch instance
# -----------------------------

# -------------------------------------------------------------
pte_path_xnn = "exported_models/tmp/duration_predictor_random-small.pte"
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
input_shapes = input_shapes[:3]     # We expect 3 shapes (3 inputs)

# Define all possible input sets
input_sets = {
    "test": lambda: (
        torch.load("original_models/data/duration_predictor_input.pt")["input_ids"],
        torch.load("original_models/data/duration_predictor_input.pt")["ref_s"],
        torch.load("original_models/data/duration_predictor_input.pt")["speed"],
    ),
    "random-small": lambda: (
        torch.randint(0, 188, size=(1, 16)),
        torch.randn(size=(1, 256)),
        torch.randn(size=(1,)),
    ),
    "random-medium": lambda: (
        torch.randint(0, 188, size=(1, 64)),
        torch.randn(size=(1, 256)),
        torch.randn(size=(1,)),
    ),
    "random-big": lambda: (
        torch.randint(0, 188, size=(1, 256)),
        torch.randn(size=(1, 256)),
        torch.randn(size=(1,)),
    ),
}

# Map input set names to their shapes
input_set_shapes = {
    "test": [
        torch.load("original_models/data/duration_predictor_input.pt")["input_ids"].shape,
        torch.load("original_models/data/duration_predictor_input.pt")["ref_s"].shape,
        torch.load("original_models/data/duration_predictor_input.pt")["speed"].shape,
    ],
    "random-small": [(1, 16), (1, 256), (1,),],
    "random-medium": [(1, 64), (1, 256), (1,),],
    "random-big": [(1, 256), (1, 256), (1,),],
}

# Find matching input set
selected_mode = None
for mode, shapes in input_set_shapes.items():
    if all(tuple(s) == tuple(ms) for s, ms in zip(shapes, input_shapes)):
        selected_mode = mode
        break

if selected_mode is None:
    raise RuntimeError(f"No matching input set for ExecuTorch input shapes: {input_shapes}")

input_ids, ref_s, speed = input_sets[selected_mode]()

print(f"Selected input mode: {selected_mode}")
print(f"Input shapes: {[x.shape for x in [input_ids, ref_s, speed]]}")

inputs = (input_ids, ref_s, speed)


# -----------------------
# Decoder - perform tests
# -----------------------

output_pytorch_pred_dur, output_pytorch_d, output_pytorch_s = duration_predictor(*inputs)

print("\nTesting ExecuTorch runtime (WITH XNNPACK)...")
outputs_xnn = method_xnn.execute((input_ids, ref_s, speed))
output_pred_dur_xnn, output_d_xnn, output_s_xnn = outputs_xnn

outputs = [
    ("pred_dur", output_pytorch_pred_dur, output_pred_dur_xnn),
    ("d", output_pytorch_d, output_d_xnn),
    ("s", output_pytorch_s, output_s_xnn),
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