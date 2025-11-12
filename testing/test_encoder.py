from kokoro import KModel
from executorch.runtime import Runtime
from export.export_encoder import TextEncoderWrapper
import torch

runtime = Runtime.get()


# --------------------------
# Encoder - PyTorch instance
# --------------------------

# Create model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)

encoder = TextEncoderWrapper(model)
encoder.eval()


# -----------------------------
# Encoder - ExecuTorch instance
# -----------------------------

# -------------------------------------------------------------
pte_path_xnn = "exported_models/tmp/text_encoder_test.pte"
# -------------------------------------------------------------

program_xnn = runtime.load_program(pte_path_xnn)
method_xnn = program_xnn.load_method("forward")


# ----------------------------
# Encoder - input data loading
# ----------------------------

import re
# Parse all input shapes at once from the string representation
meta_str = str(method_xnn.metadata)
# Updated regex to match both sizes=[] and sizes=[...]
input_shapes = []
for match in re.findall(r'sizes=\[([^\]]*)\]', meta_str):
    if match.strip() == '':
        input_shapes.append(tuple())
    else:
        input_shapes.append(tuple(int(x) for x in match.split(',') if x.strip()))
input_shapes = input_shapes[:3]     # We expect 3 shapes (3 inputs)
print(method_xnn.metadata)

# Define all possible input sets
input_sets = {
    "test": lambda: (
        torch.load("original_models/data/text_encoder_input.pt")["input_ids"],
        torch.load("original_models/data/text_encoder_input.pt")["input_lengths"],
        torch.load("original_models/data/text_encoder_input.pt")["text_mask"],
    ),
    "random-small": lambda: (
        # Ensure 0 at the beginning and end
        lambda arr: arr.index_fill_(1, torch.tensor([0, arr.shape[1]-1]), 0) or arr
        (lambda arr: arr.index_fill_(1, torch.tensor([0, arr.shape[1]-1]), 0) or arr)(
            torch.randint(0, 178, size=(1, 16))
        ),
        torch.tensor(16),
        torch.ones((1, 16), dtype=torch.bool),
    ),
    "random-medium": lambda: (
        (lambda arr: arr.index_fill_(1, torch.tensor([0, arr.shape[1]-1]), 0) or arr)(
            torch.randint(0, 178, size=(1, 64))
        ),
        torch.tensor(64),
        torch.ones((1, 64), dtype=torch.bool),
    ),
    "random-big": lambda: (
        (lambda arr: arr.index_fill_(1, torch.tensor([0, arr.shape[1]-1]), 0) or arr)(
            torch.randint(0, 178, size=(1, 256))
        ),
        torch.tensor(256),
        torch.ones((1, 256), dtype=torch.bool),
    ),
}

# Map input set names to their shapes
input_set_shapes = {
    "test": [
        torch.load("original_models/data/text_encoder_input.pt")["input_ids"].shape,
        torch.load("original_models/data/text_encoder_input.pt")["input_lengths"].shape,
        torch.load("original_models/data/text_encoder_input.pt")["text_mask"].shape
    ],
    "random-small": [(1, 16), (), (1, 16),],
    "random-medium": [(1, 64), (), (1, 64),],
    "random-big": [(1, 256), (), (1, 256),],
}

# Find matching input set
selected_mode = None
for mode, shapes in input_set_shapes.items():
    if all(tuple(s) == tuple(ms) for s, ms in zip(shapes, input_shapes)):
        selected_mode = mode
        break

if selected_mode is None:
    raise RuntimeError(f"No matching input set for ExecuTorch input shapes: {input_shapes}")

input_ids, input_lengths, text_mask = input_sets[selected_mode]()

print(f"Selected input mode: {selected_mode}")
print(f"Input shapes: {[x.shape for x in [input_ids, input_lengths, text_mask]]}")

inputs = (input_ids, input_lengths, text_mask)


# -----------------------
# Encoder - perform tests
# -----------------------

output_pytorch = encoder(*inputs)

print("\nTesting ExecuTorch runtime (WITH XNNPACK)...")
outputs_xnn = method_xnn.execute((input_ids, input_lengths, text_mask))
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

    l1 = torch.nn.functional.l1_loss(output_et_xnn, output_pytorch)
    print("  L1 loss:", l1.item())