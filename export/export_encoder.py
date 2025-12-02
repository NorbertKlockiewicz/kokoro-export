from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import argparse
import torch

# ---------------------------
# Encoder - exported interface
# ---------------------------

class TextEncoderWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids: torch.LongTensor,
                      input_lengths: torch.LongTensor,
                      text_mask: torch.BoolTensor):
        return self.text_encoder(input_ids, input_lengths, ~text_mask)

# ---------------------------------
# Encoder - model adjustments
# ---------------------------------

def remove_weight_norms(encoder: TextEncoderWrapper):
    for module in encoder.text_encoder.cnn:
        conv = module[0]
        _ = conv.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)

# -------------------------------
# Encoder - input data definition
# -------------------------------

# Predefined inputs for different sizes
inputs_dict = {
    "test": (
        torch.load("original_models/data/text_encoder_input.pt")["input_ids"],
        torch.load("original_models/data/text_encoder_input.pt")["input_lengths"],
        torch.load("original_models/data/text_encoder_input.pt")["text_mask"],
    ),
    "small": (
        torch.randint(0, 178, size=(1, 16)),
        torch.tensor(16),
        torch.ones((1, 16), dtype=torch.bool),
    ),
    "medium": (
        torch.randint(0, 178, size=(1, 64)),
        torch.tensor(64),
        torch.ones((1, 64), dtype=torch.bool),
    ),
    "big": (
        torch.randint(0, 178, size=(1, 256)),
        torch.tensor(256),
        torch.ones((1, 256), dtype=torch.bool),
    ),
}

# Map input names to token counts for filenames
input_name_mappings = {
    "test": "test",
    "small": "16",
    "medium": "64",
    "big": "256",
}

# -------------------------
# Encoder - export pipeline
# -------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True):
    # Model setup
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    encoder = TextEncoderWrapper(model)
    encoder.eval()

    # Deparametrize for XNNPACK delegation
    remove_weight_norms(encoder)

    # Export to ExportedProgram
    exported_program = torch.export.export(encoder, inputs)

    # Optionally lower to edge and partition
    if not transform_and_lower:
        return encoder, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )
    return encoder, executorch_program

# -------------------------
# Encoder - CLI entry point
# -------------------------

if __name__ == "__main__":
    # Args: input size and bundled mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["test", "small", "medium", "big"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    args = parser.parse_args()

    # Mode selection
    if args.bundled:
        input_mode = "medium"
    elif args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")

    bundled = args.bundled
    input_mode = args.input_size if not bundled else "medium"

    root_destination = "exported_models/tmp"

    if not bundled:
        # Single export
        print(f"Exporting encoder with {input_mode} input...")
        input_alias = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        # Ensure BOS/EOS for random inputs
        if input_mode in ["small", "medium", "big"]:
            input_ids = inputs[0]
            input_ids[0][0] = 0
            input_ids[0][-1] = 0

        _, edge_program = convert_to_executorch_program(inputs)
        executorch_program = edge_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        with open(f"{root_destination}/text_encoder_{input_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        # Bundled export: multiple entrypoints forward_{n_tokens}
        print("Exporting bundled encoder...")
        encoders = {}
        for name, n_tokens in input_name_mappings.items():
            inputs = inputs_dict[name]
            if name in ["small", "medium", "big"]:
                input_ids = inputs[0]
                input_ids[0][0] = 0
                input_ids[0][-1] = 0
            _, exported_enc = convert_to_executorch_program(inputs, transform_and_lower=False)
            encoders[n_tokens] = exported_enc

        edge_program = to_edge_transform_and_lower(
            {f"forward_{n_tokens}": ep for n_tokens, ep in encoders.items()},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(f"{root_destination}/text_encoder.pte", "wb") as file:
            edge_program.write_to_file(file)

    print("Finished!")