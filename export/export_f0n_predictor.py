from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import argparse
import torch

# ----------------------------------
# F0N Predictor - exported interface
# ----------------------------------

class F0NPredictorWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.shared = model.predictor.shared
        self.F0_blocks = model.predictor.F0
        self.F0_proj = model.predictor.F0_proj
        self.N_blocks = model.predictor.N
        self.N_proj = model.predictor.N_proj

    def forward(self, indices: torch.LongTensor, d: torch.FloatTensor, s: torch.FloatTensor):
        input_size = d.shape[1]

        pred_aln_trg = torch.zeros((input_size, indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg

        x = en.transpose(-1, -2)
        torch._check_is_size(x.shape[1])
        x, _ = self.shared(x)

        F0 = x.transpose(-1, -2)
        for block in self.F0_blocks:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)

        N = x.transpose(-1, -2)
        for block in self.N_blocks:
            N = block(N, s)
        N = self.N_proj(N)

        return F0.squeeze(1), N.squeeze(1), en, pred_aln_trg


# ---------------------------------
# F0N Predictor - model adjustments
# ---------------------------------

# Get rid of all weight_norm decllarations to enable delegating the convolutions to XNNPACK backend
def remove_weight_norms(f0n_predictor: F0NPredictorWrapper):
    # f0n_predictor -> F0_blocks, N_blocks -> pool, conv1, conv2, conv1x1
    modules = [module for module in f0n_predictor.F0_blocks] + [module for module in f0n_predictor.N_blocks]
    for module in modules:
        module_weight_norms = [module.conv1, module.conv2]
        if hasattr(module.pool, "weight"):
            module_weight_norms.append(module.pool)
        if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
            module_weight_norms.append(module.conv1x1)
        for conv in module_weight_norms:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)


# --------------------------------
# F0N Predictor - input definition
# --------------------------------

inputs_dict = {
    "test": (
        torch.load("original_models/data/f0n_predictor_input.pt")["en"],
        torch.load("original_models/data/f0n_predictor_input.pt")["s"],
    ),
    "small": (
        torch.randint(0, 2, (64,), dtype=torch.long),
        torch.randn(size=(1, 16, 640)),
        torch.randn(size=(1, 128)),
    ),
    "medium": (
        torch.randint(0, 2, (164,), dtype=torch.long),
        torch.randn(size=(1, 64, 640)),
        torch.randn(size=(1, 128)),
    ),
    "big": (
        torch.randint(0, 2, (556,), dtype=torch.long),
        torch.randn(size=(1, 256, 640)),
        torch.randn(size=(1, 128)),
    ),
}

input_name_mappings = {
    "test": "test",
    "small": "16",
    "medium": "64",
    "big": "256",
}

# -------------------------
# F0N Predictor - pipeline
# -------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    f0n_predictor = F0NPredictorWrapper(model)
    f0n_predictor.eval()

    # Apply deparametrization
    remove_weight_norms(f0n_predictor)

    # Export
    exported_program = torch.export.export(f0n_predictor, inputs)

    if not transform_and_lower:
        return f0n_predictor, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    return f0n_predictor, executorch_program

# ------------------------------
# F0N Predictor - CLI entrypoint
# ------------------------------

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["test", "small", "medium", "big"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    args = parser.parse_args()

    # Mode selection
    if args.bundled is False and args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")

    bundled = args.bundled
    input_mode = args.input_size if not bundled else "medium"

    root_destination = "exported_models/tmp"

    if not bundled:
        # Single export
        print(f"Exporting F0N predictor with {input_mode} input...")
        input_alias = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        _, edge_program = convert_to_executorch_program(inputs)
        executorch_program = edge_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        with open(f"{root_destination}/f0n_predictor_{input_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        # Bundled export
        print("Exporting bundled F0N predictor...")
        predictors = {}
        for name, n_tokens in input_name_mappings.items():
            _, exported_pred = convert_to_executorch_program(inputs_dict[name], transform_and_lower=False)
            predictors[n_tokens] = exported_pred

        edge_program = to_edge_transform_and_lower(
            {f"forward_{n_tokens}": ep for n_tokens, ep in predictors.items()},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(f"{root_destination}/f0n_predictor.pte", "wb") as file:
            edge_program.write_to_file(file)

    print("Finished!")