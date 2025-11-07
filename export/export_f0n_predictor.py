from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
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

    def forward(self, en: torch.FloatTensor, s: torch.FloatTensor):
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

        return F0.squeeze(1), N.squeeze(1)


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


# ----------------------------
# Decoder - input data loading
# ----------------------------

# ------------------------------------------------------------------------------------------
INPUT_MODE: Literal["test", "random-small", "random-medium", "random-big"] = "test"
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    if INPUT_MODE == "test":
        data = torch.load("original_models/data/f0n_predictor_input.pt")
        en = data["en"]
        s = data["s"]
    elif INPUT_MODE == "random-small":
        en = torch.randn(size=(1, 640, 64))
        s = torch.randn(size=(1, 128))
    elif INPUT_MODE == "random-medium":
        en = torch.randn(size=(1, 640, 256))
        s = torch.randn(size=(1, 128))
    elif INPUT_MODE == "random-big":
        en = torch.randn(size=(1, 640, 1024))
        s = torch.randn(size=(1, 128))
    else:
        raise RuntimeError("Invalid input mode!")

    inputs = (
        en,
        s
    )


# -------------------------
# Decoder - export pipeline
# -------------------------

def convert_to_executorch_program(inputs):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    f0n_predictor = F0NPredictorWrapper(model)
    f0n_predictor.eval()

    # Apply deparametrization
    remove_weight_norms(f0n_predictor)

    # Export
    exported_program = torch.export.export(f0n_predictor, inputs)

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    )

    return f0n_predictor, executorch_program

if __name__ == "__main__":
    _, executorch_program = convert_to_executorch_program(inputs)
    executorch_program = executorch_program.to_executorch()

    print_delegation_info(executorch_program.exported_program().graph_module)

    # Save exported file
    ROOT_DESTINATION = "exported_models/tmp"

    with open(f"{ROOT_DESTINATION}/f0n_predictor_{INPUT_MODE}.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Finished!")