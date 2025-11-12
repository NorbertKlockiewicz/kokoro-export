from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import torch

# ---------------------------
# Encoder- exported interface
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
# F0N Predictor - model adjustments
# ---------------------------------

# Get rid of all weight_norm decllarations to enable delegating the convolutions to XNNPACK backend
def remove_weight_norms(encoder: TextEncoderWrapper):
    for module in encoder.text_encoder.cnn:
        conv = module[0]
        _ = conv.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)


# ----------------------------
# Decoder - input data loading
# ----------------------------

# ------------------------------------------------------------------------------------------
INPUT_MODE: Literal["test", "random-small", "random-medium", "random-big"] = "random-medium"
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    if INPUT_MODE == "test":
        data = torch.load("original_models/data/text_encoder_input.pt")
        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        text_mask = data["text_mask"]
    elif INPUT_MODE == "random-small":
        input_ids = torch.randint(0, 178, size=(1, 16))
        input_ids[0][0] = 0
        input_ids[0][15] = 0
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
    elif INPUT_MODE == "random-medium":
        input_ids = torch.randint(0, 178, size=(1, 64))
        input_ids[0][0] = 0
        input_ids[0][63] = 0
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
    elif INPUT_MODE == "random-big":
        input_ids = torch.randint(0, 178, size=(1, 256))
        input_ids[0][0] = 0
        input_ids[0][255] = 0
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
    else:
        raise RuntimeError("Invalid input mode!")

    inputs = (
        input_ids,
        input_lengths,
        text_mask
    )


# -------------------------
# Decoder - export pipeline
# -------------------------

def convert_to_executorch_program(inputs):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    encoder = TextEncoderWrapper(model)
    encoder.eval()

    # Apply deparametrization
    remove_weight_norms(encoder)

    # Export
    exported_program = torch.export.export(encoder, inputs)

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    )

    return encoder, executorch_program

if __name__ == "__main__":
    _, executorch_program = convert_to_executorch_program(inputs)
    executorch_program = executorch_program.to_executorch()

    print_delegation_info(executorch_program.exported_program().graph_module)

    # Save exported file
    ROOT_DESTINATION = "exported_models/tmp"

    with open(f"{ROOT_DESTINATION}/text_encoder_{INPUT_MODE}.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Finished!")