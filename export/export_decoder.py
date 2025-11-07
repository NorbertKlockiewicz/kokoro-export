from kokoro import KModel
from kokoro.istftnet import Decoder
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from huggingface_hub import hf_hub_download
from torch.nn import Module
from typing import Literal
import torch

# ----------------------------
# Decoder - exported interface
# ----------------------------

class DecoderWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr: torch.FloatTensor, F0_pred: torch.FloatTensor,
                N_pred: torch.FloatTensor, ref_s: torch.FloatTensor):
        return self.decoder(asr, F0_pred, N_pred, ref_s)


# ---------------------------
# Decoder - model adjustments
# ---------------------------

# Get rid of all weight_norm decllarations to enable delegating the convolutions to XNNPACK backend
# NOTE: Speeds up the entire inference from ~62 seconds to 0.68 seconds
def remove_weight_norms(decoder):
    # decoder -> generator -> resblocks -> convs1, convs2
    for module in decoder.generator.resblocks:
        for conv in module.convs1:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
        for conv in module.convs2:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> generator -> noise_res -> convs1, convs2
    for module in decoder.generator.noise_res:
        for conv in module.convs1:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
        for conv in module.convs2:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> generator -> ups
    for conv_t in decoder.generator.ups:
        _ = conv_t.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv_t, "weight", leave_parametrized=True)
    # decoder -> generator -> conv_post
    _ = decoder.generator.conv_post.weight
    torch.nn.utils.parametrize.remove_parametrizations(decoder.generator.conv_post, "weight", leave_parametrized=True)
    # decoder -> encode, decode -> pool, conv1, conv2, conv1x1
    modules = [module for module in decoder.decode] + [decoder.encode]
    for module in modules:
        module_weight_norms = [module.conv1, module.conv2]
        if (hasattr(module.pool, "weight")):
            module_weight_norms.append(module.pool)
        if (hasattr(module.conv1x1, "weight")):
            module_weight_norms.append(module.conv1x1)
        for conv in module_weight_norms:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> F0_conv, N_conv, asr_res
    for conv in [decoder.F0_conv, decoder.N_conv, decoder.asr_res[0]]:
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
        data = torch.load("original_models/data/text_decoder_input.pt")
        asr = data["asr"]
        F0_pred = data["F0_pred"]
        N_pred = data["N_pred"]
        ref_s = data["ref_s"]
    elif INPUT_MODE == "random-small":
        asr = torch.randn(size=(1, 512, 64))
        F0_pred = torch.randn(size=(1, 128))
        N_pred = torch.randn(size=(1, 128))
        ref_s = torch.randn(size=(1, 128))
    elif INPUT_MODE == "random-medium":
        asr = torch.randn(size=(1, 512, 256))
        F0_pred = torch.randn(size=(1, 512))
        N_pred = torch.randn(size=(1, 512))
        ref_s = torch.randn(size=(1, 128))
    elif INPUT_MODE == "random-big":
        asr = torch.randn(size=(1, 512, 1024))
        F0_pred = torch.randn(size=(1, 4096))
        N_pred = torch.randn(size=(1, 4096))
        ref_s = torch.randn(size=(1, 128))
    else:
        raise RuntimeError("Invalid input mode!")

    inputs = (
        asr,
        F0_pred,
        N_pred,
        ref_s
    )


# -------------------------
# Decoder - export pipeline
# -------------------------

def convert_to_executorch_program(inputs):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    decoder = DecoderWrapper(model)
    decoder.eval()

    # Apply deparametrization
    remove_weight_norms(decoder.decoder)

    # Export
    exported_program = torch.export.export(decoder, inputs)

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    )

    return decoder, executorch_program

if __name__ == "__main__":
    _, executorch_program = convert_to_executorch_program(inputs)
    executorch_program = executorch_program.to_executorch()

    print_delegation_info(executorch_program.exported_program().graph_module)

    # Save exported file
    ROOT_DESTINATION = "exported_models/tmp"

    with open(f"{ROOT_DESTINATION}/text_decoder_{INPUT_MODE}.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Finished!")