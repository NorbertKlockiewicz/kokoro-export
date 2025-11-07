from kokoro.istftnet import Decoder
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import executorch
import json
import torch
from kokoro.istftnet import Decoder
from huggingface_hub import hf_hub_download
from executorch.devtools import generate_etrecord
from executorch.exir import (
    EdgeCompileConfig,
    EdgeProgramManager,
    ExecutorchProgramManager,
    to_edge,
)
import copy
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.runtime import Runtime
from executorch.devtools import Inspector


# -----
# Model
# -----

# model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()
# decoder = model.decoder

CONFIG_FILEPATH = "config.json"
with open(CONFIG_FILEPATH, 'r', encoding='utf-8') as r:
	config = json.load(r)

# Create a decoder
decoder = Decoder(
    dim_in=config["hidden_dim"],
    style_dim=config["style_dim"],
    dim_out=config["n_mels"],
    disable_complex=True,
    **config["istftnet"],
)
decoder.eval()

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

# Apply deparametrization
remove_weight_norms(decoder)


# ----------
# Input data
# ----------

data = torch.load("output/real_inputs.pt")
asr = data["asr"]
F0_pred = data["F0_pred"]
N_pred = data["N_pred"]
ref_s = data["ref_s"]

my_inputs = (
    asr,
    F0_pred,
    N_pred,
    ref_s
)

# -----------------
# Generate ETRecord
# -----------------

# exported_program = torch.export.export(decoder, my_inputs)
# executorch_program: EdgeProgramManager = to_edge_transform_and_lower(
#     exported_program,
#     partitioner = [XnnpackPartitioner()]
# )

# edge_program_manager_copy = copy.deepcopy(executorch_program)
# et_program_manager: ExecutorchProgramManager = executorch_program.to_executorch()

# etrecord_path = "debug/etrecord.bin"
# generate_etrecord(etrecord_path, edge_program_manager_copy, et_program_manager)


# ---------------
# Generate ETDump
# ---------------

# # Step 1: ExecuTorch Program Export
# m_name = "forward"
# method_graphs = {m_name: torch.export.export(decoder, my_inputs, strict=True)}

# # Step 2: Construct Method Test Suites
# # inputs = [my_inputs for _ in range(2)]

# method_test_suites = [
#     MethodTestSuite(
#         method_name=m_name,
#         test_cases=[
#             MethodTestCase(inputs=my_inputs, expected_outputs=getattr(decoder, m_name)(*my_inputs))
#         ],
#     )
# ]

# # Step 3: Generate BundledProgram
# executorch_program = to_edge_transform_and_lower(method_graphs, partitioner=[XnnpackPartitioner()]).to_executorch()
# bundled_program = BundledProgram(executorch_program, method_test_suites)

# # Step 4: Serialize BundledProgram to flatbuffer.
# serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
#     bundled_program
# )
# save_path = "bundled_program.bp"
# with open(save_path, "wb") as f:
#     f.write(serialized_bundled_program)


# -------
# Profile
# -------

etrecord_path = "debug/etrecord.bin"
etdump_path = "debug/etdump.etdp"
inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)
inspector.print_data_tabular()