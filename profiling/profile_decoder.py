from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.devtools import generate_etrecord
from executorch.exir import (
    EdgeProgramManager,
    ExecutorchProgramManager,
)
from torch.nn import Module
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from executorch.devtools import Inspector
import copy
import pandas as pd
import subprocess
import torch
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# Decoder - exported interface
# ----------------------------

print("Defining DecoderWrapper...")
class DecoderWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr: torch.FloatTensor, F0_pred: torch.FloatTensor,
                N_pred: torch.FloatTensor, ref_s: torch.FloatTensor):
        return self.decoder(asr, F0_pred, N_pred, ref_s)
    

print("DecoderWrapper defined.")

# ------------------------
# Decoder - model instance
# ------------------------

print("Creating KModel and DecoderWrapper instance...")
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
decoder = DecoderWrapper(model)
decoder.eval()
print("Model and DecoderWrapper instance created and set to eval mode.")

# ---------------------------
# Decoder - model adjustments
# ---------------------------

print("Removing weight_norm parametrizations from decoder...")
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
remove_weight_norms(decoder.decoder)
print("Weight_norm parametrizations removed.")

# ----------------------------
# Decoder - input data loading
# ----------------------------

print("Loading input data for decoder...")
data = torch.load("original_models/data/text_decoder_input.pt")
asr = data["asr"]
F0_pred = data["F0_pred"]
N_pred = data["N_pred"]
ref_s = data["ref_s"]

inputs = (
    asr,
    F0_pred,
    N_pred,
    ref_s
)
print("Input data loaded.")

# ------------
# Common paths
# ------------

PROFILE_ARTIFACTS_ROOT = "profiling/artifacts"
PROFILE_RESULTS_ROOT = "profiling/results"


# -----------------
# Generate ETRecord
# -----------------

print("Exporting program and generating ETRecord...")
exported_program = torch.export.export(decoder, inputs)
executorch_program: EdgeProgramManager = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
)

edge_program_manager_copy = copy.deepcopy(executorch_program)
et_program_manager: ExecutorchProgramManager = executorch_program.to_executorch()

etrecord_path = f"{PROFILE_ARTIFACTS_ROOT}/etrecord.bin"
generate_etrecord(etrecord_path, edge_program_manager_copy, et_program_manager)
print(f"ETRecord generated and saved to {etrecord_path}.")

# ---------------
# Generate ETDump
# ---------------

print("Generating ETDump and BundledProgram...")
m_name = "forward"
method_graphs = {m_name: torch.export.export(decoder, inputs, strict=True)}

method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(inputs=inputs, expected_outputs=getattr(decoder, m_name)(*inputs))
        ],
    )
]

# Step 3: Generate BundledProgram
executorch_program = to_edge_transform_and_lower(method_graphs, partitioner=[XnnpackPartitioner()]).to_executorch()
bundled_program = BundledProgram(executorch_program, method_test_suites)

# Step 4: Serialize BundledProgram to flatbuffer.
serialized_bundled_program = serialize_from_bundled_program_to_flatbuffer(
    bundled_program
)
bp_path = f"{PROFILE_ARTIFACTS_ROOT}/bundled_program.bp"
with open(bp_path, "wb") as f:
    f.write(serialized_bundled_program)
print(f"BundledProgram serialized and saved to {bp_path}.")

print("Running executorch example runner...")
subprocess.run(
    ["../executorch/cmake-out/examples/devtools/example_runner", f"--bundled_program_path={bp_path}"],
    check=True
)
print("executorch example runner finished.")

# -------
# Profile
# -------

print("Profiling with Inspector and saving results...")
etdump_path = f"{PROFILE_ARTIFACTS_ROOT}/etdump.etdp"

inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)
tabular_data = inspector.to_dataframe()
tabular_data.to_csv(f"{PROFILE_RESULTS_ROOT}/profiling_results.csv", index=False)
print(f"Profiling results saved to {PROFILE_RESULTS_ROOT}/profiling_results.csv")