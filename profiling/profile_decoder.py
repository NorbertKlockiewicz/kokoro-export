from kokoro import KModel
from executorch.devtools import generate_etrecord
from executorch.exir import ExecutorchProgramManager
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools import BundledProgram
from executorch.devtools.bundled_program.serialize import serialize_from_bundled_program_to_flatbuffer
from executorch.devtools import Inspector
from export.export_decoder import convert_to_executorch_program
import copy
import pandas as pd
import subprocess
import torch
import warnings

warnings.filterwarnings("ignore")


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
_, executorch_program = convert_to_executorch_program(inputs)

edge_program_manager_copy = copy.deepcopy(executorch_program)
et_program_manager: ExecutorchProgramManager = executorch_program.to_executorch()

etrecord_path = f"{PROFILE_ARTIFACTS_ROOT}/etrecord.bin"
generate_etrecord(etrecord_path, edge_program_manager_copy, et_program_manager)
print(f"ETRecord generated and saved to {etrecord_path}.")


# ---------------
# Generate ETDump
# ---------------

print("Generating ETDump and BundledProgram...")

decoder, executorch_program = convert_to_executorch_program(inputs)

m_name = "forward"
method_test_suites = [
    MethodTestSuite(
        method_name=m_name,
        test_cases=[
            MethodTestCase(inputs=inputs, expected_outputs=getattr(decoder, m_name)(*inputs))
        ],
    )
]

# Step 3: Generate BundledProgram
executorch_program = executorch_program.to_executorch()
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

# Move etdump.etdp to PROFILE_ARTIFACTS_ROOT directory
import shutil
shutil.move("etdump.etdp", f"{PROFILE_ARTIFACTS_ROOT}/etdump.etdp")

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