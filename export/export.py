from kokoro.model import KModel, KModelForONNX
from kokoro.istftnet import Decoder
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export
import executorch
import json
import torch

# Load config file
CONFIG_FILEPATH = "config.json"
MODEL_FILEPATH = "model/kokoro.pth"

# Create a model
base_model = KModel(config=CONFIG_FILEPATH, model=MODEL_FILEPATH, disable_complex=True)
model = KModelForONNX(base_model)


# Sample input
FIXED_EMBEDDING_LENGTH = 128

sample_inputs = (
    torch.randint(1, 178, (1, FIXED_EMBEDDING_LENGTH), dtype=torch.long),
    torch.randn(1, 256),
    torch.rand(1).item()
)

# Export
exported_program = torch.export.export(model, sample_inputs)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
).to_executorch()

# Save exported file
with open("output/model.pte", "wb") as file:
    file.write(executorch_program.buffer)

print("Finished!")