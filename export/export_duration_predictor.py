from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from huggingface_hub import hf_hub_download
from torch.nn import Module
from typing import Literal
import torch

# ---------------------------------------
# Duration predictor - exported interface
# ---------------------------------------

class DurationPredictorWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.text_encoder_module = model.predictor.text_encoder
        self.lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.Tensor):
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)

        bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = self.text_encoder_module(d_en, s, input_lengths, ~text_mask)
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        return pred_dur, d, s


# --------------------------------------
# Duration predictor - model adjustments
# --------------------------------------

# ...


# ---------------------------------------
# Duration predictor - input data loading
# ---------------------------------------

# ------------------------------------------------------------------------------------------
INPUT_MODE: Literal["test", "random-small", "random-medium", "random-big"] = "random-medium"
# ------------------------------------------------------------------------------------------

if __name__ == "__main__":
    if INPUT_MODE == "test":
        data = torch.load("original_models/data/duration_predictor_input.pt")
        input_ids = data["input_ids"]
        ref_s = data["ref_s"]
        speed = data["speed"]
    elif INPUT_MODE == "random-small":
        input_ids = torch.randint(0, 178, size=(1, 16))
        input_ids[0][0] = 0
        input_ids[0][15] = 0
        ref_s = torch.randn(size=(1, 256))
        speed = torch.tensor([1.0], dtype=torch.float32)
    elif INPUT_MODE == "random-medium":
        input_ids = torch.randint(0, 178, size=(1, 64))
        input_ids[0][0] = 0
        input_ids[0][63] = 0
        ref_s = torch.randn(size=(1, 256))
        speed = torch.tensor([1.0], dtype=torch.float32)
    elif INPUT_MODE == "random-big":
        input_ids = torch.randint(0, 178, size=(1, 256))
        input_ids[0][0] = 0
        input_ids[0][255] = 0
        ref_s = torch.randn(size=(1, 256))
        speed = torch.tensor([1.0], dtype=torch.float32)
    else:
        raise RuntimeError("Invalid input mode!")

    inputs = (
        input_ids,
        ref_s,
        speed
    )


# ------------------------------------
# Duration predictor - export pipeline
# ------------------------------------

def convert_to_executorch_program(inputs):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    duration_predictor = DurationPredictorWrapper(model)
    duration_predictor.eval()

    # Apply modifications
    # ...

    # Export
    exported_program = torch.export.export(duration_predictor, inputs)

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    )

    return duration_predictor, executorch_program

if __name__ == "__main__":
    _, executorch_program = convert_to_executorch_program(inputs)
    executorch_program = executorch_program.to_executorch()

    print_delegation_info(executorch_program.exported_program().graph_module)

    # Save exported file
    ROOT_DESTINATION = "exported_models/tmp"

    with open(f"{ROOT_DESTINATION}/duration_predictor_{INPUT_MODE}.pte", "wb") as file:
        file.write(executorch_program.buffer)

    print("Finished!")