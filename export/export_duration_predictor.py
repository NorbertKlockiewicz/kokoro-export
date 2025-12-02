from kokoro import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import argparse
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

    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.Tensor,
                      text_mask: torch.BoolTensor):
        input_lengths = torch.tensor(input_ids.shape[-1])
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


# --------------------------------------------
# Duration predictor - input data definition
# --------------------------------------------

# Predefined inputs for export sizes
inputs_dict = {
    "test": (
        torch.load("original_models/data/duration_predictor_input.pt")["input_ids"],
        torch.load("original_models/data/duration_predictor_input.pt")["ref_s"],
        torch.load("original_models/data/duration_predictor_input.pt")["speed"],
        torch.ones((1, torch.load("original_models/data/duration_predictor_input.pt")["input_ids"].shape[-1]), dtype=torch.bool),
    ),
    "small": (
        torch.randint(0, 178, size=(1, 16)),
        torch.randn(size=(1, 256)),
        torch.tensor([1.0], dtype=torch.float32),
        torch.ones((1, 16), dtype=torch.bool),
    ),
    "medium": (
        torch.randint(0, 178, size=(1, 64)),
        torch.randn(size=(1, 256)),
        torch.tensor([1.0], dtype=torch.float32),
        torch.ones((1, 64), dtype=torch.bool),
    ),
    "big": (
        torch.randint(0, 178, size=(1, 256)),
        torch.randn(size=(1, 256)),
        torch.tensor([1.0], dtype=torch.float32),
        torch.ones((1, 256), dtype=torch.bool),
    ),
}

# Keys: input size names -> Values: equivalent number of input tokens
input_name_mappings = {
    "test": "test",
    "small": "16",
    "medium": "64",
    "big": "256",
}

# ------------------------------------
# Duration predictor - export pipeline
# ------------------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    duration_predictor = DurationPredictorWrapper(model)
    duration_predictor.eval()

    # Apply modifications (none for duration predictor)
    # ...

    exported_program = torch.export.export(duration_predictor, inputs)

    if not transform_and_lower:
        return duration_predictor, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    return duration_predictor, executorch_program

# -------------------------------
# Duration predictor - CLI entry
# -------------------------------

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["test", "small", "medium", "big"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    args = parser.parse_args()

    # Mode selection
    if args.bundled is False and args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")

    bundled = args.bundled
    input_mode = args.input_size if not bundled else "medium"

    ROOT_DESTINATION = "exported_models/tmp"

    if not bundled:
        # Single export
        print(f"Exporting duration predictor with {input_mode} input...")
        input_alias = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        if input_mode in ["small", "medium", "big"]:
            input_ids = inputs[0]
            input_ids[0][0] = 0
            input_ids[0][-1] = 0

        _, edge_program = convert_to_executorch_program(inputs)
        executorch_program = edge_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        with open(f"{ROOT_DESTINATION}/duration_predictor_{input_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        # Bundled export: multiple entry points
        print("Exporting bundled duration predictor...")

        predictors = {}
        for name, n_tokens in input_name_mappings.items():
            inputs = inputs_dict[name]
            if name in ["small", "medium", "big"]:
                input_ids = inputs[0]
                input_ids[0][0] = 0
                input_ids[0][-1] = 0
            _, exported_predictor = convert_to_executorch_program(inputs_dict[name], transform_and_lower=False)
            predictors[n_tokens] = exported_predictor

        executorch_program = to_edge_transform_and_lower(
            {f"forward_{n_tokens}": ep for n_tokens, ep in predictors.items()},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(f"{ROOT_DESTINATION}/duration_predictor.pte", "wb") as file:
            executorch_program.write_to_file(file)

    print("Finished!")