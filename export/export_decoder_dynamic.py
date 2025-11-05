from kokoro.istftnet import Decoder
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import executorch
import json
import torch
from kokoro import KModel
from huggingface_hub import hf_hub_download
from torch.export import Dim


model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()


print("\nGenerating REAL inputs from pipeline...")

voice_file = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/af_bella.pt")
voice_style = torch.load(voice_file, weights_only=True)
if voice_style.dim() == 3:
    voice_style = voice_style.mean(dim=0)
if voice_style.dim() == 2 and voice_style.shape[0] != 1:
    if voice_style.shape[1] == 256:
        voice_style = voice_style[0:1]

phonemes = "həlˈoʊ wˈɝld"
input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
TARGET_TOKENS = 16
while len(input_ids) < (TARGET_TOKENS - 2):
    input_ids.append(0)
input_ids = input_ids[:(TARGET_TOKENS - 2)]
input_ids = torch.LongTensor([[0, *input_ids, 0]])

with torch.no_grad():
    input_lengths = torch.tensor(input_ids.shape[-1])
    text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
    bert_dur = model.bert(input_ids, attention_mask=text_mask.int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = voice_style[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, ~text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1)
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
    device = input_ids.device
    indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=device), pred_dur)
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln_trg
    x = en.transpose(-1, -2)
    x, _ = model.predictor.shared(x)
    F0 = x.transpose(-1, -2)
    for block in model.predictor.F0:
        F0 = block(F0, s)
    F0_pred = model.predictor.F0_proj(F0).squeeze(1)
    N = x.transpose(-1, -2)
    for block in model.predictor.N:
        N = block(N, s)
    N_pred = model.predictor.N_proj(N).squeeze(1)
    t_en = model.text_encoder(input_ids, input_lengths, ~text_mask)
    asr = t_en @ pred_aln_trg
    ref_s = voice_style[:, :128]

print(f"✓ Generated REAL inputs:")
print(f"  asr: {asr.shape}, range=[{asr.min():.3f}, {asr.max():.3f}]")
print(f"  F0_pred: {F0_pred.shape}, range=[{F0_pred.min():.3f}, {F0_pred.max():.3f}]")
print(f"  N_pred: {N_pred.shape}, range=[{N_pred.min():.3f}, {N_pred.max():.3f}]")
print(f"  ref_s: {ref_s.shape}, range=[{ref_s.min():.3f}, {ref_s.max():.3f}]")








# Load config file
CONFIG_FILEPATH = "config.json"
with open(CONFIG_FILEPATH, 'r', encoding='utf-8') as r:
	config = json.load(r)

# Create a decoder
decoder = Decoder(
    dim_in=config['hidden_dim'], style_dim=config['style_dim'],
    dim_out=config['n_mels'], disable_complex=True, **config['istftnet']
)
decoder.eval()

# Sample input
text_sample = "Hello world!"
voice_style = "af_heart"
speed = 1.0
# Decoder-specific random inputs
asr_example = torch.randn(1, 512, 78)
F0_pred_example = torch.randn(1, 156)
N_pred_example = torch.randn(1, 156)
ref_s_example = torch.randn(1, 128)

inputs = (
	asr_example,
	F0_pred_example,
	N_pred_example,
	ref_s_example
)

dynamic_shapes = {
    "asr": {
        2: Dim.DYNAMIC
    },
    "F0_curve": {
        1: Dim.DYNAMIC
    },
    "N": {
        1: Dim.DYNAMIC
    },
    "s": None
}

# inputs = (
#     asr,
#     F0_pred,
#     N_pred,
#     ref_s
# )

# Export
exported_program = torch.export.export(decoder, inputs, dynamic_shapes=dynamic_shapes)
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()],
).to_executorch()

# Save exported file
with open("output/model.pte", "wb") as file:
    file.write(executorch_program.buffer)

print("Finished!")