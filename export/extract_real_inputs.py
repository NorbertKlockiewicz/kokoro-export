from kokoro import KModel
from huggingface_hub import hf_hub_download
import torch

# Fixed seed for consistency
torch.manual_seed(410375)

# Create unmodified, PyTorch model
model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True).eval()


# -------------------------------------------
# INPUTS - phonemes, voice & target no tokens
# -------------------------------------------

phonemes = "həlˈoʊ wˈɝld"

voice = "af_bella"

TARGET_TOKENS = 16

# -------------------------------------------


# Go through the kokoro pipeline and save all partial models inputs
# Voice download
voice_file = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices/af_bella.pt")
voice_style: torch.FloatTensor = torch.load(voice_file, weights_only=True)
if voice_style.dim() == 3:
    voice_style = voice_style.mean(dim=0)
if voice_style.dim() == 2 and voice_style.shape[0] != 1:
    if voice_style.shape[1] == 256:
        voice_style = voice_style[0:1]

# Input tokens preprocessing
input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), phonemes)))
while len(input_ids) < (TARGET_TOKENS - 2):
    input_ids.append(0)
input_ids = input_ids[:(TARGET_TOKENS - 2)]
input_ids = torch.LongTensor([[0, *input_ids, 0]])

duration_predictor_input_ids = input_ids.clone()
duration_predictor_ref_s = voice_style.clone()
duration_predictor_speed = torch.tensor([1.0], dtype=torch.float32)

# Model main execution
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

    f0n_predictor_en = en.clone()
    f0n_predictor_s = s.clone()

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

    text_encoder_input_ids = input_ids.clone()
    text_encoder_input_lengths = input_lengths.clone()
    text_encoder_text_mask = text_mask.clone()

    t_en = model.text_encoder(input_ids, input_lengths, ~text_mask)
    asr = t_en @ pred_aln_trg
    ref_s = voice_style[:, :128]

    text_decoder_asr = asr.clone()
    text_decoder_F0_pred = F0_pred.clone()
    text_decoder_N_pred = N_pred.clone()
    text_decoder_ref_s = ref_s.clone()


# Save all input tensors
ROOT_DESTINATION = "original_models/data"
torch.save({
    "input_ids": duration_predictor_input_ids,
    "ref_s": duration_predictor_ref_s,
    "speed": duration_predictor_speed
}, f"{ROOT_DESTINATION}/duration_predictor_input.pt")
torch.save({
    "en": f0n_predictor_en,
    "s": f0n_predictor_s,
}, f"{ROOT_DESTINATION}/f0n_predictor_input.pt")
torch.save({
    "input_ids": text_encoder_input_ids,
    "input_lengths": text_encoder_input_lengths,
    "text_mask": text_encoder_text_mask,
}, f"{ROOT_DESTINATION}/text_encoder_input.pt")
torch.save({
    "asr": text_decoder_asr,
    "F0_pred": text_decoder_F0_pred,
    "N_pred": text_decoder_N_pred,
    "ref_s": text_decoder_ref_s
}, f"{ROOT_DESTINATION}/text_decoder_input.pt")
