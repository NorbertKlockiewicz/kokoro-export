from executorch.runtime import Runtime
from huggingface_hub import hf_hub_download
from kokoro import find_voice_bound, scale
from misaki import en, espeak
import numpy as np
import os
import torch

runtime = Runtime.get()

# Inference example
#
# This demo script focuses on showcasing how to utilize exported
# models (.pte) and create a full Kokoro pipeline to transform
# the input text into output audio vector.

if __name__ == "__main__":
  # Constants
  REPO_ID = "software-mansion/react-native-executorch-kokoro"
  CACHE_DIR = "."
  MODEL_VOCAB = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11,
    "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19,
    "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\uAB67": 23, "A": 24, "I": 25, "O": 31, "Q": 33,
    "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45,
    "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54,
    "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62,
    "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70,
    "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81,
    "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99,
    "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113,
    "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123,
    "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132,
    "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142,
    "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162,
    "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177
  }

  # Define paths to Kokoro submodules:
  # - DurationPredictor
  # - F0NPredictor
  # - TextEncoder
  # - TextDecoder
  # If selected paths do not exist, the script will attempt to download 
  # the files from the huggingface repository.
  MODEL_FILEPATHS = {
    "duration_predictor": ...,
    "f0n_predictor": ...,
    "text_encoder": ...,
    "text_decoder": ...
  }

  # Download missing model files if not present in selected paths
  for name, path in MODEL_FILEPATHS.items():
      filename = name + ".pte"
      if path is None or path is ... or not os.path.exists(path):
          user_input = input(f"{filename} not found. Download from HuggingFace? [y/N]: ").strip().lower()
          if user_input == "y":
              new_path = hf_hub_download(REPO_ID, "xnnpack/" + filename, cache_dir=CACHE_DIR)
              MODEL_FILEPATHS[name] = new_path
          else:
              print(f"Skipping download for {filename}.")
              MODEL_FILEPATHS[name] = None

  # Load .pte models
  # The models have 3 available static input sizes: for 32, 64 and 128 tokens.
  # The target durations are maximum likelyhood estimators for duration input shapes (for f0n_predictor, encoder and decoder).
  # Each number of input tokens maps to a different static duration: 
  # - 32 tokens -> 92 duration,
  # - 64 tokens -> 164 duration,
  # - 128 tokens -> 296 duration,
  # You can change the input size HERE:
  INPUT_SIZE = 64
  TARGET_DURATION = 164
  METHOD_NAME = f"forward_{INPUT_SIZE}"

  duration_predictor = runtime.load_program(MODEL_FILEPATHS["duration_predictor"]).load_method(METHOD_NAME)
  f0n_predictor = runtime.load_program(MODEL_FILEPATHS["f0n_predictor"]).load_method(METHOD_NAME)
  text_encoder = runtime.load_program(MODEL_FILEPATHS["text_encoder"]).load_method(METHOD_NAME)
  text_decoder = runtime.load_program(MODEL_FILEPATHS["text_decoder"]).load_method(METHOD_NAME)

  # Load sample voice from HF
  VOICE_PATH = hf_hub_download(REPO_ID, "voices/af_heart.bin", cache_dir=CACHE_DIR)
  with open(VOICE_PATH, "rb") as f:
    voice_bytes = f.read()
    voice_array = np.frombuffer(voice_bytes, dtype=np.float32).reshape(510, 1, 256)
    voice = torch.from_numpy(voice_array)
  print(voice.shape)

  # Define and phonemize the input text
  # Kokoro does not take plain text as an input - it takes phonemes (spoken representation).
  # This means you must use some phonemizer such as Misaki in this case.
  INPUT_TEXT = "Kokoro is such a wonderful model!"
  USE_FALLBACK = True

  fallback = espeak.EspeakFallback(british=False) if USE_FALLBACK else None
  phonemizer = en.G2P(trf=False, fallback=fallback, unk='')

  INPUT_PHONEMES = phonemizer(INPUT_TEXT)[0]
  print("Input phonemes:", INPUT_PHONEMES)

  # Kokoro inference starts
  # -----------------------
  # Token processing
  input_ids = [MODEL_VOCAB.get(p) for p in INPUT_PHONEMES if MODEL_VOCAB.get(p) is not None]
  original_input_length = min(len(input_ids) + 2, INPUT_SIZE)
  # Pad or truncate tokens
  input_ids = ([0] + input_ids[:INPUT_SIZE-2] + [0])
  while len(input_ids) < INPUT_SIZE:
      input_ids.append(0)
  input_tokens = torch.tensor([input_ids], dtype=torch.int64)
  # Model inputs preparation
  v_ref = voice[len(INPUT_PHONEMES) - 1][:, :128]
  v_style = voice[len(INPUT_PHONEMES) - 1][:, 128:]
  text_mask = torch.ones((1, INPUT_SIZE), dtype=torch.bool)
  text_mask[:, original_input_length:] = False
  speed = torch.tensor([1.0], dtype=torch.float32)

  # 1. Duration Predictor
  pred_dur, d = duration_predictor.execute((input_tokens, text_mask, v_style, speed))

  # Native duration scaling
  pred_dur_scaled = scale(pred_dur, TARGET_DURATION)
  indices = torch.repeat_interleave(torch.arange(INPUT_SIZE), pred_dur_scaled.view(-1))[:TARGET_DURATION]

  # 2. F0 Predictor
  f0_pred, n_pred, _, pred_aln_trg = f0n_predictor.execute((indices, d, v_style))

  # Calculate efficient duration to trim silence
  first_index = None
  for i, val in enumerate(indices):
      if val.item() == original_input_length:
          first_index = i
          break
  eff_dur = int((first_index if first_index is not None else TARGET_DURATION) * 0.95)

  # 3. Text Encoder
  asr = text_encoder.execute((input_tokens, text_mask, pred_aln_trg))[0]

  # 4. Text Decoder
  audio_out = text_decoder.execute((asr, f0_pred, n_pred, v_ref))[0]
  audio_out = audio_out[:, :, :600 * eff_dur].squeeze().cpu()
  # -------------------------
  # Kokoro inference finishes

  # Post-processing
  # Crop the audio to remove silent fragments from both ends.
  v_start = find_voice_bound(audio_out, from_end=False)
  v_end = find_voice_bound(audio_out, from_end=True)
  final_audio = audio_out[v_start:v_end]

  print("Inference succeeded!")
  print("Result:", final_audio)
