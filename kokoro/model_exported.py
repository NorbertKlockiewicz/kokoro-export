from .modules import CustomAlbert
from .temporal_scaling import scale
from dataclasses import dataclass
from executorch.runtime import Runtime
from huggingface_hub import hf_hub_download
from .postprocessing import find_voice_bound
from typing import Dict, Optional, Union
import json
import torch


runtime = Runtime.get()


class KModelPTE(torch.nn.Module):
    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        model: Optional[str] = None,
        disable_complex: bool = False,
    ):
        super().__init__()
        if repo_id is None:
            repo_id = "hexgrad/Kokoro-82M"
            print(
                f"WARNING: Defaulting repo_id to {repo_id}. Pass repo_id='{repo_id}' to suppress this warning."
            )
        self.repo_id = repo_id
        if not isinstance(config, dict):
            if not config:
                # logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config, "r", encoding="utf-8") as r:
                config = json.load(r)
                # logger.debug(f"Loaded config: {config}")
        self.vocab = config["vocab"]

        self.context_length = 510

        # Use exported .pte models
        self.TARGET_TOKENS = 128
        self.TARGET_LEN = 296
        # method_name = "forward"
        method_name = f"forward_{self.TARGET_TOKENS}"
        # self.duration_predictor = runtime.load_program("exported_models/tmp/duration_predictor_64.pte").load_method(method_name)
        # self.f0n_predictor = runtime.load_program("exported_models/tmp/f0n_predictor_64.pte").load_method(method_name)
        # self.text_encoder = runtime.load_program("exported_models/tmp/text_encoder_64.pte").load_method(method_name)
        # self.text_decoder = runtime.load_program("exported_models/tmp/text_decoder_64.pte").load_method(method_name)
        self.duration_predictor = runtime.load_program("exported_models/xnnpack-newsizes/duration_predictor.pte").load_method(method_name)
        self.f0n_predictor = runtime.load_program("exported_models/xnnpack-newsizes/f0n_predictor.pte").load_method(method_name)
        self.text_encoder = runtime.load_program("exported_models/xnnpack-newsizes/text_encoder.pte").load_method(method_name)
        # self.text_decoder = runtime.load_program("exported_models/xnnpack-newsizes/text_decoder.pte").load_method(method_name)
        self.text_decoder = runtime.load_program("exported_models/dynamic-shapes/text_decoder.pte").load_method("forward")

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float, original_input_length: int
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # To fix the repeated parts of code'
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)
        text_mask[:, original_input_length:] = 0

        ref = ref_s[:, :128]
        s = ref_s[:, 128:]

        # ------------------------------------------------------------------------------
        # DurationPredictor (duration_predictor.pte) - returns pred_dur, d, en
        # ------------------------------------------------------------------------------
        pred_dur, d = self.duration_predictor.execute((input_ids, text_mask, s, torch.tensor([speed], dtype=torch.float32)))
        # ------------------------------------------------------------------------------

        # NATIVE ------------------------------------------------------------------------
        pred_dur = scale(pred_dur, self.TARGET_LEN)
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1]), pred_dur
        )
        # NATIVE ------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # F0NPredictor (f0n_predictor.pte) - returns F0_pred, N_pred
        # ------------------------------------------------------------------------------
        # en = (1, 640, 16) x (1, 16, 64)
        F0_pred, N_pred, en, pred_aln_trg = self.f0n_predictor.execute((indices, d, s))
        # ------------------------------------------------------------------------------

        # NATIVE --------------------------------------------------------------------------------------------------
        first_index = None
        for i in range(indices.shape[0]):
            if indices[i].item() == original_input_length:
                first_index = i
        efficient_duration = first_index if first_index is not None else self.TARGET_LEN
        efficient_duration = int(efficient_duration * 0.95)
        # NATIVE --------------------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # TextEncoderWrapper (text_encoder.pte) - returns asr
        # ------------------------------------------------------------------------------
        asr = self.text_encoder.execute((input_ids, text_mask, pred_aln_trg))[0]
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # TextDecoderWrapper (text_decoder_16_xxx.pte) - returns audio
        # ------------------------------------------------------------------------------
        audio = self.text_decoder.execute((asr, F0_pred, N_pred, ref))[0]
        audio = audio[:, :, :600*efficient_duration]
        # ------------------------------------------------------------------------------

        return audio, pred_dur

    def forward(
        self,
        phonemes: str,
        ref_s: torch.FloatTensor,
        speed: float = 1,
        return_output: bool = False,
    ) -> Union["KModelPTE.Output", torch.FloatTensor]:
        input_ids = list(
            filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes))
        )
        assert len(input_ids) + 2 <= self.context_length, (
            len(input_ids) + 2,
            self.context_length,
        )

        # Cut the number of tokens (as models are being exported with static input)
        original_input_length = min(len(input_ids) + 2, self.TARGET_TOKENS)
        while len(input_ids) < (self.TARGET_TOKENS - 2):
            input_ids.append(0)
        input_ids = input_ids[:(self.TARGET_TOKENS - 2)]
        input_ids = torch.LongTensor([[0, *input_ids, 0]])

        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed, original_input_length)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        
        voice_beg = find_voice_bound(audio, from_end=False)
        voice_end = find_voice_bound(audio, from_end=True)

        audio = audio[voice_beg:voice_end]
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio
