from .istftnet import Decoder
from .modules import CustomAlbert
from .temporal_scaling import scale
from dataclasses import dataclass
from executorch.runtime import Runtime
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AlbertConfig
from typing import Dict, Optional, Union
import json
import torch


runtime = Runtime.get()


class KModelPTE(torch.nn.Module):
    """
    KModel is a torch.nn.Module with 2 main responsibilities:
    1. Init weights, downloading config.json + model.pth from HF if needed
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    You likely only need one KModel instance, and it can be reused across
    multiple KPipelines to avoid redundant memory allocation.

    Unlike KPipeline, KModel is language-blind.

    KModel stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModel.
    """

    MODEL_NAMES = {
        "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
        "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
    }

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
                logger.debug("No config provided, downloading from HF")
                config = hf_hub_download(repo_id=repo_id, filename="config.json")
            with open(config, "r", encoding="utf-8") as r:
                config = json.load(r)
                logger.debug(f"Loaded config: {config}")
        self.vocab = config["vocab"]
        self.bert = CustomAlbert(
            AlbertConfig(vocab_size=config["n_token"], **config["plbert"])
        )
        self.bert_encoder = torch.nn.Linear(
            self.bert.config.hidden_size, config["hidden_dim"]
        )
        self.context_length = self.bert.config.max_position_embeddings

        # Use exported .pte models
        # self.duration_predictor = runtime.load_program("exported_models/duration_predictor_16.pte").load_method("forward")
        # self.f0n_predictor = runtime.load_program("exported_models/f0n_predictor_16.pte").load_method("forward")
        # self.text_encoder = runtime.load_program("exported_models/text_encoder_16.pte").load_method("forward")
        # self.text_decoder = runtime.load_program("exported_models/text_decoder_16.pte").load_method("forward")
        self.duration_predictor = runtime.load_program("exported_models/duration_predictor_16.pte").load_method("forward")
        self.f0n_predictor = runtime.load_program("exported_models/f0n_predictor_16.pte").load_method("forward")
        self.text_encoder = runtime.load_program("exported_models/text_encoder_16.pte").load_method("forward")
        self.text_decoder = runtime.load_program("exported_models/text_decoder_16.pte").load_method("forward")

    @property
    def device(self):
        return self.bert.device

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # To fix the repeated parts of code'
        input_lengths = torch.tensor(input_ids.shape[-1])
        text_mask = torch.ones((1, input_ids.shape[-1]), dtype=torch.bool)

        # ------------------------------------------------------------------------------
        # DurationPredictor (duration_predictor.pte) - returns pred_dur, d, s
        # ------------------------------------------------------------------------------
        pred_dur, d, s = self.duration_predictor.execute((input_ids, ref_s, torch.tensor([speed], dtype=torch.float32)))
        # ------------------------------------------------------------------------------

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=self.device), pred_dur
        )

        # Adjustment to fixate the indices length
        # This is a LLM generated shit, replace it with a proper code and algorithm
        print("Original duration length:", len(indices))
        target_len = 64

        indices = scale(indices, target_len)

        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        # # Force batch size = 1.
        # pred_aln_trg = pred_aln_trg.to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg

        # ------------------------------------------------------------------------------
        # F0NPredictor (f0n_predictor.pte) - returns F0_pred, N_pred
        # ------------------------------------------------------------------------------
        F0_pred, N_pred = self.f0n_predictor.execute((en, s))
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # TextEncoderWrapper (text_encoder.pte) - returns t_en, takes only input_ids
        # ------------------------------------------------------------------------------
        t_en = self.text_encoder.execute((input_ids, input_lengths, text_mask))[0]
        # ------------------------------------------------------------------------------

        asr = t_en @ pred_aln_trg

        # ------------------------------------------------------------------------------
        # TextDecoderWrapper (text_decoder_16_xxx.pte) - returns audio
        # ------------------------------------------------------------------------------
        audio = self.text_decoder.execute((asr, F0_pred, N_pred, ref_s[:, :128]))[0]
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
        logger.debug(f"phonemes: {phonemes} -> input_ids: {input_ids}")
        assert len(input_ids) + 2 <= self.context_length, (
            len(input_ids) + 2,
            self.context_length,
        )

        print("Original number of tokens:", len(input_ids))

        # Cut the number of tokens (as models are being exported with static input)
        TARGET_TOKENS = 16
        while len(input_ids) < (TARGET_TOKENS - 2):
            input_ids.append(0)
        input_ids = input_ids[:(TARGET_TOKENS - 2)]
        input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(self.device)

        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio
