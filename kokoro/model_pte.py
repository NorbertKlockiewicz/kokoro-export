import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from huggingface_hub import hf_hub_download

# executorch imports
from executorch.runtime import Runtime

logger = logging.getLogger(__name__)
runtime = Runtime.get()


class KModelPTE(torch.nn.Module):
    """
    KModelPTE is a torch.nn.Module with 2 main responsibilities:
    1. Load pre-exported .pte modules
    2. forward(phonemes: str, ref_s: FloatTensor) -> (audio: FloatTensor)

    Unlike KPipeline, KModelPTE is language-blind.

    KModelPTE stores self.vocab and thus knows how to map phonemes -> input_ids,
    so there is no need to repeatedly download config.json outside of KModelPTE.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        config: Union[Dict, str, None] = None,
        pte_dir: str = "exported_pte",
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
        # Assuming context_length is needed from config
        self.context_length = config["plbert"]["max_position_embeddings"]

        program = runtime.load_program(f"{pte_dir}/duration_predictor.pte")
        self.duration_predictor = program.load_method("forward")

        program = runtime.load_program(f"{pte_dir}/f0n_predictor.pte")
        self.f0n_predictor = program.load_method("forward")

        program = runtime.load_program(f"{pte_dir}/text_encoder.pte")
        self.text_encoder = program.load_method("forward")

        program = runtime.load_program(f"{pte_dir}/text_decoder_16_det.pte")
        self.decoder = program.load_method("forward")

    @property
    def device(self):
        # ExecuTorch modules are device-agnostic at this level
        return torch.device("cpu")

    @dataclass
    class Output:
        audio: torch.FloatTensor
        pred_dur: Optional[torch.LongTensor] = None

    @torch.no_grad()
    def forward_with_tokens(
        self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: float = 1.0
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        # ------------------------------------------------------------------------------
        # DurationPredictor (duration_predictor.pte) - returns pred_dur, d, s
        # ------------------------------------------------------------------------------
        outputs = self.duration_predictor.execute(
            (input_ids, ref_s, torch.tensor([speed], dtype=torch.float))
        )
        pred_dur, d, s = outputs
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # External code
        # ------------------------------------------------------------------------------
        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=self.device), pred_dur
        )
        torch._check(indices.shape[0] > 0)
        torch._check_is_size(indices.shape[0])
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)
        en = d.transpose(-1, -2) @ pred_aln_trg
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # F0NPredictor (f0n_predictor.pte) - returns F0_pred, N_pred
        # ------------------------------------------------------------------------------
        # PROBLEM 1: en might vary on the last dimension
        if en.shape[-1] != 63:
          if en.shape[-1] > 63:
              en = en[..., :63].contiguous()
          else:
              en = torch.nn.functional.pad(en, (0, 63 - en.shape[-1])).contiguous()
        F0_pred, N_pred = self.f0n_predictor.execute((en, s))
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # TextEncoderWrapper (text_encoder.pte) - returns t_en, takes only input_ids
        # ------------------------------------------------------------------------------
        # PROBLEM 2: requies a squeeze
        (t_en,) = self.text_encoder.execute((input_ids,))
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # External code
        # ------------------------------------------------------------------------------
        asr = t_en @ pred_aln_trg
        # ------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------
        # TextDecoderWrapper (text_decoder_16_xxx.pte) - returns audio
        # ------------------------------------------------------------------------------
        # PROBLEM 3: Decoder's last shape also varies

        print(asr.shape)
        print(F0_pred.shape)
        print(N_pred.shape)

        # Ensure asr.shape[2] == 78
        if asr.shape[2] != 78:
            if asr.shape[2] > 78:
                asr = asr[..., :78].contiguous()
            else:
                asr = torch.nn.functional.pad(asr, (0, 78 - asr.shape[2])).contiguous()

        # Ensure F0_pred.shape[1] == 156
        if F0_pred.shape[1] != 156:
            if F0_pred.shape[1] > 156:
                F0_pred = F0_pred[:, :156, ...].contiguous()
            else:
                pad_size = 156 - F0_pred.shape[1]
                F0_pred = torch.nn.functional.pad(F0_pred, (0, pad_size)).contiguous()

        # Ensure N_pred.shape[1] == 156
        if N_pred.shape[1] != 156:
            if N_pred.shape[1] > 156:
                N_pred = N_pred[:, :156, ...].contiguous()
            else:
                pad_size = 156 - N_pred.shape[1]
                N_pred = torch.nn.functional.pad(N_pred, (0, pad_size)).contiguous()
        
        print(asr.shape)
        print(F0_pred.shape)
        print(N_pred.shape)

        (audio,) = self.decoder.execute((asr, F0_pred, N_pred, ref_s[:, :128]))
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

        TARGET_TOKENS = 16
        while len(input_ids) < (TARGET_TOKENS - 2):
            input_ids.append(0)
        input_ids = input_ids[:(TARGET_TOKENS - 2)]
        input_ids = torch.LongTensor([[0, *input_ids, 0]])

        ref_s = ref_s.to(self.device)
        audio, pred_dur = self.forward_with_tokens(input_ids, ref_s, speed)
        audio = audio.squeeze().cpu()
        pred_dur = pred_dur.cpu() if pred_dur is not None else None
        logger.debug(f"pred_dur: {pred_dur}")
        return self.Output(audio=audio, pred_dur=pred_dur) if return_output else audio
