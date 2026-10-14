# SPDX-License-Identifier: Apache-2.0
"""TPU Autoregressive (AR) Model Runner for Omni models."""

from typing import Any

import torch
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner


class OmniTPUARModelRunner(OmniConnectorModelRunnerMixin, OmniTPUModelRunner):
    """TPU model runner for autoregressive stages (thinker, talker) in Omni models."""

    def _gather_mm_embeddings(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_multimodal = super()._gather_mm_embeddings(*args, **kwargs)
        self._omni_mm_embeds_list = mm_embeds
        if mm_embeds:
            total = sum(embed.shape[0] for embed in mm_embeds)
            return [torch.arange(total, dtype=torch.int64)], is_multimodal
        return mm_embeds, is_multimodal

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> Any:
        if mm_embed_inputs is not None:
            chunk_embeds, is_multimodal = mm_embed_inputs
            mm_embeds = getattr(self, "_omni_mm_embeds_list", None)
            if chunk_embeds and mm_embeds and chunk_embeds[0].ndim == 1:
                chunk_start = int(chunk_embeds[0][0])
                chunk_end = chunk_start + chunk_embeds[0].shape[0]
                sliced_embeds, offset = [], 0
                for embed in mm_embeds:
                    rows = embed.shape[0]
                    start = max(0, chunk_start - offset)
                    end = min(rows, chunk_end - offset)
                    if start < end:
                        sliced = embed[start:end]
                        if hasattr(embed, "modality"):
                            sliced.modality = embed.modality
                        sliced_embeds.append(sliced)
                    offset += rows
                mm_embed_inputs = (sliced_embeds, is_multimodal)
        return super()._get_model_inputs(input_ids, mm_embed_inputs)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        output = super().sample_tokens(grammar_output)
        if getattr(self, "_omni_connector_initialized", False):
            return self.attach_omni_connector_output(output)
        return output
