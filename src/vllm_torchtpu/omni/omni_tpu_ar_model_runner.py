# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU Autoregressive (AR) Model Runner for Omni models."""

from typing import Any

import torch

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUModelRunnerOutput


class OmniTPUARModelRunner(OmniTPUModelRunner):
    """TPU model runner for autoregressive stages (thinker, talker) in Omni models.

    The connector lifecycle lives in ``OmniTPUModelRunner``; this class only
    attaches the connector signals to every output the engine consumes.
    """

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

    def forward_model(self, *args: Any, **kwargs: Any) -> Any:
        out, aux = super().forward_model(*args, **kwargs)
        return getattr(out, "text_hidden_states", out), aux

    def _with_omni_connector_output(self, output: Any) -> Any:
        """Attach connector signals to the ``ModelRunnerOutput`` the engine reads.

        Under async scheduling ``sample_tokens`` returns an
        ``AsyncTPUModelRunnerOutput`` and the engine consumes ``get_output()``,
        so the signals have to live on the wrapped ``ModelRunnerOutput``
        (upstream GPUARModelRunner sets them on the output before wrapping it),
        not on the wrapper, where they would be dropped.
        """
        if output is None or not self._omni_connector_initialized:
            return output
        if isinstance(output, AsyncTPUModelRunnerOutput):
            output._model_runner_output = self.attach_omni_connector_output(
                output._model_runner_output
            )
            return output
        return self.attach_omni_connector_output(output)

    def execute_model(
        self, scheduler_output: Any, intermediate_tensors: Any = None
    ) -> Any:
        output = super().execute_model(scheduler_output, intermediate_tensors)
        # ``None`` is the normal two-phase path: the engine calls sample_tokens()
        # next, which attaches the signals. A zero-token step (and the
        # KV-connector-only variant) returns an output right here and the engine
        # skips sample_tokens(), so attach on this path as well.
        return self._with_omni_connector_output(output)

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        return self._with_omni_connector_output(super().sample_tokens(grammar_output))
