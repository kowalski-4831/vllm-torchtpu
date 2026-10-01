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

    def forward_model(self, *args: Any, **kwargs: Any) -> Any:
        out, aux = super().forward_model(*args, **kwargs)
        if hasattr(out, "text_hidden_states"):
            self._omni_last_model_output = out
            return out.text_hidden_states, aux
        return out, aux

    def _on_chunk_prepared(self, start: int, num_reqs: int) -> None:
        if hasattr(super(), "_on_chunk_prepared"):
            super()._on_chunk_prepared(start, num_reqs)

    def _sample_chunk(
        self, logits: torch.Tensor, start: int, end: int, **kwargs: Any
    ) -> Any:
        if hasattr(self.model, "sample") and callable(self.model.sample):
            return self.model.sample(logits, start, end, **kwargs)
        if hasattr(super(), "_sample_chunk"):
            return super()._sample_chunk(logits, start, end, **kwargs)
        return None

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        output = super().sample_tokens(grammar_output)
        if getattr(self, "_omni_connector_initialized", False):
            return self.attach_omni_connector_output(output)
        return self.build_omni_output(output)
