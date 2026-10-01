# SPDX-License-Identifier: Apache-2.0
"""TPU Autoregressive (AR) Model Runner for Omni models."""

from typing import Any

from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
)

from vllm_torchtpu.omni.omni_tpu_model_runner import OmniTPUModelRunner


class OmniTPUARModelRunner(OmniConnectorModelRunnerMixin, OmniTPUModelRunner):
    """TPU model runner for autoregressive stages (thinker, talker) in Omni models."""

    def forward_model(self, *args: Any, **kwargs: Any) -> Any:
        out, aux = super().forward_model(*args, **kwargs)
        return getattr(out, "text_hidden_states", out), aux

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        output = super().sample_tokens(grammar_output)
        if getattr(self, "_omni_connector_initialized", False):
            return self.attach_omni_connector_output(output)
        return output
