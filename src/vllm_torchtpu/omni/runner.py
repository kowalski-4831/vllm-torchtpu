# SPDX-License-Identifier: Apache-2.0
"""TPU model runners for vLLM-Omni."""

from typing import Any

import torch
from vllm.logger import init_logger

try:
    from vllm_omni.worker.omni_connector_model_runner_mixin import (
        OmniConnectorModelRunnerMixin,
    )
except (ImportError, ModuleNotFoundError):

    class OmniConnectorModelRunnerMixin:  # type: ignore[no-redef]
        pass

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class VocoderStep:
    """Vocoder execution step for decoding audio codes into waveforms (Stage 2)."""

    def __init__(self, model: Any = None) -> None:
        self.model = model

    def warmup(self) -> None:
        # TODO(qwen3-omni-stage2): Implement TPU vocoder bucket warmup for Code2Wav stage.
        raise NotImplementedError(
            "VocoderStep.warmup is not implemented for Thinker-only stage on TPU."
        )

    def run(self, payloads: list[Any]) -> list[Any]:
        # TODO(qwen3-omni-stage2): Implement TPU vocoder decoding for Code2Wav stage.
        raise NotImplementedError(
            "VocoderStep.run is not implemented for Thinker-only stage on TPU."
        )


class OmniTPUModelRunner(OmniConnectorModelRunnerMixin, TPUModelRunner):
    """Base Omni TPU Model Runner with Thinker support and stage stubs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self._omni_last_model_output: Any = None
        self._omni_last_captured_layers: dict[str, Any] | None = None
        self._in_profile_run: bool = False
        if hasattr(self, "model_config"):
            try:
                self.init_omni_connectors(self.model_config)
            except Exception as e:
                logger.debug("[OmniTPUModelRunner] init_omni_connectors skipped: %s", e)

    @property
    def _model_has_multimodal_outputs(self) -> bool:
        return bool(getattr(self.model, "has_multimodal_outputs", False))

    def profile_run(self, *args: Any, **kwargs: Any) -> None:
        if getattr(self.model, "model_stage", None) == "code2wav":
            return
        self._in_profile_run = True
        try:
            super().profile_run(*args, **kwargs)
        finally:
            self._in_profile_run = False

    def capture_model(self, *args: Any, **kwargs: Any) -> int:
        if self.model_config.enforce_eager or getattr(
            self.model, "model_stage", None
        ) in ("talker", "code2wav"):
            return 0
        return super().capture_model(*args, **kwargs)

    def forward_model(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
    ) -> tuple[Any, Any]:
        out, aux = super().forward_model(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )
        self._omni_last_model_output = None
        self._omni_last_captured_layers = aux if isinstance(aux, dict) else None
        if isinstance(aux, dict):
            aux = None
        if hasattr(out, "text_hidden_states"):
            self._omni_last_model_output = out
            out = out.text_hidden_states
        return out, aux

    def _sync_local_stage_payloads(self, scheduler_output: Any = None) -> None:
        # TODO(qwen3-omni-stage1): Implement inter-stage payload sync from Thinker to Talker.
        raise NotImplementedError(
            "_sync_local_stage_payloads is not implemented for Thinker-only stage on TPU."
        )

    def _execute_talker_mtp(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # TODO(qwen3-omni-stage1): Implement Talker multi-token code predictor execution.
        raise NotImplementedError(
            "_execute_talker_mtp is not implemented for Thinker-only stage on TPU."
        )


class OmniTPUARModelRunner(OmniTPUModelRunner):
    """TPU AR Model Runner for Omni Thinker stage."""

    def _attach_connector(self, output: Any) -> Any:
        if hasattr(self, "attach_omni_connector_output"):
            try:
                return self.attach_omni_connector_output(output)
            except Exception:
                pass
        return output

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None) -> Any:
        return self._attach_connector(super().execute_model(scheduler_output, intermediate_tensors))

    def sample_tokens(self, grammar_output: Any = None) -> Any:
        return self._attach_connector(super().sample_tokens(grammar_output))


class OmniTPUGenerationModelRunner(OmniTPUModelRunner):
    """TPU Generation Model Runner stub for Stage 2 audio/vocoder generation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vocoder_step = VocoderStep()

    def get_kv_cache_spec(self) -> dict[str, Any]:
        return {}

    def execute_model(self, scheduler_output: Any, intermediate_tensors: Any = None) -> Any:
        # TODO(qwen3-omni-stage2): Implement non-AR generation runner for Code2Wav stage.
        raise NotImplementedError(
            "OmniTPUGenerationModelRunner.execute_model is not implemented for Thinker-only stage on TPU."
        )


TPUOmniModelRunner = OmniTPUARModelRunner
