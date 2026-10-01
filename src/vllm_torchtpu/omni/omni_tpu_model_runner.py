# SPDX-License-Identifier: Apache-2.0
"""Base TPU Model Runner for Omni models."""

from typing import Any

import torch

from vllm_torchtpu.omni.qwen3_omni_moe_thinker_patch import apply_omni_ar_patches
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class OmniTPUModelRunner(TPUModelRunner):
    """Shared base for every Omni Runner on TPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self._omni_last_model_output: Any = None
        self._sampled_token_ids_cpu_override: Any = None
        self._omni_query_start_loc_model_kwarg: bool = False

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        apply_omni_ar_patches()
        super().load_model(*args, **kwargs)
        model = getattr(self, "model", None)
        if bool(getattr(model, "supports_sampled_token_ids_cpu_override", False)):
            candidate = getattr(model, "consume_sampled_token_ids_cpu_override", None)
            if callable(candidate):
                self._sampled_token_ids_cpu_override = candidate
        self._omni_query_start_loc_model_kwarg = bool(
            getattr(model, "supports_omni_query_start_loc", False)
        )
        if hasattr(self, "init_omni_connectors") and hasattr(self, "model_config"):
            self.init_omni_connectors(self.model_config)

    def _to_host(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu").contiguous()
        if isinstance(value, dict):
            return {k: self._to_host(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_host(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._to_host(v) for v in value)
        return value

    def _update_states(self, scheduler_output: Any) -> Any:
        cleanup = (
            getattr(self, "cleanup_finished_request", None)
            if getattr(self, "_omni_connector_initialized", False)
            else None
        )
        for req_id in getattr(scheduler_output, "finished_req_ids", ()):
            self.model_intermediate_buffer.pop(req_id, None)
            if cleanup is not None:
                cleanup(req_id)
        return super()._update_states(scheduler_output)

    def build_omni_output(self, output: Any) -> Any:
        return output
