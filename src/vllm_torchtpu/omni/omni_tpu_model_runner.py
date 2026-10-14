# SPDX-License-Identifier: Apache-2.0
"""Base TPU Model Runner for Omni models."""

from typing import Any

from vllm_torchtpu.omni.qwen3_omni_moe_thinker_patch import apply_omni_ar_patches
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class OmniTPUModelRunner(TPUModelRunner):
    """Shared base for every Omni Runner on TPU."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        apply_omni_ar_patches()
        super().load_model(*args, **kwargs)
        if hasattr(self, "init_omni_connectors") and hasattr(self, "model_config"):
            self.init_omni_connectors(self.model_config)

    def _update_states(self, scheduler_output: Any) -> Any:
        if getattr(self, "_omni_connector_initialized", False):
            for req_id in getattr(scheduler_output, "finished_req_ids", ()):
                self.cleanup_finished_request(req_id)
        return super()._update_states(scheduler_output)
