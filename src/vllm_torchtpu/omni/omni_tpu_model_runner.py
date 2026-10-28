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
"""Base TPU Model Runner for Omni models."""

from typing import Any

import torch
from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.worker.omni_connector_model_runner_mixin import (
    OmniConnectorModelRunnerMixin,
    needs_omni_connector,
)

from vllm_torchtpu.omni.qwen3_omni_moe_talker_patch import apply_omni_talker_patches
from vllm_torchtpu.omni.qwen3_omni_moe_thinker_patch import apply_omni_ar_patches
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _decode_additional_info(info: Any) -> dict[str, Any]:
    if not info:
        return {}
    if isinstance(info, dict):
        return info
    decoded = deserialize_additional_information(info)
    return decoded if isinstance(decoded, dict) else {}


class OmniTPUModelRunner(OmniConnectorModelRunnerMixin, TPUModelRunner):
    """Shared base for every Omni runner on TPU.

    Owns the Omni connector lifecycle (initialisation after the model loads,
    per-request cleanup) and per-request ``model_intermediate_buffer`` state so
    that the AR runner, and the generation runner that follows, only add their
    stage-specific hooks. Mirrors upstream, where the mixin sits on both
    GPUARModelRunner and GPUGenerationModelRunner.
    """

    # The mixin only annotates this attribute; ``init_omni_connectors`` sets it.
    _omni_connector_initialized: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}

    @property
    def _intermediate_buffer(self) -> dict[str, dict[str, Any]]:
        buf = self.__dict__.get("model_intermediate_buffer")
        if buf is None:
            buf = {}
            self.model_intermediate_buffer = buf
        return buf

    def _store_buffer_value(
        self, dest: dict[str, Any], key: str, value: Any, resident_keys: set[str]
    ) -> None:
        if isinstance(value, torch.Tensor):
            t = value.detach()
            dest[key] = (
                t.clone()
                if key in resident_keys
                else (
                    t.cpu().contiguous() if t.device.type != "cpu" else t.contiguous()
                )
            )
        elif isinstance(value, list):
            dest[key] = [
                item.detach().cpu().contiguous()
                if isinstance(item, torch.Tensor)
                else item
                for item in value
            ]
        else:
            dest[key] = value

    def _update_intermediate_buffer(
        self, req_id: str, update_dict: dict[str, Any] | None
    ) -> None:
        if not isinstance(update_dict, dict) or not update_dict:
            return
        orig_model = getattr(
            getattr(self, "model", None), "_orig_mod", getattr(self, "model", None)
        )
        gpu_keys: set[tuple[str, str]] = getattr(
            orig_model, "gpu_resident_buffer_keys", set()
        )
        existing = self._intermediate_buffer.setdefault(req_id, {})
        for key, val in update_dict.items():
            if isinstance(val, dict):
                sub = existing.setdefault(key, {})
                sub_gpu_keys = {q for tk, q in gpu_keys if tk == key}
                for qual, sub_val in val.items():
                    self._store_buffer_value(sub, qual, sub_val, sub_gpu_keys)
            else:
                self._store_buffer_value(existing, key, val, set())

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        apply_omni_ar_patches()
        apply_omni_talker_patches()
        super().load_model(*args, **kwargs)
        # Same gate as upstream GPUARModelRunner: a stage owns connectors only
        # when it consumes a full payload, produces inter-stage data
        # (custom_process_next_stage_input_func) or is given an explicit
        # connector role. A plain stage must not start the connector threads.
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(self.model_config)

    def _update_states(self, scheduler_output: Any) -> Any:
        finished_reqs = getattr(scheduler_output, "finished_req_ids", ())
        for req_id in finished_reqs:
            self._intermediate_buffer.pop(req_id, None)
            if self._omni_connector_initialized:
                self.cleanup_finished_request(req_id)
        res = super()._update_states(scheduler_output)
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
            req_id = getattr(new_req, "req_id", None)
            if req_id is None:
                continue
            self._intermediate_buffer[req_id] = {}
            self._update_intermediate_buffer(
                req_id,
                _decode_additional_info(
                    getattr(new_req, "additional_information", None)
                ),
            )
            self._update_intermediate_buffer(
                req_id,
                _decode_additional_info(
                    getattr(new_req, "model_intermediate_buffer", None)
                ),
            )
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        cached_infos = getattr(cached_reqs, "additional_information", None)
        if isinstance(cached_infos, dict):
            for req_id in getattr(cached_reqs, "req_ids", ()):
                self._update_intermediate_buffer(
                    req_id, _decode_additional_info(cached_infos.get(req_id))
                )
        return res
