# SPDX-License-Identifier: Apache-2.0
"""TPU Workers for vLLM-Omni."""

from typing import Any

try:
    from vllm_omni.worker.mixins import OmniWorkerMixin
except (ImportError, ModuleNotFoundError):

    class OmniWorkerMixin:  # type: ignore[no-redef]
        pass

from vllm_torchtpu.omni.patches import apply_omni_tpu_patches
from vllm_torchtpu.omni.runner import (
    OmniTPUARModelRunner,
    OmniTPUGenerationModelRunner,
)
from vllm_torchtpu.worker.tpu_worker import TPUWorker


class TPUARWorker(OmniWorkerMixin, TPUWorker):
    """TPU AR worker for Omni Thinker stage."""

    model_runner_cls = OmniTPUARModelRunner

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_omni_tpu_patches()
        super().__init__(*args, **kwargs)


class TPUGenerationWorker(OmniWorkerMixin, TPUWorker):
    """TPU Generation worker stub for Stage 2 audio/vocoder decoding."""

    model_runner_cls = OmniTPUGenerationModelRunner

    def determine_available_memory(self) -> int:
        # TODO(qwen3-omni-stage2): Implement KV-cache-free memory accounting for Code2Wav stage.
        raise NotImplementedError(
            "TPUGenerationWorker.determine_available_memory is not implemented for Thinker-only stage on TPU."
        )

    def initialize_from_config(self, kv_cache_Config: Any) -> None:
        # TODO(qwen3-omni-stage2): Implement non-KV initialization for Code2Wav stage.
        raise NotImplementedError(
            "TPUGenerationWorker.initialize_from_config is not implemented for Thinker-only stage on TPU."
        )

    def compile_or_warm_up_model(self) -> None:
        # TODO(qwen3-omni-stage2): Implement vocoder compilation/warmup for Code2Wav stage.
        raise NotImplementedError(
            "TPUGenerationWorker.compile_or_warm_up_model is not implemented for Thinker-only stage on TPU."
        )


TPUOmniWorker = TPUARWorker
