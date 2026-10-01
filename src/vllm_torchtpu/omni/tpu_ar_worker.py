# SPDX-License-Identifier: Apache-2.0
"""TPU Autoregressive (AR) Worker for Omni models."""

from vllm_omni.worker.mixins import OmniWorkerMixin

from vllm_torchtpu.omni.omni_tpu_ar_model_runner import OmniTPUARModelRunner
from vllm_torchtpu.worker.tpu_worker import TPUWorker


class TPUARWorker(OmniWorkerMixin, TPUWorker):
    """TPU AR worker for autoregressive stages (thinker, talker) in Omni models."""

    model_runner_cls = OmniTPUARModelRunner
